from pathlib import Path
import random
import torch
import numpy as np
import pandas as pd
from src.utils.plots import plot_models_gan

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

class Trainer:
    def __init__(
        self, 
        generator,
        discriminator, 
        loss_fn,
        loss_recon,
        epochs, 
        optimizer_gen,
        optimizer_disc, 
        train_loader, 
        val_loader, 
        device,
        path_to_save_plots,
        path_to_save_models,
        path_to_save_tables,
        seed,
        input_dim_gen_noise=80
    ):
        
        self.generator = generator
        self.discriminator = discriminator
        self.loss_fn = loss_fn
        self.loss_recon = loss_recon
        self.epochs = epochs
        self.optimizer_gen = optimizer_gen
        self.optimizer_disc = optimizer_disc
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.path_to_save_plots = path_to_save_plots
        self.path_to_save_models = path_to_save_models
        self.path_to_save_tables = path_to_save_tables
        self.input_dim_gen_noise = input_dim_gen_noise
        
        set_seed(seed=seed)
        
        
    def train_step(self):
    
        epoch_d_loss = 0
        epoch_g_loss = 0
        rec_loss = 0
        
        self.generator.train()
        self.discriminator.train()

        for i, batch in enumerate(self.train_loader):
            real_features = batch['features'].to(self.device)
            real_vah = batch['vah'].to(self.device)
            batch_size = real_vah.size(0)
            z = torch.randn(batch_size, self.input_dim_gen_noise).to(self.device)
            
            real_labels = torch.full((batch_size, 1), 0.9).to(self.device)  # Label smoothing
            fake_labels = torch.full((batch_size, 1), 0.1).to(self.device)

            # ==========================================
            #  Обучение генератора
            # ==========================================
            self.optimizer_gen.zero_grad()
            
            gen_vah = self.generator(z.unsqueeze(1), real_features.unsqueeze(1))
            fake_outputs = self.discriminator(gen_vah.unsqueeze(1))
            
            g_loss_adv = self.loss_fn(fake_outputs, real_labels)
            
            g_loss_adv = g_loss_adv + 0.05 * self.loss_recon(gen_vah, real_vah)
            
            g_loss_adv.backward()
            self.optimizer_gen.step()
            
            epoch_g_loss += g_loss_adv.item()
            rec_loss += self.loss_recon(gen_vah, real_vah).item()
            
            # ===========================================
            #  Обучение дискриминатора
            # ===========================================
            self.optimizer_disc.zero_grad()
            
            real_outputs = self.discriminator(real_vah.unsqueeze(1))
            d_loss_real = self.loss_fn(real_outputs, real_labels)
            
            gen_vah = self.generator(z.unsqueeze(1), real_features.unsqueeze(1))
            fake_outputs = self.discriminator(gen_vah.unsqueeze(1).detach())
            d_loss_fake = self.loss_fn(fake_outputs, fake_labels)
            
            d_loss = d_loss_fake + d_loss_real
            
            d_loss.backward()
            self.optimizer_disc.step()

            epoch_d_loss += d_loss.item()
            
        epoch_d_loss /= len(self.train_loader)
        epoch_g_loss /= len(self.train_loader)
        rec_loss /= len(self.train_loader)
        
        return epoch_d_loss, epoch_g_loss, rec_loss
        

    def val_step(self):
        
        val_d_loss = 0
        val_g_loss = 0
        val_rec_loss = 0
        
        self.generator.eval()
        self.discriminator.eval()
        
        with torch.no_grad():
            for val_batch in self.val_loader:
                real_features = val_batch['features'].to(self.device)
                real_vah = val_batch['vah'].to(self.device)
                batch_size = real_vah.size(0)
                z = torch.randn(batch_size, self.input_dim_gen_noise).to(self.device)

                real_labels = torch.full((batch_size, 1), 0.9).to(self.device)
                fake_labels = torch.full((batch_size, 1), 0.1).to(self.device)
                
                real_outputs = self.discriminator(real_vah.unsqueeze(1))
                gen_vah = self.generator(z.unsqueeze(1), real_features.unsqueeze(1))
                fake_outputs = self.discriminator(gen_vah.unsqueeze(1))
                
                d_loss_real = self.loss_fn(real_outputs, real_labels)
                d_loss_fake = self.loss_fn(fake_outputs, fake_labels)
                val_d_loss += (d_loss_real + d_loss_fake).item()
                
                val_gg_loss = self.loss_fn(fake_outputs, real_labels)
                val_gg_loss = val_gg_loss + 0.05 * self.loss_recon(gen_vah, real_vah)
                
                val_rec_loss += self.loss_recon(gen_vah, real_vah).item()
                
                val_g_loss += val_gg_loss.item()
        
        val_d_loss /= len(self.val_loader)
        val_g_loss /= len(self.val_loader)
        val_rec_loss /= len(self.val_loader)
            
        return val_d_loss, val_g_loss, val_rec_loss, gen_vah, real_vah
    
    
    def train_model(self):
        
        train_disc_losses, val_disc_losses = [], []
        train_gen_losses, val_gen_losses = [], []
        train_rec_losses, val_rec_losses = [], []
        
        for epoch in range(self.epochs):
            
            train_disc_loss, train_gen_loss, train_rec_loss = self.train_step()
            val_disc_loss, val_gen_loss, val_rec_loss, gen_vah, real_vah = self.val_step()
            
            train_disc_losses.append(train_disc_loss)
            train_gen_losses.append(train_gen_loss)
            train_rec_losses.append(train_rec_loss)
            
            val_disc_losses.append(val_disc_loss)
            val_gen_losses.append(val_gen_loss)
            val_rec_losses.append(val_rec_loss)
            
            if epoch % 10 == 0:
            
                plot_models_gan(
                    epoch=epoch,
                    path_to_save=Path(self.path_to_save_plots)/f"epoch_{epoch}.png",
                    gen_vah=gen_vah,
                    real_vah=real_vah,
                    train_disc_losses=train_gen_losses,
                    val_disc_losses=val_disc_losses,
                    train_gen_losses=train_gen_losses,
                    val_gen_losses=val_gen_losses,
                    train_rec_losses=train_rec_losses,
                    val_rec_losses=val_rec_losses
                )
                
            print(f"Epoch [{epoch:03d}] | D Loss: {train_disc_loss:.4f}({val_disc_loss:.4f}) | G Loss: {train_gen_loss:.4f}({val_gen_loss:.4f}) ")
            
        
        torch.save(self.generator.state_dict(), Path(self.path_to_save_models)/"best_generator_model.pt")
        torch.save(self.discriminator.state_dict(), Path(self.path_to_save_models)/"best_discriminator_model.pt")

        pd.DataFrame({
            "Train_D_Loss": train_disc_losses,
            "Val_D_Loss": val_disc_losses,
            "Train_G_Loss": train_gen_losses,
            "Val_D_Loss": val_gen_losses,
        }).reset_index(drop=True).to_csv(f"{self.path_to_save_tables}/losses.csv")