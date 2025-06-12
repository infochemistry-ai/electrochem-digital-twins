from pathlib import Path
import random
import torch
import numpy as np
import pandas as pd
from src.utils.plots import plot_models

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
        model, 
        loss_fn,
        epochs, 
        optimizer, 
        train_loader, 
        val_loader, 
        device,
        path_to_save_plots,
        path_to_save_models,
        path_to_save_tables,
        seed
    ):
        
        self.model = model
        self.loss_fn = loss_fn
        self.epochs = epochs
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.path_to_save_plots = path_to_save_plots
        self.path_to_save_models = path_to_save_models
        self.path_to_save_tables = path_to_save_tables
        
        set_seed(seed=seed)
        
        
    def train_step(self):
        train_true_vah = []
        train_predicted_vah = []
        train_loss = 0.0

        self.model.train()
        for batch in self.train_loader:
            vah, desc = batch["vah"].to(self.device), batch["features"].to(self.device)
            self.optimizer.zero_grad()
            vah_hat, m, lv = self.model(vah, desc)
            loss = self.loss_fn(vah_hat, vah, m, lv)
            loss.backward()
            self.optimizer.step()

            train_loss += loss.item()

            vah_np = vah.detach().cpu().numpy()
            pred_np = vah_hat.detach().cpu().numpy()
            if pred_np.ndim == 3:
                pred_np = pred_np.squeeze(-1)

        train_true_vah.extend(np.transpose(vah_np[0]))
        train_predicted_vah.extend(np.transpose(pred_np[0]))

        train_loss /= len(self.train_loader.dataset)
        return train_true_vah, train_predicted_vah, train_loss


    def val_step(self):
        val_true_vah = []
        val_predicted_vah = []
        val_loss = 0.0

        self.model.eval()
        with torch.no_grad():
            for batch in self.val_loader:
                vah, desc = batch["vah"].to(self.device), batch["features"].to(self.device)
                vah_hat, m, lv = self.model(vah, desc)
                val_loss += self.loss_fn(vah_hat, vah, m, lv).item()

                vah_np = vah.detach().cpu().numpy()
                pred_np = vah_hat.detach().cpu().numpy()
                if pred_np.ndim == 3:
                    pred_np = pred_np.squeeze(-1)

        val_true_vah.extend(np.transpose(vah_np[0]))
        val_predicted_vah.extend(np.transpose(pred_np[0]))

        val_loss /= len(self.val_loader.dataset)
        return val_true_vah, val_predicted_vah, val_loss
    
    
    def train_model(self):
        
        train_losses = []
        val_losses = []
        
        best_val_loss = float("inf")
        
        for epoch in range(self.epochs):
            
            train_true_cva, train_pred_cva, train_loss = self.train_step()
            val_true_cva, val_pred_cva, val_loss = self.val_step()
            
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            
            
            if epoch % 10 == 0:
            
                plot_models(
                    epoch,
                    Path(self.path_to_save_plots)/f"epoch_{epoch}.png",
                    train_true_cva, 
                    train_pred_cva, 
                    val_true_cva, 
                    val_pred_cva, 
                    train_losses, 
                    val_losses
                )
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(self.model.state_dict(), self.path_to_save_models)
                
            print(f"Epoch {epoch:03d} — Train: {train_loss:.4f}, Val: {val_loss:.4f}")
        
        pd.DataFrame({
            "Train_loss": train_losses,
            "Val_loss": val_losses
        }).reset_index(drop=True).to_csv(f"{self.path_to_save_tables}/losses.csv")