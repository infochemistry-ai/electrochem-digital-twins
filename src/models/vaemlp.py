import torch
import torch.nn as nn

class VAE(nn.Module):
    def __init__(self, input_dim, latent_dim, desc_dim):
        super().__init__()
        
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 512), 
            nn.LeakyReLU(),
            
            nn.Linear(512, 256),
            nn.LeakyReLU(),
            
            nn.Linear(256, 162),       
            nn.LeakyReLU(),
            
            nn.Linear(162, 120),       
            nn.LeakyReLU(),
        )
        
        self.enc_mean   = nn.Linear(120, latent_dim)
        self.enc_logvar = nn.Linear(120, latent_dim)

        self.decoder_pre = nn.Sequential(
            nn.Linear(latent_dim + desc_dim, 120),
            nn.LeakyReLU(),
            
            nn.Linear(120, 162),     
            nn.LeakyReLU()
        )
        
        self.vah_head = nn.Sequential(
            nn.Linear(162, 256),  
            nn.LeakyReLU(),
                                
            nn.Linear(256, 512),      
            nn.LeakyReLU(),
            
            nn.Linear(512, input_dim),
            
            )

    def encode(self, x):
        h = self.encoder(x)
        return self.enc_mean(h), self.enc_logvar(h)

    def reparameterize(self, mean, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mean + eps * std

    def decode(self, z, desc):
        h = torch.cat([z, desc], dim=1)
        h = self.decoder_pre(h)
        
        vah_hat = self.vah_head(h)
        return vah_hat

    def forward(self, x, desc):
        m, lv = self.encode(x)
        z = self.reparameterize(m, lv)
        return self.decode(z, desc), m, lv
    
    
def vae_loss(vah_hat, vah, mean, logvar):
    recon_vah = nn.functional.l1_loss(vah_hat, vah, reduction="sum")
    kl = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp())
    return recon_vah + kl
