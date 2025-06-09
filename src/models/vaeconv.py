import torch
import torch.nn as nn
import torch.nn.functional as F

class VAE_CONV(nn.Module):
    def __init__(self, in_channels: int, latent_dim: int, seq_len: int, desc_dim: int):
        super().__init__()
        self.seq_len = seq_len

        self.encoder = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(True),

            nn.Conv1d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(True),

            nn.Conv1d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(True)
        )

        with torch.no_grad():
            dummy = torch.zeros(1, in_channels, seq_len)
            enc_out = self.encoder(dummy)
        self.enc_out_channels = enc_out.size(1)
        self.enc_out_len      = enc_out.size(2)
        self.flatten_dim      = self.enc_out_channels * self.enc_out_len

        self.fc_mu     = nn.Linear(self.flatten_dim, latent_dim)
        self.fc_logvar = nn.Linear(self.flatten_dim, latent_dim)

        self.decoder_input = nn.Sequential(
            nn.Linear(latent_dim + desc_dim, latent_dim*2),
            nn.LeakyReLU(),
            
            nn.Linear(latent_dim*2, latent_dim*4),
            nn.LeakyReLU(),
            
            nn.Linear(latent_dim*4, latent_dim*6),
            nn.LeakyReLU(),
            
            nn.Linear(latent_dim*6, self.flatten_dim),
            nn.LeakyReLU(),
            
            )

        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(128, 64, kernel_size=3, stride=2,
                               padding=1, output_padding=1),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(True),

            nn.ConvTranspose1d(64, 32, kernel_size=3, stride=2,
                               padding=1, output_padding=1),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(True),
        )

        self.final_layer = nn.ConvTranspose1d(
            32, in_channels, kernel_size=3, stride=2,
            padding=1, output_padding=1
        )

    def encode(self, x: torch.Tensor):        
        x = self.encoder(x)
        x = torch.flatten(x, start_dim=1)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar

    @staticmethod
    def reparameterize(mu: torch.Tensor, logvar: torch.Tensor):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z, desc):
        h = torch.cat([z, desc], dim=1)
        
        x = self.decoder_input(h)
        x = x.view(h.size(0), self.enc_out_channels, self.enc_out_len)
        x = self.decoder(x)
        return self.final_layer(x)

    def forward(self, x: torch.Tensor, desc: torch.Tensor):
        x = x.unsqueeze(1)
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decode(z, desc)
        return x_recon, mu, logvar


def vae_loss(recon_x: torch.Tensor, x: torch.Tensor, mu: torch.Tensor, logvar: torch.Tensor, beta: float = 1.0):
    """Базовая β-VAE потеря (MSE + β * KL)."""
    x = x.unsqueeze(1)
    recon_loss = F.l1_loss(recon_x, x, reduction='sum')
    kld = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + beta * kld