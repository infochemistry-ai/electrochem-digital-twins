import torch
import torch.nn as nn
import torch.nn.functional as F

class VAE_LSTM(nn.Module):
    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        hidden_dim: int,
        num_layers: int,
        seq_len: int,
        desc_dim: int,
    ):
        super().__init__()
        self.seq_len    = seq_len
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.input_dim  = input_dim

        self.encoder_lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=False,
        )
        self.fc_mu     = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

        self.init_h = nn.Linear(latent_dim + desc_dim, num_layers * hidden_dim)

        self.decoder_lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
        )
        self.output_layer = nn.Linear(hidden_dim, input_dim)

    def encode(self, x):
        _, (h_n, _) = self.encoder_lstm(x)
        h_last = h_n[-1]
        mu     = self.fc_mu(h_last)
        logvar = self.fc_logvar(h_last)
        return mu, logvar

    @staticmethod
    def reparameterize(mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z, desc, x=None, teacher_forcing_ratio=1.0):
        B = z.size(0)
        h0c0 = self.init_h(torch.cat([z, desc], dim=1))
        h0 = h0c0.view(self.num_layers, B, self.hidden_dim)
        c0 = torch.zeros_like(h0)

        if x is not None and teacher_forcing_ratio > 0.5:
            outputs, _ = self.decoder_lstm(x, (h0, c0))
            return self.output_layer(outputs)

        prev = torch.zeros(B, 1, self.input_dim, device=z.device)
        outputs = []
        h, c = h0, c0
        for t in range(self.seq_len):
            out, (h, c) = self.decoder_lstm(prev, (h, c))
            step = self.output_layer(out)
            outputs.append(step)
            if x is not None and torch.rand(1).item() < teacher_forcing_ratio:
                prev = x[:, t:t+1, :]
            else:
                prev = step
        return torch.cat(outputs, dim=1)

    def forward(self, x, desc, teacher_forcing_ratio=1.0):
        x_in = x.unsqueeze(-1)            # (B, seq_len, 1)
        mu, logvar = self.encode(x_in)
        z          = self.reparameterize(mu, logvar)
        x_recon    = self.decode(z, desc, x_in, teacher_forcing_ratio)
        return x_recon, mu, logvar


def vae_loss(recon_x, x, mu, logvar, beta=1.0):
    x = x.unsqueeze(-1)
    recon_loss = F.l1_loss(recon_x, x, reduction='sum')
    kld = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + beta * kld
