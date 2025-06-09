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
        self.seq_len = seq_len
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

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
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
        )

        self.output_layer = nn.Linear(hidden_dim, input_dim)

    def encode(self, x: torch.Tensor):
        _, (h_n, _) = self.encoder_lstm(x)
        h_last = h_n[-1]
        mu     = self.fc_mu(h_last)
        logvar = self.fc_logvar(h_last)
        return mu, logvar

    @staticmethod
    def reparameterize(mu: torch.Tensor, logvar: torch.Tensor):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor, desc: torch.Tensor):
        init = torch.cat([z, desc], dim=1)
        h0c0 = self.init_h(init)
        h0 = h0c0.view(self.num_layers, -1, self.hidden_dim).contiguous()
        c0 = torch.zeros_like(h0)
        dec_input = h0[0].unsqueeze(1).repeat(1, self.seq_len, 1)

        outputs, _ = self.decoder_lstm(dec_input, (h0, c0))
        x_recon = self.output_layer(outputs)
        return x_recon

    def forward(self, x: torch.Tensor, desc: torch.Tensor):
        x = x.unsqueeze(-1)
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decode(z, desc)
        return x_recon, mu, logvar


def vae_loss(
    recon_x: torch.Tensor,
    x: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    beta: float = 1.0
):
    x = x.unsqueeze(-1)
    recon_loss = F.l1_loss(recon_x, x, reduction='sum')
    kld = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + beta * kld
