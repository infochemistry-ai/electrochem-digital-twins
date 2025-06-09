import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, in_dim, vah_dim, desc_dim):
        super(Generator, self).__init__()

        self.input_dim = in_dim + desc_dim

        self.gen = nn.Sequential(

            nn.Linear(self.input_dim, 128),
            nn.LeakyReLU(),

            nn.Linear(128, 256),
            nn.LeakyReLU(),

            nn.Linear(256, 512),
            nn.LeakyReLU(),
            
            nn.Linear(512, 742),
            nn.LeakyReLU(),
            
            nn.Linear(742, vah_dim),
        )

    def forward(self, x, z):
        d = torch.cat([x, z], dim=1)
        return self.gen(d)


class Discriminator(nn.Module):
    def __init__(self, vah_dim):
        super(Discriminator, self).__init__()

        self.disc = nn.Sequential(
            nn.Linear(vah_dim, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(),
            
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(),
            
            nn.Linear(256, 64),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(),
            
            # nn.Linear(128, 64),
            # nn.LayerNorm(64),
            # nn.LeakyReLU(),
            
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.disc(x)
