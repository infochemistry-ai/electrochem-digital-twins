import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, in_dim, vah_dim, desc_dim):
        super(Generator, self).__init__()

        self.input_dim = in_dim + desc_dim
        self.vah_dim = vah_dim

        self.fc = nn.Linear(self.input_dim, 128 * 16)
        self.net = nn.Sequential(
            nn.LeakyReLU(0.2),

            nn.ConvTranspose1d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.2),

            nn.ConvTranspose1d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.2),

            nn.ConvTranspose1d(32, 16, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(16),
            nn.LeakyReLU(0.2),

            nn.ConvTranspose1d(16, 1, kernel_size=4, stride=2, padding=1),
        )

        self.upsample = nn.Upsample(size=vah_dim, mode='linear')

    def forward(self, x, z):
        d = torch.cat([x, z], dim=2)
        x = self.fc(d)
        x = x.view(x.size(0), 128, 16)
        x = self.net(x)
        x = self.upsample(x)
        return x.squeeze(1)


class Discriminator(nn.Module):
    def __init__(self, vah_dim):
        super(Discriminator, self).__init__()

        self.net = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            
            nn.Conv1d(16, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.2),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * (vah_dim // 4), 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.net(x)
        return self.classifier(x)
