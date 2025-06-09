import os
import sys
sys.path.append('..')
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import *
from sklearn.preprocessing import MinMaxScaler
import optuna
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from src.utils.paths import get_project_path
from src.data.data_preprocessing import get_all_data


def seed_everything(seed: int):
    import random, os
    import numpy as np
    import torch
    
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


# Определение вашего кастомного Dataset
class CVDataset(Dataset):
    def __init__(self, dataframe):
        self.data = dataframe
        self.features = self.data.values
        self.targets = None

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = torch.tensor(self.features[idx], dtype=torch.float32)
        return x

class VAE(nn.Module):
    def __init__(self, input_dim, latent_dim, num_layers):
        super(VAE, self).__init__()
        
        encoder_dims = np.linspace(input_dim, latent_dim * 2, num_layers + 1, dtype=int)
        decoder_dims = np.linspace(latent_dim, input_dim, num_layers + 1, dtype=int)

        encoder_layers = []
        for i in range(num_layers):
            encoder_layers.append(nn.Linear(encoder_dims[i], encoder_dims[i+1]))
        self.encoder_layers = nn.ModuleList(encoder_layers)
        
        self.encoder_mean = nn.Linear(encoder_dims[-1], latent_dim)
        self.encoder_logvar = nn.Linear(encoder_dims[-1], latent_dim)
        
        decoder_layers = [nn.Linear(latent_dim, decoder_dims[1])]
        for i in range(1, num_layers):
            decoder_layers.append(nn.Linear(decoder_dims[i], decoder_dims[i+1]))
        self.decoder_layers = nn.ModuleList(decoder_layers)

    def encode(self, x):
        h = x
        for layer in self.encoder_layers:
            h = torch.relu(layer(h))
        
        mean = self.encoder_mean(h)
        logvar = self.encoder_logvar(h)
        return mean, logvar

    def reparameterize(self, mean, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mean + eps * std

    def decode(self, z):
        h = z
        for i, layer in enumerate(self.decoder_layers):
            if i < len(self.decoder_layers) - 1:
                h = torch.relu(layer(h))
            else:
                h = layer(h)
        return h

    def forward(self, x):
        mean, logvar = self.encode(x)
        z = self.reparameterize(mean, logvar)
        return self.decode(z), mean, logvar
    


def vae_loss(recon_x, x, mean, logvar):
    l1_loss = nn.L1Loss(reduction='mean')
    recon_loss = l1_loss(recon_x, x)
    
    kl_loss = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp())
    
    return recon_loss + kl_loss


def objective(trial):
    # Гиперпараметры для оптимизации
    latent_dim = trial.suggest_int('latent_dim', 16, 256, step=16)
    num_layers = trial.suggest_int('num_layers', 2, 5)
    learning_rate = trial.suggest_float('learning_rate', 1e-6, 1e-3, log=True)
    max_lr = trial.suggest_float('max_lr', 1e-4, 1e-2, log=True)
    batch_size = trial.suggest_categorical('batch_size', [16, 32, 64, 128])
    epochs = 1000

    seed_everything(42)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    input_dim = benz_2.shape[1]

    vae = VAE(input_dim, latent_dim, num_layers).to(device)
    optimizer = optim.Adam(vae.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.CyclicLR(optimizer, base_lr=learning_rate, max_lr=max_lr,
                                            step_size_up=50, step_size_down=50)

    best_loss = float('inf')

    for epoch in range(epochs):
        vae.train()
        total_loss = 0.0
        for features in train_dataloader:
            features = features.to(torch.float32).to(device)
            optimizer.zero_grad()
            recon_batch, mean, logvar = vae(features)
            loss = vae_loss(recon_batch, features, mean, logvar)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        # Валидация
        vae.eval()
        val_loss = 0.0
        with torch.no_grad():
            for features in val_dataloader:
                features = features.to(torch.float32).to(device)
                recon_batch, mean, logvar = vae(features)
                loss = vae_loss(recon_batch, features, mean, logvar)
                val_loss += loss.item()

        avg_train_loss = total_loss / len(train_dataloader.dataset)
        avg_val_loss = val_loss / len(val_dataloader.dataset)

        # Шаг шедулера
        scheduler.step(avg_val_loss)

        # Проверка улучшения
        if avg_val_loss < best_loss:
            best_loss = avg_val_loss

        # Можем добавить возможность прерывания оптимизации, если loss не улучшается
        trial.report(best_loss, epoch)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    return best_loss


if __name__ == '__main__':
    volt, cur = get_all_data()

    scal = MinMaxScaler()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    for inh in ['benzotriazole', '2-mercaptobenzimidazole', 'benzimidazole', '4-benzylpiperidine']:

        benz = cur[cur['Inhibitor'] == inh]
        benz = benz.iloc[[idx for idx in range(4, len(benz), 5)]]

        benz_2 = benz.drop(columns=['Inhibitor', 'ppm'])
        benz_2 = pd.DataFrame(scal.fit_transform(benz_2.reset_index(drop=True).T)).T

        train, val = train_test_split(benz_2, test_size=0.2, shuffle=True, random_state=42)

        train_dataset = CVDataset(train)
        val_dataset = CVDataset(val)

        input_dim = benz_2.shape[1]

        study = optuna.create_study(
            directions=["minimize"],
            study_name=f"{inh}_vae",
            pruner=optuna.pruners.MedianPruner(),
            load_if_exists=True,
            storage=f"sqlite:///vae_opt.db")

        study.optimize(objective, n_trials=1000)
