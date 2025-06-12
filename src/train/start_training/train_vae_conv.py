import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from src.models.vaeconv import *
from src.train.trainer.TrainerVAE import Trainer
from src.data.preprocessing.pipeline import Pipeline
from src.data.datasets.universal_dataset import CVADataset
from src.data.preprocessing.splitter import select_test_inh
from src.utils.paths import get_project_path


def load_data(drop_inhib: str):
    prep = Pipeline(
        num_cycle=[1, 2, 3, 4], 
        inhibitor_name="all", 
        split="all",
        norm_feat=True
    )
    data = prep.full_data

    train_data, valid_data = select_test_inh(data, drop_inhib)
    
    return train_data, valid_data


if "__main__" == __name__:

    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    for inh_name in ["2-mercaptobenzimidazole", "4-benzylpiperidine", "benzimidazole", "benzotriazole"]:
    
        train, val = load_data(drop_inhib=inh_name)
        
        train_dataset = CVADataset(train)
        val_dataset = CVADataset(val)

        train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=256, shuffle=True)
        
        vae = VAE_CONV(
            in_channels=1,
            latent_dim=64,
            seq_len=968,
            desc_dim=41
        ).to(device)

        opt_vae = torch.optim.Adam(vae.parameters(), lr=5e-5)
        num_epoch = 500

        trainer = Trainer(
            model=vae,
            loss_fn=vae_loss,
            epochs=num_epoch,
            optimizer=opt_vae,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            path_to_save_plots=os.path.join(get_project_path(), "reports", "vae_conv", inh_name),
            path_to_save_models=os.path.join(get_project_path(), "models", "vae_conv", inh_name, "best_model.pt"),
            path_to_save_tables=os.path.join(get_project_path(), "reports", "vae_conv", inh_name),
            seed=42
        )
        
        trainer.train_model()