import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from src.models.ganconv import *
from src.train.trainer.TrainerGAN import Trainer
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
        test_dataset = CVADataset(val)

        train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
        val_loader = DataLoader(test_dataset, batch_size=256, shuffle=True)
        
        input_dim=80
        output_vah = 968

        generator = Generator(in_dim=input_dim, vah_dim=output_vah, desc_dim=41).to(device)
        discriminator = Discriminator(output_vah).to(device)

        optimizer_G = torch.optim.Adam(generator.parameters(), lr=1e-4, betas=(0.99, 0.899))
        optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=1e-5, betas=(0.99, 0.999))


        criterion = nn.BCELoss(reduction='sum')
        recon_loss = nn.L1Loss(reduction="sum")

        num_epochs=500

        trainer = Trainer(
            generator=generator,
            discriminator=discriminator,
            loss_fn=criterion,
            loss_recon=recon_loss,
            epochs=num_epochs,
            optimizer_gen=optimizer_G,
            optimizer_disc=optimizer_D,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            path_to_save_plots=os.path.join(get_project_path(), "reports", "gan_conv", inh_name),
            path_to_save_models=os.path.join(get_project_path(), "models", "gan_conv", inh_name),
            path_to_save_tables=os.path.join(get_project_path(), "reports", "gan_conv", inh_name),
            seed=42
        )
        
        trainer.train_model()