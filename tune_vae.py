import os
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import ray
from ray import tune
from ray.air import session
from ray.air.integrations.wandb import WandbLoggerCallback
import ray.train.torch

import tempfile
from dotenv import load_dotenv
from src.data.preprocessing.pipeline import Pipeline
from src.data.datasets.universal_dataset import VAEDataset
from src.models.vae_mlp.VAE_MLP import *
from src.utils.paths import get_project_path

load_dotenv()

def tune_vae(config):
    # wandb = None

    # import wandb
    # wandb.login(key=os.getenv("WANDB_KEY"), anonymous="never")
    # wandb.init(
    #     project="vae-cva-leaky-relu-train-val-test",
    #     entity="vinavolo-itmo",
    #     config=config,
    #     tags=["hyperopt", "vae"],
    #     group="ray-tune-experiment"
    # )

    prep = Pipeline(
        num_cycle=[0, 1, 2, 3, 4],
        inhibitor_name="all",
        split="train"
    )

    vol = prep.vol_orig
    cur = prep.cur_orig

    prep_test = Pipeline(
        num_cycle=[0, 1, 2, 3, 4],
        inhibitor_name="all",
        split="test"
    )

    vol_test = prep_test.vol_orig
    cur_test = prep_test.cur_orig

    torch.manual_seed(42)
    np.random.seed(42)

    train_cur, val_cur = train_test_split(cur, test_size=0.1, shuffle=True, random_state=42)

    train_vol = vol.iloc[train_cur.index]
    val_vol = vol.iloc[val_cur.index]

    train_dataset = VAEDataset(train_vol, train_cur)
    val_dataset = VAEDataset(val_vol, val_cur)
    test_dataset = VAEDataset(vol_test, cur_test)

    train_dataloader = DataLoader(train_dataset, batch_size=256, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    input_dim = train_cur.shape[1]
    latent_dim = config["latent_dim"]
    num_layers = config["num_layers"]
    learning_rate = config["learning_rate"]
    epochs = 2000

    best_loss = float('inf')

    vae = VAE(input_dim, latent_dim, num_layers).to(ray.train.torch.get_device())
    optimizer = optim.Adam(
        vae.parameters(), 
        lr=learning_rate, 
        betas=(config['beta1'], config['beta2']), 
        weight_decay=config['weight_decay']
    )

    for epoch in range(epochs):
        vae.train()
        total_train_loss = 0
        for features in train_dataloader:
            features = features.to(torch.float32).to(ray.train.torch.get_device())
            optimizer.zero_grad()
            recon_batch, mean, logvar = vae(features)
            loss = vae_loss(recon_batch, features, mean, logvar)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()
        
        avg_train_loss = total_train_loss / len(train_dataloader.dataset)

        vae.eval()
        total_val_loss = 0
        total_test_loss = 0
        with torch.inference_mode():
            for features in val_dataloader:
                features = features.to(torch.float32).to(ray.train.torch.get_device())
                recon_batch, mean, logvar = vae(features)
                loss = vae_loss(recon_batch, features, mean, logvar)
                total_val_loss += loss.item()
            
            for features_test in test_dataloader:
                features_test = features_test.to(torch.float32).to(ray.train.torch.get_device())
                recon_batch_test, mean_test, logvar_test = vae(features_test)
                loss_test = vae_loss(recon_batch_test, features_test, mean_test, logvar_test)
                total_test_loss += loss_test.item()

        avg_val_loss = total_val_loss / len(val_dataloader.dataset)
        avg_test_loss = total_test_loss / len(test_dataloader.dataset)
            
        if avg_val_loss < best_loss:
            best_loss = avg_val_loss

        metrics = {
            "train_loss": avg_train_loss,
            "val_loss": avg_val_loss,
            "test_loss": avg_test_loss,
            "best_val_loss": best_loss,
            "epoch": epoch
        }
        
        # if wandb:
        #     wandb.log(metrics)

        session.report(metrics)

        print(f"Epoch {epoch+1}: Train Loss {avg_train_loss:.4f}, Val Loss {avg_val_loss:.4f}")


if __name__ == "__main__":
    ray.init(ignore_reinit_error=True)

    config = {
        "latent_dim": tune.choice([int(i)**2 for i in range(4, 23)]),
        "num_layers": tune.choice([int(i)*2 for i in range(1, 17)]),
        "learning_rate": tune.loguniform(1e-6, 1e-3),
        "beta1": tune.loguniform(0.99, 0.29),
        "beta2": tune.loguniform(0.99, 0.29),
        "weight_decay": tune.loguniform(1e-6, 1e-2)
    }
    
    analysis = tune.run(
        tune_vae,
        config=config,
        resources_per_trial={"cpu": 20,"gpu": 1},
        num_samples=75,
        callbacks=[
            WandbLoggerCallback(
                api_key=os.getenv("WANDB_KEY"),
                project="vae-cva-leaky-relu-train-val-test",
                entity="vinavolo-itmo",
                upload_checkpoints=True,
                log_config=True
            )
        ],

        metric="val_loss",
        mode="min",
        storage_path=os.path.join(get_project_path(), "reports", "vae", "checkpoints", "vae-cva-leaky-relu-train-val-test")
    )

    print("Best config found:", analysis.get_best_config(metric="val_loss", mode="min"))

    