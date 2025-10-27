from pathlib import Path
import random
import torch
import numpy as np
import os
import pandas as pd
from utils.plots import plot_models
from torch.optim.lr_scheduler import StepLR


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


class Trainer:
    def __init__(
        self,
        model,
        loss_fn,
        epochs,
        optimizer,
        train_loader,
        val_loader,
        device,
        path_to_save_plots,
        path_to_save_models,
        path_to_save_tables,
        scheduler=None,
        train_denorm_fn=None,
        val_denorm_fn=None,
        seed=42
    ):
        self.model = model
        self.loss_fn = loss_fn
        self.epochs = epochs
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.scheduler = scheduler

        self.path_to_save_plots = Path(path_to_save_plots)
        self.path_to_save_models = Path(path_to_save_models)
        self.path_to_save_tables = Path(path_to_save_tables)

        self.train_denorm_fn = train_denorm_fn
        self.val_denorm_fn = val_denorm_fn

        set_seed(seed=seed)
        self.path_to_save_plots.mkdir(parents=True, exist_ok=True)
        self.path_to_save_models.mkdir(parents=True, exist_ok=True)
        self.path_to_save_tables.mkdir(parents=True, exist_ok=True)

        self.model.to(self.device)


    def train_step(self):
        train_loss = 0.0
        train_true = []
        train_pred = []

        self.model.train()
        for batch in self.train_loader:
            y = batch["vah"].to(self.device)                # [B, seq_len]
            c = batch["features"].to(self.device)           # [B, cond_dim]

            y_input = y[:, :-1]
            if random.random() < 0.5:     # 50% батчей — с шумом
                noise = torch.randn_like(y_input) * 0.05    # настройте scale!
                y_input = y_input + noise                   # вход (y_0 .. y_{T-2})
            y_target = y[:, 1:]                             # цель (y_1 .. y_{T-1})

            self.optimizer.zero_grad()
            y_hat = self.model(c, y_input)                  # [B, seq_len - 1]
            loss = self.loss_fn(y_hat, y_target)
            loss.backward()
            self.optimizer.step()

            train_loss += loss.item() * y.size(0)

            if len(train_true) == 0:
                train_true = y_target[0].detach().cpu().numpy()
                train_pred = y_hat[0].detach().cpu().numpy()

        train_loss /= len(self.train_loader.dataset)
        return train_true, train_pred, train_loss


    def val_step(self):
        val_loss = 0.0
        val_true = []
        val_pred = []

        self.model.eval()
        with torch.no_grad():
            for batch in self.val_loader:
                y = batch["vah"].to(self.device)
                c = batch["features"].to(self.device)

                y_input = y[:, :-1]
                y_target = y[:, 1:]

                y_hat = self.model(c, y_input)
                loss = self.loss_fn(y_hat, y_target)
                val_loss += loss.item() * y.size(0)

                if torch.isnan(y_hat).any():
                    print("NaN in y_hat during validation!")
                if torch.isnan(loss):
                    print("NaN in loss during validation!")

                if len(val_true) == 0:
                    val_true = y_target[0].detach().cpu().numpy()
                    val_pred = y_hat[0].detach().cpu().numpy()

        val_loss /= len(self.val_loader.dataset)
        return val_true, val_pred, val_loss


    def train_model(self):
        train_losses = []
        val_losses = []
        self.best_val_loss = float("inf")

        for epoch in range(self.epochs):
            train_true, train_pred, train_loss = self.train_step()
            val_true, val_pred, val_loss = self.val_step()

            train_losses.append(train_loss)
            val_losses.append(val_loss)

            # Визуализация
            if epoch % 10 == 0 or epoch == self.epochs - 1:
                if self.train_denorm_fn:
                    train_true_cva=self.train_denorm_fn(train_true)
                    train_pred_cva=self.train_denorm_fn(train_pred)
                    val_true_cva=self.train_denorm_fn(val_true)
                    val_pred_cva=self.train_denorm_fn(val_pred)
                else:
                    train_true_cva = train_true
                    train_pred_cva = train_pred
                    val_true_cva=val_true
                    val_pred_cva=val_pred
                plot_models(
                    epoch=epoch,
                    path_to_save=self.path_to_save_plots / f"epoch_{epoch:03d}.png",
                    train_true_cva=train_true_cva,
                    train_pred_cva=train_pred_cva,
                    val_true_cva=val_true_cva,
                    val_pred_cva=val_pred_cva,
                    train_loss=train_losses,
                    val_loss=val_losses
                )

            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                torch.save(self.model.state_dict(), self.path_to_save_models / "best_model.pt")

            print(f"Epoch {epoch:03d} — Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}, LR: {self.scheduler.get_last_lr()}")

            if self.scheduler:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()

        loss_df = pd.DataFrame({
            "train_loss": train_losses,
            "val_loss": val_losses
        })
        loss_df.to_csv(self.path_to_save_tables / "losses.csv", index=False)
