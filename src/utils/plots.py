import math
import matplotlib.pyplot as plt

def plot_interpolations(vol, cur, vol_inter, cur_inter):
    n = len(vol)
    n_cols = 5
    n_rows = math.ceil(n / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 4 * n_rows))
    if n_rows == 1:
        axes = axes.flatten()
    else:
        axes = axes.flatten()
    
    for i in range(n):
        ax = axes[i]
        ax.plot(vol[i], cur[i], label="Original")
        ax.plot(vol_inter[i], cur_inter[i], label="Interpolated")
        ax.set_title(f"Index {i}")
        ax.legend()
    
    for j in range(n, len(axes)):
        fig.delaxes(axes[j])
    
    plt.tight_layout()
    plt.show()

def plot_valid(vol, cur, vol_inter, cur_inter):
    n = len(vol)
    n_cols = 5
    n_rows = math.ceil(n / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 4 * n_rows))
    if n_rows == 1:
        axes = axes.flatten()
    else:
        axes = axes.flatten()
    
    for i in range(n):
        ax = axes[i]
        ax.plot(vol[i], cur[i], label="Original")
        ax.plot(vol_inter[i], cur_inter[i], label="Reduction")
        ax.set_title(f"Index {i}")
        ax.legend()
    
    for j in range(n, len(axes)):
        fig.delaxes(axes[j])
    
    plt.tight_layout()
    plt.show()


def plot_full(vol, cur, num_columns):
    n = len(vol)
    n_cols = num_columns
    n_rows = math.ceil(n / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 4 * n_rows))
    if n_rows == 1:
        axes = axes.flatten()
    else:
        axes = axes.flatten()
    
    for i in range(n):
        ax = axes[i]
        ax.plot(vol[i], cur[i])
        ax.set_title(f"Index {i}")
        ax.legend()
    
    for j in range(n, len(axes)):
        fig.delaxes(axes[j])
    
    plt.tight_layout()
    plt.show()


def plot_models(
    epoch,
    path_to_save,
    train_true_cva, 
    train_pred_cva, 
    val_true_cva, 
    val_pred_cva, 
    train_loss, 
    val_loss
    ):
        
        fig, axs = plt.subplots(ncols=2, nrows=2, figsize=(14, 10))

        axs[0, 0].plot(train_true_cva, "r",)
        axs[0, 0].set_title("Train results for epoch {}".format(epoch))
        axs[0, 0].scatter(range(len(train_pred_cva)), train_pred_cva, s=1.5)
        axs[0, 0].set_xlabel("True values")
        axs[0, 0].set_ylabel("Predicted values")

        axs[0, 1].plot(val_true_cva,"r",)
        axs[0, 1].set_title("Test results for epoch {}".format(epoch))
        axs[0, 1].scatter( range(len(val_pred_cva)), val_pred_cva, s=1.5)
        axs[0, 1].set_xlabel("True values")
        axs[0, 1].set_ylabel("Predicted values")

        axs[1, 0].plot(train_loss, label="Train loss")
        axs[1, 0].plot(val_loss, label="Test loss")
        axs[1, 0].set_ylabel("EBLOSS Loss")
        axs[1, 0].set_xlabel("Epochs")
        axs[1, 0].legend()

        plt.tight_layout()
        plt.savefig(path_to_save)
        plt.close(fig)
        
        
def plot_models_gan(
    epoch,
    path_to_save,
    real_vah,
    gen_vah,
    train_disc_losses,
    val_disc_losses,
    train_gen_losses,
    val_gen_losses,
    train_rec_losses,
    val_rec_losses,
    ):
        
    fig, axs = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(f'Training Progress - Epoch {epoch+1}')
        
    axs[0, 0].plot(real_vah.detach().cpu().numpy()[0], label='Real VAH', alpha=0.7)
    axs[0, 0].plot(gen_vah.detach().cpu().numpy()[0], label='Generated VAH', alpha=0.7)
    axs[0, 0].set_title('VAH Comparison')
    axs[0, 0].legend()
    axs[0, 0].grid(True)

    axs[0, 1].plot(train_disc_losses, label='Train D Loss')
    axs[0, 1].plot(val_disc_losses, label='Val D Loss', linestyle='--')
    axs[0, 1].set_title('Discriminator Loss')
    axs[0, 1].legend()
    axs[0, 1].grid(True)
    
    axs[1, 0].plot(train_gen_losses, label='Train G Loss')
    axs[1, 0].plot(val_gen_losses, label='Val G Loss', linestyle='--')
    axs[1, 0].set_title('Generator Loss')
    axs[1, 0].legend()
    axs[1, 0].grid(True)
    
    axs[1, 1].plot(train_rec_losses, label='Train Rec Loss')
    axs[1, 1].plot(val_rec_losses, label='Val Rec Loss', linestyle='--')
    axs[1, 1].set_title('Recon Loss')
    axs[1, 1].legend()
    axs[1, 1].grid(True)

    plt.tight_layout()
    plt.savefig(path_to_save)
    plt.close(fig)