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