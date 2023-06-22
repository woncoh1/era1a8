import torch
import matplotlib.pyplot as plt


def inspect_batch(dataloader: torch.utils.data.DataLoader) -> None:
    """Visualize sample images from a batch."""
    batch_data, batch_label = next(iter(dataloader))
    fig = plt.figure()
    for i in range(12):
        plt.subplot(3, 4, i+1)
        plt.tight_layout()
        plt.imshow(batch_data[i].squeeze(0).permute(1, 2, 0))
        plt.title(batch_label[i].item())
        plt.xticks([])
        plt.yticks([])


def plot_lr(lrs:list[float], losses:list[float]) -> None:
    """Plot loss vs learning rate for the learning rate finder."""
    x = lrs
    y = losses
    plt.plot(x, y)
    plt.xscale('log')
    plt.xticks([10**-exponent for exponent in range(-1, 9)])
    plt.xlabel("Learning rate")
    plt.ylabel("Loss")
    ymin = min(y)
    xmin = x[y.index(ymin)]
    plt.title(f"Minimum loss = {ymin:.02f} at learning rate = {xmin:.02f}")
    plt.axhline(y=ymin, color='red', linestyle='dotted', alpha=0.5)
    plt.axvline(x=xmin, color='red', linestyle='dotted', alpha=0.5)
    plt.show()


def plot_curves(results: dict[str, list[float]], epoch:int) -> None:
    """Plot training and test losses and accuracies."""
    epochs = range(1, epoch+1)
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    axs[0].plot(epochs, results['train_loss'], label='Train', marker='.')
    axs[0].plot(epochs, results['test_loss'], label='Test', marker='.')
    axs[0].set_title("Loss")
    axs[0].set_xlabel("Epoch")
    axs[0].set_ylabel("Average loss")
    axs[0].set_ylim(bottom=0, top=None)
    axs[0].grid()
    axs[1].plot(epochs, results['train_acc'], label='Train', marker='.')
    axs[1].plot(epochs, results['test_acc'], label='Test', marker='.')
    axs[1].set_title("Accuracy")
    axs[1].set_xlabel("Epoch")
    axs[1].set_ylabel("Accuracy (%)")
    axs[1].set_ylim(bottom=None, top=100)
    axs[1].grid()
    plt.setp(axs, xticks=epochs)
    plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))
    plt.show()