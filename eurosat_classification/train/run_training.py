from pathlib import Path

import matplotlib.pyplot as plt
import torch

from ..data.datasets import create_dataloaders
from ..models.cnn import CNNConfig, ConvBlockConfig, Kernel
from .train import train_model

CHANNELS = {"rgb": 3, "ms": 13}

KERNEL_OPTIONS = {
    "single_3":      [Kernel(3)],
    "single_5":      [Kernel(5)],
    "multi_3_5":     [Kernel(3), Kernel(5)],
    "multi_3_7":     [Kernel(3), Kernel(7)],
    "multi_3_5_7_9": [Kernel(3), Kernel(5), Kernel(7), Kernel(9)],
}


def build_config_from_params(image_type: str, params: dict) -> CNNConfig:
    n_blocks = params["n_conv_blocks"]
    base = params["base_channels"]

    conv_blocks = []
    for i in range(n_blocks):
        # Support both per-block keys (kernels_block_0, ...) and a single "kernels" key
        kernel_choice = params.get(f"kernels_block_{i}", params.get("kernels", "single_3"))
        conv_blocks.append(ConvBlockConfig(
            out_channels=base * (2**i),
            kernels=KERNEL_OPTIONS[kernel_choice],
            batch_norm=True,
            pool_size=2,
        ))

    n_fc_layers = params.get("n_fc_layers", 1)
    fc_hidden = params["fc_hidden"]

    return CNNConfig(
        in_channels=CHANNELS[image_type],
        input_height=64,
        input_width=64,
        conv_blocks=conv_blocks,
        fc_layers=[fc_hidden] * n_fc_layers + [10],
        dropout=params["dropout"],
        activation=params["activation"],
    )


def plot_history(history: dict, output_path: Path) -> None:
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(epochs, history["train_loss"], label="Train loss")
    ax1.plot(epochs, history["val_loss"], label="Val loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Loss curves")
    ax1.legend()

    ax2.plot(epochs, history["val_f1"], color="tab:green", label="Val F1")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Macro F1")
    ax2.set_title("Validation F1")
    ax2.legend()

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved plot: {output_path.resolve()}")


def train_from_params(
    image_type: str,
    params: dict,
    output: str | Path,
    epochs: int = 30,
    patience: int = 5,
    batch_size: int = 64,
) -> None:
    config = build_config_from_params(image_type, params)
    train_loader, val_loader, _ = create_dataloaders(image_type, batch_size=batch_size)

    model, best_val_f1, history = train_model(
        config, train_loader, val_loader,
        lr=params["lr"],
        epochs=epochs,
        patience=patience,
        track_history=True,
    )

    print(f"Best val F1: {best_val_f1:.4f}")

    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model, output)
    print(f"Saved model: {output.resolve()}")

    plot_history(history, output.with_suffix(".png"))


if __name__ == "__main__":
    IMAGE_TYPE = "ms"
    OUTPUT = "models/ms_model_final.pkl"

    PARAMS = {
        "n_conv_blocks": 4,
        "base_channels": 64,
        "fc_hidden": 64,
        "dropout": 0.38217102063179526,
        "activation": "relu",
        "lr": 0.00010200960558027954,
    }

    train_from_params(IMAGE_TYPE, PARAMS, OUTPUT)
