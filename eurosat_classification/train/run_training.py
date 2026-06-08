import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn as nn

from ..data.datasets import create_dataloaders
from ..models.cnn import CNNConfig, ConvBlockConfig, Kernel
from .train import evaluate, train_model

CHANNELS = {"rgb": 3, "ms": 13}

KERNEL_OPTIONS = {
    "single_3": [Kernel(3)],
    "single_5": [Kernel(5)],
    "multi_3_5": [Kernel(3), Kernel(5)],
    "multi_3_7": [Kernel(3), Kernel(7)],
    "multi_3_5_7_9": [Kernel(3), Kernel(5), Kernel(7), Kernel(9)],
}

# Keep in sync with tune.py: channel counts are capped during tuning, so the
# final config must apply the same cap to reproduce the tuned architecture.
MAX_CHANNELS = 512


def build_config_from_params(image_type: str, params: dict) -> CNNConfig:
    n_blocks = params["n_conv_blocks"]
    base = params["base_channels"]

    conv_blocks = []
    for i in range(n_blocks):
        # Support both per-block keys (kernels_block_0, ...) and a single "kernels" key
        kernel_choice = params.get(
            f"kernels_block_{i}", params.get("kernels", "single_3")
        )
        conv_blocks.append(
            ConvBlockConfig(
                out_channels=min(base * (2**i), MAX_CHANNELS),
                kernels=KERNEL_OPTIONS[kernel_choice],
                batch_norm=True,
                pool_size=2,
            )
        )

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
        config,
        train_loader,
        val_loader,
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


def train_final_from_params(
    image_type: str,
    params: dict,
    output: str | Path,
    epochs: int,
    batch_size: int = 64,
) -> float:
    """Final fit: train on train+val and evaluate once on the test set."""

    from .compare_models import make_trainval_loader

    config = build_config_from_params(image_type, params)
    trainval_loader, test_loader = make_trainval_loader(image_type, batch_size)

    model, _ = train_model(
        config,
        trainval_loader,
        val_loader=None,
        lr=params["lr"],
        epochs=epochs,
    )

    test_loss, test_f1 = evaluate(model, test_loader, nn.CrossEntropyLoss())
    print(f"Test loss: {test_loss:.4f}, Test macro F1: {test_f1:.4f}")

    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model, output)
    print(f"Saved model: {output.resolve()}")
    return test_f1


BEST_PARAMS = {
    "ms": {
        "n_conv_blocks": 4,
        "base_channels": 64,
        "kernels_block_0": "single_3",
        "kernels_block_1": "multi_3_5_7_9",
        "kernels_block_2": "single_5",
        "kernels_block_3": "single_3",
        "n_fc_layers": 2,
        "fc_hidden": 64,
        "dropout": 0.28939638652358357,
        "activation": "gelu",
        "lr": 0.00029190915346677794,
    },
    "rgb": {
        "n_conv_blocks": 5,
        "base_channels": 64,
        "kernels_block_0": "single_3",
        "kernels_block_1": "single_5",
        "kernels_block_2": "single_3",
        "kernels_block_3": "single_3",
        "kernels_block_4": "multi_3_5",
        "n_fc_layers": 2,
        "fc_hidden": 256,
        "dropout": 0.4378661192801057,
        "activation": "relu",
        "lr": 0.00015605712150871704,
    },
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("modality", choices=["rgb", "ms"])
    parser.add_argument(
        "--final",
        action="store_true",
    )  # final run -> trains on train+val
    parser.add_argument(
        "--epochs",
        type=int,
        default=30,
        help="Number of training epochs. For --final, this should be set to -> rgb: 25, ms: 22; which is when the validation F1's peaked",
    )
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    params = BEST_PARAMS[args.modality]

    if args.final:
        train_final_from_params(
            args.modality,
            params,
            f"models/{args.modality}_model_final.pkl",
            epochs=args.epochs,
            batch_size=args.batch_size,
        )
    else:
        train_from_params(
            args.modality,
            params,
            f"models/{args.modality}_model_val.pkl",
            epochs=args.epochs,
            batch_size=args.batch_size,
        )
