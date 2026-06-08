"""This script tunes the CNN hyperparameters for a given modality (rgb or ms)
using Optuna. Each trial samples an architecture and learning rate, trains on
the train set with early stopping, and is scored by its best validation macro
F1. A median pruner prunes underperforming trials early. The best F1 and the
corresponding parameters are printed at the end (the chosen values are copied
into BEST_PARAMS in run_training.py).

Usage:
    # Tune the ms model with 30 trials:
    python -m eurosat_classification.train.tune ms 30
"""

import argparse

import optuna
from torch.utils.data import DataLoader

from ..data.datasets import create_dataloaders
from ..models.cnn import CNNConfig, ConvBlockConfig, Kernel
from .train import train_model

CHANNELS = {"rgb": 3, "ms": 13}


KERNEL_OPTIONS = {
    "single_3": [Kernel(3)],
    "single_5": [Kernel(5)],
    "multi_3_5": [Kernel(3), Kernel(5)],
    "multi_3_7": [Kernel(3), Kernel(7)],
    "multi_3_5_7_9": [Kernel(3), Kernel(5), Kernel(7), Kernel(9)],
}


MAX_CHANNELS = 512


def build_config(trial: optuna.Trial, image_type: str) -> CNNConfig:
    """Builds a CNNConfig from the given Optuna trial and image type.

    Args:
        trial (optuna.Trial): The Optuna trial.
        image_type (str): The type of image (either "rgb" or "ms").

    Returns:
        CNNConfig: The built CNN configuration.
    """
    n_blocks = trial.suggest_int("n_conv_blocks", 2, 6)
    base = trial.suggest_categorical("base_channels", [16, 32, 64])
    conv_blocks = []
    for i in range(n_blocks):
        kernel_choice = trial.suggest_categorical(
            f"kernels_block_{i}", list(KERNEL_OPTIONS)
        )
        conv_blocks.append(
            ConvBlockConfig(
                out_channels=min(base * (2**i), MAX_CHANNELS),
                kernels=KERNEL_OPTIONS[kernel_choice],
                batch_norm=True,
                pool_size=2,
            )
        )

    n_fc_layers = trial.suggest_int("n_fc_layers", 1, 10)
    fc_hidden_size = trial.suggest_categorical(
        "fc_hidden", [64, 128, 256, 512, 1024, 2048]
    )
    fc_layers = [fc_hidden_size] * n_fc_layers + [10]

    return CNNConfig(
        in_channels=CHANNELS[image_type],
        input_height=64,
        input_width=64,
        conv_blocks=conv_blocks,
        fc_layers=fc_layers,
        dropout=trial.suggest_float("dropout", 0.2, 0.6),
        activation=trial.suggest_categorical("activation", ["relu", "gelu", "silu"]),
    )


def objective(
    trial: optuna.Trial,
    image_type: str,
    train_loader: DataLoader,
    val_loader: DataLoader,
) -> float:
    """The objective function for Optuna hyperparameter tuning. Builds a CNNConfig from the trial and image type,
       trains the model using train_model, and reports the best validation F1 score back to Optuna for pruning.

    Args:
        trial (optuna.Trial): The Optuna trial.
        image_type (str): The type of image (either "rgb" or "ms").
        train_loader (DataLoader): The training data loader.
        val_loader (DataLoader): The validation data loader.

    Raises:
        optuna.TrialPruned: If the trial should be pruned based on the reported validation F1 score.

    Returns:
        float: The best validation F1 score.
    """
    config = build_config(trial, image_type)
    lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)

    def report(epoch, val_f1):
        """Reports the validation F1 score to Optuna to decide whether current trial should be pruned.

        Args:
            epoch (int): The current epoch.
            val_f1 (float): The validation F1 score.

        Raises:
            optuna.TrialPruned: If the trial should be pruned based on the reported validation F1 score.
        """
        trial.report(val_f1, epoch)
        if trial.should_prune():
            raise optuna.TrialPruned()

    _, best_val_f1 = train_model(
        config,
        train_loader,
        val_loader,
        lr=lr,
        epochs=30,
        patience=5,
        check_prune=report,
    )
    return best_val_f1


def tune_image_type(image_type: str, n_trials: int = 30) -> optuna.Study:
    """Tunes the hyperparameters for a given image type using Optuna.
       Creates the training and validation data loaders, sets up the Optuna study with a median pruner,
       and runs the optimization for the specified number of trials.

    Args:
        image_type (str): The type of image ("rgb" or "ms").
        n_trials (int, optional): The number of trials to run. Defaults to 30.

    Returns:
        optuna.Study: The optimized Optuna study.
    """
    train_loader, val_loader, _ = create_dataloaders(image_type, batch_size=64)
    study = optuna.create_study(
        study_name=image_type,
        direction="maximize",
        pruner=optuna.pruners.MedianPruner(),
    )
    study.optimize(
        lambda trial: objective(trial, image_type, train_loader, val_loader),
        n_trials=n_trials,
    )

    return study


def main():
    """Main function to run the hyperparameter tuning. Parses command-line arguments for the image type and number of trials,"""
    parser = argparse.ArgumentParser(description="Hyperparameter tuning")
    parser.add_argument("image_type", choices=("rgb", "ms"))
    parser.add_argument("n_trials", type=int)
    args = parser.parse_args()

    print(f"\n=== Tuning {args.image_type} ===")
    study = tune_image_type(args.image_type, args.n_trials)
    print(
        f"{args.image_type}: best F1 = {study.best_value:.4f}, params = {study.best_params}"
    )


if __name__ == "__main__":
    main()
