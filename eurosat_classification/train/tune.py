import optuna
import torch
from optuna.trial import FixedTrial

from ..data.datasets import create_dataloaders
from ..models.cnn import CNNConfig, ConvBlockConfig
from .train import train_model

CHANNELS = {"rgb": 3, "ms": 13}


def build_config(trial, image_type):
    n_blocks = trial.suggest_int("n_conv_blocks", 2, 4)
    base = trial.suggest_categorical("base_channels", [16, 32, 64])
    conv_blocks = [
        ConvBlockConfig(out_channels=base * (2**i), batch_norm=True, pool_size=2)
        for i in range(n_blocks)
    ]
    return CNNConfig(
        in_channels=CHANNELS[image_type],
        input_height=64,
        input_width=64,
        conv_blocks=conv_blocks,
        fc_layers=[trial.suggest_categorical("fc_hidden", [64, 128, 256]), 10],
        dropout=trial.suggest_float("dropout", 0.2, 0.6),
        activation=trial.suggest_categorical("activation", ["relu", "gelu", "silu"]),
    )


def objective(trial, image_type, train_loader, val_loader):
    config = build_config(trial, image_type)
    lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)

    def report(epoch, val_f1):
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


def tune_image_type(image_type, n_trials=30):
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

    # Retrain and save the best models for each
    best_config = build_config(FixedTrial(study.best_params), image_type)
    model, _ = train_model(
        best_config,
        train_loader,
        val_loader,
        lr=study.best_params["lr"],
        epochs=30,
        patience=5,
    )
    torch.save(model.state_dict(), f"models/best_{image_type}.pt")
    return study


def main():
    studies = {}
    for image_type in ("rgb", "ms"):
        print(f"\n=== Tuning {image_type} ===")
        studies[image_type] = tune_image_type(image_type)

    print("\n=== Comparison ===")
    for image_type, study in studies.items():
        print(
            f"{image_type}: best F1 = {study.best_value:.4f}, params = {study.best_params}"
        )

    best = max(studies, key=lambda t: studies[t].best_value)
    print(f"\nOverall best: {best} (F1 = {studies[best].best_value:.4f})")


if __name__ == "__main__":
    main()
