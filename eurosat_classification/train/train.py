import copy
from typing import Callable

import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader

from ..models.cnn import CNN, CNNConfig


def evaluate(
    model: CNN,
    loader: DataLoader,
    loss_fn: nn.Module,
    return_preds: bool = False,
):
    """Evaluates the model on a given dataset.

    Args:
        model (CNN): The CNN model to evaluate.
        loader (DataLoader): The data loader for the dataset.
        loss_fn (nn.Module): The loss function.

        return_preds (bool, optional): Whether to return
        predictions and labels. Defaults to False.

    Returns:
        tuple: A tuple containing the loss and F1 score,
        or a tuple with the predictions and labels
        if return_preds is True.
    """

    model.eval()
    device = next(model.parameters()).device
    total_loss = 0
    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            total_loss += loss_fn(outputs, labels).item()
            all_preds.extend(outputs.argmax(dim=1).tolist())
            all_labels.extend(labels.tolist())
    loss = total_loss / len(loader)
    f1 = f1_score(all_labels, all_preds, average="macro")
    if return_preds:
        return loss, f1, all_labels, all_preds
    return loss, f1


def train_model(
    config: CNNConfig,
    train_loader,
    val_loader=None,
    lr=1e-3,
    epochs: int = 30,
    patience: int = 5,
    check_prune: Callable | None = None,
    track_history: bool = False,
    eval_loader=None,
) -> tuple[CNN, float] | tuple[CNN, float, dict]:
    """Method to train a CNN. When val_loader is None, the model
       is trained for a fixed number of epochs with no validation
       nor early stopping and the final-epoch model is returned.
       Otherwise the model is validated each epoch, early stopping
       is applied, and the best-val-F1 checkpoint is restored
       before returning. Eval_loader is just for tracking the
       test loss in the history, it is not used for early
       stopping or model selection.

    Args:
        config (CNNConfig): The configuration for the CNN model.
        train_loader (DataLoader): The data loader for the training dataset.

        val_loader (DataLoader, optional): The data loader for
        the validation dataset. Defaults to None.

        lr (float, optional): The learning rate for the optimizer.
        Defaults to 1e-3.

        epochs (int, optional): The number of epochs to train for.
        Defaults to 30.

        patience (int, optional): The number of epochs to wait
        for improvement before stopping. Defaults to 5.

        check_prune (Callable | None, optional): A function to
        check if pruning should be applied. Defaults to None.

        track_history (bool, optional): Whether to track the
        training history. Defaults to False.

        eval_loader (DataLoader, optional): The data loader
        for the evaluation dataset. Defaults to None.

    Returns:
        tuple[CNN, float] | tuple[CNN, float, dict]: The trained model,
        best validation F1 score, and optionally the training history.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CNN(config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    best_val_f1 = 0.0
    best_state = None
    epochs_no_improve = 0
    history: dict = (
        {"train_loss": [], "val_loss": [], "val_f1": [], "test_loss": []}
        if track_history
        else {}
    )

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = loss_fn(model(images), labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)

        if val_loader is None:
            msg = f"Epoch {epoch + 1}, Train loss: {train_loss:.4f}"
            if track_history:
                history["train_loss"].append(train_loss)
                # eval_loader is just for history and plotting only
                if eval_loader is not None:
                    test_loss, _ = evaluate(model, eval_loader, loss_fn)
                    history["test_loss"].append(test_loss)
                    msg += f", Test loss: {test_loss:.4f}"
            print(msg)
            continue

        val_loss, val_f1 = evaluate(model, val_loader, loss_fn)

        if track_history:
            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            history["val_f1"].append(val_f1)
            print(
                f"Epoch {epoch + 1}, Train loss: {train_loss:.4f}, \
                Val loss: {val_loss:.4f}, Val F1: {val_f1:.4f}"
            )
        else:
            print(
                f"Epoch {epoch + 1}, Val loss: {val_loss:.4f}, \
            Val F1: {val_f1:.4f}"
            )

        if check_prune is not None:
            check_prune(epoch, val_f1)

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_state = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    if track_history:
        return model, best_val_f1, history
    return model, best_val_f1
