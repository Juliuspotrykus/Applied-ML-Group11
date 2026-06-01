from typing import Callable

import torch
import torch.nn as nn
from sklearn.metrics import f1_score

from ..models.cnn import CNN, CNNConfig


def evaluate(model, loader, loss_fn):
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
    return total_loss / len(loader), f1_score(all_labels, all_preds, average="macro")


def train_model(
    config: CNNConfig,
    train_loader,
    val_loader,
    lr,
    epochs: int,
    patience: int,
    check_prune: Callable | None = None,
    track_history: bool = False,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CNN(config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    best_val_f1 = 0.0
    epochs_no_improve = 0
    history: dict = {"train_loss": [], "val_loss": [], "val_f1": []} if track_history else {}

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = loss_fn(model(images), labels)
            loss.backward()
            optimizer.step()
            if track_history:
                train_loss += loss.item()

        val_loss, val_f1 = evaluate(model, val_loader, loss_fn)

        if track_history:
            train_loss /= len(train_loader)
            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            history["val_f1"].append(val_f1)
            print(f"Epoch {epoch + 1}, Train loss: {train_loss:.4f}, Val loss: {val_loss:.4f}, Val F1: {val_f1:.4f}")
        else:
            print(f"Epoch {epoch + 1}, Val loss: {val_loss:.4f}, Val F1: {val_f1:.4f}")

        if check_prune is not None:
            check_prune(epoch, val_f1)

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                break

    if track_history:
        return model, best_val_f1, history
    return model, best_val_f1
