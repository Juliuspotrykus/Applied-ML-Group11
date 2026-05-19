import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from ..models.cnn import CNN, CNNConfig, ConvBlockConfig
from ..data.datasets import create_dataloaders

# Example training script for CNN model.

# Config
config = CNNConfig(
    in_channels=3, # 3 Channels for RGB images
    input_height=64, 
    input_width=64,
    conv_blocks=[
        ConvBlockConfig(out_channels=32, kernel_size=3, stride=1, padding=1, batch_norm=True, pool_size=2),
        ConvBlockConfig(out_channels=64, kernel_size=3, stride=1, padding=1, batch_norm=True, pool_size=2),
        ConvBlockConfig(out_channels=128, kernel_size=3, stride=1, padding=1, batch_norm=True, pool_size=2),
    ],
    fc_layers=[128, 10], # 128 units in hidden layer, 10 output classes --> You can also specify this as [128, 64, 10] for an additional hidden layer
    dropout=0.5,
    activation="relu"
)

model = CNN(config)
train_loader, val_loader, test_loader = create_dataloaders(image_type="rgb", batch_size=64)

# Training setup 
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.CrossEntropyLoss()

# Training loop
train_losses = []
val_losses = []

print("Starting training...")

for epoch in range(0):
    model.train()
    total_loss_train = 0
    for images, labels in train_loader:
        optimizer.zero_grad()
        loss = loss_fn(model(images), labels)
        loss.backward()
        optimizer.step()
        total_loss_train += loss.item()

    model.eval()
    total_loss_val = 0
    with torch.no_grad():
        for images, labels in val_loader:
            loss = loss_fn(model(images), labels)
            total_loss_val += loss.item()

    avg_train_loss = total_loss_train / len(train_loader)
    avg_val_loss = total_loss_val / len(val_loader)
    train_losses.append(avg_train_loss)
    val_losses.append(avg_val_loss)
    print(f"Epoch {epoch+1}, Train loss: {avg_train_loss:.4f}, Val loss: {avg_val_loss:.4f}")

torch.save(model, "models/model1.pkl")

# Test evaluation (after training)
model.eval()
total_loss_test = 0
with torch.no_grad():
    for images, labels in test_loader:
        loss = loss_fn(model(images), labels)
        total_loss_test += loss.item()
print(f"Test loss: {total_loss_test / len(test_loader):.4f}")

plt.plot(train_losses, label="Train")
plt.plot(val_losses, label="Val")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.show()