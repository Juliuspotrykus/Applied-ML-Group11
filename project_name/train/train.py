import torch
import torch.nn as nn

from project_name.models.cnn import CNN, CNNConfig, ConvBlockConfig

# Example training script for CNN model.

# Config
config = CNNConfig(
    in_channels=3, # 3 Channels for RGB images
    input_height=256, 
    input_width=256,
    conv_blocks=[
        ConvBlockConfig(out_channels=32, kernel_size=3, stride=1, padding=1, batch_norm=True, pool_size=2),
        ConvBlockConfig(out_channels=64, kernel_size=3, stride=1, padding=1, batch_norm=True, pool_size=2),
        ConvBlockConfig(out_channels=128, kernel_size=3, stride=1, padding=1, batch_norm=True, pool_size=None),  # No pooling in last block
    ],
    fc_layers=[128, 10], # 128 units in hidden layer, 10 output classes --> You can also specify this as [128, 64, 10] for an additional hidden layer
    dropout=0.5,
    activation="relu"
)

model = CNN(config)

# Data - To be implemented, use torch DataLoader
train_loader = ... 

# Training setup 
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.CrossEntropyLoss()

# Training loop
model.train()
for epoch in range(10):
    total_loss = 0
    for images, labels in train_loader:
        optimizer.zero_grad()
        predictions = model(images)
        loss = loss_fn(predictions, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}, loss: {total_loss / len(train_loader):.4f}")
