from typing import Literal

import torch
import torch.nn as nn


class ConvBlockConfig:
    def __init__(
        self,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        batch_norm: bool = True,
        pool_size: int | None = 2,
    ) -> None:
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.batch_norm = batch_norm
        self.pool_size = pool_size  


class CNNConfig:
    def __init__(
        self,
        in_channels: int = 3,
        input_height: int = 32,
        input_width: int = 32,
        conv_blocks: list[ConvBlockConfig] | None = None,
        fc_layers: list[int] | None = None,
        dropout: float = 0.5,
        activation: Literal["relu", "gelu", "leaky_relu", "silu"] = "relu",
    ) -> None:
        self.in_channels = in_channels
        self.input_height = input_height
        self.input_width = input_width
        self.conv_blocks = conv_blocks or [
            ConvBlockConfig(out_channels=32),
            ConvBlockConfig(out_channels=64),
            ConvBlockConfig(out_channels=128),
        ]
        self.fc_layers = fc_layers or [256, 10]
        self.dropout = dropout
        self.activation = activation


_ACTIVATIONS: dict[str, type[nn.Module]] = {
    "relu": nn.ReLU,
    "gelu": nn.GELU,
    "leaky_relu": nn.LeakyReLU,
    "silu": nn.SiLU,
}


def _build_activation(name: str) -> nn.Module:
    try:
        return _ACTIVATIONS[name]()
    except KeyError:
        raise ValueError(f"Unknown activation '{name}'. Choose from {list(_ACTIVATIONS)}")


class CNN(nn.Module):
    def __init__(self, config: CNNConfig | None = None) -> None:
        super().__init__()
        self.config = config or CNNConfig()

        self.backbone = self._build_backbone()
        flat_dim = self._infer_flat_dim()
        self.classifier = self._build_classifier(flat_dim)

    def _build_backbone(self) -> nn.Sequential:
        layers: list[nn.Module] = []
        in_ch = self.config.in_channels

        for block_cfg in self.config.conv_blocks:
            layers.append(
                nn.Conv2d(
                    in_ch,
                    block_cfg.out_channels,
                    kernel_size=block_cfg.kernel_size,
                    stride=block_cfg.stride,
                    padding=block_cfg.padding,
                )
            )
            if block_cfg.batch_norm:
                layers.append(nn.BatchNorm2d(block_cfg.out_channels))
            layers.append(_build_activation(self.config.activation))
            if block_cfg.pool_size is not None:
                layers.append(nn.MaxPool2d(kernel_size=block_cfg.pool_size))
            in_ch = block_cfg.out_channels

        return nn.Sequential(*layers)

    def _infer_flat_dim(self) -> int:
        with torch.no_grad():
            dummy = torch.zeros(
                1, self.config.in_channels, self.config.input_height, self.config.input_width
            )
            return int(self.backbone(dummy).numel())

    def _build_classifier(self, flat_dim: int) -> nn.Sequential:
        layers: list[nn.Module] = []
        in_features = flat_dim

        for i, out_features in enumerate(self.config.fc_layers):
            layers.append(nn.Linear(in_features, out_features))
            is_last = i == len(self.config.fc_layers) - 1
            if not is_last:
                layers.append(_build_activation(self.config.activation))
                layers.append(nn.Dropout(self.config.dropout))
            in_features = out_features

        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.backbone(x)
        x = x.flatten(start_dim=1)
        return self.classifier(x)
