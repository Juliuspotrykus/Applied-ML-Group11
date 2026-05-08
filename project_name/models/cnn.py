from typing import Literal

import torch
import torch.nn as nn


class ConvBlockConfig:
    """Configuration for a single convolutional block (Conv2d → BN → activation → MaxPool)."""

    def __init__(
        self,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        batch_norm: bool = True,
        pool_size: int | None = 2,
    ) -> None:
        """
        Args:
            out_channels: Number of filters produced by the conv layer.
            kernel_size: Size of the convolving kernel.
            stride: Step size of the convolution.
            padding: Zero-padding added to both sides of the input.
            batch_norm: Whether to add BatchNorm2d after the conv layer.
            pool_size: Kernel size for MaxPool2d. None disables pooling.
        """
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.batch_norm = batch_norm
        self.pool_size = pool_size


class CNNConfig:
    """Top-level configuration for the CNN architecture."""

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
        """
        Args:
            in_channels: Number of input channels (e.g. 1 for grayscale, 3 for RGB).
            input_height: Height of the input image in pixels.
            input_width: Width of the input image in pixels.
            conv_blocks: List of ConvBlockConfig objects defining the backbone.
                Defaults to three blocks with 32, 64, and 128 filters.
            fc_layers: Widths of the fully-connected head layers. The last value
                is the number of output classes. Defaults to [256, 10].
            dropout: Dropout probability applied between FC layers.
            activation: Activation function used after conv and FC layers.
        """
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
    """Configurable CNN with a convolutional backbone and fully-connected classifier head."""

    def __init__(self, config: CNNConfig | None = None) -> None:
        """
        Args:
            config: CNNConfig instance. Defaults to CNNConfig() if not provided.
        """
        super().__init__()
        self.config = config or CNNConfig()

        self.backbone = self._build_backbone()
        flat_dim = self._infer_flat_dim()
        self.classifier = self._build_classifier(flat_dim)

    def _build_backbone(self) -> nn.Sequential:
        """Stacks conv blocks as defined in config.conv_blocks."""
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
        """Runs a dummy forward pass to determine the flattened backbone output size."""
        with torch.no_grad():
            dummy = torch.zeros(
                1, self.config.in_channels, self.config.input_height, self.config.input_width
            )
            return int(self.backbone(dummy).numel())

    def _build_classifier(self, flat_dim: int) -> nn.Sequential:
        """Builds the FC head from flat_dim to the final output size."""
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
        """Forward pass: backbone → flatten → classifier."""
        x = self.backbone(x)
        x = x.flatten(start_dim=1)
        return self.classifier(x)
