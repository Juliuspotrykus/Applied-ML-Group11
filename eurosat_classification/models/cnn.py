from typing import Literal

import torch
from torch import nn


class Kernel:
    """Geometry of a single convolutional kernel branch."""

    def __init__(
        self,
        kernel_size: int,
        stride: int = 1,
        padding: int | None = None,
    ) -> None:
        """
        Args:
            kernel_size: Size of the convolving kernel.
            stride: Step size of the convolution.
            padding: Zero-padding on both sides. Defaults to kernel_size // 2.
        """
        self.kernel_size = kernel_size
        self.stride = stride

        if padding is not None:
            self.padding = padding
        else:
            self.padding = kernel_size // 2


class ConvBlockConfig:
    """Configuration for a convolutional block
    (Conv2d → BN → activation → MaxPool).

    Pass a list of Kernel objects to get an inception-style block where
    each kernel runs in parallel and the outputs are concatenated along
    the channel axis out_channels is then split evenly across branches,
    so it must be divisible by the number of kernels.
    """

    def __init__(
        self,
        out_channels: int,
        kernels: Kernel | list[Kernel] = Kernel(3),
        batch_norm: bool = True,
        pool_size: int | None = 2,
    ) -> None:
        """
        Args:
            out_channels: Total output filters. For multiple kernels
            this must be divisible by the number of kernels
            (each branch gets an equal share).
            kernels: A single Kernel or a list of Kernel objects.
            batch_norm: Whether to add BatchNorm2d after the conv output.
            pool_size: Kernel size for MaxPool2d. None disables pooling.
        """
        self.out_channels = out_channels
        self.batch_norm = batch_norm
        self.pool_size = pool_size

        if isinstance(kernels, Kernel):
            kernels = [kernels]

        for k in kernels:
            if not isinstance(k, Kernel):
                raise TypeError(
                    f"Expected a Kernel instance, got {type(k).__name__}. "
                    "Wrap kernel sizes in a Kernel object, e.g. Kernel(3)."
                )
        self.kernels: list[Kernel] = list(kernels)

        # Validate multi-kernel constraints
        num_kernels = len(self.kernels)
        if num_kernels > 1:
            if out_channels % num_kernels != 0:
                raise ValueError(
                    f"out_channels ({out_channels}) must be divisible "
                    "by the number "
                    f"of kernels ({num_kernels}) for multi-kernel blocks."
                )
            strides = {k.stride for k in self.kernels}
            if len(strides) > 1:
                raise ValueError(
                    "All kernels in a multi-kernel block must share the "
                    "same stride "
                    f"so their outputs can be concatenated. Got: {strides}"
                )


class CNNConfig:
    """Top-level configuration for the CNN architecture."""

    def __init__(
        self,
        in_channels: int = 3,
        input_height: int = 64,
        input_width: int = 64,
        conv_blocks: list[ConvBlockConfig] | None = None,
        fc_layers: list[int] | None = None,
        dropout: float = 0.5,
        activation: Literal["relu", "gelu", "leaky_relu", "silu"] = "relu",
    ) -> None:
        """
        Args:
            in_channels: Number of input channels
                (e.g. 1 for grayscale, 3 for RGB).
            input_height: Height of the input image in pixels.
            input_width: Width of the input image in pixels.
            conv_blocks: List of ConvBlockConfig objects defining the backbone.
                Defaults to three blocks with 32, 64, and 128 filters.
            fc_layers: Widths of the fully-connected head layers.
                The last value is the number of output classes.
                Defaults to [256, 10].
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
    """
    Build NN module activation modules from string representation of activation
    functions.

    Args:
        name (str): Activation fucntion name

    Raises:
        ValueError: Unknown activation function given.

    Returns:
        nn.Module: Torch NN module of requested activation function.
    """
    try:
        return _ACTIVATIONS[name]()
    except KeyError:
        raise ValueError(
            f"Unknown activation '{name}'. Choose from {list(_ACTIVATIONS)}"
        )


class _MultiKernelBlock(nn.Module):
    """Inception-style block: parallel Conv2d branches
    concatenated along the channel axis."""

    def __init__(
        self,
        in_channels: int,
        block_cfg: ConvBlockConfig,
        activation: nn.Module,
    ) -> None:
        """
        Initializes the multi-kernel parallel convolutional block.

        Args:
            in_channels (int): Number of input channels.
            block_cfg (ConvBlockConfig): Configuration object containing
                kernels, out_channels, batch_norm, and pool_size settings.
            activation (nn.Module): Activation function layer.
        """
        super().__init__()

        branch_channels = block_cfg.out_channels // len(block_cfg.kernels)

        branches = []
        for kernel in block_cfg.kernels:
            branch = nn.Conv2d(
                in_channels,
                branch_channels,
                kernel_size=kernel.kernel_size,
                stride=kernel.stride,
                padding=kernel.padding,
            )
            branches.append(branch)
        self.branches = nn.ModuleList(branches)

        post_layers: list[nn.Module] = []
        if block_cfg.batch_norm:
            post_layers.append(nn.BatchNorm2d(block_cfg.out_channels))
        post_layers.append(activation)
        if block_cfg.pool_size is not None:
            post_layers.append(nn.MaxPool2d(kernel_size=block_cfg.pool_size))
        self.post = nn.Sequential(*post_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Passes input through parallel branches and applies post-processing.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Processed tensor.
        """
        branch_outputs = [branch(x) for branch in self.branches]
        concatenated = torch.cat(branch_outputs, dim=1)
        return self.post(concatenated)


class CNN(nn.Module):
    """Configurable CNN with a convolutional backbone
    and fully-connected classifier head."""

    def __init__(self, config: CNNConfig | None = None) -> None:
        """
        Initializes the network topology using a provided or
        default configuration.

        Args:
            config: CNNConfig instance. Defaults to CNNConfig()
            if not provided.
        """
        super().__init__()
        self.config = config or CNNConfig()

        self.backbone = self._build_backbone()
        flat_dim = self._infer_flat_dim()
        self.classifier = self._build_classifier(flat_dim)

    def _build_backbone(self) -> nn.Sequential:
        """
        Stacks conv blocks as defined in config.conv_blocks.

        Returns:
            nn.Sequential: Compiled convolutional feature extractor.

        Raises:
            IndexError: If a configuration block contains an
            empty kernels list.
        """
        layers: list[nn.Module] = []
        in_ch = self.config.in_channels

        for block_cfg in self.config.conv_blocks:
            if len(block_cfg.kernels) > 1:
                activation = _build_activation(self.config.activation)
                block = _MultiKernelBlock(in_ch, block_cfg, activation)
                layers.append(block)
            else:
                kernel = block_cfg.kernels[0]
                layers.append(
                    nn.Conv2d(
                        in_ch,
                        block_cfg.out_channels,
                        kernel_size=kernel.kernel_size,
                        stride=kernel.stride,
                        padding=kernel.padding,
                    )
                )
                if block_cfg.batch_norm:
                    layers.append(nn.BatchNorm2d(block_cfg.out_channels))
                layers.append(_build_activation(self.config.activation))
                if block_cfg.pool_size is not None:
                    layers.append(
                        nn.MaxPool2d(kernel_size=block_cfg.pool_size)
                    )

            in_ch = block_cfg.out_channels

        return nn.Sequential(*layers)

    def _infer_flat_dim(self) -> int:
        """
        Runs a dummy forward pass to determine the flattened backbone
        output size.

        Returns:
            int: Flattened size feature dimension (total elements per sample)

        Raises:
            RuntimeError: If spatial size configurations cause dimension
                reduction down to zero or negative dimensions during the pass.
        """
        with torch.no_grad():
            dummy = torch.zeros(
                1,
                self.config.in_channels,
                self.config.input_height,
                self.config.input_width,
            )
            return int(self.backbone(dummy).numel())

    def _build_classifier(self, flat_dim: int) -> nn.Sequential:
        """
        Builds the FC head from flat_dim to the final output size.

        Args:
            flat_dim (int): Flattened size input feature dimension.

        Returns:
            nn.Sequential: Multi-layer linear head with dropout and
                            activation functions.
        """
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
        """
        Forward pass:
        - backbone
        - flatten
        - classifier

        Args:
            x (torch.Tensor): Raw image input tensor.

        Returns:
            torch.Tensor: Model prediction logits.
        """
        x = self.backbone(x)
        x = x.flatten(start_dim=1)
        return self.classifier(x)
