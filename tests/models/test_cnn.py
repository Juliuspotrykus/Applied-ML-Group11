import unittest

import torch

from eurosat_classification.models.cnn import (
    CNN,
    CNNConfig,
    ConvBlockConfig,
    Kernel,
)


class TestKernel(unittest.TestCase):
    def test_default_padding_is_half_kernel_size(self):
        self.assertEqual(Kernel(3).padding, 1)
        self.assertEqual(Kernel(5).padding, 2)
        self.assertEqual(Kernel(7).padding, 3)

    def test_explicit_padding_overrides_default(self):
        k = Kernel(3, padding=0)
        self.assertEqual(k.padding, 0)

    def test_stride_stored(self):
        k = Kernel(3, stride=2)
        self.assertEqual(k.stride, 2)


class TestConvBlockConfig(unittest.TestCase):
    def test_single_kernel_wrapped_in_list(self):
        cfg = ConvBlockConfig(out_channels=32, kernels=Kernel(3))
        self.assertIsInstance(cfg.kernels, list)
        self.assertEqual(len(cfg.kernels), 1)

    def test_multi_kernel_list_stored(self):
        cfg = ConvBlockConfig(out_channels=32, kernels=[Kernel(3), Kernel(5)])
        self.assertEqual(len(cfg.kernels), 2)

    def test_raises_type_error_on_non_kernel(self):
        with self.assertRaises(TypeError):
            ConvBlockConfig(out_channels=32, kernels=[Kernel(3), 5])

    def test_raises_value_error_when_channels_not_divisible_by_kernels(self):
        with self.assertRaises(ValueError):
            ConvBlockConfig(out_channels=32, kernels=[Kernel(3), Kernel(5), Kernel(7)])

    def test_raises_value_error_on_mixed_strides(self):
        with self.assertRaises(ValueError):
            ConvBlockConfig(
                out_channels=32,
                kernels=[Kernel(3, stride=1), Kernel(5, stride=2)],
            )

    def test_valid_multi_kernel_does_not_raise(self):
        ConvBlockConfig(out_channels=32, kernels=[Kernel(3), Kernel(5)])


class TestCNN(unittest.TestCase):
    def _forward(self, config: CNNConfig, batch_size: int = 2) -> torch.Tensor:
        model = CNN(config)
        model.eval()
        x = torch.randn(batch_size, config.in_channels, config.input_height, config.input_width)
        with torch.no_grad():
            return model(x)

    def test_output_shape_rgb(self):
        config = CNNConfig(in_channels=3, fc_layers=[64, 10])
        out = self._forward(config)
        self.assertEqual(out.shape, (2, 10))

    def test_output_shape_ms(self):
        config = CNNConfig(in_channels=13, fc_layers=[64, 10])
        out = self._forward(config)
        self.assertEqual(out.shape, (2, 10))

    def test_output_classes_match_fc_last_layer(self):
        config = CNNConfig(in_channels=3, fc_layers=[32, 7])
        out = self._forward(config)
        self.assertEqual(out.shape[1], 7)

    def test_inception_block_output_shape(self):
        config = CNNConfig(
            in_channels=3,
            conv_blocks=[ConvBlockConfig(out_channels=32, kernels=[Kernel(3), Kernel(5)])],
            fc_layers=[64, 10],
        )
        out = self._forward(config)
        self.assertEqual(out.shape, (2, 10))

    def test_all_activations_run(self):
        for activation in ("relu", "gelu", "leaky_relu", "silu"):
            with self.subTest(activation=activation):
                config = CNNConfig(activation=activation, fc_layers=[32, 10])
                out = self._forward(config)
                self.assertEqual(out.shape[1], 10)

    def test_invalid_activation_raises(self):
        with self.assertRaises(ValueError):
            CNN(CNNConfig(activation="tanh"))

    def test_no_pooling_block(self):
        config = CNNConfig(
            in_channels=3,
            conv_blocks=[ConvBlockConfig(out_channels=16, pool_size=None)],
            fc_layers=[32, 10],
        )
        out = self._forward(config)
        self.assertEqual(out.shape, (2, 10))

    def test_no_batch_norm_block(self):
        config = CNNConfig(
            in_channels=3,
            conv_blocks=[ConvBlockConfig(out_channels=16, batch_norm=False)],
            fc_layers=[32, 10],
        )
        out = self._forward(config)
        self.assertEqual(out.shape, (2, 10))

    def test_default_config_runs(self):
        out = self._forward(CNNConfig())
        self.assertEqual(out.shape[1], 10)


if __name__ == "__main__":
    unittest.main()
