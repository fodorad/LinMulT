import unittest

import torch

from linmult.core.tcn import TCN, TCNLayer


class TestTCNLayer(unittest.TestCase):
    def setUp(self):
        self.B, self.T, self.d = 2, 20, 16

    def test_shape_preserved(self):
        layer = TCNLayer(d_model=self.d, kernel_size=3, dilation=1)
        x = torch.rand(self.B, self.T, self.d)
        out = layer(x)
        self.assertEqual(out.shape, x.shape)

    def test_shape_various_lengths(self):
        layer = TCNLayer(d_model=self.d, kernel_size=3, dilation=4)
        for t in [1, 5, 50, 200]:
            x = torch.rand(self.B, t, self.d)
            out = layer(x)
            self.assertEqual(out.shape, (self.B, t, self.d))

    def test_causality(self):
        """Changing future inputs must not affect past outputs."""
        layer = TCNLayer(d_model=self.d, kernel_size=3, dilation=2)
        layer.eval()
        t_split = 10
        x = torch.rand(self.B, self.T, self.d)
        out1 = layer(x)

        # Modify future positions
        x_modified = x.clone()
        x_modified[:, t_split:, :] = torch.rand(self.B, self.T - t_split, self.d)
        out2 = layer(x_modified)

        torch.testing.assert_close(out1[:, :t_split, :], out2[:, :t_split, :])

    def test_gradient_flow(self):
        layer = TCNLayer(d_model=self.d, kernel_size=3, dilation=1)
        x = torch.rand(self.B, self.T, self.d)
        out = layer(x)
        out.sum().backward()
        for p in layer.parameters():
            self.assertIsNotNone(p.grad)

    def test_no_nan_train(self):
        layer = TCNLayer(d_model=self.d, kernel_size=3, dilation=1, dropout=0.5)
        layer.train()
        x = torch.rand(self.B, self.T, self.d)
        out = layer(x)
        self.assertFalse(torch.isnan(out).any())

    def test_no_nan_eval(self):
        layer = TCNLayer(d_model=self.d, kernel_size=3, dilation=1, dropout=0.5)
        layer.eval()
        x = torch.rand(self.B, self.T, self.d)
        out = layer(x)
        self.assertFalse(torch.isnan(out).any())


class TestTCN(unittest.TestCase):
    def setUp(self):
        self.B, self.T, self.d = 2, 30, 16

    def test_shape_preserved(self):
        block = TCN(d_model=self.d, num_layers=3, kernel_size=3)
        x = torch.rand(self.B, self.T, self.d)
        out = block(x)
        self.assertEqual(out.shape, x.shape)

    def test_single_layer(self):
        block = TCN(d_model=self.d, num_layers=1, kernel_size=3)
        x = torch.rand(self.B, self.T, self.d)
        out = block(x)
        self.assertEqual(out.shape, x.shape)

    def test_four_layers(self):
        block = TCN(d_model=self.d, num_layers=4, kernel_size=5)
        x = torch.rand(self.B, self.T, self.d)
        out = block(x)
        self.assertEqual(out.shape, x.shape)

    def test_causality(self):
        """Changing future inputs must not affect past outputs."""
        block = TCN(d_model=self.d, num_layers=3, kernel_size=3)
        block.eval()
        t_split = 15
        x = torch.rand(self.B, self.T, self.d)
        out1 = block(x)

        x_modified = x.clone()
        x_modified[:, t_split:, :] = torch.rand(self.B, self.T - t_split, self.d)
        out2 = block(x_modified)

        torch.testing.assert_close(out1[:, :t_split, :], out2[:, :t_split, :])

    def test_dilation_pattern(self):
        block = TCN(d_model=self.d, num_layers=3, kernel_size=3)
        expected_dilations = [1, 2, 4]
        for layer, expected in zip(block.layers, expected_dilations):
            self.assertEqual(layer.conv.dilation[0], expected)

    def test_gradient_flow(self):
        block = TCN(d_model=self.d, num_layers=3, kernel_size=3)
        x = torch.rand(self.B, self.T, self.d)
        out = block(x)
        out.sum().backward()
        for p in block.parameters():
            self.assertIsNotNone(p.grad)

    def test_short_sequence(self):
        """TCN should work even with T=1."""
        block = TCN(d_model=self.d, num_layers=3, kernel_size=3)
        x = torch.rand(self.B, 1, self.d)
        out = block(x)
        self.assertEqual(out.shape, (self.B, 1, self.d))


if __name__ == "__main__":
    unittest.main()
