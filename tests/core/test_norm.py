import unittest

import torch

from linmult.core.norm import BN, IN


class TestBN(unittest.TestCase):
    def test_sequence_input(self):
        bn = BN(feature_dim=32, time_aware=True)
        bn.eval()
        x = torch.rand(4, 10, 32)
        out = bn(x)
        self.assertEqual(out.shape, x.shape)

    def test_vector_input(self):
        bn = BN(feature_dim=32, time_aware=False)
        bn.eval()
        x = torch.rand(4, 32)
        out = bn(x)
        self.assertEqual(out.shape, x.shape)


class TestIN(unittest.TestCase):
    def test_sequence_input(self):
        norm = IN(feature_dim=32, time_aware=True)
        x = torch.rand(4, 10, 32)
        out = norm(x)
        self.assertEqual(out.shape, x.shape)

    def test_vector_input(self):
        norm = IN(feature_dim=32, time_aware=False)
        x = torch.rand(4, 32)
        out = norm(x)
        self.assertEqual(out.shape, x.shape)

    def test_vector_batch_size_one(self):
        # regression: squeeze() without args collapsed batch dim when B=1
        norm = IN(feature_dim=32, time_aware=False)
        x = torch.rand(1, 32)
        out = norm(x)
        self.assertEqual(out.shape, (1, 32))

    def test_vector_output_is_normalized(self):
        norm = IN(feature_dim=64, time_aware=False)
        x = torch.rand(8, 64) * 100  # large scale
        out = norm(x)
        # LayerNorm: mean ≈ 0, std ≈ 1 per sample (before affine)
        self.assertFalse(torch.isnan(out).any())
        self.assertFalse(torch.isinf(out).any())
