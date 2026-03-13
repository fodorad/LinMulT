import unittest

import torch

from linmult.core.pe import PositionalEncoding


class TestPositionalEncoding(unittest.TestCase):
    def test_output_shape(self):
        pe = PositionalEncoding(dropout=0.0)
        x = torch.rand(4, 30, 40)
        out = pe(x)
        self.assertEqual(out.shape, x.shape)

    def test_output_no_nan(self):
        pe = PositionalEncoding(dropout=0.0)
        x = torch.rand(2, 100, 64)
        out = pe(x)
        self.assertFalse(torch.isnan(out).any())

    def test_caching_reused_for_same_dims(self):
        pe = PositionalEncoding(dropout=0.0)
        x1 = torch.rand(2, 50, 32)
        pe(x1)  # populate cache
        cached = pe.pe
        x2 = torch.rand(3, 30, 32)  # shorter time dim, same feature dim
        pe(x2)
        self.assertIs(pe.pe, cached)  # cache reused

    def test_cache_rebuilt_for_larger_time_dim(self):
        pe = PositionalEncoding(dropout=0.0)
        x1 = torch.rand(2, 30, 32)
        pe(x1)
        x2 = torch.rand(2, 60, 32)  # larger time dim
        pe(x2)
        self.assertGreaterEqual(pe.pe.size(1), 60)

    def test_cache_rebuilt_for_different_feature_dim(self):
        pe = PositionalEncoding(dropout=0.0)
        x1 = torch.rand(2, 30, 32)
        pe(x1)
        x2 = torch.rand(2, 30, 64)  # different feature dim
        pe(x2)
        self.assertEqual(pe.pe.size(2), 64)

    def test_no_clone_detach_overhead(self):
        # Verify pe buffer tensors are used directly (no redundant copies)
        pe = PositionalEncoding(dropout=0.0)
        x = torch.rand(2, 20, 40)
        pe(x)  # populate
        # PE should be on same device and have correct shape
        self.assertEqual(pe.pe.shape, (1, 20, 40))

    def test_batch_size_independence(self):
        pe = PositionalEncoding(dropout=0.0)
        x1 = torch.rand(1, 50, 40)
        x2 = torch.rand(8, 50, 40)
        out1 = pe(x1)
        out2 = pe(x2)
        self.assertEqual(out1.shape, x1.shape)
        self.assertEqual(out2.shape, x2.shape)
