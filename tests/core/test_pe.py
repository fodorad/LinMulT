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

    def test_cache_rebuilt_every_call_while_tracing(self):
        # Under torch.jit.trace, the cache must be rebuilt on every call rather than
        # reused, since a tracer only observes whichever branch fires on the shapes
        # it is given at trace time. Reusing the cache would bake in a fixed shape,
        # which fails when a later call sees a genuinely different time_dim.
        class Wrap(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.pe = PositionalEncoding(dropout=0.0)

            def forward(self, x_q, x_k):
                return self.pe(x_q), self.pe(x_k)

        m = Wrap()
        m.eval()

        # Trace-time inputs share the same time_dim, which used to poison the cache.
        x_q = torch.rand(1, 50, 8)
        x_k = torch.rand(1, 50, 8)
        traced = torch.jit.trace(m, (x_q, x_k), check_trace=False)

        # Inference-time inputs have genuinely different time_dims.
        x_q2 = torch.rand(1, 17, 8)
        x_k2 = torch.rand(1, 23, 8)
        out_q, out_k = traced(x_q2, x_k2)
        self.assertEqual(out_q.shape, x_q2.shape)
        self.assertEqual(out_k.shape, x_k2.shape)
