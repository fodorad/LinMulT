import unittest

import torch

from linmult.core.config import HeadConfig
from linmult.core.heads import (
    DownsampleHead,
    HeadFactory,
    SequenceAggregationHead,
    SequenceHead,
    SimpleHead,
    UpsampleHead,
    VectorHead,
)


class TestHeadFactory(unittest.TestCase):
    def test_registered_types(self):
        for head_type in ("sequence_aggregation", "sequence", "vector", "simple"):
            cfg = HeadConfig(type=head_type, output_dim=5)
            head = HeadFactory.create_head(head_type, input_dim=32, output_dim=5, config=cfg)
            self.assertIsNotNone(head)
        for head_type, cfg in [
            (
                "upsample",
                HeadConfig(type="upsample", output_dim=16, input_time_dim=10, output_time_dim=40),
            ),
            (
                "downsample",
                HeadConfig(type="downsample", output_dim=16, input_time_dim=40, output_time_dim=10),
            ),
        ]:
            head = HeadFactory.create_head(head_type, input_dim=32, output_dim=16, config=cfg)
            self.assertIsNotNone(head)

    def test_unknown_type_raises(self):
        with self.assertRaises(ValueError):
            cfg = HeadConfig(type="nonexistent", output_dim=5)
            HeadFactory.create_head("nonexistent", input_dim=32, output_dim=5, config=cfg)

    def test_register_custom_head(self):
        import torch.nn as nn

        from linmult.core.heads import BaseHead

        class MyHead(BaseHead):
            def __init__(self, input_dim, output_dim, config: HeadConfig):
                super().__init__(input_dim, output_dim, config)
                self.linear = nn.Linear(input_dim, output_dim)

            def forward(self, x, **_kwargs):
                return self.linear(x)

        HeadFactory.register_head("my_custom", MyHead)
        cfg = HeadConfig(type="my_custom", output_dim=3)
        head = HeadFactory.create_head("my_custom", input_dim=8, output_dim=3, config=cfg)
        x = torch.rand(2, 8)
        out = head(x)
        self.assertEqual(out.shape, (2, 3))


class TestSequenceAggregationHead(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.B, cls.T, cls.F = 4, 20, 32
        cls.x = torch.rand(cls.B, cls.T, cls.F)
        cls.mask = torch.ones(cls.B, cls.T, dtype=torch.bool)
        cls.mask[:, 15:] = False

    def test_gap_pooling(self):
        cfg = HeadConfig(type="sequence_aggregation", output_dim=5, pooling="gap")
        head = SequenceAggregationHead(self.F, 5, cfg)
        out = head(self.x)
        self.assertEqual(out.shape, (self.B, 5))

    def test_gmp_pooling(self):
        cfg = HeadConfig(type="sequence_aggregation", output_dim=5, pooling="gmp")
        head = SequenceAggregationHead(self.F, 5, cfg)
        out = head(self.x)
        self.assertEqual(out.shape, (self.B, 5))

    def test_attention_pooling(self):
        cfg = HeadConfig(type="sequence_aggregation", output_dim=5, pooling="attentionpool")
        head = SequenceAggregationHead(self.F, 5, cfg)
        out = head(self.x)
        self.assertEqual(out.shape, (self.B, 5))

    def test_with_mask(self):
        cfg = HeadConfig(type="sequence_aggregation", output_dim=5, pooling="attentionpool")
        head = SequenceAggregationHead(self.F, 5, cfg)
        out = head(self.x, mask=self.mask)
        self.assertEqual(out.shape, (self.B, 5))
        self.assertFalse(torch.isnan(out).any())

    def test_with_in_norm(self):
        cfg = HeadConfig(type="sequence_aggregation", output_dim=5, norm="in")
        head = SequenceAggregationHead(self.F, 5, cfg)
        out = head(self.x)
        self.assertEqual(out.shape, (self.B, 5))

    def test_invalid_pooling_raises(self):
        with self.assertRaises(ValueError):
            cfg = HeadConfig(type="sequence_aggregation", output_dim=5, pooling="unknown")
            SequenceAggregationHead(self.F, 5, cfg)

    def test_invalid_norm_raises(self):
        with self.assertRaises(ValueError):
            cfg = HeadConfig(type="sequence_aggregation", output_dim=5, norm="layer")
            SequenceAggregationHead(self.F, 5, cfg)


class TestSequenceHead(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.B, cls.T, cls.F = 4, 20, 32
        cls.x = torch.rand(cls.B, cls.T, cls.F)
        cls.mask = torch.ones(cls.B, cls.T, dtype=torch.bool)
        cls.mask[:, 15:] = False

    def test_output_shape(self):
        head = SequenceHead(self.F, 5, HeadConfig(type="sequence", output_dim=5))
        out = head(self.x)
        self.assertEqual(out.shape, (self.B, self.T, 5))

    def test_with_mask(self):
        head = SequenceHead(self.F, 5, HeadConfig(type="sequence", output_dim=5))
        out = head(self.x, mask=self.mask)
        self.assertEqual(out.shape, (self.B, self.T, 5))
        # Masked positions should be zeroed out
        self.assertTrue(torch.allclose(out[:, 15:, :], torch.zeros(self.B, 5, 5)))

    def test_with_in_norm(self):
        head = SequenceHead(self.F, 5, HeadConfig(type="sequence", output_dim=5, norm="in"))
        out = head(self.x)
        self.assertEqual(out.shape, (self.B, self.T, 5))

    def test_invalid_norm_raises(self):
        with self.assertRaises(ValueError):
            SequenceHead(self.F, 5, HeadConfig(type="sequence", output_dim=5, norm="layer"))


class TestVectorHead(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.B, cls.F = 4, 32
        cls.x = torch.rand(cls.B, cls.F)

    def test_output_shape(self):
        head = VectorHead(self.F, 5, HeadConfig(type="vector", output_dim=5))
        out = head(self.x)
        self.assertEqual(out.shape, (self.B, 5))

    def test_with_in_norm(self):
        head = VectorHead(self.F, 5, HeadConfig(type="vector", output_dim=5, norm="in"))
        out = head(self.x)
        self.assertEqual(out.shape, (self.B, 5))
        self.assertFalse(torch.isnan(out).any())

    def test_batch_size_one(self):
        head = VectorHead(self.F, 5, HeadConfig(type="vector", output_dim=5))
        head.eval()  # BN requires B>1 in training mode
        x = torch.rand(1, self.F)
        out = head(x)
        self.assertEqual(out.shape, (1, 5))

    def test_invalid_norm_raises(self):
        with self.assertRaises(ValueError):
            VectorHead(self.F, 5, HeadConfig(type="vector", output_dim=5, norm="layer"))


class TestSimpleHead(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.B, cls.T, cls.F = 4, 20, 32
        cls.x_seq = torch.rand(cls.B, cls.T, cls.F)
        cls.x_vec = torch.rand(cls.B, cls.F)

    def test_attentionpool_pooling(self):
        head = SimpleHead(
            self.F, 5, HeadConfig(type="simple", output_dim=5, pooling="attentionpool")
        )
        out = head(self.x_seq)
        self.assertEqual(out.shape, (self.B, 5))

    def test_gap_pooling(self):
        head = SimpleHead(self.F, 5, HeadConfig(type="simple", output_dim=5, pooling="gap"))
        out = head(self.x_seq)
        self.assertEqual(out.shape, (self.B, 5))

    def test_gap_pooling_with_mask(self):
        # regression: mask was silently ignored before (now uses GlobalAvgPooling)
        head = SimpleHead(self.F, 5, HeadConfig(type="simple", output_dim=5, pooling="gap"))
        mask = torch.ones(self.B, self.T, dtype=torch.bool)
        mask[:, 15:] = False
        out = head(self.x_seq, mask=mask)
        self.assertEqual(out.shape, (self.B, 5))
        self.assertFalse(torch.isnan(out).any())

    def test_gmp_pooling(self):
        head = SimpleHead(self.F, 5, HeadConfig(type="simple", output_dim=5, pooling="gmp"))
        out = head(self.x_seq)
        self.assertEqual(out.shape, (self.B, 5))

    def test_gmp_pooling_with_mask(self):
        # regression: mask was silently ignored before (now uses GlobalMaxPooling)
        head = SimpleHead(self.F, 5, HeadConfig(type="simple", output_dim=5, pooling="gmp"))
        mask = torch.ones(self.B, self.T, dtype=torch.bool)
        mask[:, 15:] = False
        out = head(self.x_seq, mask=mask)
        self.assertEqual(out.shape, (self.B, 5))
        self.assertFalse(torch.isnan(out).any())

    def test_invalid_pool_raises(self):
        with self.assertRaises(ValueError):
            SimpleHead(self.F, 5, HeadConfig(type="simple", output_dim=5, pooling="unknown"))


class TestUpsampleHead(unittest.TestCase):
    def test_missing_input_time_dim_raises(self):
        with self.assertRaises(ValueError):
            UpsampleHead(16, 8, HeadConfig(type="upsample", output_dim=8, output_time_dim=40))

    def test_missing_output_time_dim_raises(self):
        with self.assertRaises(ValueError):
            UpsampleHead(16, 8, HeadConfig(type="upsample", output_dim=8, input_time_dim=10))

    def test_output_shape(self):
        head = UpsampleHead(
            input_dim=32,
            output_dim=16,
            config=HeadConfig(
                type="upsample", output_dim=16, input_time_dim=10, output_time_dim=40
            ),
        )
        x = torch.rand(2, 10, 32)
        out = head(x)
        self.assertEqual(out.shape, (2, 40, 16))

    def test_with_mask(self):
        head = UpsampleHead(
            input_dim=32,
            output_dim=16,
            config=HeadConfig(
                type="upsample", output_dim=16, input_time_dim=10, output_time_dim=40
            ),
        )
        x = torch.rand(2, 10, 32)
        mask = torch.ones(2, 10, dtype=torch.bool)
        out = head(x, mask=mask)
        self.assertEqual(out.shape, (2, 40, 16))
        self.assertFalse(torch.isnan(out).any())

    def test_no_upsample_needed(self):
        # When output_time_dim <= input_time_dim*2 range but exact target
        head = UpsampleHead(
            input_dim=16,
            output_dim=8,
            config=HeadConfig(type="upsample", output_dim=8, input_time_dim=10, output_time_dim=10),
        )
        x = torch.rand(2, 10, 16)
        out = head(x)
        self.assertEqual(out.shape, (2, 10, 8))


class TestDownsampleHead(unittest.TestCase):
    def test_missing_input_time_dim_raises(self):
        with self.assertRaises(ValueError):
            DownsampleHead(16, 8, HeadConfig(type="downsample", output_dim=8, output_time_dim=10))

    def test_missing_output_time_dim_raises(self):
        with self.assertRaises(ValueError):
            DownsampleHead(16, 8, HeadConfig(type="downsample", output_dim=8, input_time_dim=40))

    def test_output_shape(self):
        head = DownsampleHead(
            input_dim=32,
            output_dim=16,
            config=HeadConfig(
                type="downsample", output_dim=16, input_time_dim=40, output_time_dim=10
            ),
        )
        x = torch.rand(2, 40, 32)
        out = head(x)
        self.assertEqual(out.shape, (2, 10, 16))

    def test_with_mask(self):
        head = DownsampleHead(
            input_dim=32,
            output_dim=16,
            config=HeadConfig(
                type="downsample", output_dim=16, input_time_dim=40, output_time_dim=10
            ),
        )
        x = torch.rand(2, 40, 32)
        mask = torch.ones(2, 40, dtype=torch.bool)
        out = head(x, mask=mask)
        self.assertEqual(out.shape, (2, 10, 16))
        self.assertFalse(torch.isnan(out).any())

    def test_no_downsample_needed(self):
        head = DownsampleHead(
            input_dim=16,
            output_dim=8,
            config=HeadConfig(
                type="downsample", output_dim=8, input_time_dim=10, output_time_dim=10
            ),
        )
        x = torch.rand(2, 10, 16)
        out = head(x)
        self.assertEqual(out.shape, (2, 10, 8))
