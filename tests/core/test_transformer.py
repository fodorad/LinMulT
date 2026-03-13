import unittest

import torch

from linmult.core.attention import AttentionConfig
from linmult.core.transformer import TransformerEncoder, TransformerEncoderLayer


class TestTransformerEncoderLayerValidation(unittest.TestCase):
    """Tests for the corrected mask validation logic."""

    def setUp(self):
        self.layer = TransformerEncoderLayer(d_model=16)
        self.x = torch.rand(2, 10, 16)

    def test_wrong_dtype_mask_raises(self):
        bad_mask = torch.ones(2, 10, dtype=torch.float32)  # float, not bool
        with self.assertRaises(ValueError):
            self.layer(self.x, query_mask=bad_mask)

    def test_wrong_shape_mask_raises(self):
        bad_mask = torch.ones(2, 5, dtype=torch.bool)  # wrong T
        with self.assertRaises(ValueError):
            self.layer(self.x, query_mask=bad_mask)

    def test_correct_mask_passes(self):
        good_mask = torch.ones(2, 10, dtype=torch.bool)
        out = self.layer(self.x, query_mask=good_mask)
        self.assertEqual(out.shape, (2, 10, 16))

    def test_key_mask_with_no_key_raises(self):
        # key_mask provided but x_k is None
        key_mask = torch.ones(2, 10, dtype=torch.bool)
        with self.assertRaises(ValueError):
            self.layer(self.x, key_mask=key_mask)

    def test_key_mask_wrong_shape_raises(self):
        # key_mask shape (B, T_2) must match x_k shape[:2]; wrong shape raises ValueError.
        x_k = torch.rand(2, 8, 16)
        bad_key_mask = torch.ones(2, 5, dtype=torch.bool)  # wrong T
        with self.assertRaises(ValueError) as ctx:
            self.layer(self.x, x_k, x_k, key_mask=bad_key_mask)
        self.assertIn("key_mask", str(ctx.exception))

    def test_key_mask_wrong_dtype_raises(self):
        # key_mask must be bool dtype; float raises ValueError.
        x_k = torch.rand(2, 8, 16)
        bad_key_mask = torch.ones(2, 8, dtype=torch.float32)
        with self.assertRaises(ValueError) as ctx:
            self.layer(self.x, x_k, x_k, key_mask=bad_key_mask)
        self.assertIn("key_mask", str(ctx.exception))


class TestTransformerEncoderMHA(unittest.TestCase):
    """Tests MHA attention type now correctly passes masks."""

    def test_mha_self_attention_with_mask(self):
        encoder = TransformerEncoder(
            d_model=16, attention_config=AttentionConfig(type="mha"), num_layers=1
        )
        x = torch.rand(2, 10, 16)
        mask = torch.ones(2, 10, dtype=torch.bool)
        mask[:, 8:] = False
        out = encoder(x, query_mask=mask)
        self.assertEqual(out.shape, (2, 10, 16))
        self.assertFalse(torch.isnan(out).any())

    def test_mha_cross_attention_with_mask(self):
        encoder = TransformerEncoder(
            d_model=16, attention_config=AttentionConfig(type="mha"), num_layers=1
        )
        xq = torch.rand(2, 10, 16)
        xk = torch.rand(2, 15, 16)
        q_mask = torch.ones(2, 10, dtype=torch.bool)
        k_mask = torch.ones(2, 15, dtype=torch.bool)
        k_mask[:, 12:] = False
        out = encoder(xq, xk, xk, query_mask=q_mask, key_mask=k_mask)
        self.assertEqual(out.shape, (2, 10, 16))
        self.assertFalse(torch.isnan(out).any())


class TestTemporalFactoryValidation(unittest.TestCase):
    """Tests for TemporalFactory error handling."""

    def test_invalid_reducer_raises(self):
        from linmult.core.temporal import TemporalFactory

        with self.assertRaises(ValueError) as ctx:
            TemporalFactory.create_reducer(d_model=40, reducer="unknown_reducer")
        self.assertIn("unknown_reducer", str(ctx.exception))

    def test_temporal_factory_padding_aligner(self):
        from linmult.core.temporal import TemporalFactory

        padder = TemporalFactory.create_aligner("padding")
        x = torch.rand(2, 5, 8)
        out, mask = padder(x, time_dim=10)
        self.assertEqual(out.shape, (2, 10, 8))
        self.assertEqual(mask.shape, (2, 10))
        self.assertTrue(mask[:, :5].all())
        self.assertFalse(mask[:, 5:].any())


class TestTransformerEncoder(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.batch_size = 8
        cls.time_dim_1 = 1500
        cls.time_dim_2 = 450
        cls.time_dim_3 = 450
        cls.d_model = 40
        cls.d_1 = torch.rand((cls.batch_size, cls.time_dim_1, cls.d_model))
        cls.d_2 = torch.rand((cls.batch_size, cls.time_dim_2, cls.d_model))
        cls.d_3 = torch.rand((cls.batch_size, cls.time_dim_3, cls.d_model))
        cls.mask_1 = (
            (torch.arange(cls.time_dim_1).unsqueeze(0) < cls.time_dim_1 - 10)
            .expand(cls.batch_size, -1)
            .bool()
        )
        cls.mask_2 = (
            (torch.arange(cls.time_dim_2).unsqueeze(0) < cls.time_dim_2 - 10)
            .expand(cls.batch_size, -1)
            .bool()
        )
        cls.mask_3f = torch.zeros((cls.batch_size, cls.time_dim_3), dtype=torch.bool)

    # --- Without masks ---

    def test_self_attention(self):
        encoder = TransformerEncoder(d_model=self.d_model)
        output = encoder(self.d_1)
        self.assertEqual(output.shape, (self.batch_size, self.time_dim_1, self.d_model))

    def test_cross_attention(self):
        encoder = TransformerEncoder(d_model=self.d_model)
        output = encoder(self.d_1, self.d_2, self.d_2)
        self.assertEqual(output.shape, (self.batch_size, self.time_dim_1, self.d_model))

    # --- With masks ---

    def test_self_attention_with_mask(self):
        encoder = TransformerEncoder(d_model=self.d_model)
        output = encoder(self.d_1, query_mask=self.mask_1)
        self.assertEqual(output.shape, (self.batch_size, self.time_dim_1, self.d_model))

    def test_cross_attention_with_mask(self):
        encoder = TransformerEncoder(d_model=self.d_model)
        output = encoder(self.d_1, self.d_2, self.d_2, query_mask=self.mask_1, key_mask=self.mask_2)
        self.assertEqual(output.shape, (self.batch_size, self.time_dim_1, self.d_model))

    def test_self_attention_with_full_query_mask(self):
        encoder = TransformerEncoder(d_model=self.d_model)
        output = encoder(self.d_3, query_mask=self.mask_3f)
        self.assertEqual(output.shape, (self.batch_size, self.time_dim_3, self.d_model))

    def test_cross_attention_with_full_query_mask(self):
        encoder = TransformerEncoder(d_model=self.d_model)
        output = encoder(
            self.d_3, self.d_2, self.d_2, query_mask=self.mask_3f, key_mask=self.mask_2
        )
        self.assertEqual(output.shape, (self.batch_size, self.time_dim_3, self.d_model))

    def test_cross_attention_with_full_key_mask(self):
        encoder = TransformerEncoder(d_model=self.d_model)
        output = encoder(
            self.d_2, self.d_3, self.d_3, query_mask=self.mask_2, key_mask=self.mask_3f
        )
        self.assertEqual(output.shape, (self.batch_size, self.time_dim_2, self.d_model))

    # --- Softmax/BigBird NaN regression (Issue 2) ---

    def test_softmax_no_nan_with_partial_query_mask(self):
        # Regression: softmax attention over all-masked query rows produced NaN, which
        # survived masked pooling (NaN * 0 = NaN in IEEE 754).
        encoder = TransformerEncoder(
            d_model=self.d_model, attention_config=AttentionConfig(type="softmax")
        )
        output = encoder(self.d_1, query_mask=self.mask_1)
        self.assertEqual(output.shape, (self.batch_size, self.time_dim_1, self.d_model))
        self.assertFalse(torch.isnan(output).any())

    def test_bigbird_no_nan_with_partial_query_mask(self):
        # Same regression for bigbird attention.
        encoder = TransformerEncoder(
            d_model=self.d_model, attention_config=AttentionConfig(type="bigbird")
        )
        output = encoder(self.d_1, query_mask=self.mask_1)
        self.assertEqual(output.shape, (self.batch_size, self.time_dim_1, self.d_model))
        self.assertFalse(torch.isnan(output).any())

    def test_softmax_cross_no_nan_with_partial_query_mask(self):
        # Cross-attention variant: masked query positions should not produce NaN.
        encoder = TransformerEncoder(
            d_model=self.d_model, attention_config=AttentionConfig(type="softmax")
        )
        output = encoder(self.d_1, self.d_2, self.d_2, query_mask=self.mask_1, key_mask=self.mask_2)
        self.assertEqual(output.shape, (self.batch_size, self.time_dim_1, self.d_model))
        self.assertFalse(torch.isnan(output).any())
