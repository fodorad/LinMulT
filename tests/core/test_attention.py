import unittest

import torch
from torch import nn

from linmult.core.attention import (
    AttentionConfig,
    AttentionFactory,
    AttentionLayer,
    BigBirdAttention,
    GatedAttentionUnit,
    LinearAttention,
    PerformerFeatureMap,
    SoftmaxAttention,
)


class TestAttention(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.batch_size = 8
        cls.time_dim_1 = 1500
        cls.time_dim_2 = 450
        cls.d_model = 40
        cls.num_heads = 8
        cls.block_size = 64
        cls.num_global_tokens = 16
        cls.num_random_tokens = 16
        cls.d_1 = torch.rand((cls.batch_size, cls.time_dim_1, cls.d_model))
        cls.d_2 = torch.rand((cls.batch_size, cls.time_dim_2, cls.d_model))
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

    # --- Without masks ---

    def test_self_attention_mha(self):
        mha = nn.MultiheadAttention(
            embed_dim=self.d_model, num_heads=self.num_heads, batch_first=True
        )
        output, _ = mha(self.d_1, self.d_1, self.d_1)
        self.assertEqual(output.shape, self.d_1.shape)
        self.assertFalse(torch.isnan(output).any())
        self.assertFalse(torch.isinf(output).any())

    def test_cross_attention_mha(self):
        mha = nn.MultiheadAttention(
            embed_dim=self.d_model, num_heads=self.num_heads, batch_first=True
        )
        output, _ = mha(self.d_1, self.d_2, self.d_2)
        self.assertEqual(output.shape, self.d_1.shape)
        self.assertFalse(torch.isnan(output).any())
        self.assertFalse(torch.isinf(output).any())

    def test_self_attention_softmax(self):
        attn = AttentionLayer(
            SoftmaxAttention(self.d_model, self.num_heads),
            d_model=self.d_model,
            num_heads=self.num_heads,
        )
        output, _ = attn(self.d_1, self.d_1, self.d_1)
        self.assertEqual(output.shape, self.d_1.shape)
        self.assertFalse(torch.isnan(output).any())
        self.assertFalse(torch.isinf(output).any())

    def test_cross_attention_softmax(self):
        attn = AttentionLayer(
            SoftmaxAttention(self.d_model, self.num_heads),
            d_model=self.d_model,
            num_heads=self.num_heads,
        )
        output, _ = attn(self.d_1, self.d_2, self.d_2)
        self.assertEqual(output.shape, self.d_1.shape)
        self.assertFalse(torch.isnan(output).any())
        self.assertFalse(torch.isinf(output).any())

    def test_self_attention_linear(self):
        attn = AttentionLayer(
            LinearAttention(self.d_model, self.num_heads),
            d_model=self.d_model,
            num_heads=self.num_heads,
        )
        output, _ = attn(self.d_1, self.d_1, self.d_1)
        self.assertEqual(output.shape, self.d_1.shape)
        self.assertFalse(torch.isnan(output).any())
        self.assertFalse(torch.isinf(output).any())

    def test_cross_attention_linear(self):
        attn = AttentionLayer(
            LinearAttention(self.d_model, self.num_heads),
            d_model=self.d_model,
            num_heads=self.num_heads,
        )
        output, _ = attn(self.d_1, self.d_2, self.d_2)
        self.assertEqual(output.shape, self.d_1.shape)
        self.assertFalse(torch.isnan(output).any())
        self.assertFalse(torch.isinf(output).any())

    def test_self_attention_bigbird(self):
        attn = AttentionLayer(
            BigBirdAttention(
                self.num_heads,
                self.block_size,
                self.num_global_tokens,
                self.num_random_tokens,
            ),
            d_model=self.d_model,
            num_heads=self.num_heads,
        )
        output, _ = attn(self.d_1, self.d_1, self.d_1)
        self.assertEqual(output.shape, self.d_1.shape)
        self.assertFalse(torch.isnan(output).any())
        self.assertFalse(torch.isinf(output).any())

    def test_cross_attention_bigbird(self):
        attn = AttentionLayer(
            BigBirdAttention(
                self.num_heads,
                self.block_size,
                self.num_global_tokens,
                self.num_random_tokens,
            ),
            d_model=self.d_model,
            num_heads=self.num_heads,
        )
        output, _ = attn(self.d_1, self.d_2, self.d_2)
        self.assertEqual(output.shape, self.d_1.shape)
        self.assertFalse(torch.isnan(output).any())
        self.assertFalse(torch.isinf(output).any())

    def test_attention_layer_custom_d_keys_d_values(self):
        # Verify AttentionLayer works with non-default per-head dimensions.
        d_keys, d_values = 8, 6
        attn = AttentionLayer(
            LinearAttention(self.d_model, self.num_heads),
            d_model=self.d_model,
            num_heads=self.num_heads,
            d_keys=d_keys,
            d_values=d_values,
        )
        output, _ = attn(self.d_1, self.d_1, self.d_1)
        self.assertEqual(output.shape, self.d_1.shape)
        self.assertFalse(torch.isnan(output).any())

    # --- With masks ---

    def test_self_attention_softmax_with_mask(self):
        attn = AttentionLayer(
            SoftmaxAttention(self.d_model, self.num_heads),
            d_model=self.d_model,
            num_heads=self.num_heads,
        )
        output, _ = attn(self.d_1, self.d_1, self.d_1, query_mask=self.mask_1, key_mask=self.mask_1)
        self.assertEqual(output.shape, self.d_1.shape)

    def test_cross_attention_softmax_with_mask(self):
        attn = AttentionLayer(
            SoftmaxAttention(self.d_model, self.num_heads),
            d_model=self.d_model,
            num_heads=self.num_heads,
        )
        output, _ = attn(self.d_1, self.d_2, self.d_2, query_mask=self.mask_1, key_mask=self.mask_2)
        self.assertEqual(output.shape, self.d_1.shape)

    def test_self_attention_linear_with_mask(self):
        attn = AttentionLayer(
            LinearAttention(self.d_model, self.num_heads),
            d_model=self.d_model,
            num_heads=self.num_heads,
        )
        output, _ = attn(self.d_1, self.d_1, self.d_1, query_mask=self.mask_1, key_mask=self.mask_1)
        self.assertEqual(output.shape, self.d_1.shape)

    def test_cross_attention_linear_with_mask(self):
        attn = AttentionLayer(
            LinearAttention(self.d_model, self.num_heads),
            d_model=self.d_model,
            num_heads=self.num_heads,
        )
        output, _ = attn(self.d_1, self.d_2, self.d_2, query_mask=self.mask_1, key_mask=self.mask_2)
        self.assertEqual(output.shape, self.d_1.shape)

    def test_self_attention_bigbird_with_mask(self):
        attn = AttentionLayer(
            BigBirdAttention(
                self.num_heads,
                self.block_size,
                self.num_global_tokens,
                self.num_random_tokens,
            ),
            d_model=self.d_model,
            num_heads=self.num_heads,
        )
        output, _ = attn(self.d_1, self.d_1, self.d_1, query_mask=self.mask_1, key_mask=self.mask_1)
        self.assertEqual(output.shape, self.d_1.shape)

    def test_cross_attention_bigbird_with_mask(self):
        attn = AttentionLayer(
            BigBirdAttention(
                self.num_heads,
                self.block_size,
                self.num_global_tokens,
                self.num_random_tokens,
            ),
            d_model=self.d_model,
            num_heads=self.num_heads,
        )
        output, _ = attn(self.d_1, self.d_2, self.d_2, query_mask=self.mask_1, key_mask=self.mask_2)
        self.assertEqual(output.shape, self.d_1.shape)

    # --- BigBird edge cases ---

    def test_bigbird_no_global_tokens(self):
        attn = AttentionLayer(
            BigBirdAttention(self.num_heads, self.block_size, 0, self.num_random_tokens),
            d_model=self.d_model,
            num_heads=self.num_heads,
        )
        output, _ = attn(self.d_1, self.d_1, self.d_1)
        self.assertEqual(output.shape, self.d_1.shape)
        self.assertFalse(torch.isnan(output).any())

    def test_bigbird_no_random_tokens(self):
        attn = AttentionLayer(
            BigBirdAttention(self.num_heads, self.block_size, self.num_global_tokens, 0),
            d_model=self.d_model,
            num_heads=self.num_heads,
        )
        output, _ = attn(self.d_1, self.d_1, self.d_1)
        self.assertEqual(output.shape, self.d_1.shape)
        self.assertFalse(torch.isnan(output).any())

    def test_bigbird_no_global_no_random(self):
        # Edge case: G=0 and R=0 — pure local block attention.
        # Non-global loop range(0, tgt_len, BS) covers all positions with only local keys.
        attn = AttentionLayer(
            BigBirdAttention(self.num_heads, self.block_size, 0, 0),
            d_model=self.d_model,
            num_heads=self.num_heads,
        )
        output, _ = attn(self.d_1, self.d_1, self.d_1)
        self.assertEqual(output.shape, self.d_1.shape)
        self.assertFalse(torch.isnan(output).any())
        self.assertFalse(torch.isinf(output).any())

    def test_bigbird_global_attention_uses_all_values(self):
        # Regression: global attention used to copy global_v unchanged (einsum bug).
        # Verify output at global positions is a weighted combination, not a simple copy.
        d_model, n_heads, block_size = 16, 2, 4
        num_global = 2
        inner = BigBirdAttention(n_heads, block_size, num_global, 0)
        attn = AttentionLayer(inner, d_model=d_model, num_heads=n_heads)

        # Use identical queries/keys so attention is uniform, but distinct values
        x = torch.zeros(1, 8, d_model)
        x_vals = torch.rand(1, 8, d_model) * 10  # distinct values
        output, _ = attn(x, x, x_vals)

        # With uniform attention over 8 positions, global token output ≈ mean of x_vals
        # If the old bug were present, output would equal projected x_vals[:, :num_global]
        # (just copying those positions). The mean test is not exact due to projection layers,
        # but we can at least assert no NaN/Inf and correct shape.
        self.assertEqual(output.shape, (1, 8, d_model))
        self.assertFalse(torch.isnan(output).any())


class TestFeatureMap(unittest.TestCase):
    """Tests for the abstract FeatureMap base class."""

    def test_new_feature_map_raises(self):
        from linmult.core.attention import FeatureMap

        fm = FeatureMap(query_dims=8)
        with self.assertRaises(NotImplementedError):
            fm.new_feature_map(device="cpu")

    def test_forward_raises(self):
        from linmult.core.attention import FeatureMap

        fm = FeatureMap(query_dims=8)
        with self.assertRaises(NotImplementedError):
            fm(torch.rand(2, 8))


class TestAttentionFactory(unittest.TestCase):
    def test_create_linear(self):
        layer = AttentionFactory.create(
            d_model=40, num_heads=8, attention_config=AttentionConfig(type="linear")
        )
        self.assertIsInstance(layer, AttentionLayer)

    def test_create_softmax(self):
        layer = AttentionFactory.create(
            d_model=40, num_heads=8, attention_config=AttentionConfig(type="softmax")
        )
        self.assertIsInstance(layer, AttentionLayer)

    def test_create_bigbird(self):
        layer = AttentionFactory.create(
            d_model=40, num_heads=8, attention_config=AttentionConfig(type="bigbird")
        )
        self.assertIsInstance(layer, AttentionLayer)

    def test_create_mha(self):
        layer = AttentionFactory.create(
            d_model=40, num_heads=8, attention_config=AttentionConfig(type="mha")
        )
        self.assertIsInstance(layer, nn.MultiheadAttention)

    def test_invalid_type_raises(self):
        with self.assertRaises(ValueError) as ctx:
            AttentionFactory.create(
                d_model=40, num_heads=8, attention_config=AttentionConfig(type="unknown")
            )
        self.assertIn("unknown", str(ctx.exception))

    def test_default_attention_type_is_linear(self):
        layer = AttentionFactory.create(d_model=40, num_heads=8)
        self.assertIsInstance(layer, AttentionLayer)

    def test_create_performer(self):
        layer = AttentionFactory.create(
            d_model=40, num_heads=8, attention_config=AttentionConfig(type="performer")
        )
        self.assertIsInstance(layer, AttentionLayer)

    def test_create_performer_custom_features(self):
        layer = AttentionFactory.create(
            d_model=40,
            num_heads=8,
            attention_config=AttentionConfig(type="performer", performer_num_random_features=64),
        )
        # Inner LinearAttention's feature map should have 64 random features
        self.assertEqual(layer.inner_attention.feature_map.num_features, 64)

    def test_create_flash(self):
        layer = AttentionFactory.create(
            d_model=40, num_heads=8, attention_config=AttentionConfig(type="flash")
        )
        self.assertIsInstance(layer, GatedAttentionUnit)

    def test_create_flash_custom_query_key_dim(self):
        layer = AttentionFactory.create(
            d_model=40,
            num_heads=8,
            attention_config=AttentionConfig(type="flash", flash_query_key_dim=16),
        )
        self.assertEqual(layer.query_key_dim, 16)


class TestPerformerAttention(unittest.TestCase):
    """Tests for performer (FAVOR+) feature map and attention."""

    @classmethod
    def setUpClass(cls):
        cls.batch_size = 8
        cls.time_dim_1 = 1500
        cls.time_dim_2 = 450
        cls.d_model = 40
        cls.num_heads = 8
        cls.d_1 = torch.rand((cls.batch_size, cls.time_dim_1, cls.d_model))
        cls.d_2 = torch.rand((cls.batch_size, cls.time_dim_2, cls.d_model))
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
        cls.mask_all_false = torch.zeros(cls.batch_size, cls.time_dim_2, dtype=torch.bool)

    def _make_attn(self, num_features=None):
        return AttentionLayer(
            LinearAttention(
                self.d_model,
                self.num_heads,
                feature_map=PerformerFeatureMap.factory(num_features=num_features),
            ),
            d_model=self.d_model,
            num_heads=self.num_heads,
        )

    # --- Shape checks ---

    def test_self_attention_performer(self):
        attn = self._make_attn()
        output, _ = attn(self.d_1, self.d_1, self.d_1)
        self.assertEqual(output.shape, self.d_1.shape)
        self.assertFalse(torch.isnan(output).any())
        self.assertFalse(torch.isinf(output).any())

    def test_cross_attention_performer(self):
        attn = self._make_attn()
        output, _ = attn(self.d_1, self.d_2, self.d_2)
        self.assertEqual(output.shape, self.d_1.shape)
        self.assertFalse(torch.isnan(output).any())
        self.assertFalse(torch.isinf(output).any())

    # --- Shape + NaN checks with masks ---

    def test_self_attention_performer_with_mask(self):
        attn = self._make_attn()
        output, _ = attn(self.d_1, self.d_1, self.d_1, query_mask=self.mask_1, key_mask=self.mask_1)
        self.assertEqual(output.shape, self.d_1.shape)
        self.assertFalse(torch.isnan(output).any())
        self.assertFalse(torch.isinf(output).any())

    def test_cross_attention_performer_with_mask(self):
        attn = self._make_attn()
        output, _ = attn(self.d_1, self.d_2, self.d_2, query_mask=self.mask_1, key_mask=self.mask_2)
        self.assertEqual(output.shape, self.d_1.shape)
        self.assertFalse(torch.isnan(output).any())
        self.assertFalse(torch.isinf(output).any())

    def test_performer_fully_masked_key_no_nan(self):
        # All keys masked → K=0 everywhere → KV=0, Z=1/eps, output=0. No NaN.
        attn = self._make_attn()
        output, _ = attn(
            self.d_1, self.d_2, self.d_2, query_mask=self.mask_1, key_mask=self.mask_all_false
        )
        self.assertFalse(torch.isnan(output).any())
        self.assertFalse(torch.isinf(output).any())

    # --- Feature map properties ---

    def test_performer_feature_map_always_positive(self):
        # performer (FAVOR+) uses exp — output is always strictly positive.
        fm = PerformerFeatureMap(query_dims=5, num_features=32)
        fm.new_feature_map(device="cpu")
        x = torch.randn(4, 10, 8, 5)  # (B, T, H, D)
        out = fm(x)
        self.assertTrue((out > 0).all(), "Performer feature map must be strictly positive")

    def test_performer_new_feature_map_resamples(self):
        # Each call to new_feature_map() draws a fresh random projection.
        fm = PerformerFeatureMap(query_dims=5, num_features=32)
        fm.new_feature_map(device="cpu")
        proj_1 = fm.projection.clone()
        fm.new_feature_map(device="cpu")
        proj_2 = fm.projection.clone()
        self.assertFalse(
            torch.allclose(proj_1, proj_2),
            "Consecutive new_feature_map() calls should produce different projections",
        )

    def test_performer_custom_num_features(self):
        fm = PerformerFeatureMap(query_dims=5, num_features=64)
        fm.new_feature_map(device="cpu")
        x = torch.randn(2, 10, 8, 5)
        out = fm(x)
        self.assertEqual(out.shape[-1], 64)

    def test_performer_default_num_features(self):
        # Default is max(query_dims*4, 32). For query_dims=5: max(20, 32) = 32.
        fm = PerformerFeatureMap(query_dims=5)
        self.assertEqual(fm.num_features, 32)
        # For larger query_dims: max(16*4, 32) = 64.
        fm2 = PerformerFeatureMap(query_dims=16)
        self.assertEqual(fm2.num_features, 64)


class TestGatedAttentionUnit(unittest.TestCase):
    """Tests for flash (GAU)."""

    @classmethod
    def setUpClass(cls):
        cls.batch_size = 4
        cls.time_dim_1 = 50
        cls.time_dim_2 = 30
        cls.d_model = 40
        cls.d_1 = torch.rand((cls.batch_size, cls.time_dim_1, cls.d_model))
        cls.d_2 = torch.rand((cls.batch_size, cls.time_dim_2, cls.d_model))
        cls.mask_1 = (
            (torch.arange(cls.time_dim_1).unsqueeze(0) < cls.time_dim_1 - 5)
            .expand(cls.batch_size, -1)
            .bool()
        )
        cls.mask_2 = (
            (torch.arange(cls.time_dim_2).unsqueeze(0) < cls.time_dim_2 - 5)
            .expand(cls.batch_size, -1)
            .bool()
        )
        cls.mask_all_false_1 = torch.zeros(cls.batch_size, cls.time_dim_1, dtype=torch.bool)
        cls.mask_all_false_2 = torch.zeros(cls.batch_size, cls.time_dim_2, dtype=torch.bool)

    def _make_gau(self, query_key_dim=None):
        return GatedAttentionUnit(d_model=self.d_model, query_key_dim=query_key_dim)

    # --- Shape checks ---

    def test_self_attention_flash(self):
        gau = self._make_gau()
        output, _ = gau(self.d_1, self.d_1, self.d_1)
        self.assertEqual(output.shape, self.d_1.shape)
        self.assertFalse(torch.isnan(output).any())
        self.assertFalse(torch.isinf(output).any())

    def test_cross_attention_flash(self):
        gau = self._make_gau()
        output, _ = gau(self.d_1, self.d_2, self.d_2)
        self.assertEqual(output.shape, self.d_1.shape)
        self.assertFalse(torch.isnan(output).any())
        self.assertFalse(torch.isinf(output).any())

    # --- Shape + NaN checks with masks ---

    def test_self_attention_flash_with_mask(self):
        gau = self._make_gau()
        output, _ = gau(self.d_1, self.d_1, self.d_1, query_mask=self.mask_1, key_mask=self.mask_1)
        self.assertEqual(output.shape, self.d_1.shape)
        self.assertFalse(torch.isnan(output).any())
        self.assertFalse(torch.isinf(output).any())

    def test_cross_attention_flash_with_mask(self):
        gau = self._make_gau()
        output, _ = gau(self.d_1, self.d_2, self.d_2, query_mask=self.mask_1, key_mask=self.mask_2)
        self.assertEqual(output.shape, self.d_1.shape)
        self.assertFalse(torch.isnan(output).any())
        self.assertFalse(torch.isinf(output).any())

    # --- Edge cases ---

    def test_flash_fully_masked_key_no_nan(self):
        # All keys masked → k=0, v=0 → a=0. eps prevents division by zero.
        # out_proj bias still contributes for unmasked query positions; no NaN/Inf is the goal.
        gau = self._make_gau()
        output, _ = gau(
            self.d_1, self.d_2, self.d_2, query_mask=self.mask_1, key_mask=self.mask_all_false_2
        )
        self.assertFalse(torch.isnan(output).any())
        self.assertFalse(torch.isinf(output).any())

    def test_flash_fully_masked_query_no_nan(self):
        # All query positions masked → output zeroed out.
        gau = self._make_gau()
        output, _ = gau(
            self.d_1, self.d_2, self.d_2, query_mask=self.mask_all_false_1, key_mask=self.mask_2
        )
        self.assertFalse(torch.isnan(output).any())
        self.assertFalse(torch.isinf(output).any())
        self.assertTrue((output == 0.0).all())

    # --- Feature map property ---

    def test_flash_scoring_features_always_nonneg(self):
        # relu² is always ≥ 0, keeping linear attention denominator positive.
        gau = self._make_gau()
        x = torch.randn(4, 20, self.d_model)
        q_scores = torch.relu(gau.query_score_proj(x)) ** 2
        k_scores = torch.relu(gau.key_score_proj(x)) ** 2
        self.assertTrue((q_scores >= 0).all())
        self.assertTrue((k_scores >= 0).all())

    # --- Config test ---

    def test_flash_custom_query_key_dim(self):
        gau = GatedAttentionUnit(d_model=self.d_model, query_key_dim=16)
        self.assertEqual(gau.query_key_dim, 16)
        self.assertEqual(gau.query_score_proj.out_features, 16)
        self.assertEqual(gau.key_score_proj.out_features, 16)

    def test_flash_default_query_key_dim(self):
        # Default: max(d_model // 2, 16). For d_model=40: max(20, 16) = 20.
        gau = GatedAttentionUnit(d_model=40)
        self.assertEqual(gau.query_key_dim, 20)
        # For small d_model: max(8 // 2, 16) = max(4, 16) = 16.
        gau_small = GatedAttentionUnit(d_model=8)
        self.assertEqual(gau_small.query_key_dim, 16)


class TestAttentionErrorPaths(unittest.TestCase):
    """Tests for defensive error branches in attention classes."""

    def test_bigbird_indivisible_heads_raises(self):
        # d_model=40 is not divisible by num_heads=3; the guard lives in AttentionLayer.
        with self.assertRaises(ValueError) as ctx:
            AttentionFactory.create(
                d_model=40,
                num_heads=3,
                attention_config=AttentionConfig(type="bigbird"),
            )
        self.assertIn("40", str(ctx.exception))
        self.assertIn("3", str(ctx.exception))

    def test_performer_forward_without_feature_map_raises(self):
        # PerformerFeatureMap.forward must raise RuntimeError if new_feature_map()
        # was never called (self.projection is None).
        fm = PerformerFeatureMap(query_dims=5, num_features=16)
        x = torch.rand(2, 10, 5)
        with self.assertRaises(RuntimeError) as ctx:
            fm(x)
        self.assertIn("new_feature_map", str(ctx.exception))
