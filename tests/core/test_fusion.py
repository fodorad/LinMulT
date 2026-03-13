import unittest

import torch

from linmult.core.attention import AttentionConfig
from linmult.core.fusion import FusionModule


class TestFusionModuleConcat(unittest.TestCase):
    """Tests for the concatenation path (no TAM)."""

    def setUp(self):
        self.B, self.T, self.d = 2, 10, 8
        self.n = 3  # number of branches
        self.x_list = [torch.rand(self.B, self.T, self.d) for _ in range(self.n)]
        self.mask_list = [None] * self.n

    def test_plain_concat_no_modules(self):
        fusion = FusionModule(input_dim=self.d, n_branches=self.n, d_model=self.d)
        x, mask = fusion(self.x_list, self.mask_list)
        self.assertEqual(x.shape, (self.B, self.T, self.n * self.d))
        self.assertIsNone(mask)

    def test_concat_with_mask_propagation(self):
        mask_list = [torch.ones(self.B, self.T, dtype=torch.bool) for _ in range(self.n)]
        mask_list[0][:, 8:] = False
        fusion = FusionModule(input_dim=self.d, n_branches=self.n, d_model=self.d)
        x, mask = fusion(self.x_list, mask_list)
        self.assertEqual(x.shape, (self.B, self.T, self.n * self.d))
        self.assertIsNotNone(mask)
        # AND: positions 8-9 should be False
        self.assertFalse(mask[:, 8:].any())

    def test_concat_with_sat(self):
        fusion = FusionModule(
            input_dim=self.d,
            n_branches=self.n,
            d_model=self.d,
            num_heads=4,
            attention_config=AttentionConfig(type="linear"),
            add_sat_fusion=True,
            fusion_sat_num_layers=1,
        )
        x, mask = fusion(self.x_list, self.mask_list)
        self.assertEqual(x.shape, (self.B, self.T, self.n * self.d))

    def test_concat_with_ffn(self):
        fusion = FusionModule(
            input_dim=self.d, n_branches=self.n, d_model=self.d, add_ffn_fusion=True
        )
        x, mask = fusion(self.x_list, self.mask_list)
        self.assertEqual(x.shape, (self.B, self.T, self.n * self.d))

    def test_concat_with_sat_and_ffn(self):
        fusion = FusionModule(
            input_dim=self.d,
            n_branches=self.n,
            d_model=self.d,
            num_heads=4,
            attention_config=AttentionConfig(type="linear"),
            add_sat_fusion=True,
            fusion_sat_num_layers=1,
            add_ffn_fusion=True,
        )
        x, mask = fusion(self.x_list, self.mask_list)
        self.assertEqual(x.shape, (self.B, self.T, self.n * self.d))

    def test_no_nan_output(self):
        fusion = FusionModule(input_dim=self.d, n_branches=self.n, d_model=self.d)
        x, _ = fusion(self.x_list, self.mask_list)
        self.assertFalse(torch.isnan(x).any())

    def test_output_dim_property(self):
        fusion = FusionModule(input_dim=self.d, n_branches=self.n, d_model=self.d)
        self.assertEqual(fusion.output_dim, self.n * self.d)


class TestFusionModuleTRM(unittest.TestCase):
    """Tests for the concat path with per-branch TRM (time reduction)."""

    def setUp(self):
        self.B, self.T, self.d = 2, 10, 8
        self.n = 2
        self.x_list = [torch.rand(self.B, self.T, self.d) for _ in range(self.n)]
        self.mask_list = [None] * self.n

    def test_trm_reduces_time_dim(self):
        fusion = FusionModule(
            input_dim=self.d, n_branches=self.n, d_model=self.d, time_dim_reducer="gap"
        )
        x, mask = fusion(self.x_list, self.mask_list)
        # TRM removes time axis → (B, n*d)
        self.assertEqual(x.shape, (self.B, self.n * self.d))
        self.assertIsNone(mask)

    def test_trm_attentionpool(self):
        fusion = FusionModule(
            input_dim=self.d,
            n_branches=self.n,
            d_model=self.d,
            time_dim_reducer="attentionpool",
        )
        x, mask = fusion(self.x_list, self.mask_list)
        self.assertEqual(x.shape, (self.B, self.n * self.d))
        self.assertFalse(torch.isnan(x).any())

    def test_trm_with_mask(self):
        masks = [torch.ones(self.B, self.T, dtype=torch.bool) for _ in range(self.n)]
        masks[0][:, 7:] = False
        fusion = FusionModule(
            input_dim=self.d, n_branches=self.n, d_model=self.d, time_dim_reducer="gap"
        )
        x, mask = fusion(self.x_list, masks)
        self.assertEqual(x.shape, (self.B, self.n * self.d))
        self.assertFalse(torch.isnan(x).any())

    def test_output_dim_with_trm(self):
        fusion = FusionModule(
            input_dim=self.d, n_branches=self.n, d_model=self.d, time_dim_reducer="gap"
        )
        self.assertEqual(fusion.output_dim, self.n * self.d)


class TestFusionModuleTAM(unittest.TestCase):
    """Tests for the TAM fusion path."""

    def setUp(self):
        self.B, self.T, self.d = 2, 10, 8
        self.n = 2
        self.x_list = [torch.rand(self.B, self.T, self.d) for _ in range(self.n)]
        self.mask_list = [None] * self.n

    def test_tam_path(self):
        fusion = FusionModule(
            input_dim=self.d,
            n_branches=self.n,
            d_model=self.d,
            num_heads=4,
            attention_config=AttentionConfig(type="linear"),
            add_tam_fusion=True,
            tam_aligner="aap",
            tam_time_dim=self.T,
            fusion_num_layers=1,
        )
        x, mask = fusion(self.x_list, self.mask_list)
        # TAM output: (B, T, n_branches * d_model)
        self.assertEqual(x.shape, (self.B, self.T, self.n * self.d))

    def test_tam_path_no_nan(self):
        fusion = FusionModule(
            input_dim=self.d,
            n_branches=self.n,
            d_model=self.d,
            num_heads=4,
            attention_config=AttentionConfig(type="linear"),
            add_tam_fusion=True,
            tam_aligner="aap",
            tam_time_dim=self.T,
            fusion_num_layers=1,
        )
        x, _ = fusion(self.x_list, self.mask_list)
        self.assertFalse(torch.isnan(x).any())

    def test_tam_output_dim_property(self):
        fusion = FusionModule(
            input_dim=self.d,
            n_branches=self.n,
            d_model=self.d,
            num_heads=4,
            attention_config=AttentionConfig(type="linear"),
            add_tam_fusion=True,
            tam_aligner="aap",
            tam_time_dim=self.T,
            fusion_num_layers=1,
        )
        # TAM changes dim to n_branches * d_model
        self.assertEqual(fusion.output_dim, self.n * self.d)

    def test_tam_with_sat(self):
        fusion = FusionModule(
            input_dim=self.d,
            n_branches=self.n,
            d_model=self.d,
            num_heads=4,
            attention_config=AttentionConfig(type="linear"),
            add_tam_fusion=True,
            tam_aligner="aap",
            tam_time_dim=self.T,
            fusion_num_layers=1,
            add_sat_fusion=True,
            fusion_sat_num_layers=1,
        )
        x, _ = fusion(self.x_list, self.mask_list)
        self.assertEqual(x.shape, (self.B, self.T, self.n * self.d))


if __name__ == "__main__":
    unittest.main()
