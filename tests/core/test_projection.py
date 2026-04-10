import unittest

import torch

from linmult.core.projection import ProjectionModule


class TestProjectionModule(unittest.TestCase):
    def setUp(self):
        self.B, self.T = 2, 10

    def test_basic_projection(self):
        stage = ProjectionModule(input_feature_dims=[25, 35], d_model=40)
        x1 = torch.rand(self.B, self.T, 25)
        x2 = torch.rand(self.B, self.T, 35)
        out = stage([x1, x2])
        self.assertEqual(len(out), 2)
        self.assertEqual(out[0].shape, (self.B, self.T, 40))
        self.assertEqual(out[1].shape, (self.B, self.T, 40))

    def test_single_modality(self):
        stage = ProjectionModule(input_feature_dims=[25], d_model=16)
        x = torch.rand(self.B, self.T, 25)
        out = stage([x])
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0].shape, (self.B, self.T, 16))

    def test_dropout_zero(self):
        stage = ProjectionModule(input_feature_dims=[10], d_model=8, dropout=0.0)
        x = torch.rand(self.B, self.T, 10)
        out = stage([x])
        self.assertFalse(torch.isnan(out[0]).any())

    def test_dropout_nonzero(self):
        stage = ProjectionModule(input_feature_dims=[10], d_model=8, dropout=0.5)
        stage.train()
        x = torch.rand(self.B, self.T, 10)
        out = stage([x])
        self.assertEqual(out[0].shape, (self.B, self.T, 8))

    def test_special_handling_weighted_sum(self):
        special = {"feat": {"type": "weighted_sum", "start_layer": 0, "end_layer": 3}}
        stage = ProjectionModule(input_feature_dims=[10], d_model=8, special_handling=special)
        # 4D input: (B, N_layers, T, F)
        x = torch.rand(self.B, 5, self.T, 10)
        out = stage([x], names=["feat"])
        self.assertEqual(out[0].shape, (self.B, self.T, 8))

    def test_special_handling_ignored_without_name(self):
        special = {"feat": {"type": "weighted_sum", "start_layer": 0, "end_layer": 3}}
        stage = ProjectionModule(input_feature_dims=[10], d_model=8, special_handling=special)
        # 3D input without name — special handling not applied
        x = torch.rand(self.B, self.T, 10)
        out = stage([x])
        self.assertEqual(out[0].shape, (self.B, self.T, 8))

    def test_no_nan_output(self):
        stage = ProjectionModule(input_feature_dims=[25, 35], d_model=40)
        x1 = torch.rand(self.B, self.T, 25)
        x2 = torch.rand(self.B, self.T, 35)
        out = stage([x1, x2])
        for o in out:
            self.assertFalse(torch.isnan(o).any())

    def test_empty_special_handling(self):
        stage = ProjectionModule(input_feature_dims=[10], d_model=8, special_handling={})
        x = torch.rand(self.B, self.T, 10)
        out = stage([x])
        self.assertEqual(out[0].shape, (self.B, self.T, 8))

    def test_none_special_handling(self):
        stage = ProjectionModule(input_feature_dims=[10], d_model=8, special_handling=None)
        x = torch.rand(self.B, self.T, 10)
        out = stage([x])
        self.assertEqual(out[0].shape, (self.B, self.T, 8))

    # --- TCN ---

    def test_tcn_multi_modality(self):
        stage = ProjectionModule(
            input_feature_dims=[25, 35],
            d_model=40,
            add_tcn=True,
        )
        x1 = torch.rand(self.B, self.T, 25)
        x2 = torch.rand(self.B, self.T, 35)
        out = stage([x1, x2])
        self.assertEqual(len(out), 2)
        self.assertEqual(out[0].shape, (self.B, self.T, 40))
        self.assertEqual(out[1].shape, (self.B, self.T, 40))

    def test_tcn_single_modality(self):
        stage = ProjectionModule(
            input_feature_dims=[25],
            d_model=16,
            add_tcn=True,
            tcn_num_layers=2,
        )
        x = torch.rand(self.B, self.T, 25)
        out = stage([x])
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0].shape, (self.B, self.T, 16))

    def test_tcn_enabled_by_default(self):
        stage = ProjectionModule(input_feature_dims=[10], d_model=8)
        self.assertIsNotNone(stage.tcn_blocks)

    def test_tcn_per_modality_blocks(self):
        stage = ProjectionModule(
            input_feature_dims=[10, 20, 30],
            d_model=8,
            add_tcn=True,
        )
        self.assertIsNotNone(stage.tcn_blocks)
        self.assertEqual(len(stage.tcn_blocks), 3)

    def test_tcn_with_special_handling(self):
        special = {"feat": {"type": "weighted_sum", "start_layer": 0, "end_layer": 3}}
        stage = ProjectionModule(
            input_feature_dims=[10],
            d_model=8,
            special_handling=special,
            add_tcn=True,
        )
        x = torch.rand(self.B, 5, self.T, 10)
        out = stage([x], names=["feat"])
        self.assertEqual(out[0].shape, (self.B, self.T, 8))


if __name__ == "__main__":
    unittest.main()
