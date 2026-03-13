import unittest

import torch

from linmult import FIXTURE_DIR, LinT
from linmult.core.config import LinTConfig

CONFIGS_DIR = FIXTURE_DIR / "configs"


class TestLinT(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.batch_size = 8
        cls.time_dim_1 = 1500
        cls.feature_dim_1 = 25
        cls.output_dim_1 = 5
        cls.x_1 = torch.rand((cls.batch_size, cls.time_dim_1, cls.feature_dim_1))

    def test_sequential_output(self):
        model = LinT(
            LinTConfig.from_dict(
                {
                    "input_feature_dim": self.feature_dim_1,
                    "heads": [{"type": "simple", "output_dim": self.output_dim_1}],
                }
            )
        )
        output = list(model(self.x_1).values())
        self.assertEqual(output[0].shape, (self.batch_size, self.time_dim_1, self.output_dim_1))

    def test_with_time_reducer(self):
        model = LinT(
            LinTConfig.from_dict(
                {
                    "input_feature_dim": self.feature_dim_1,
                    "heads": [{"type": "simple", "output_dim": self.output_dim_1}],
                    "time_dim_reducer": "attentionpool",
                }
            )
        )
        output = list(model(self.x_1).values())
        self.assertEqual(output[0].shape, (self.batch_size, self.output_dim_1))

    def test_with_mask(self):
        model = LinT(
            LinTConfig.from_dict(
                {
                    "input_feature_dim": self.feature_dim_1,
                    "heads": [{"type": "simple", "output_dim": self.output_dim_1}],
                    "time_dim_reducer": "gap",
                }
            )
        )
        mask = torch.ones(self.batch_size, self.time_dim_1, dtype=torch.bool)
        mask[:, self.time_dim_1 - 50 :] = False
        output = list(model(self.x_1, mask=mask).values())
        self.assertEqual(output[0].shape, (self.batch_size, self.output_dim_1))
        self.assertFalse(torch.isnan(output[0]).any())

    def test_with_fully_masked_modality(self):
        model = LinT(
            LinTConfig.from_dict(
                {
                    "input_feature_dim": self.feature_dim_1,
                    "heads": [{"type": "simple", "output_dim": self.output_dim_1}],
                    "time_dim_reducer": "attentionpool",
                }
            )
        )
        mask = torch.zeros(self.batch_size, self.time_dim_1, dtype=torch.bool)
        output = list(model(self.x_1, mask=mask).values())
        self.assertEqual(output[0].shape, (self.batch_size, self.output_dim_1))
        self.assertFalse(torch.isnan(output[0]).any())

    def test_ffn_fusion(self):
        model = LinT(
            LinTConfig.from_dict(
                {
                    "input_feature_dim": self.feature_dim_1,
                    "heads": [{"type": "simple", "output_dim": self.output_dim_1}],
                    "add_module_ffn_fusion": True,
                    "time_dim_reducer": "gap",
                }
            )
        )
        output = list(model(self.x_1).values())
        self.assertEqual(output[0].shape, (self.batch_size, self.output_dim_1))

    def test_multiple_heads(self):
        model = LinT(
            LinTConfig.from_dict(
                {
                    "input_feature_dim": self.feature_dim_1,
                    "heads": [
                        {"type": "simple", "output_dim": self.output_dim_1, "name": "valence"},
                        {"type": "simple", "output_dim": 1, "name": "arousal"},
                    ],
                    "time_dim_reducer": "gap",
                }
            )
        )
        outputs = model(self.x_1)
        self.assertIn("valence", outputs)
        self.assertIn("arousal", outputs)

    def test_sequence_aggregation_head(self):
        model = LinT(
            LinTConfig.from_dict(
                {
                    "input_feature_dim": self.feature_dim_1,
                    "heads": [{"type": "sequence_aggregation", "output_dim": self.output_dim_1}],
                }
            )
        )
        output = list(model(self.x_1).values())
        self.assertEqual(output[0].shape, (self.batch_size, self.output_dim_1))

    def test_list_input(self):
        model = LinT(
            LinTConfig.from_dict(
                {
                    "input_feature_dim": self.feature_dim_1,
                    "heads": [{"type": "simple", "output_dim": self.output_dim_1}],
                    "time_dim_reducer": "gap",
                }
            )
        )
        output = list(model([self.x_1]).values())
        self.assertEqual(output[0].shape, (self.batch_size, self.output_dim_1))

    def test_str_config_path(self):
        model = LinT(str(CONFIGS_DIR / "LinT.yaml"))
        self.assertIsNotNone(model)

    def test_path_config(self):
        model = LinT(CONFIGS_DIR / "LinT.yaml")
        self.assertIsNotNone(model)

    def test_list_input_multiple_tensors_raises(self):
        # LinT.forward: list with >1 element should raise
        model = LinT(
            LinTConfig.from_dict(
                {
                    "input_feature_dim": self.feature_dim_1,
                    "heads": [{"type": "simple", "output_dim": self.output_dim_1}],
                }
            )
        )
        with self.assertRaises(Exception):
            model([self.x_1, self.x_1])

    def test_list_name_single_unwraps(self):
        model = LinT(
            LinTConfig.from_dict(
                {
                    "input_feature_dim": self.feature_dim_1,
                    "heads": [{"type": "simple", "output_dim": self.output_dim_1}],
                    "time_dim_reducer": "gap",
                }
            )
        )
        output = list(model(self.x_1, name=["some_name"]).values())
        self.assertEqual(output[0].shape, (self.batch_size, self.output_dim_1))

    def test_list_name_multiple_raises(self):
        model = LinT(
            LinTConfig.from_dict(
                {
                    "input_feature_dim": self.feature_dim_1,
                    "heads": [{"type": "simple", "output_dim": self.output_dim_1}],
                }
            )
        )
        with self.assertRaises(Exception):
            model(self.x_1, name=["a", "b"])

    def test_list_mask_single_unwraps(self):
        model = LinT(
            LinTConfig.from_dict(
                {
                    "input_feature_dim": self.feature_dim_1,
                    "heads": [{"type": "simple", "output_dim": self.output_dim_1}],
                    "time_dim_reducer": "gap",
                }
            )
        )
        mask = torch.ones(self.batch_size, self.time_dim_1, dtype=torch.bool)
        output = list(model(self.x_1, mask=[mask]).values())
        self.assertEqual(output[0].shape, (self.batch_size, self.output_dim_1))

    def test_list_mask_multiple_raises(self):
        model = LinT(
            LinTConfig.from_dict(
                {
                    "input_feature_dim": self.feature_dim_1,
                    "heads": [{"type": "simple", "output_dim": self.output_dim_1}],
                }
            )
        )
        mask = torch.ones(self.batch_size, self.time_dim_1, dtype=torch.bool)
        with self.assertRaises(Exception):
            model(self.x_1, mask=[mask, mask])

    def test_special_handling_4d_tensor(self):
        # LinT._apply_projection lines 123-129: weighted_sum on (B,N,T,F) input
        model = LinT(
            LinTConfig.from_dict(
                {
                    "input_feature_dim": self.feature_dim_1,
                    "heads": [{"type": "simple", "output_dim": self.output_dim_1}],
                    "time_dim_reducer": "gap",
                    "special_handling": {
                        "wavlm": {"type": "weighted_sum", "start_layer": 0, "end_layer": 4}
                    },
                }
            )
        )
        x_4d = torch.rand(self.batch_size, 6, self.time_dim_1 // 10, self.feature_dim_1)
        output = list(model(x_4d, name="wavlm").values())
        self.assertEqual(output[0].shape, (self.batch_size, self.output_dim_1))

    def test_list_input_feature_dim_raises(self):
        # LinT.__init__: input_feature_dim must be int, not list → ValueError
        from linmult.core.config import HeadConfig

        cfg = LinTConfig(
            input_feature_dim=[256, 128],
            heads=[HeadConfig(type="simple", output_dim=5)],
        )
        with self.assertRaises(ValueError) as ctx:
            LinT(cfg)
        self.assertIn("LinT", str(ctx.exception))

    def test_special_modules_are_registered_parameters(self):
        # regression: special_modules used plain dict so params were not tracked
        model = LinT(
            LinTConfig.from_dict(
                {
                    "input_feature_dim": self.feature_dim_1,
                    "heads": [{"type": "simple", "output_dim": self.output_dim_1}],
                    "time_dim_reducer": "gap",
                    "special_handling": {
                        "bert": {"type": "weighted_sum", "start_layer": 0, "end_layer": 4}
                    },
                }
            )
        )
        param_names = [n for n, _ in model.named_parameters()]
        self.assertTrue(any("special_modules" in n for n in param_names))
