import unittest

import torch

from linmult import FIXTURE_DIR, LinMulT
from linmult.core.config import LinMulTConfig
from linmult.core.utils import apply_logit_aggregation

CONFIGS_DIR = FIXTURE_DIR / "configs"


class TestLinMulT(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.batch_size = 8
        cls.time_dim_1 = 1500
        cls.time_dim_2 = 450
        cls.time_dim_3 = 450
        cls.feature_dim_1 = 25
        cls.feature_dim_2 = 35
        cls.feature_dim_3 = 256
        cls.output_dim_1 = 5
        cls.output_dim_2 = 1
        cls.x_1 = torch.rand((cls.batch_size, cls.time_dim_1, cls.feature_dim_1))
        cls.x_2 = torch.rand((cls.batch_size, cls.time_dim_2, cls.feature_dim_2))
        cls.x_3 = torch.rand((cls.batch_size, cls.time_dim_3, cls.feature_dim_3))
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
        cls.mask_3 = (
            (torch.arange(cls.time_dim_3).unsqueeze(0) < cls.time_dim_3 - 10)
            .expand(cls.batch_size, -1)
            .bool()
        )
        cls.mask_3f = torch.zeros((cls.batch_size, cls.time_dim_3), dtype=torch.bool)

    # --- Attention types ---

    def test_attention_linear(self):
        model = LinMulT(
            LinMulTConfig.from_dict(
                {
                    "input_feature_dim": [self.feature_dim_2, self.feature_dim_3],
                    "heads": [{"type": "simple", "output_dim": self.output_dim_1}],
                    "attention_type": "linear",
                }
            )
        )
        self.assertEqual(
            model.cross_modal.branches[0].cross_transformers[0].layers[0].attention_type, "linear"
        )

    def test_attention_bigbird(self):
        model = LinMulT(
            LinMulTConfig.from_dict(
                {
                    "input_feature_dim": [self.feature_dim_2, self.feature_dim_3],
                    "heads": [{"type": "simple", "output_dim": self.output_dim_1}],
                    "attention_type": "bigbird",
                }
            )
        )
        self.assertEqual(
            model.cross_modal.branches[0].cross_transformers[0].layers[0].attention_type, "bigbird"
        )

    def test_attention_mha(self):
        model = LinMulT(
            LinMulTConfig.from_dict(
                {
                    "input_feature_dim": [self.feature_dim_2, self.feature_dim_3],
                    "heads": [{"type": "simple", "output_dim": self.output_dim_1}],
                    "attention_type": "mha",
                }
            )
        )
        self.assertEqual(
            model.cross_modal.branches[0].cross_transformers[0].layers[0].attention_type, "mha"
        )

    def test_attention_softmax_forward(self):
        # softmax was never verified with a real forward pass, only attribute inspection.
        model = LinMulT(
            LinMulTConfig.from_dict(
                {
                    "input_feature_dim": [self.feature_dim_2, self.feature_dim_3],
                    "heads": [{"type": "simple", "output_dim": self.output_dim_1}],
                    "attention_type": "softmax",
                    "time_dim_reducer": "gap",
                }
            )
        )
        output = list(model([self.x_2, self.x_3]).values())
        self.assertEqual(output[0].shape, (self.batch_size, self.output_dim_1))
        self.assertFalse(torch.isnan(output[0]).any())

    def test_attention_softmax_forward_with_mask(self):
        # softmax + partial masks exercises the NaN-prevention masked_fill path.
        model = LinMulT(
            LinMulTConfig.from_dict(
                {
                    "input_feature_dim": [self.feature_dim_2, self.feature_dim_3],
                    "heads": [{"type": "simple", "output_dim": self.output_dim_1}],
                    "attention_type": "softmax",
                    "time_dim_reducer": "gap",
                }
            )
        )
        output = list(model([self.x_2, self.x_3], masks=[self.mask_2, self.mask_3]).values())
        self.assertEqual(output[0].shape, (self.batch_size, self.output_dim_1))
        self.assertFalse(torch.isnan(output[0]).any())

    def test_attention_bigbird_forward(self):
        # bigbird was never verified with a real forward pass, only attribute inspection.
        model = LinMulT(
            LinMulTConfig.from_dict(
                {
                    "input_feature_dim": [self.feature_dim_2, self.feature_dim_3],
                    "heads": [{"type": "simple", "output_dim": self.output_dim_1}],
                    "attention_type": "bigbird",
                    "time_dim_reducer": "gap",
                }
            )
        )
        output = list(model([self.x_2, self.x_3]).values())
        self.assertEqual(output[0].shape, (self.batch_size, self.output_dim_1))
        self.assertFalse(torch.isnan(output[0]).any())

    def test_attention_bigbird_forward_with_mask(self):
        model = LinMulT(
            LinMulTConfig.from_dict(
                {
                    "input_feature_dim": [self.feature_dim_2, self.feature_dim_3],
                    "heads": [{"type": "simple", "output_dim": self.output_dim_1}],
                    "attention_type": "bigbird",
                    "time_dim_reducer": "gap",
                }
            )
        )
        output = list(model([self.x_2, self.x_3], masks=[self.mask_2, self.mask_3]).values())
        self.assertEqual(output[0].shape, (self.batch_size, self.output_dim_1))
        self.assertFalse(torch.isnan(output[0]).any())

    def test_attention_performer_forward(self):
        model = LinMulT(
            LinMulTConfig.from_dict(
                {
                    "input_feature_dim": [self.feature_dim_2, self.feature_dim_3],
                    "heads": [{"type": "simple", "output_dim": self.output_dim_1}],
                    "attention_type": "performer",
                    "time_dim_reducer": "gap",
                }
            )
        )
        output = list(model([self.x_2, self.x_3]).values())
        self.assertEqual(output[0].shape, (self.batch_size, self.output_dim_1))
        self.assertFalse(torch.isnan(output[0]).any())

    def test_attention_performer_forward_with_mask(self):
        model = LinMulT(
            LinMulTConfig.from_dict(
                {
                    "input_feature_dim": [self.feature_dim_2, self.feature_dim_3],
                    "heads": [{"type": "simple", "output_dim": self.output_dim_1}],
                    "attention_type": "performer",
                    "time_dim_reducer": "gap",
                }
            )
        )
        output = list(model([self.x_2, self.x_3], masks=[self.mask_2, self.mask_3]).values())
        self.assertEqual(output[0].shape, (self.batch_size, self.output_dim_1))
        self.assertFalse(torch.isnan(output[0]).any())

    def test_attention_performer_custom_features(self):
        # num_random_features config key is respected end-to-end.
        model = LinMulT(
            LinMulTConfig.from_dict(
                {
                    "input_feature_dim": [self.feature_dim_2, self.feature_dim_3],
                    "heads": [{"type": "simple", "output_dim": self.output_dim_1}],
                    "attention_type": "performer",
                    "performer_num_random_features": 64,
                    "time_dim_reducer": "gap",
                }
            )
        )
        fm = (
            model.cross_modal.branches[0]
            .cross_transformers[0]
            .layers[0]
            .attention.inner_attention.feature_map
        )
        self.assertEqual(fm.num_features, 64)
        output = list(model([self.x_2, self.x_3]).values())
        self.assertEqual(output[0].shape, (self.batch_size, self.output_dim_1))

    def test_attention_flash_forward(self):
        model = LinMulT(
            LinMulTConfig.from_dict(
                {
                    "input_feature_dim": [self.feature_dim_2, self.feature_dim_3],
                    "heads": [{"type": "simple", "output_dim": self.output_dim_1}],
                    "attention_type": "flash",
                    "time_dim_reducer": "gap",
                }
            )
        )
        output = list(model([self.x_2, self.x_3]).values())
        self.assertEqual(output[0].shape, (self.batch_size, self.output_dim_1))
        self.assertFalse(torch.isnan(output[0]).any())

    def test_attention_flash_forward_with_mask(self):
        model = LinMulT(
            LinMulTConfig.from_dict(
                {
                    "input_feature_dim": [self.feature_dim_2, self.feature_dim_3],
                    "heads": [{"type": "simple", "output_dim": self.output_dim_1}],
                    "attention_type": "flash",
                    "time_dim_reducer": "gap",
                }
            )
        )
        output = list(model([self.x_2, self.x_3], masks=[self.mask_2, self.mask_3]).values())
        self.assertEqual(output[0].shape, (self.batch_size, self.output_dim_1))
        self.assertFalse(torch.isnan(output[0]).any())

    def test_attention_flash_custom_query_key_dim(self):
        # query_key_dim config key is respected end-to-end.
        from linmult.core.attention import GatedAttentionUnit

        model = LinMulT(
            LinMulTConfig.from_dict(
                {
                    "input_feature_dim": [self.feature_dim_2, self.feature_dim_3],
                    "heads": [{"type": "simple", "output_dim": self.output_dim_1}],
                    "attention_type": "flash",
                    "flash_query_key_dim": 16,
                    "time_dim_reducer": "gap",
                }
            )
        )
        gau = model.cross_modal.branches[0].cross_transformers[0].layers[0].attention
        self.assertIsInstance(gau, GatedAttentionUnit)
        self.assertEqual(gau.query_key_dim, 16)
        output = list(model([self.x_2, self.x_3]).values())
        self.assertEqual(output[0].shape, (self.batch_size, self.output_dim_1))

    # --- Input/output shapes ---

    def test_same_time_dim(self):
        model = LinMulT(
            LinMulTConfig.from_dict(
                {
                    "input_feature_dim": [self.feature_dim_2, self.feature_dim_3],
                    "heads": [{"type": "simple", "output_dim": self.output_dim_1}],
                }
            )
        )
        output_seq = list(model([self.x_2, self.x_3]).values())
        output_cls = apply_logit_aggregation(x=output_seq[0], method="meanpooling")
        self.assertEqual(output_seq[0].shape, (self.batch_size, self.time_dim_2, self.output_dim_1))
        self.assertEqual(output_cls.shape, (self.batch_size, self.output_dim_1))

    def test_2i(self):
        model = LinMulT(
            LinMulTConfig.from_dict(
                {
                    "input_feature_dim": [self.feature_dim_1, self.feature_dim_2],
                    "heads": [{"type": "simple", "output_dim": self.output_dim_1}],
                    "time_dim_reducer": "gmp",
                }
            )
        )
        output = list(model([self.x_1, self.x_2]).values())
        self.assertEqual(output[0].shape, (self.batch_size, self.output_dim_1))

    def test_2o(self):
        model = LinMulT(
            LinMulTConfig.from_dict(
                {
                    "input_feature_dim": [self.feature_dim_1, self.feature_dim_2],
                    "heads": [
                        {"type": "simple", "output_dim": self.output_dim_1},
                        {"type": "simple", "output_dim": self.output_dim_2},
                    ],
                    "time_dim_reducer": "gap",
                }
            )
        )
        output = list(model([self.x_1, self.x_2]).values())
        self.assertEqual(output[0].shape, (self.batch_size, self.output_dim_1))
        self.assertEqual(output[1].shape, (self.batch_size, self.output_dim_2))

    def test_3i(self):
        model = LinMulT(
            LinMulTConfig.from_dict(
                {
                    "input_feature_dim": [
                        self.feature_dim_1,
                        self.feature_dim_2,
                        self.feature_dim_3,
                    ],
                    "heads": [{"type": "simple", "output_dim": self.output_dim_1}],
                    "time_dim_reducer": "gap",
                }
            )
        )
        output = list(model([self.x_1, self.x_2, self.x_3]).values())
        self.assertEqual(output[0].shape, (self.batch_size, self.output_dim_1))

    def test_4i(self):
        # 4 modalities: combined_dim = (4-1) * 4 * d_model = 480.
        # Verifies dimension calculations don't break beyond 3 modalities.
        x_4 = torch.rand(self.batch_size, self.time_dim_2, self.feature_dim_2)
        model = LinMulT(
            LinMulTConfig.from_dict(
                {
                    "input_feature_dim": [
                        self.feature_dim_1,
                        self.feature_dim_2,
                        self.feature_dim_3,
                        self.feature_dim_2,
                    ],
                    "heads": [{"type": "simple", "output_dim": self.output_dim_1}],
                    "time_dim_reducer": "gap",
                }
            )
        )
        output = list(model([self.x_1, self.x_2, self.x_3, x_4]).values())
        self.assertEqual(output[0].shape, (self.batch_size, self.output_dim_1))
        self.assertFalse(torch.isnan(output[0]).any())

    def test_4i_with_masks(self):
        x_4 = torch.rand(self.batch_size, self.time_dim_2, self.feature_dim_2)
        mask_4 = (
            (torch.arange(self.time_dim_2).unsqueeze(0) < self.time_dim_2 - 10)
            .expand(self.batch_size, -1)
            .bool()
        )
        model = LinMulT(
            LinMulTConfig.from_dict(
                {
                    "input_feature_dim": [
                        self.feature_dim_1,
                        self.feature_dim_2,
                        self.feature_dim_3,
                        self.feature_dim_2,
                    ],
                    "heads": [{"type": "simple", "output_dim": self.output_dim_1}],
                    "time_dim_reducer": "gap",
                }
            )
        )
        output = list(
            model(
                [self.x_1, self.x_2, self.x_3, x_4],
                masks=[self.mask_1, self.mask_2, self.mask_3, mask_4],
            ).values()
        )
        self.assertEqual(output[0].shape, (self.batch_size, self.output_dim_1))
        self.assertFalse(torch.isnan(output[0]).any())

    def test_2i_upsample(self):
        model = LinMulT(
            LinMulTConfig.from_dict(
                {
                    "input_feature_dim": [self.feature_dim_1, self.feature_dim_2],
                    "heads": [
                        {
                            "type": "upsample",
                            "input_time_dim": 300,
                            "output_time_dim": self.time_dim_1,
                            "output_dim": self.feature_dim_1,
                        },
                        {
                            "type": "upsample",
                            "input_time_dim": 300,
                            "output_time_dim": self.time_dim_2,
                            "output_dim": self.feature_dim_2,
                        },
                    ],
                    "add_module_multimodal_signal": True,
                    "tam_aligner": "aap",
                    "tam_time_dim": 300,
                    "add_module_tam_fusion": True,
                }
            )
        )
        output = list(model([self.x_1, self.x_2]).values())
        self.assertEqual(output[0].shape, (self.batch_size, self.time_dim_1, self.feature_dim_1))
        self.assertEqual(output[1].shape, (self.batch_size, self.time_dim_2, self.feature_dim_2))

    def test_2i_downsample(self):
        model = LinMulT(
            LinMulTConfig.from_dict(
                {
                    "input_feature_dim": [self.feature_dim_1, self.feature_dim_2],
                    "heads": [
                        {
                            "type": "downsample",
                            "input_time_dim": 2000,
                            "output_time_dim": self.time_dim_1,
                            "output_dim": self.feature_dim_1,
                        },
                        {
                            "type": "downsample",
                            "input_time_dim": 2000,
                            "output_time_dim": self.time_dim_2,
                            "output_dim": self.feature_dim_2,
                        },
                    ],
                    "add_module_multimodal_signal": True,
                    "tam_aligner": "aap",
                    "tam_time_dim": 2000,
                    "add_module_tam_fusion": True,
                }
            )
        )
        output = list(model([self.x_1, self.x_2]).values())
        self.assertEqual(output[0].shape, (self.batch_size, self.time_dim_1, self.feature_dim_1))
        self.assertEqual(output[1].shape, (self.batch_size, self.time_dim_2, self.feature_dim_2))

    def test_upsample_head_with_masks(self):
        # upsample/downsample heads were never tested with padding masks.
        model = LinMulT(
            LinMulTConfig.from_dict(
                {
                    "input_feature_dim": [self.feature_dim_1, self.feature_dim_2],
                    "heads": [
                        {
                            "type": "upsample",
                            "input_time_dim": 300,
                            "output_time_dim": self.time_dim_1,
                            "output_dim": self.feature_dim_1,
                        },
                        {
                            "type": "upsample",
                            "input_time_dim": 300,
                            "output_time_dim": self.time_dim_2,
                            "output_dim": self.feature_dim_2,
                        },
                    ],
                    "add_module_multimodal_signal": True,
                    "tam_aligner": "aap",
                    "tam_time_dim": 300,
                    "add_module_tam_fusion": True,
                }
            )
        )
        output = list(model([self.x_1, self.x_2], masks=[self.mask_1, self.mask_2]).values())
        self.assertEqual(output[0].shape, (self.batch_size, self.time_dim_1, self.feature_dim_1))
        self.assertEqual(output[1].shape, (self.batch_size, self.time_dim_2, self.feature_dim_2))
        self.assertFalse(torch.isnan(output[0]).any())
        self.assertFalse(torch.isnan(output[1]).any())

    def test_downsample_head_with_masks(self):
        model = LinMulT(
            LinMulTConfig.from_dict(
                {
                    "input_feature_dim": [self.feature_dim_1, self.feature_dim_2],
                    "heads": [
                        {
                            "type": "downsample",
                            "input_time_dim": 2000,
                            "output_time_dim": self.time_dim_1,
                            "output_dim": self.feature_dim_1,
                        },
                        {
                            "type": "downsample",
                            "input_time_dim": 2000,
                            "output_time_dim": self.time_dim_2,
                            "output_dim": self.feature_dim_2,
                        },
                    ],
                    "add_module_multimodal_signal": True,
                    "tam_aligner": "aap",
                    "tam_time_dim": 2000,
                    "add_module_tam_fusion": True,
                }
            )
        )
        output = list(model([self.x_1, self.x_2], masks=[self.mask_1, self.mask_2]).values())
        self.assertEqual(output[0].shape, (self.batch_size, self.time_dim_1, self.feature_dim_1))
        self.assertEqual(output[1].shape, (self.batch_size, self.time_dim_2, self.feature_dim_2))
        self.assertFalse(torch.isnan(output[0]).any())
        self.assertFalse(torch.isnan(output[1]).any())

    # --- Modules ---

    def test_module_time_aligner_aap(self):
        model = LinMulT(
            LinMulTConfig.from_dict(
                {
                    "input_feature_dim": [self.feature_dim_1, self.feature_dim_2],
                    "heads": [{"type": "simple", "output_dim": self.output_dim_1}],
                    "add_module_tam_fusion": True,
                    "tam_aligner": "aap",
                    "tam_time_dim": 450,
                }
            )
        )
        output = list(model([self.x_1, self.x_2]).values())
        self.assertEqual(output[0].shape, (self.batch_size, 450, self.output_dim_1))

    def test_module_time_aligner_padding(self):
        # padding aligner was tested at TemporalFactory level but never end-to-end in LinMulT.
        # Pads both 450-length sequences to 500.
        model = LinMulT(
            LinMulTConfig.from_dict(
                {
                    "input_feature_dim": [self.feature_dim_2, self.feature_dim_3],
                    "heads": [{"type": "simple", "output_dim": self.output_dim_1}],
                    "add_module_tam_fusion": True,
                    "tam_aligner": "padding",
                    "tam_time_dim": 500,
                }
            )
        )
        output = list(model([self.x_2, self.x_3]).values())
        self.assertEqual(output[0].shape, (self.batch_size, 500, self.output_dim_1))
        self.assertFalse(torch.isnan(output[0]).any())

    def test_module_time_reducer_gap(self):
        model = LinMulT(
            LinMulTConfig.from_dict(
                {
                    "input_feature_dim": [self.feature_dim_1, self.feature_dim_2],
                    "heads": [{"type": "simple", "output_dim": self.output_dim_1}],
                    "time_dim_reducer": "gap",
                }
            )
        )
        output = list(model([self.x_1, self.x_2]).values())
        self.assertEqual(output[0].shape, (self.batch_size, self.output_dim_1))

    def test_module_mms_time_aligner_amp(self):
        model = LinMulT(
            LinMulTConfig.from_dict(
                {
                    "input_feature_dim": [self.feature_dim_1, self.feature_dim_2],
                    "heads": [{"type": "simple", "output_dim": self.output_dim_1}],
                    "add_module_multimodal_signal": True,
                    "add_module_tam_fusion": True,
                    "tam_aligner": "amp",
                    "tam_time_dim": 450,
                }
            )
        )
        output = list(model([self.x_1, self.x_2]).values())
        self.assertEqual(output[0].shape, (self.batch_size, 450, self.output_dim_1))

    def test_module_mms_time_reduce_gap(self):
        model = LinMulT(
            LinMulTConfig.from_dict(
                {
                    "input_feature_dim": [
                        self.feature_dim_1,
                        self.feature_dim_2,
                        self.feature_dim_3,
                    ],
                    "heads": [{"type": "simple", "output_dim": self.output_dim_1}],
                    "add_module_multimodal_signal": True,
                    "tam_aligner": "amp",
                    "tam_time_dim": 450,
                    "time_dim_reducer": "gap",
                }
            )
        )
        output = list(model([self.x_1, self.x_2, self.x_3]).values())
        self.assertEqual(output[0].shape, (self.batch_size, self.output_dim_1))

    def test_module_mms_time_reduce_ap(self):
        model = LinMulT(
            LinMulTConfig.from_dict(
                {
                    "input_feature_dim": [self.feature_dim_1, self.feature_dim_2],
                    "heads": [{"type": "simple", "output_dim": self.output_dim_1}],
                    "add_module_multimodal_signal": True,
                    "tam_aligner": "aap",
                    "tam_time_dim": 450,
                    "time_dim_reducer": "attentionpool",
                }
            )
        )
        output = list(model([self.x_1, self.x_2]).values())
        self.assertEqual(output[0].shape, (self.batch_size, self.output_dim_1))

    # --- Masks ---

    def test_2i_with_masks(self):
        model = LinMulT(
            LinMulTConfig.from_dict(
                {
                    "input_feature_dim": [self.feature_dim_1, self.feature_dim_2],
                    "heads": [{"type": "simple", "output_dim": self.output_dim_1}],
                    "time_dim_reducer": "gap",
                }
            )
        )
        output = list(model([self.x_1, self.x_2], masks=[self.mask_1, self.mask_2]).values())
        self.assertEqual(output[0].shape, (self.batch_size, self.output_dim_1))

    def test_3i_with_masks(self):
        model = LinMulT(
            LinMulTConfig.from_dict(
                {
                    "input_feature_dim": [
                        self.feature_dim_1,
                        self.feature_dim_2,
                        self.feature_dim_3,
                    ],
                    "heads": [{"type": "simple", "output_dim": self.output_dim_1}],
                    "time_dim_reducer": "gap",
                }
            )
        )
        output = list(
            model(
                [self.x_1, self.x_2, self.x_3], masks=[self.mask_1, self.mask_2, self.mask_3]
            ).values()
        )
        self.assertEqual(output[0].shape, (self.batch_size, self.output_dim_1))

    def test_modality_masking(self):
        model = LinMulT(
            LinMulTConfig.from_dict(
                {
                    "input_feature_dim": [
                        self.feature_dim_1,
                        self.feature_dim_2,
                        self.feature_dim_3,
                    ],
                    "heads": [{"type": "simple", "output_dim": self.output_dim_1}],
                    "time_dim_reducer": "gap",
                }
            )
        )
        output = list(
            model(
                [self.x_1, self.x_2, self.x_3], masks=[self.mask_1, self.mask_2, self.mask_3f]
            ).values()
        )
        self.assertEqual(output[0].shape, (self.batch_size, self.output_dim_1))

    def test_modality_sequence_masking(self):
        model = LinMulT(
            LinMulTConfig.from_dict(
                {
                    "input_feature_dim": [
                        self.feature_dim_2,
                        self.feature_dim_2,
                        self.feature_dim_3,
                    ],
                    "heads": [{"type": "simple", "output_dim": self.output_dim_1}],
                }
            )
        )
        output_seq = list(
            model(
                [self.x_2, self.x_2, self.x_3], masks=[self.mask_2, self.mask_2, self.mask_3f]
            ).values()
        )
        output_cls = apply_logit_aggregation(
            x=output_seq[0], mask=self.mask_2, method="meanpooling"
        )
        self.assertEqual(output_seq[0].shape, (self.batch_size, self.time_dim_2, self.output_dim_1))
        self.assertEqual(output_cls.shape, (self.batch_size, self.output_dim_1))

    # --- Fusion modules ---

    def test_sat_fusion(self):
        # add_module_sat_fusion applies a SA transformer over the concatenated sequence outputs;
        # it requires sequence (3D) inputs, so no time_dim_reducer and same-T inputs.
        model = LinMulT(
            LinMulTConfig.from_dict(
                {
                    "input_feature_dim": [self.feature_dim_2, self.feature_dim_3],
                    "heads": [{"type": "simple", "output_dim": self.output_dim_1}],
                    "add_module_sat_fusion": True,
                }
            )
        )
        output = list(model([self.x_2, self.x_3]).values())
        self.assertEqual(output[0].shape, (self.batch_size, self.time_dim_2, self.output_dim_1))

    def test_ffn_fusion(self):
        model = LinMulT(
            LinMulTConfig.from_dict(
                {
                    "input_feature_dim": [self.feature_dim_1, self.feature_dim_2],
                    "heads": [{"type": "simple", "output_dim": self.output_dim_1}],
                    "add_module_ffn_fusion": True,
                    "time_dim_reducer": "gap",
                }
            )
        )
        output = list(model([self.x_1, self.x_2]).values())
        self.assertEqual(output[0].shape, (self.batch_size, self.output_dim_1))

    def test_unimodal_sat(self):
        model = LinMulT(
            LinMulTConfig.from_dict(
                {
                    "input_feature_dim": [self.feature_dim_2, self.feature_dim_3],
                    "heads": [{"type": "simple", "output_dim": self.output_dim_1}],
                    "add_module_unimodal_sat": True,
                    "time_dim_reducer": "gap",
                }
            )
        )
        output = list(model([self.x_2, self.x_3]).values())
        self.assertEqual(output[0].shape, (self.batch_size, self.output_dim_1))

    # --- Auxiliary heads ---

    def test_auxiliary_heads(self):
        model = LinMulT(
            LinMulTConfig.from_dict(
                {
                    "input_feature_dim": [self.feature_dim_1, self.feature_dim_2],
                    "heads": [{"type": "simple", "output_dim": self.output_dim_1}],
                    "auxiliary_heads": [{"type": "simple", "output_dim": self.output_dim_2}],
                    "time_dim_reducer": "gap",
                }
            )
        )
        outputs, outputs_aux = model([self.x_1, self.x_2])
        self.assertIsInstance(outputs, dict)
        self.assertIsInstance(outputs_aux, list)
        self.assertEqual(len(outputs_aux), 2)  # one per input modality

    def test_auxiliary_heads_with_masks(self):
        model = LinMulT(
            LinMulTConfig.from_dict(
                {
                    "input_feature_dim": [self.feature_dim_1, self.feature_dim_2],
                    "heads": [{"type": "simple", "output_dim": self.output_dim_1}],
                    "auxiliary_heads": [{"type": "simple", "output_dim": self.output_dim_2}],
                    "time_dim_reducer": "gap",
                }
            )
        )
        outputs, outputs_aux = model([self.x_1, self.x_2], masks=[self.mask_1, self.mask_2])
        self.assertEqual(list(outputs.values())[0].shape, (self.batch_size, self.output_dim_1))

    def test_auxiliary_heads_3i(self):
        # Auxiliary heads with 3 modalities: one aux dict per input branch (3 total).
        model = LinMulT(
            LinMulTConfig.from_dict(
                {
                    "input_feature_dim": [
                        self.feature_dim_1,
                        self.feature_dim_2,
                        self.feature_dim_3,
                    ],
                    "heads": [{"type": "simple", "output_dim": self.output_dim_1}],
                    "auxiliary_heads": [{"type": "simple", "output_dim": self.output_dim_2}],
                    "time_dim_reducer": "gap",
                }
            )
        )
        outputs, outputs_aux = model([self.x_1, self.x_2, self.x_3])
        self.assertIsInstance(outputs, dict)
        self.assertIsInstance(outputs_aux, list)
        self.assertEqual(len(outputs_aux), 3)
        self.assertFalse(torch.isnan(list(outputs.values())[0]).any())

    def test_auxiliary_heads_softmax_with_mask(self):
        # auxiliary heads + softmax attention + masks: exercises the NaN-free path end-to-end.
        model = LinMulT(
            LinMulTConfig.from_dict(
                {
                    "input_feature_dim": [self.feature_dim_2, self.feature_dim_3],
                    "heads": [{"type": "simple", "output_dim": self.output_dim_1}],
                    "auxiliary_heads": [{"type": "simple", "output_dim": self.output_dim_2}],
                    "attention_type": "softmax",
                    "time_dim_reducer": "gap",
                }
            )
        )
        outputs, outputs_aux = model([self.x_2, self.x_3], masks=[self.mask_2, self.mask_3])
        self.assertEqual(list(outputs.values())[0].shape, (self.batch_size, self.output_dim_1))
        self.assertFalse(torch.isnan(list(outputs.values())[0]).any())

    # --- Named heads ---

    def test_named_heads(self):
        model = LinMulT(
            LinMulTConfig.from_dict(
                {
                    "input_feature_dim": [self.feature_dim_1, self.feature_dim_2],
                    "heads": [
                        {"type": "simple", "output_dim": self.output_dim_1, "name": "valence"},
                        {"type": "simple", "output_dim": self.output_dim_2, "name": "arousal"},
                    ],
                    "time_dim_reducer": "gap",
                }
            )
        )
        outputs = model([self.x_1, self.x_2])
        self.assertIn("valence", outputs)
        self.assertIn("arousal", outputs)
        self.assertEqual(outputs["valence"].shape, (self.batch_size, self.output_dim_1))
        self.assertEqual(outputs["arousal"].shape, (self.batch_size, self.output_dim_2))

    # --- Additional head types ---

    def test_sequence_aggregation_head(self):
        model = LinMulT(
            LinMulTConfig.from_dict(
                {
                    "input_feature_dim": [self.feature_dim_2, self.feature_dim_3],
                    "heads": [{"type": "sequence_aggregation", "output_dim": self.output_dim_1}],
                }
            )
        )
        output = list(model([self.x_2, self.x_3]).values())
        self.assertEqual(output[0].shape, (self.batch_size, self.output_dim_1))

    def test_sequence_head(self):
        model = LinMulT(
            LinMulTConfig.from_dict(
                {
                    "input_feature_dim": [self.feature_dim_2, self.feature_dim_2],
                    "heads": [{"type": "sequence", "output_dim": self.output_dim_1}],
                }
            )
        )
        output = list(model([self.x_2, self.x_2]).values())
        self.assertEqual(output[0].shape, (self.batch_size, self.time_dim_2, self.output_dim_1))

    # --- Additional reducers ---

    def test_attentionpool_reducer(self):
        model = LinMulT(
            LinMulTConfig.from_dict(
                {
                    "input_feature_dim": [self.feature_dim_2, self.feature_dim_3],
                    "heads": [{"type": "simple", "output_dim": self.output_dim_1}],
                    "time_dim_reducer": "attentionpool",
                }
            )
        )
        output = list(model([self.x_2, self.x_3]).values())
        self.assertEqual(output[0].shape, (self.batch_size, self.output_dim_1))

    def test_last_reducer(self):
        model = LinMulT(
            LinMulTConfig.from_dict(
                {
                    "input_feature_dim": [self.feature_dim_2, self.feature_dim_3],
                    "heads": [{"type": "simple", "output_dim": self.output_dim_1}],
                    "time_dim_reducer": "last",
                }
            )
        )
        output = list(model([self.x_2, self.x_3]).values())
        self.assertEqual(output[0].shape, (self.batch_size, self.output_dim_1))

    def test_gmp_reducer_with_mask(self):
        model = LinMulT(
            LinMulTConfig.from_dict(
                {
                    "input_feature_dim": [self.feature_dim_1, self.feature_dim_2],
                    "heads": [{"type": "simple", "output_dim": self.output_dim_1}],
                    "time_dim_reducer": "gmp",
                }
            )
        )
        output = list(model([self.x_1, self.x_2], masks=[self.mask_1, self.mask_2]).values())
        self.assertEqual(output[0].shape, (self.batch_size, self.output_dim_1))
        self.assertFalse(torch.isnan(output[0]).any())

    def test_gmp_reducer_fully_masked_no_nan(self):
        # GlobalMaxPooling with an all-False mask must return 0, not -inf.
        model = LinMulT(
            LinMulTConfig.from_dict(
                {
                    "input_feature_dim": [
                        self.feature_dim_1,
                        self.feature_dim_2,
                        self.feature_dim_3,
                    ],
                    "heads": [{"type": "simple", "output_dim": self.output_dim_1}],
                    "time_dim_reducer": "gmp",
                }
            )
        )
        output = list(
            model(
                [self.x_1, self.x_2, self.x_3],
                masks=[self.mask_1, self.mask_2, self.mask_3f],
            ).values()
        )
        self.assertEqual(output[0].shape, (self.batch_size, self.output_dim_1))
        self.assertFalse(torch.isnan(output[0]).any())
        self.assertFalse(torch.isinf(output[0]).any())

    def test_last_reducer_with_mask(self):
        model = LinMulT(
            LinMulTConfig.from_dict(
                {
                    "input_feature_dim": [self.feature_dim_1, self.feature_dim_2],
                    "heads": [{"type": "simple", "output_dim": self.output_dim_1}],
                    "time_dim_reducer": "last",
                }
            )
        )
        output = list(model([self.x_1, self.x_2], masks=[self.mask_1, self.mask_2]).values())
        self.assertEqual(output[0].shape, (self.batch_size, self.output_dim_1))
        self.assertFalse(torch.isnan(output[0]).any())

    def test_last_reducer_fully_masked_no_nan(self):
        # LastTimestamp with an all-False mask must return zeros, not garbage values.
        model = LinMulT(
            LinMulTConfig.from_dict(
                {
                    "input_feature_dim": [
                        self.feature_dim_1,
                        self.feature_dim_2,
                        self.feature_dim_3,
                    ],
                    "heads": [{"type": "simple", "output_dim": self.output_dim_1}],
                    "time_dim_reducer": "last",
                }
            )
        )
        output = list(
            model(
                [self.x_1, self.x_2, self.x_3],
                masks=[self.mask_1, self.mask_2, self.mask_3f],
            ).values()
        )
        self.assertEqual(output[0].shape, (self.batch_size, self.output_dim_1))
        self.assertFalse(torch.isnan(output[0]).any())

    def test_attentionpool_reducer_with_fully_masked_modality_no_nan(self):
        # regression: AttentionPooling with all-masked input must return zeros not NaN
        model = LinMulT(
            LinMulTConfig.from_dict(
                {
                    "input_feature_dim": [
                        self.feature_dim_1,
                        self.feature_dim_2,
                        self.feature_dim_3,
                    ],
                    "heads": [{"type": "simple", "output_dim": self.output_dim_1}],
                    "time_dim_reducer": "attentionpool",
                }
            )
        )
        output = list(
            model(
                [self.x_1, self.x_2, self.x_3],
                masks=[self.mask_1, self.mask_2, self.mask_3f],
            ).values()
        )
        self.assertEqual(output[0].shape, (self.batch_size, self.output_dim_1))
        self.assertFalse(torch.isnan(output[0]).any())

    def test_str_config_path(self):
        model = LinMulT(str(CONFIGS_DIR / "LinMulT.yaml"))
        self.assertIsNotNone(model)

    def test_path_config(self):
        model = LinMulT(CONFIGS_DIR / "LinMulT.yaml")
        self.assertIsNotNone(model)

    def test_wrong_input_count_raises(self):
        model = LinMulT(
            LinMulTConfig.from_dict(
                {
                    "input_feature_dim": [self.feature_dim_2, self.feature_dim_3],
                    "heads": [{"type": "simple", "output_dim": self.output_dim_1}],
                }
            )
        )
        with self.assertRaises(ValueError):
            model([self.x_2])  # expects 2 inputs, got 1

    def test_aux_yaml_end_to_end(self):
        model = LinMulT(str(CONFIGS_DIR / "LinMulT_with_aux.yaml"))
        B = 2
        x_1 = torch.rand((B, 300, 25))
        x_2 = torch.rand((B, 300, 41))
        x_3 = torch.rand((B, 500, 768))
        m_1 = torch.ones((B, 300), dtype=torch.bool)
        m_2 = torch.ones((B, 300), dtype=torch.bool)
        m_3 = torch.zeros((B, 500), dtype=torch.bool)
        outputs, outputs_aux = model([x_1, x_2, x_3], [m_1, m_2, m_3])
        self.assertIsInstance(outputs, dict)
        self.assertIsInstance(outputs_aux, list)
        self.assertFalse(any(torch.isnan(v).any() for v in outputs.values()))

    def test_unimodal_sat_with_auxiliary_heads(self):
        # Covers LinMulT._init_auxiliary_heads line 243: unimodal_sat adds extra d_model
        model = LinMulT(
            LinMulTConfig.from_dict(
                {
                    "input_feature_dim": [self.feature_dim_2, self.feature_dim_3],
                    "heads": [{"type": "simple", "output_dim": self.output_dim_1}],
                    "auxiliary_heads": [{"type": "simple", "output_dim": self.output_dim_2}],
                    "add_module_unimodal_sat": True,
                    "time_dim_reducer": "gap",
                }
            )
        )
        outputs, outputs_aux = model([self.x_2, self.x_3])
        self.assertEqual(list(outputs.values())[0].shape, (self.batch_size, self.output_dim_1))
        self.assertEqual(len(outputs_aux), 2)

    def test_special_handling_4d_tensor(self):
        # Covers LinMulT._apply_projections lines 278-284: weighted_sum on (B,N,T,F) input
        # LinMulT requires >=2 modalities; use the 4D tensor as the first input.
        model = LinMulT(
            LinMulTConfig.from_dict(
                {
                    "input_feature_dim": [self.feature_dim_3, self.feature_dim_2],
                    "heads": [{"type": "simple", "output_dim": self.output_dim_1}],
                    "time_dim_reducer": "gap",
                    "special_handling": {
                        "wavlm": {"type": "weighted_sum", "start_layer": 0, "end_layer": 4}
                    },
                }
            )
        )
        # (B, N_layers, T, F) — e.g. WavLM hidden states
        x_4d = torch.rand(self.batch_size, 6, self.time_dim_2, self.feature_dim_3)
        x_2d = torch.rand(self.batch_size, self.time_dim_2, self.feature_dim_2)
        output = list(model([x_4d, x_2d], names=["wavlm", None]).values())
        self.assertEqual(output[0].shape, (self.batch_size, self.output_dim_1))

    def test_special_modules_are_registered_parameters(self):
        # regression: special_modules used plain dict so params were not tracked
        model = LinMulT(
            LinMulTConfig.from_dict(
                {
                    "input_feature_dim": [self.feature_dim_3, self.feature_dim_2],
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

    # --- Bug regression tests ---

    def test_unimodal_sat_with_attentionpool_reducer(self):
        # Regression Bug 1: AttentionPooling dim mismatch when unimodal_sat=True.
        # TemporalFactory.time_dim_reducer must add +1 to multiplier for the unimodal SAT branch.
        model = LinMulT(
            LinMulTConfig.from_dict(
                {
                    "input_feature_dim": [self.feature_dim_2, self.feature_dim_3],
                    "heads": [{"type": "simple", "output_dim": self.output_dim_1}],
                    "add_module_unimodal_sat": True,
                    "time_dim_reducer": "attentionpool",
                }
            )
        )
        output = list(model([self.x_2, self.x_3]).values())
        self.assertEqual(output[0].shape, (self.batch_size, self.output_dim_1))
        self.assertFalse(torch.isnan(output[0]).any())

    def test_3i_tam_fusion_no_mms(self):
        # Regression Bug 3: _init_output_heads dimension formula was wrong for
        # tam_fusion + n_sequences >= 3 without multimodal_signal.
        model = LinMulT(
            LinMulTConfig.from_dict(
                {
                    "input_feature_dim": [
                        self.feature_dim_1,
                        self.feature_dim_2,
                        self.feature_dim_3,
                    ],
                    "heads": [{"type": "simple", "output_dim": self.output_dim_1}],
                    "add_module_tam_fusion": True,
                    "tam_aligner": "aap",
                    "tam_time_dim": 450,
                }
            )
        )
        output = list(model([self.x_1, self.x_2, self.x_3]).values())
        self.assertEqual(output[0].shape, (self.batch_size, 450, self.output_dim_1))

    def test_tam_fusion_with_sa_fusion(self):
        # Regression Bug 2: SA fusion initialized with pre-TAM combined_dim.
        # After TAM, combined_dim must be updated to n_sequences * d_model.
        model = LinMulT(
            LinMulTConfig.from_dict(
                {
                    "input_feature_dim": [self.feature_dim_2, self.feature_dim_3],
                    "heads": [{"type": "simple", "output_dim": self.output_dim_1}],
                    "add_module_tam_fusion": True,
                    "tam_aligner": "aap",
                    "tam_time_dim": 450,
                    "add_module_sat_fusion": True,
                }
            )
        )
        output = list(model([self.x_2, self.x_3]).values())
        self.assertEqual(output[0].shape, (self.batch_size, 450, self.output_dim_1))

    def test_tam_fusion_with_ffn_fusion(self):
        # Regression Bug 2: FFN projection layers initialized with pre-TAM combined_dim.
        model = LinMulT(
            LinMulTConfig.from_dict(
                {
                    "input_feature_dim": [self.feature_dim_2, self.feature_dim_3],
                    "heads": [{"type": "simple", "output_dim": self.output_dim_1}],
                    "add_module_tam_fusion": True,
                    "tam_aligner": "aap",
                    "tam_time_dim": 450,
                    "add_module_ffn_fusion": True,
                }
            )
        )
        output = list(model([self.x_2, self.x_3]).values())
        self.assertEqual(output[0].shape, (self.batch_size, 450, self.output_dim_1))

    def test_tam_fusion_with_unimodal_sat(self):
        # add_module_tam_fusion + add_module_unimodal_sat:
        # TAM correctly receives (n_cmt+1)*d_model per branch
        # and subsequent modules use updated combined_dim.
        model = LinMulT(
            LinMulTConfig.from_dict(
                {
                    "input_feature_dim": [self.feature_dim_2, self.feature_dim_3],
                    "heads": [{"type": "simple", "output_dim": self.output_dim_1}],
                    "add_module_tam_fusion": True,
                    "tam_aligner": "aap",
                    "tam_time_dim": 450,
                    "add_module_unimodal_sat": True,
                }
            )
        )
        output = list(model([self.x_2, self.x_3]).values())
        self.assertEqual(output[0].shape, (self.batch_size, 450, self.output_dim_1))

    # --- Fusion combinations ---

    def test_sa_fusion_with_ffn_fusion(self):
        # SA and FFN fusion together; requires same-time-dim inputs.
        model = LinMulT(
            LinMulTConfig.from_dict(
                {
                    "input_feature_dim": [self.feature_dim_2, self.feature_dim_3],
                    "heads": [{"type": "simple", "output_dim": self.output_dim_1}],
                    "add_module_sat_fusion": True,
                    "add_module_ffn_fusion": True,
                }
            )
        )
        output = list(model([self.x_2, self.x_3]).values())
        self.assertEqual(output[0].shape, (self.batch_size, self.time_dim_2, self.output_dim_1))

    def test_sa_fusion_with_unimodal_sat(self):
        # SA fusion + unimodal SAT; requires same-time-dim inputs.
        model = LinMulT(
            LinMulTConfig.from_dict(
                {
                    "input_feature_dim": [self.feature_dim_2, self.feature_dim_3],
                    "heads": [{"type": "simple", "output_dim": self.output_dim_1}],
                    "add_module_sat_fusion": True,
                    "add_module_unimodal_sat": True,
                }
            )
        )
        output = list(model([self.x_2, self.x_3]).values())
        self.assertEqual(output[0].shape, (self.batch_size, self.time_dim_2, self.output_dim_1))

    def test_ffn_fusion_with_unimodal_sat(self):
        # FFN fusion + unimodal SAT with time reduction.
        model = LinMulT(
            LinMulTConfig.from_dict(
                {
                    "input_feature_dim": [self.feature_dim_2, self.feature_dim_3],
                    "heads": [{"type": "simple", "output_dim": self.output_dim_1}],
                    "add_module_ffn_fusion": True,
                    "add_module_unimodal_sat": True,
                    "time_dim_reducer": "gap",
                }
            )
        )
        output = list(model([self.x_2, self.x_3]).values())
        self.assertEqual(output[0].shape, (self.batch_size, self.output_dim_1))

    def test_all_fusions_combined(self):
        # sat_fusion + ffn_fusion + unimodal_sat all active; requires same-time-dim inputs.
        model = LinMulT(
            LinMulTConfig.from_dict(
                {
                    "input_feature_dim": [self.feature_dim_2, self.feature_dim_3],
                    "heads": [{"type": "simple", "output_dim": self.output_dim_1}],
                    "add_module_sat_fusion": True,
                    "add_module_ffn_fusion": True,
                    "add_module_unimodal_sat": True,
                }
            )
        )
        output = list(model([self.x_2, self.x_3]).values())
        self.assertEqual(output[0].shape, (self.batch_size, self.time_dim_2, self.output_dim_1))

    def test_multimodal_signal_with_unimodal_sat(self):
        # add_module_multimodal_signal + add_module_unimodal_sat:
        # each branch attends to N+1 sources.
        model = LinMulT(
            LinMulTConfig.from_dict(
                {
                    "input_feature_dim": [self.feature_dim_1, self.feature_dim_2],
                    "heads": [{"type": "simple", "output_dim": self.output_dim_1}],
                    "add_module_multimodal_signal": True,
                    "tam_aligner": "aap",
                    "tam_time_dim": 450,
                    "add_module_unimodal_sat": True,
                    "time_dim_reducer": "gap",
                }
            )
        )
        output = list(model([self.x_1, self.x_2]).values())
        self.assertEqual(output[0].shape, (self.batch_size, self.output_dim_1))

    # --- Mask edge cases ---

    def test_mixed_none_and_real_masks(self):
        # Covers the else-branch in _apply_fusion: when some masks are None (all-valid)
        # and some are real, the AND logic must treat None as an all-True tensor.
        model = LinMulT(
            LinMulTConfig.from_dict(
                {
                    "input_feature_dim": [
                        self.feature_dim_1,
                        self.feature_dim_2,
                        self.feature_dim_3,
                    ],
                    "heads": [{"type": "simple", "output_dim": self.output_dim_1}],
                    "time_dim_reducer": "gap",
                }
            )
        )
        output = list(
            model(
                [self.x_1, self.x_2, self.x_3],
                masks=[self.mask_1, None, self.mask_3],
            ).values()
        )
        self.assertEqual(output[0].shape, (self.batch_size, self.output_dim_1))
        self.assertFalse(torch.isnan(output[0]).any())

    def test_fully_valid_masks_handled_as_no_mask(self):
        # All-True masks are converted to None for efficiency; output shape must match.
        model = LinMulT(
            LinMulTConfig.from_dict(
                {
                    "input_feature_dim": [self.feature_dim_2, self.feature_dim_3],
                    "heads": [{"type": "simple", "output_dim": self.output_dim_1}],
                    "time_dim_reducer": "gap",
                }
            )
        )
        all_true_2 = torch.ones((self.batch_size, self.time_dim_2), dtype=torch.bool)
        all_true_3 = torch.ones((self.batch_size, self.time_dim_3), dtype=torch.bool)
        output = list(model([self.x_2, self.x_3], masks=[all_true_2, all_true_3]).values())
        self.assertEqual(output[0].shape, (self.batch_size, self.output_dim_1))
        self.assertFalse(torch.isnan(output[0]).any())

    def test_single_modality_raises(self):
        # Regression: LinMulT with n_sequences < 2 crashed with cryptic RuntimeError.
        with self.assertRaises(ValueError) as ctx:
            LinMulT(LinMulTConfig.from_dict({"input_feature_dim": [self.feature_dim_1]}))
        self.assertIn("at least 2", str(ctx.exception))
