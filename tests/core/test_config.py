import unittest

from linmult import FIXTURE_DIR
from linmult.core.config import HeadConfig, LinMulTConfig, LinTConfig

CONFIGS_DIR = FIXTURE_DIR / "configs"


class TestHeadConfigDefaults(unittest.TestCase):
    def test_required_fields(self):
        cfg = HeadConfig(type="simple", output_dim=5)
        self.assertEqual(cfg.type, "simple")
        self.assertEqual(cfg.output_dim, 5)

    def test_default_name_is_empty_string(self):
        cfg = HeadConfig(type="simple", output_dim=5)
        self.assertEqual(cfg.name, "")

    def test_default_norm(self):
        cfg = HeadConfig(type="simple", output_dim=5)
        self.assertEqual(cfg.norm, "bn")

    def test_default_pooling(self):
        cfg = HeadConfig(type="simple", output_dim=5)
        # by default we preserve the time dimension unless user opts into pooling
        self.assertIsNone(cfg.pooling)

    def test_default_hidden_dim(self):
        cfg = HeadConfig(type="simple", output_dim=5)
        self.assertEqual(cfg.hidden_dim, 256)

    def test_default_dropout(self):
        cfg = HeadConfig(type="simple", output_dim=5)
        self.assertAlmostEqual(cfg.dropout, 0.1)

    def test_pooling_none_allowed(self):
        cfg = HeadConfig(type="simple", output_dim=5, pooling=None)
        self.assertIsNone(cfg.pooling)

    def test_default_time_dims_none(self):
        cfg = HeadConfig(type="upsample", output_dim=8)
        self.assertIsNone(cfg.input_time_dim)
        self.assertIsNone(cfg.output_time_dim)

    def test_custom_fields(self):
        cfg = HeadConfig(
            type="sequence_aggregation",
            output_dim=10,
            name="valence",
            norm="in",
            pooling="attentionpool",
            hidden_dim=128,
            dropout=0.2,
        )
        self.assertEqual(cfg.name, "valence")
        self.assertEqual(cfg.norm, "in")
        self.assertEqual(cfg.pooling, "attentionpool")
        self.assertEqual(cfg.hidden_dim, 128)
        self.assertAlmostEqual(cfg.dropout, 0.2)


class TestHeadConfigFromDict(unittest.TestCase):
    def test_basic_from_dict(self):
        cfg = HeadConfig.from_dict({"type": "simple", "output_dim": 5})
        self.assertEqual(cfg.type, "simple")
        self.assertEqual(cfg.output_dim, 5)

    def test_unknown_keys_ignored(self):
        cfg = HeadConfig.from_dict({"type": "simple", "output_dim": 5, "nonexistent_key": 99})
        self.assertEqual(cfg.output_dim, 5)
        self.assertFalse(hasattr(cfg, "nonexistent_key"))

    def test_all_fields_from_dict(self):
        cfg = HeadConfig.from_dict(
            {
                "type": "sequence_aggregation",
                "output_dim": 7,
                "name": "emotion",
                "norm": "in",
                "pooling": "gmp",
                "hidden_dim": 512,
                "dropout": 0.3,
            }
        )
        self.assertEqual(cfg.type, "sequence_aggregation")
        self.assertEqual(cfg.output_dim, 7)
        self.assertEqual(cfg.name, "emotion")
        self.assertEqual(cfg.norm, "in")
        self.assertEqual(cfg.pooling, "gmp")
        self.assertEqual(cfg.hidden_dim, 512)
        self.assertAlmostEqual(cfg.dropout, 0.3)

    def test_upsample_time_dims(self):
        cfg = HeadConfig.from_dict(
            {"type": "upsample", "output_dim": 16, "input_time_dim": 10, "output_time_dim": 40}
        )
        self.assertEqual(cfg.input_time_dim, 10)
        self.assertEqual(cfg.output_time_dim, 40)

    def test_pooling_from_dict(self):
        cfg = HeadConfig.from_dict({"type": "simple", "output_dim": 5, "pooling": "gap"})
        self.assertEqual(cfg.pooling, "gap")

    def test_from_dict_preserves_existing_headconfig(self):
        # from_dict on LinTConfig: already-HeadConfig instances must pass through as-is
        existing = HeadConfig(type="vector", output_dim=3, name="existing")
        cfg = LinTConfig.from_dict(
            {
                "input_feature_dim": 32,
                "heads": [existing],
            }
        )
        self.assertIs(cfg.heads[0], existing)


class TestHeadConfigEquality(unittest.TestCase):
    def test_equal_configs(self):
        cfg1 = HeadConfig(type="simple", output_dim=5)
        cfg2 = HeadConfig(type="simple", output_dim=5)
        self.assertEqual(cfg1, cfg2)

    def test_unequal_configs(self):
        cfg1 = HeadConfig(type="simple", output_dim=5)
        cfg2 = HeadConfig(type="simple", output_dim=10)
        self.assertNotEqual(cfg1, cfg2)


class TestLinTConfigDefaults(unittest.TestCase):
    def test_required_field(self):
        cfg = LinTConfig(input_feature_dim=32)
        self.assertEqual(cfg.input_feature_dim, 32)

    def test_default_name(self):
        cfg = LinTConfig(input_feature_dim=32)
        self.assertEqual(cfg.name, "")

    def test_default_d_model(self):
        cfg = LinTConfig(input_feature_dim=32)
        self.assertEqual(cfg.d_model, 40)

    def test_default_num_heads(self):
        cfg = LinTConfig(input_feature_dim=32)
        self.assertEqual(cfg.num_heads, 8)

    def test_default_cmt_num_layers(self):
        cfg = LinTConfig(input_feature_dim=32)
        self.assertEqual(cfg.cmt_num_layers, 6)

    def test_default_attention_type(self):
        cfg = LinTConfig(input_feature_dim=32)
        self.assertEqual(cfg.attention_type, "linear")

    def test_default_dropout_fields(self):
        cfg = LinTConfig(input_feature_dim=32)
        self.assertAlmostEqual(cfg.dropout_input, 0.0)
        self.assertAlmostEqual(cfg.dropout_output, 0.0)
        self.assertAlmostEqual(cfg.dropout_pe, 0.0)
        self.assertAlmostEqual(cfg.dropout_ffn, 0.1)
        self.assertAlmostEqual(cfg.dropout_attention, 0.0)

    def test_default_time_dim_reducer_none(self):
        cfg = LinTConfig(input_feature_dim=32)
        self.assertIsNone(cfg.time_dim_reducer)

    def test_default_add_module_ffn_fusion_false(self):
        cfg = LinTConfig(input_feature_dim=32)
        self.assertFalse(cfg.add_module_ffn_fusion)

    def test_default_heads_empty(self):
        cfg = LinTConfig(input_feature_dim=32)
        self.assertEqual(cfg.heads, [])

    def test_default_special_handling_empty(self):
        cfg = LinTConfig(input_feature_dim=32)
        self.assertEqual(cfg.special_handling, {})

    def test_heads_not_shared_between_instances(self):
        cfg1 = LinTConfig(input_feature_dim=32)
        cfg2 = LinTConfig(input_feature_dim=32)
        cfg1.heads.append(HeadConfig(type="simple", output_dim=1))
        self.assertEqual(len(cfg2.heads), 0)


class TestLinTConfigFromDict(unittest.TestCase):
    def test_basic_from_dict(self):
        cfg = LinTConfig.from_dict({"input_feature_dim": 25, "d_model": 64})
        self.assertEqual(cfg.input_feature_dim, 25)
        self.assertEqual(cfg.d_model, 64)

    def test_unknown_keys_ignored(self):
        cfg = LinTConfig.from_dict(
            {"input_feature_dim": 25, "unknown_key": 123, "another_unknown": "value"}
        )
        self.assertEqual(cfg.input_feature_dim, 25)
        self.assertFalse(hasattr(cfg, "unknown_key"))

    def test_heads_converted_from_dicts(self):
        cfg = LinTConfig.from_dict(
            {
                "input_feature_dim": 25,
                "heads": [
                    {"type": "simple", "output_dim": 5},
                    {"type": "sequence", "output_dim": 3},
                ],
            }
        )
        self.assertEqual(len(cfg.heads), 2)
        self.assertIsInstance(cfg.heads[0], HeadConfig)
        self.assertIsInstance(cfg.heads[1], HeadConfig)
        self.assertEqual(cfg.heads[0].type, "simple")
        self.assertEqual(cfg.heads[1].type, "sequence")

    def test_heads_already_headconfig_pass_through(self):
        head = HeadConfig(type="vector", output_dim=4)
        cfg = LinTConfig.from_dict({"input_feature_dim": 32, "heads": [head]})
        self.assertIs(cfg.heads[0], head)

    def test_attention_keys(self):
        cfg = LinTConfig.from_dict(
            {
                "input_feature_dim": 32,
                "attention_type": "performer",
                "performer_num_random_features": 64,
                "flash_query_key_dim": 20,
                "bigbird_block_size": 32,
                "bigbird_num_global_tokens": 8,
                "bigbird_num_random_tokens": 5,
            }
        )
        self.assertEqual(cfg.attention_type, "performer")
        self.assertEqual(cfg.performer_num_random_features, 64)
        self.assertEqual(cfg.flash_query_key_dim, 20)
        self.assertEqual(cfg.bigbird_block_size, 32)
        self.assertEqual(cfg.bigbird_num_global_tokens, 8)
        self.assertEqual(cfg.bigbird_num_random_tokens, 5)

    def test_dropout_keys(self):
        cfg = LinTConfig.from_dict(
            {
                "input_feature_dim": 32,
                "dropout_input": 0.1,
                "dropout_output": 0.2,
                "dropout_pe": 0.05,
                "dropout_ffn": 0.3,
                "dropout_attention": 0.05,
            }
        )
        self.assertAlmostEqual(cfg.dropout_input, 0.1)
        self.assertAlmostEqual(cfg.dropout_output, 0.2)
        self.assertAlmostEqual(cfg.dropout_pe, 0.05)
        self.assertAlmostEqual(cfg.dropout_ffn, 0.3)
        self.assertAlmostEqual(cfg.dropout_attention, 0.05)

    def test_special_handling(self):
        cfg = LinTConfig.from_dict(
            {
                "input_feature_dim": 41,
                "special_handling": {
                    "wavlm": {"type": "weighted_sum", "start_layer": 0, "end_layer": 12}
                },
            }
        )
        self.assertIn("wavlm", cfg.special_handling)
        self.assertEqual(cfg.special_handling["wavlm"]["type"], "weighted_sum")

    def test_equality(self):
        cfg1 = LinTConfig.from_dict({"input_feature_dim": 32, "d_model": 40})
        cfg2 = LinTConfig.from_dict({"input_feature_dim": 32, "d_model": 40})
        self.assertEqual(cfg1, cfg2)

    def test_inequality(self):
        cfg1 = LinTConfig.from_dict({"input_feature_dim": 32, "d_model": 40})
        cfg2 = LinTConfig.from_dict({"input_feature_dim": 32, "d_model": 80})
        self.assertNotEqual(cfg1, cfg2)


class TestLinTConfigDirectConstruction(unittest.TestCase):
    def test_heads_coerced_from_dicts_directly(self):
        cfg = LinTConfig(
            input_feature_dim=32,
            heads=[{"type": "simple", "output_dim": 5}],
        )
        self.assertIsInstance(cfg.heads[0], HeadConfig)
        self.assertEqual(cfg.heads[0].type, "simple")

    def test_heads_headconfig_pass_through_directly(self):
        head = HeadConfig(type="vector", output_dim=4)
        cfg = LinTConfig(input_feature_dim=32, heads=[head])
        self.assertIs(cfg.heads[0], head)


class TestLinTConfigFromYaml(unittest.TestCase):
    def test_from_yaml_lint(self):
        cfg = LinTConfig.from_yaml(CONFIGS_DIR / "LinT.yaml")
        self.assertIsInstance(cfg, LinTConfig)
        self.assertIsInstance(cfg.input_feature_dim, int)
        self.assertGreater(len(cfg.heads), 0)
        for head in cfg.heads:
            self.assertIsInstance(head, HeadConfig)

    def test_from_yaml_str_path(self):
        cfg = LinTConfig.from_yaml(str(CONFIGS_DIR / "LinT.yaml"))
        self.assertIsInstance(cfg, LinTConfig)

    def test_from_yaml_special_handling_loaded(self):
        cfg = LinTConfig.from_yaml(CONFIGS_DIR / "LinT.yaml")
        # LinT.yaml defines wavlm_baseplus in special_handling
        self.assertIn("wavlm_baseplus", cfg.special_handling)


class TestLinMulTConfigDefaults(unittest.TestCase):
    def test_required_field(self):
        cfg = LinMulTConfig(input_feature_dim=[25, 35])
        self.assertEqual(cfg.input_feature_dim, [25, 35])

    def test_default_name(self):
        cfg = LinMulTConfig(input_feature_dim=[25, 35])
        self.assertEqual(cfg.name, "")

    def test_default_d_model(self):
        cfg = LinMulTConfig(input_feature_dim=[25, 35])
        self.assertEqual(cfg.d_model, 40)

    def test_default_num_heads(self):
        cfg = LinMulTConfig(input_feature_dim=[25, 35])
        self.assertEqual(cfg.num_heads, 8)

    def test_default_cmt_num_layers(self):
        cfg = LinMulTConfig(input_feature_dim=[25, 35])
        self.assertEqual(cfg.cmt_num_layers, 6)

    def test_default_branch_sat_num_layers(self):
        cfg = LinMulTConfig(input_feature_dim=[25, 35])
        self.assertEqual(cfg.branch_sat_num_layers, 6)

    def test_default_attention_type(self):
        cfg = LinMulTConfig(input_feature_dim=[25, 35])
        self.assertEqual(cfg.attention_type, "linear")

    def test_default_dropout_fields(self):
        cfg = LinMulTConfig(input_feature_dim=[25, 35])
        self.assertAlmostEqual(cfg.dropout_input, 0.0)
        self.assertAlmostEqual(cfg.dropout_output, 0.0)
        self.assertAlmostEqual(cfg.dropout_pe, 0.0)
        self.assertAlmostEqual(cfg.dropout_ffn, 0.1)
        self.assertAlmostEqual(cfg.dropout_attention, 0.0)
        self.assertAlmostEqual(cfg.dropout_tam, 0.1)

    def test_default_time_fields_none(self):
        cfg = LinMulTConfig(input_feature_dim=[25, 35])
        self.assertIsNone(cfg.time_dim_reducer)
        self.assertIsNone(cfg.tam_aligner)
        self.assertIsNone(cfg.tam_time_dim)

    def test_default_module_flags_false(self):
        cfg = LinMulTConfig(input_feature_dim=[25, 35])
        self.assertFalse(cfg.add_module_ffn_fusion)
        self.assertFalse(cfg.add_module_tam_fusion)
        self.assertFalse(cfg.add_module_multimodal_signal)
        self.assertFalse(cfg.add_module_sat_fusion)
        self.assertFalse(cfg.add_module_unimodal_sat)

    def test_default_heads_empty(self):
        cfg = LinMulTConfig(input_feature_dim=[25, 35])
        self.assertEqual(cfg.heads, [])

    def test_default_auxiliary_heads_empty(self):
        cfg = LinMulTConfig(input_feature_dim=[25, 35])
        self.assertEqual(cfg.auxiliary_heads, [])

    def test_default_special_handling_empty(self):
        cfg = LinMulTConfig(input_feature_dim=[25, 35])
        self.assertEqual(cfg.special_handling, {})

    def test_heads_not_shared_between_instances(self):
        cfg1 = LinMulTConfig(input_feature_dim=[25, 35])
        cfg2 = LinMulTConfig(input_feature_dim=[25, 35])
        cfg1.heads.append(HeadConfig(type="simple", output_dim=1))
        self.assertEqual(len(cfg2.heads), 0)


class TestLinMulTConfigFromDict(unittest.TestCase):
    def test_basic_from_dict(self):
        cfg = LinMulTConfig.from_dict({"input_feature_dim": [25, 35], "d_model": 64})
        self.assertEqual(cfg.input_feature_dim, [25, 35])
        self.assertEqual(cfg.d_model, 64)

    def test_unknown_keys_ignored(self):
        cfg = LinMulTConfig.from_dict(
            {"input_feature_dim": [25, 35], "unknown_key": 123, "another_unknown": "value"}
        )
        self.assertEqual(cfg.input_feature_dim, [25, 35])
        self.assertFalse(hasattr(cfg, "unknown_key"))

    def test_heads_converted_from_dicts(self):
        cfg = LinMulTConfig.from_dict(
            {
                "input_feature_dim": [25, 35],
                "heads": [
                    {"type": "simple", "output_dim": 5},
                    {"type": "sequence", "output_dim": 3},
                ],
            }
        )
        self.assertEqual(len(cfg.heads), 2)
        self.assertIsInstance(cfg.heads[0], HeadConfig)
        self.assertIsInstance(cfg.heads[1], HeadConfig)
        self.assertEqual(cfg.heads[0].type, "simple")
        self.assertEqual(cfg.heads[1].type, "sequence")

    def test_auxiliary_heads_converted_from_dicts(self):
        cfg = LinMulTConfig.from_dict(
            {
                "input_feature_dim": [25, 35],
                "heads": [{"type": "simple", "output_dim": 5}],
                "auxiliary_heads": [{"type": "simple", "output_dim": 1}],
            }
        )
        self.assertEqual(len(cfg.auxiliary_heads), 1)
        self.assertIsInstance(cfg.auxiliary_heads[0], HeadConfig)

    def test_heads_already_headconfig_pass_through(self):
        head = HeadConfig(type="vector", output_dim=4)
        cfg = LinMulTConfig.from_dict({"input_feature_dim": [32, 16], "heads": [head]})
        self.assertIs(cfg.heads[0], head)

    def test_all_architecture_keys(self):
        cfg = LinMulTConfig.from_dict(
            {
                "input_feature_dim": [25, 35],
                "d_model": 80,
                "num_heads": 4,
                "cmt_num_layers": 3,
                "branch_sat_num_layers": 2,
                "mms_num_layers": 2,
                "fusion_num_layers": 2,
                "fusion_sat_num_layers": 2,
            }
        )
        self.assertEqual(cfg.d_model, 80)
        self.assertEqual(cfg.num_heads, 4)
        self.assertEqual(cfg.cmt_num_layers, 3)
        self.assertEqual(cfg.branch_sat_num_layers, 2)
        self.assertEqual(cfg.mms_num_layers, 2)
        self.assertEqual(cfg.fusion_num_layers, 2)
        self.assertEqual(cfg.fusion_sat_num_layers, 2)

    def test_attention_keys(self):
        cfg = LinMulTConfig.from_dict(
            {
                "input_feature_dim": [25, 35],
                "attention_type": "performer",
                "performer_num_random_features": 64,
                "flash_query_key_dim": 20,
                "bigbird_block_size": 32,
                "bigbird_num_global_tokens": 8,
                "bigbird_num_random_tokens": 5,
            }
        )
        self.assertEqual(cfg.attention_type, "performer")
        self.assertEqual(cfg.performer_num_random_features, 64)
        self.assertEqual(cfg.flash_query_key_dim, 20)
        self.assertEqual(cfg.bigbird_block_size, 32)
        self.assertEqual(cfg.bigbird_num_global_tokens, 8)
        self.assertEqual(cfg.bigbird_num_random_tokens, 5)

    def test_dropout_keys(self):
        cfg = LinMulTConfig.from_dict(
            {
                "input_feature_dim": [25, 35],
                "dropout_input": 0.1,
                "dropout_output": 0.2,
                "dropout_pe": 0.05,
                "dropout_ffn": 0.3,
                "dropout_attention": 0.05,
                "dropout_tam": 0.2,
            }
        )
        self.assertAlmostEqual(cfg.dropout_input, 0.1)
        self.assertAlmostEqual(cfg.dropout_output, 0.2)
        self.assertAlmostEqual(cfg.dropout_pe, 0.05)
        self.assertAlmostEqual(cfg.dropout_ffn, 0.3)
        self.assertAlmostEqual(cfg.dropout_attention, 0.05)
        self.assertAlmostEqual(cfg.dropout_tam, 0.2)

    def test_temporal_keys(self):
        cfg = LinMulTConfig.from_dict(
            {
                "input_feature_dim": [25, 35],
                "time_dim_reducer": "gap",
                "tam_aligner": "aap",
                "tam_time_dim": 300,
            }
        )
        self.assertEqual(cfg.time_dim_reducer, "gap")
        self.assertEqual(cfg.tam_aligner, "aap")
        self.assertEqual(cfg.tam_time_dim, 300)

    def test_module_flags(self):
        cfg = LinMulTConfig.from_dict(
            {
                "input_feature_dim": [25, 35],
                "add_module_ffn_fusion": True,
                "add_module_tam_fusion": True,
                "add_module_multimodal_signal": True,
                "add_module_sat_fusion": True,
                "add_module_unimodal_sat": True,
                "tam_time_dim": 300,
            }
        )
        self.assertTrue(cfg.add_module_ffn_fusion)
        self.assertTrue(cfg.add_module_tam_fusion)
        self.assertTrue(cfg.add_module_multimodal_signal)
        self.assertTrue(cfg.add_module_sat_fusion)
        self.assertTrue(cfg.add_module_unimodal_sat)

    def test_special_handling(self):
        cfg = LinMulTConfig.from_dict(
            {
                "input_feature_dim": [25, 41],
                "special_handling": {
                    "wavlm": {"type": "weighted_sum", "start_layer": 0, "end_layer": 12}
                },
            }
        )
        self.assertIn("wavlm", cfg.special_handling)
        self.assertEqual(cfg.special_handling["wavlm"]["type"], "weighted_sum")

    def test_equality(self):
        cfg1 = LinMulTConfig.from_dict({"input_feature_dim": [25, 35], "d_model": 40})
        cfg2 = LinMulTConfig.from_dict({"input_feature_dim": [25, 35], "d_model": 40})
        self.assertEqual(cfg1, cfg2)

    def test_inequality(self):
        cfg1 = LinMulTConfig.from_dict({"input_feature_dim": [25, 35], "d_model": 40})
        cfg2 = LinMulTConfig.from_dict({"input_feature_dim": [25, 35], "d_model": 80})
        self.assertNotEqual(cfg1, cfg2)


class TestLinMulTConfigDirectConstruction(unittest.TestCase):
    def test_heads_coerced_from_dicts_directly(self):
        cfg = LinMulTConfig(
            input_feature_dim=[25, 35],
            heads=[{"type": "simple", "output_dim": 5}],
        )
        self.assertIsInstance(cfg.heads[0], HeadConfig)
        self.assertEqual(cfg.heads[0].type, "simple")

    def test_auxiliary_heads_coerced_from_dicts_directly(self):
        cfg = LinMulTConfig(
            input_feature_dim=[25, 35],
            auxiliary_heads=[{"type": "simple", "output_dim": 1}],
        )
        self.assertIsInstance(cfg.auxiliary_heads[0], HeadConfig)

    def test_heads_headconfig_pass_through_directly(self):
        head = HeadConfig(type="vector", output_dim=4)
        cfg = LinMulTConfig(input_feature_dim=[25, 35], heads=[head])
        self.assertIs(cfg.heads[0], head)


class TestLinMulTConfigTAMValidation(unittest.TestCase):
    def test_tam_fusion_requires_tam_time_dim(self):
        with self.assertRaises(ValueError) as ctx:
            LinMulTConfig(input_feature_dim=[25, 35], add_module_tam_fusion=True)
        self.assertIn("tam_time_dim", str(ctx.exception))

    def test_multimodal_signal_requires_tam_time_dim(self):
        with self.assertRaises(ValueError) as ctx:
            LinMulTConfig(input_feature_dim=[25, 35], add_module_multimodal_signal=True)
        self.assertIn("tam_time_dim", str(ctx.exception))

    def test_both_tam_modules_error_lists_both(self):
        with self.assertRaises(ValueError) as ctx:
            LinMulTConfig(
                input_feature_dim=[25, 35],
                add_module_tam_fusion=True,
                add_module_multimodal_signal=True,
            )
        msg = str(ctx.exception)
        self.assertIn("add_module_tam_fusion", msg)
        self.assertIn("add_module_multimodal_signal", msg)

    def test_tam_modules_valid_with_tam_time_dim(self):
        cfg = LinMulTConfig(
            input_feature_dim=[25, 35],
            add_module_tam_fusion=True,
            add_module_multimodal_signal=True,
            tam_time_dim=300,
        )
        self.assertTrue(cfg.add_module_tam_fusion)
        self.assertTrue(cfg.add_module_multimodal_signal)


class TestLinMulTConfigFromYaml(unittest.TestCase):
    def test_from_yaml_linmult(self):
        cfg = LinMulTConfig.from_yaml(CONFIGS_DIR / "LinMulT.yaml")
        self.assertIsInstance(cfg, LinMulTConfig)
        self.assertIsInstance(cfg.input_feature_dim, list)
        self.assertEqual(len(cfg.input_feature_dim), 3)
        self.assertGreater(len(cfg.heads), 0)
        for head in cfg.heads:
            self.assertIsInstance(head, HeadConfig)

    def test_from_yaml_str_path(self):
        cfg = LinMulTConfig.from_yaml(str(CONFIGS_DIR / "LinMulT.yaml"))
        self.assertIsInstance(cfg, LinMulTConfig)

    def test_from_yaml_linmult_with_aux(self):
        cfg = LinMulTConfig.from_yaml(CONFIGS_DIR / "LinMulT_with_aux.yaml")
        self.assertIsInstance(cfg, LinMulTConfig)
        self.assertGreater(len(cfg.auxiliary_heads), 0)
        for head in cfg.auxiliary_heads:
            self.assertIsInstance(head, HeadConfig)

    def test_from_yaml_special_handling_loaded(self):
        cfg = LinMulTConfig.from_yaml(CONFIGS_DIR / "LinMulT.yaml")
        # LinMulT.yaml defines wavlm_baseplus in special_handling
        self.assertIn("wavlm_baseplus", cfg.special_handling)
