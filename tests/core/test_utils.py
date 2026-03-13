import unittest

import torch

from linmult import FIXTURE_DIR
from linmult.core.utils import apply_logit_aggregation, load_config

CONFIGS_DIR = FIXTURE_DIR / "configs"


class TestLoadConfig(unittest.TestCase):
    def test_load_lint_config(self):
        config = load_config(CONFIGS_DIR / "LinT.yaml")
        self.assertIsInstance(config, dict)
        self.assertIn("input_feature_dim", config)

    def test_load_linmult_config(self):
        config = load_config(CONFIGS_DIR / "LinMulT.yaml")
        self.assertIsInstance(config, dict)
        self.assertIsInstance(config["input_feature_dim"], list)


class TestApplyLogitAggregation(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.B, cls.T, cls.F = 4, 20, 8
        cls.x = torch.rand(cls.B, cls.T, cls.F)
        cls.mask = torch.ones(cls.B, cls.T, dtype=torch.bool)
        cls.mask[:, 15:] = False  # last 5 positions masked

    def test_meanpooling_no_mask(self):
        out = apply_logit_aggregation(self.x, method="meanpooling")
        self.assertEqual(out.shape, (self.B, self.F))
        expected = self.x.mean(dim=1)
        self.assertTrue(torch.allclose(out, expected, atol=1e-5))

    def test_meanpooling_with_mask(self):
        out = apply_logit_aggregation(self.x, mask=self.mask, method="meanpooling")
        self.assertEqual(out.shape, (self.B, self.F))
        # Only first 15 positions contribute
        expected = self.x[:, :15, :].mean(dim=1)
        self.assertTrue(torch.allclose(out, expected, atol=1e-5))

    def test_maxpooling_no_mask(self):
        out = apply_logit_aggregation(self.x, method="maxpooling")
        self.assertEqual(out.shape, (self.B, self.F))
        expected = self.x.max(dim=1)[0]
        self.assertTrue(torch.allclose(out, expected, atol=1e-5))

    def test_maxpooling_with_mask(self):
        out = apply_logit_aggregation(self.x, mask=self.mask, method="maxpooling")
        self.assertEqual(out.shape, (self.B, self.F))
        # Only first 15 positions are unmasked; verify no -inf bleeds through
        self.assertFalse(torch.isinf(out).any())
        expected = self.x[:, :15, :].max(dim=1)[0]
        self.assertTrue(torch.allclose(out, expected, atol=1e-5))

    def test_invalid_method_raises(self):
        with self.assertRaises(ValueError):
            apply_logit_aggregation(self.x, method="unknown")

    def test_none_mask_treated_as_all_valid(self):
        out_no_mask = apply_logit_aggregation(self.x, mask=None, method="meanpooling")
        mask_all = torch.ones(self.B, self.T, dtype=torch.bool)
        out_full_mask = apply_logit_aggregation(self.x, mask=mask_all, method="meanpooling")
        self.assertTrue(torch.allclose(out_no_mask, out_full_mask, atol=1e-5))

    def test_maxpooling_fully_masked_returns_zero(self):
        # Regression: fully-masked samples returned -inf instead of 0.
        fully_masked = torch.zeros(self.B, self.T, dtype=torch.bool)
        out = apply_logit_aggregation(self.x, mask=fully_masked, method="maxpooling")
        self.assertEqual(out.shape, (self.B, self.F))
        self.assertFalse(torch.isinf(out).any())
        self.assertTrue((out == 0.0).all())

    def test_meanpooling_fully_masked_returns_zero(self):
        fully_masked = torch.zeros(self.B, self.T, dtype=torch.bool)
        out = apply_logit_aggregation(self.x, mask=fully_masked, method="meanpooling")
        self.assertEqual(out.shape, (self.B, self.F))
        self.assertTrue((out == 0.0).all())
