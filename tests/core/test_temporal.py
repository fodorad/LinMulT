import unittest

import torch

from linmult.core.temporal import (
    TAM,
    TRM,
    AdaptiveAvgPooling,
    AdaptiveMaxPooling,
    AttentionPooling,
    GlobalAvgPooling,
    GlobalMaxPooling,
    LastTimestamp,
    TemporalPadding,
)


class TestTemporalPadding(unittest.TestCase):
    def setUp(self):
        self.padder = TemporalPadding()

    def test_truncate(self):
        x = torch.rand(2, 20, 8)
        out, mask = self.padder(x, time_dim=10)
        self.assertEqual(out.shape, (2, 10, 8))
        self.assertEqual(mask.shape, (2, 10))
        self.assertTrue(mask.all())

    def test_pad(self):
        x = torch.rand(2, 5, 8)
        out, mask = self.padder(x, time_dim=10)
        self.assertEqual(out.shape, (2, 10, 8))
        self.assertEqual(mask.shape, (2, 10))
        self.assertTrue(mask[:, :5].all())
        self.assertFalse(mask[:, 5:].any())

    def test_same_size(self):
        x = torch.rand(2, 10, 8)
        out, mask = self.padder(x, time_dim=10)
        self.assertEqual(out.shape, (2, 10, 8))
        self.assertTrue(mask.all())

    def test_with_mask_truncate(self):
        x = torch.rand(2, 20, 8)
        mask = torch.ones(2, 20, dtype=torch.bool)
        mask[:, 15:] = False
        out, out_mask = self.padder(x, time_dim=10, mask=mask)
        self.assertEqual(out_mask.shape, (2, 10))
        self.assertTrue(out_mask.all())  # first 10 were valid

    def test_with_mask_pad(self):
        x = torch.rand(2, 5, 8)
        mask = torch.ones(2, 5, dtype=torch.bool)
        out, out_mask = self.padder(x, time_dim=10, mask=mask)
        self.assertTrue(out_mask[:, :5].all())
        self.assertFalse(out_mask[:, 5:].any())


class TestAdaptiveAvgPooling(unittest.TestCase):
    def setUp(self):
        self.pool = AdaptiveAvgPooling()

    def test_without_mask(self):
        x = torch.rand(2, 20, 8)
        out, mask = self.pool(x, time_dim=10)
        self.assertEqual(out.shape, (2, 10, 8))
        self.assertTrue(mask.all())

    def test_with_mask(self):
        x = torch.rand(2, 20, 8)
        mask = torch.ones(2, 20, dtype=torch.bool)
        mask[:, 18:] = False
        out, out_mask = self.pool(x, time_dim=10, mask=mask)
        self.assertEqual(out.shape, (2, 10, 8))
        self.assertEqual(out_mask.shape, (2, 10))
        self.assertFalse(torch.isnan(out).any())


class TestAdaptiveMaxPooling(unittest.TestCase):
    def setUp(self):
        self.pool = AdaptiveMaxPooling()

    def test_without_mask(self):
        x = torch.rand(2, 20, 8)
        out, mask = self.pool(x, time_dim=10)
        self.assertEqual(out.shape, (2, 10, 8))
        self.assertTrue(mask.all())

    def test_with_mask(self):
        x = torch.rand(2, 20, 8)
        mask = torch.ones(2, 20, dtype=torch.bool)
        mask[:, 18:] = False
        out, out_mask = self.pool(x, time_dim=10, mask=mask)
        self.assertEqual(out.shape, (2, 10, 8))
        self.assertFalse(torch.isinf(out).any())


class TestGlobalAvgPooling(unittest.TestCase):
    def setUp(self):
        self.pool = GlobalAvgPooling()

    def test_without_mask(self):
        x = torch.rand(4, 10, 16)
        out = self.pool(x)
        self.assertEqual(out.shape, (4, 16))

    def test_with_mask(self):
        x = torch.rand(4, 10, 16)
        mask = torch.ones(4, 10, dtype=torch.bool)
        mask[:, 8:] = False
        out = self.pool(x, mask)
        self.assertEqual(out.shape, (4, 16))
        # avg should equal mean of first 8 elements
        expected = x[:, :8, :].mean(dim=1)
        self.assertTrue(torch.allclose(out, expected, atol=1e-5))

    def test_fully_masked_no_div_zero(self):
        x = torch.rand(4, 10, 16)
        mask = torch.zeros(4, 10, dtype=torch.bool)
        out = self.pool(x, mask)
        self.assertFalse(torch.isnan(out).any())


class TestGlobalMaxPooling(unittest.TestCase):
    def setUp(self):
        self.pool = GlobalMaxPooling()

    def test_without_mask(self):
        x = torch.rand(4, 10, 16)
        out = self.pool(x)
        self.assertEqual(out.shape, (4, 16))

    def test_with_mask(self):
        x = torch.rand(4, 10, 16)
        mask = torch.ones(4, 10, dtype=torch.bool)
        mask[:, 8:] = False
        out = self.pool(x, mask)
        self.assertEqual(out.shape, (4, 16))
        expected = x[:, :8, :].max(dim=1)[0]
        self.assertTrue(torch.allclose(out, expected, atol=1e-5))

    def test_fully_masked_returns_zeros_not_inf(self):
        # regression: was returning -inf for fully-masked samples
        x = torch.rand(4, 10, 16) + 1.0
        mask = torch.zeros(4, 10, dtype=torch.bool)
        out = self.pool(x, mask)
        self.assertEqual(out.shape, (4, 16))
        self.assertFalse(torch.isinf(out).any())
        self.assertTrue(torch.allclose(out, torch.zeros(4, 16)))

    def test_mixed_mask_partial_and_full(self):
        x = torch.rand(4, 10, 16) + 1.0
        mask = torch.ones(4, 10, dtype=torch.bool)
        mask[2, :] = False  # sample 2 is fully masked
        mask[:, 8:] = False
        out = self.pool(x, mask)
        self.assertTrue(torch.allclose(out[2], torch.zeros(16)))
        self.assertFalse(torch.isinf(out).any())


class TestLastTimestamp(unittest.TestCase):
    def setUp(self):
        self.module = LastTimestamp()

    def test_without_mask(self):
        x = torch.rand(4, 10, 16)
        out = self.module(x)
        self.assertEqual(out.shape, (4, 16))
        # Should return x[:, -1, :]
        self.assertTrue(torch.allclose(out, x[:, -1, :]))

    def test_with_partial_mask(self):
        x = torch.rand(4, 10, 16)
        mask = torch.ones(4, 10, dtype=torch.bool)
        mask[:, 6:] = False  # valid up to index 5 (6 valid tokens)
        out = self.module(x, mask)
        self.assertEqual(out.shape, (4, 16))
        # Last valid token is at index 5
        self.assertTrue(torch.allclose(out, x[:, 5, :]))

    def test_with_fully_masked_returns_zeros(self):
        # regression: was returning x[:, -1, :] (last element) for fully-masked samples
        x = torch.rand(4, 10, 16) + 1.0  # ensure non-zero
        mask = torch.zeros(4, 10, dtype=torch.bool)
        out = self.module(x, mask)
        self.assertEqual(out.shape, (4, 16))
        self.assertTrue(torch.allclose(out, torch.zeros(4, 16)))

    def test_mixed_mask_partial_and_full(self):
        x = torch.rand(4, 10, 16) + 1.0
        mask = torch.ones(4, 10, dtype=torch.bool)
        mask[2, :] = False  # sample 2 is fully masked
        mask[:, 7:] = False  # all samples: valid up to index 6
        out = self.module(x, mask)
        self.assertEqual(out[2].sum().item(), 0.0)  # sample 2 → zeros
        self.assertTrue(torch.allclose(out[0], x[0, 6, :]))


class TestAttentionPooling(unittest.TestCase):
    def setUp(self):
        self.pool = AttentionPooling(d_model=16)

    def test_without_mask(self):
        x = torch.rand(4, 10, 16)
        out = self.pool(x)
        self.assertEqual(out.shape, (4, 16))
        self.assertFalse(torch.isnan(out).any())

    def test_with_partial_mask(self):
        x = torch.rand(4, 10, 16)
        mask = torch.ones(4, 10, dtype=torch.bool)
        mask[:, 8:] = False
        out = self.pool(x, mask)
        self.assertEqual(out.shape, (4, 16))
        self.assertFalse(torch.isnan(out).any())

    def test_fully_masked_returns_zeros_not_nan(self):
        # regression: was producing NaN for all-masked samples
        x = torch.rand(4, 10, 16) + 1.0
        mask = torch.zeros(4, 10, dtype=torch.bool)
        out = self.pool(x, mask)
        self.assertEqual(out.shape, (4, 16))
        self.assertFalse(torch.isnan(out).any())
        self.assertTrue(torch.allclose(out, torch.zeros(4, 16)))

    def test_mixed_mask_some_fully_masked(self):
        x = torch.rand(4, 10, 16) + 1.0
        mask = torch.ones(4, 10, dtype=torch.bool)
        mask[1, :] = False  # sample 1 fully masked
        mask[3, :] = False  # sample 3 fully masked
        out = self.pool(x, mask)
        self.assertFalse(torch.isnan(out).any())
        self.assertTrue(torch.allclose(out[1], torch.zeros(16)))
        self.assertTrue(torch.allclose(out[3], torch.zeros(16)))
        # Valid samples should have non-zero output
        self.assertFalse(torch.allclose(out[0], torch.zeros(16)))


class TestTAM(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.batch_size = 2
        cls.d_model = 16
        cls.aligned_time_dim = 20

    def _make_tam(self):
        return TAM(
            input_dim=2 * self.d_model,
            output_dim=self.d_model,
            aligner="aap",
            time_dim=self.aligned_time_dim,
            num_layers=1,
        )

    def test_output_shape(self):
        tam = self._make_tam()
        x1 = torch.rand(self.batch_size, 15, self.d_model)
        x2 = torch.rand(self.batch_size, 30, self.d_model)
        out, mask = tam([x1, x2], [None, None])
        self.assertEqual(out.shape, (self.batch_size, self.aligned_time_dim, self.d_model))
        self.assertEqual(mask.shape, (self.batch_size, self.aligned_time_dim))

    def test_with_masks(self):
        tam = self._make_tam()
        x1 = torch.rand(self.batch_size, 15, self.d_model)
        x2 = torch.rand(self.batch_size, 30, self.d_model)
        mask1 = torch.ones(self.batch_size, 15, dtype=torch.bool)
        mask2 = torch.ones(self.batch_size, 30, dtype=torch.bool)
        out, mask = tam([x1, x2], [mask1, mask2])
        self.assertEqual(out.shape, (self.batch_size, self.aligned_time_dim, self.d_model))
        self.assertFalse(torch.isnan(out).any())


class TestTRM(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.batch_size = 4
        cls.time_dim = 20
        cls.d_model = 16
        cls.x = torch.rand(cls.batch_size, cls.time_dim, cls.d_model)
        cls.mask = torch.ones(cls.batch_size, cls.time_dim, dtype=torch.bool)

    def test_attentionpool(self):
        trm = TRM(d_model=self.d_model, reducer="attentionpool")
        out = trm(self.x, self.mask)
        self.assertEqual(out.shape, (self.batch_size, self.d_model))

    def test_gap(self):
        trm = TRM(d_model=self.d_model, reducer="gap")
        out = trm(self.x, self.mask)
        self.assertEqual(out.shape, (self.batch_size, self.d_model))

    def test_gmp(self):
        trm = TRM(d_model=self.d_model, reducer="gmp")
        out = trm(self.x, self.mask)
        self.assertEqual(out.shape, (self.batch_size, self.d_model))

    def test_last(self):
        trm = TRM(d_model=self.d_model, reducer="last")
        out = trm(self.x, self.mask)
        self.assertEqual(out.shape, (self.batch_size, self.d_model))

    def test_invalid_reducer_raises(self):
        with self.assertRaises(Exception):
            TRM(d_model=self.d_model, reducer="unknown")

    def test_tam_invalid_aligner_raises(self):
        with self.assertRaises(ValueError):
            TAM(input_dim=16, output_dim=8, aligner="unknown", time_dim=20, num_layers=1)

    def test_apply_to_list(self):
        trm = TRM(d_model=self.d_model, reducer="gap")
        x_list = [self.x, self.x]
        mask_list = [self.mask, self.mask]
        out_list = trm.apply_to_list(x_list, mask_list)
        self.assertEqual(len(out_list), 2)
        for out in out_list:
            self.assertEqual(out.shape, (self.batch_size, self.d_model))

    def test_attentionpool_with_larger_dim(self):
        # Regression Bug 1: AttentionPooling must be built with the actual input d_model.
        # When unimodal_sat=True and 2 modalities (no MMS), branch_feat_dim = 2*d_model.
        # The caller (LinMulT) computes this and passes it directly to TRM.
        branch_feat_dim = 2 * self.d_model
        trm = TRM(d_model=branch_feat_dim, reducer="attentionpool")
        x = torch.rand(self.batch_size, self.time_dim, branch_feat_dim)
        out = trm(x, self.mask)
        self.assertEqual(out.shape, (self.batch_size, branch_feat_dim))
        self.assertFalse(torch.isnan(out).any())
