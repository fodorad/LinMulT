import unittest

import torch
from torch import nn

from linmult.core.attention import AttentionConfig
from linmult.core.branch import CrossModalBranch, CrossModalModule, MultimodalSignal
from linmult.core.temporal import TAM
from linmult.core.transformer import TransformerEncoder


def _make_encoder(d_model: int, is_cross_modal: bool = False) -> TransformerEncoder:
    return TransformerEncoder(
        d_model=d_model,
        num_heads=4,
        num_layers=1,
        attention_config=AttentionConfig(type="linear"),
        is_cross_modal=is_cross_modal,
    )


class TestCrossModalBranch(unittest.TestCase):
    def setUp(self):
        self.B, self.T, self.d = 2, 10, 8
        self.x_q = torch.rand(self.B, self.T, self.d)
        self.x_s1 = torch.rand(self.B, 12, self.d)
        self.x_s2 = torch.rand(self.B, 15, self.d)

    def _make_branch(self, n_sources: int, with_unimodal: bool = False) -> CrossModalBranch:
        cross = nn.ModuleList(
            [_make_encoder(self.d, is_cross_modal=True) for _ in range(n_sources)]
        )
        sat = _make_encoder(n_sources * self.d)
        uni = _make_encoder(self.d) if with_unimodal else None
        return CrossModalBranch(cross_transformers=cross, branch_sat=sat, unimodal_sat=uni)

    def test_two_sources_no_mask(self):
        branch = self._make_branch(n_sources=2)
        out = branch(self.x_q, [self.x_s1, self.x_s2])
        # branch_dim = 2 * d
        self.assertEqual(out.shape, (self.B, self.T, 2 * self.d))

    def test_two_sources_with_mask(self):
        branch = self._make_branch(n_sources=2)
        mask_q = torch.ones(self.B, self.T, dtype=torch.bool)
        mask_q[:, 8:] = False
        mask_s1 = torch.ones(self.B, 12, dtype=torch.bool)
        mask_s2 = torch.ones(self.B, 15, dtype=torch.bool)
        out = branch(
            self.x_q, [self.x_s1, self.x_s2], mask_query=mask_q, mask_sources=[mask_s1, mask_s2]
        )
        self.assertEqual(out.shape, (self.B, self.T, 2 * self.d))
        self.assertFalse(torch.isnan(out).any())

    def test_one_source_no_mask(self):
        branch = self._make_branch(n_sources=1)
        out = branch(self.x_q, [self.x_s1])
        self.assertEqual(out.shape, (self.B, self.T, self.d))

    def test_unimodal_sat_appended(self):
        branch = self._make_branch(n_sources=2, with_unimodal=True)
        out = branch(self.x_q, [self.x_s1, self.x_s2])
        # full_branch_dim = 2*d + d
        self.assertEqual(out.shape, (self.B, self.T, 3 * self.d))

    def test_none_mask_sources_defaults(self):
        branch = self._make_branch(n_sources=2)
        # mask_sources=None should work (defaults to all None)
        out = branch(self.x_q, [self.x_s1, self.x_s2], mask_sources=None)
        self.assertEqual(out.shape, (self.B, self.T, 2 * self.d))

    def test_no_nan_output(self):
        branch = self._make_branch(n_sources=2)
        out = branch(self.x_q, [self.x_s1, self.x_s2])
        self.assertFalse(torch.isnan(out).any())


class TestMultimodalSignal(unittest.TestCase):
    def setUp(self):
        self.B, self.T, self.d = 2, 20, 8
        self.x_list = [torch.rand(self.B, self.T, self.d) for _ in range(3)]
        self.mask_list = [None, None, None]

    def _make_mms(self) -> MultimodalSignal:
        tam = TAM(
            input_dim=3 * self.d,
            output_dim=self.d,
            aligner="aap",
            time_dim=self.T,
            num_layers=1,
            num_heads=4,
            attention_config=AttentionConfig(type="linear"),
        )
        return MultimodalSignal(tam=tam)

    def test_extends_x_list(self):
        mms = self._make_mms()
        new_x, new_mask = mms(self.x_list, self.mask_list)
        self.assertEqual(len(new_x), 4)
        self.assertEqual(len(new_mask), 4)

    def test_mms_output_shape(self):
        mms = self._make_mms()
        new_x, new_mask = mms(self.x_list, self.mask_list)
        # MMS output: (B, time_dim, d_model)
        self.assertEqual(new_x[-1].shape, (self.B, self.T, self.d))

    def test_original_inputs_unchanged(self):
        mms = self._make_mms()
        orig = [x.clone() for x in self.x_list]
        new_x, _ = mms(self.x_list, self.mask_list)
        for i, (orig_x, new_xi) in enumerate(zip(orig, new_x[:3])):
            self.assertTrue(torch.equal(orig_x, new_xi), f"x_list[{i}] was mutated")

    def test_no_nan_output(self):
        mms = self._make_mms()
        new_x, _ = mms(self.x_list, self.mask_list)
        for x in new_x:
            self.assertFalse(torch.isnan(x).any())


class TestCrossModalModule(unittest.TestCase):
    def setUp(self):
        self.B, self.T, self.d = 2, 20, 8

    def _make_stage(
        self,
        n_modalities: int = 2,
        add_mms: bool = False,
        add_unimodal_sat: bool = False,
    ) -> CrossModalModule:
        return CrossModalModule(
            num_modalities=n_modalities,
            d_model=self.d,
            num_heads=4,
            branch_cmt_num_layers=1,
            branch_sat_num_layers=1,
            attention_config=AttentionConfig(type="linear"),
            add_mms=add_mms,
            mms_num_layers=1,
            tam_aligner="aap",
            tam_time_dim=self.T,
            add_unimodal_sat=add_unimodal_sat,
            unimodal_sat_num_layers=1,
        )

    def test_two_modalities_basic(self):
        stage = self._make_stage(n_modalities=2)
        x_list = [torch.rand(self.B, self.T, self.d) for _ in range(2)]
        mask_list = [None, None]
        out = stage(x_list, mask_list)
        self.assertEqual(len(out), 2)
        # branch_dim = (n_modalities - 1) * d_model = 1 * 8 = 8
        self.assertEqual(out[0].shape, (self.B, self.T, self.d))
        self.assertEqual(out[1].shape, (self.B, self.T, self.d))

    def test_three_modalities(self):
        stage = self._make_stage(n_modalities=3)
        x_list = [torch.rand(self.B, self.T, self.d) for _ in range(3)]
        mask_list = [None, None, None]
        out = stage(x_list, mask_list)
        self.assertEqual(len(out), 3)
        # branch_dim = (3 - 1) * 8 = 16
        expected_dim = 2 * self.d
        for o in out:
            self.assertEqual(o.shape, (self.B, self.T, expected_dim))

    def test_with_mms(self):
        stage = self._make_stage(n_modalities=2, add_mms=True)
        x_list = [torch.rand(self.B, self.T, self.d) for _ in range(2)]
        mask_list = [None, None]
        out = stage(x_list, mask_list)
        self.assertEqual(len(out), 2)
        # with MMS: n_sources = n_modalities = 2, branch_dim = 2 * d = 16
        expected_dim = 2 * self.d
        self.assertEqual(out[0].shape, (self.B, self.T, expected_dim))

    def test_with_unimodal_sat(self):
        stage = self._make_stage(n_modalities=2, add_unimodal_sat=True)
        x_list = [torch.rand(self.B, self.T, self.d) for _ in range(2)]
        mask_list = [None, None]
        out = stage(x_list, mask_list)
        # full_branch_dim = branch_dim + d = 8 + 8 = 16
        expected_dim = 2 * self.d
        self.assertEqual(out[0].shape, (self.B, self.T, expected_dim))

    def test_with_mms_and_unimodal_sat(self):
        stage = self._make_stage(n_modalities=2, add_mms=True, add_unimodal_sat=True)
        x_list = [torch.rand(self.B, self.T, self.d) for _ in range(2)]
        mask_list = [None, None]
        out = stage(x_list, mask_list)
        # branch_dim = 2*d, full = 2*d + d = 3*d
        expected_dim = 3 * self.d
        self.assertEqual(out[0].shape, (self.B, self.T, expected_dim))

    def test_output_dim_property(self):
        stage_basic = self._make_stage(n_modalities=2)
        self.assertEqual(stage_basic.output_dim, self.d)  # (2-1)*8

        stage_mms = self._make_stage(n_modalities=2, add_mms=True)
        self.assertEqual(stage_mms.output_dim, 2 * self.d)  # 2*8

        stage_uni = self._make_stage(n_modalities=2, add_unimodal_sat=True)
        self.assertEqual(stage_uni.output_dim, 2 * self.d)  # (2-1)*8 + 8

        stage_both = self._make_stage(n_modalities=3, add_mms=True, add_unimodal_sat=True)
        self.assertEqual(stage_both.output_dim, 4 * self.d)  # 3*8 + 8

    def test_with_masks(self):
        stage = self._make_stage(n_modalities=2)
        x_list = [torch.rand(self.B, self.T, self.d) for _ in range(2)]
        mask = torch.ones(self.B, self.T, dtype=torch.bool)
        mask[:, 15:] = False
        mask_list = [mask, None]
        out = stage(x_list, mask_list)
        self.assertEqual(len(out), 2)
        self.assertFalse(torch.isnan(out[0]).any())

    def test_no_nan_output(self):
        stage = self._make_stage(n_modalities=2)
        x_list = [torch.rand(self.B, self.T, self.d) for _ in range(2)]
        mask_list = [None, None]
        out = stage(x_list, mask_list)
        for o in out:
            self.assertFalse(torch.isnan(o).any())

    def test_branches_attribute(self):
        stage = self._make_stage(n_modalities=3)
        self.assertEqual(len(stage.branches), 3)
        self.assertIsInstance(stage.branches[0], CrossModalBranch)

    def test_mms_attribute(self):
        stage_no_mms = self._make_stage(n_modalities=2, add_mms=False)
        self.assertIsNone(stage_no_mms.mms)

        stage_with_mms = self._make_stage(n_modalities=2, add_mms=True)
        self.assertIsInstance(stage_with_mms.mms, MultimodalSignal)


if __name__ == "__main__":
    unittest.main()
