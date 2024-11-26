import unittest
import torch
from linmult import LinT, LinMulT
from linmult.models.transformer import TransformerEncoder


class TestArchitecture(unittest.TestCase):


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
        cls.x_1 = torch.rand((cls.batch_size, cls.time_dim_1, cls.feature_dim_1)) # Shape: (B, T_1, F_1)
        cls.x_2 = torch.rand((cls.batch_size, cls.time_dim_2, cls.feature_dim_2)) # Shape: (B, T_2, F_2)
        cls.x_3 = torch.rand((cls.batch_size, cls.time_dim_3, cls.feature_dim_3)) # Shape: (B, T_3, F_3)


    def test_same_time_dim(self):
        model = LinMulT(
            {
                'input_modality_channels': [self.feature_dim_2, self.feature_dim_3],
                'output_dim': [self.output_dim_1]
            }
        )
        output_seq = model([self.x_2, self.x_3])
        output_cls = LinMulT.apply_logit_aggregation(output_seq, 'meanpooling')
        self.assertEqual(output_cls[0].shape, (self.batch_size, self.output_dim_1))
        self.assertEqual(output_seq[0].shape, (self.batch_size, self.time_dim_2, self.output_dim_1))


    def test_attention_linear(self):
        model = LinMulT(
            {
                'input_modality_channels': [self.feature_dim_2, self.feature_dim_3],
                'output_dim': [self.output_dim_1],
                'attention_type': 'linear'
            }
        )
        self.assertEqual(model.branch_crossmodal_transformers[0][0].layers[0].attention_type, 'linear')


    def test_attention_bigbird(self):
        model = LinMulT(
            {
                'input_modality_channels': [self.feature_dim_2, self.feature_dim_3],
                'output_dim': [self.output_dim_1],
                'attention_type': 'bigbird'
            }
        )
        self.assertEqual(model.branch_crossmodal_transformers[0][0].layers[0].attention_type, 'bigbird')


    def test_attention_mha(self):
        model = LinMulT(
            {
                'input_modality_channels': [self.feature_dim_2, self.feature_dim_3],
                'output_dim': [self.output_dim_1],
                'attention_type': 'mha'
            }
        )
        self.assertEqual(model.branch_crossmodal_transformers[0][0].layers[0].attention_type, 'mha')


    def test_module_time_reduce_gap(self):
        model = LinMulT(
            {
                'input_modality_channels': [self.feature_dim_1, self.feature_dim_2],
                'output_dim': [self.output_dim_1],
                'module_time_reduce': 'gap'
            }
        )
        output = model([self.x_1, self.x_2])
        self.assertEqual(output[0].shape, (self.batch_size, self.output_dim_1))


    def test_module_time_reduce_gmp(self):
        model = LinMulT(
            {
                'input_modality_channels': [self.feature_dim_1, self.feature_dim_2],
                'output_dim': [self.output_dim_1],
                'module_time_reduce': 'gmp'
            }
        )
        output = model([self.x_1, self.x_2])
        self.assertEqual(output[0].shape, (self.batch_size, self.output_dim_1))


    def test_module_time_reduce_attentionpool(self):
        model = LinMulT(
            {
                'input_modality_channels': [self.feature_dim_1, self.feature_dim_2],
                'output_dim': [self.output_dim_1],
                'module_time_reduce': 'attentionpool'
            }
        )
        output = model([self.x_1, self.x_2])
        self.assertEqual(output[0].shape, (self.batch_size, self.output_dim_1))


    def test_multimodal_signal_type_padding(self):
        model = LinMulT(
            {
                'input_modality_channels': [self.feature_dim_1, self.feature_dim_2],
                'output_dim': [self.output_dim_1],
                'module_time_reduce': 'gap',
                'multimodal_signal_type': 'padding',
                'multimodal_signal_time_dim': self.time_dim_1
            }
        )
        output = model([self.x_1, self.x_2])
        self.assertEqual(output[0].shape, (self.batch_size, self.output_dim_1))


    def test_multimodal_signal_type_truncating(self):
        model = LinMulT(
            {
                'input_modality_channels': [self.feature_dim_1, self.feature_dim_2],
                'output_dim': [self.output_dim_1],
                'module_time_reduce': 'gap',
                'multimodal_signal_type': 'padding',
                'multimodal_signal_time_dim': self.time_dim_2
            }
        )
        output = model([self.x_1, self.x_2])
        self.assertEqual(output[0].shape, (self.batch_size, self.output_dim_1))


    def test_multimodal_signal_type_gap(self):
        model = LinMulT(
            {
                'input_modality_channels': [self.feature_dim_1, self.feature_dim_2],
                'output_dim': [self.output_dim_1],
                'module_time_reduce': 'gap',
                'multimodal_signal_type': 'gap',
                'multimodal_signal_time_dim': self.time_dim_2
            }
        )
        output = model([self.x_1, self.x_2])
        self.assertEqual(output[0].shape, (self.batch_size, self.output_dim_1))


    def test_multimodal_signal_type_gmp(self):
        model = LinMulT(
            {
                'input_modality_channels': [self.feature_dim_1, self.feature_dim_2],
                'output_dim': [self.output_dim_1],
                'module_time_reduce': 'gap',
                'multimodal_signal_type': 'gmp',
                'multimodal_signal_time_dim': self.time_dim_2
            }
        )
        output = model([self.x_1, self.x_2])
        self.assertEqual(output[0].shape, (self.batch_size, self.output_dim_1))


    def test_lint(self):
        model = LinT(
            {
                'input_modality_channels': self.feature_dim_1,
                'output_dim': [self.output_dim_1],
                'module_time_reduce': 'attentionpool',
            }
        )
        output = model(self.x_1)
        self.assertEqual(output[0].shape, (self.batch_size, self.output_dim_1))


if __name__ == '__main__':
    unittest.main()