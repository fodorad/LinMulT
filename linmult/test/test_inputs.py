import unittest
import torch
from linmult import LinT, LinMulT
from linmult.models.transformer import TransformerEncoder


class TestInputs(unittest.TestCase):


    @classmethod
    def setUpClass(cls):
        cls.batch_size = 8
        cls.time_dim_1 = 1500
        cls.time_dim_2 = 450
        cls.time_dim_3 = 450
        cls.feature_dim_1 = 25
        cls.feature_dim_2 = 35
        cls.feature_dim_3 = 256
        cls.d_model = 40
        cls.output_dim_1 = 5
        cls.output_dim_2 = 1
        cls.x_1 = torch.rand((cls.batch_size, cls.time_dim_1, cls.feature_dim_1)) # Shape: (B, T_1, F_1)
        cls.x_2 = torch.rand((cls.batch_size, cls.time_dim_2, cls.feature_dim_2)) # Shape: (B, T_2, F_2)
        cls.x_3 = torch.rand((cls.batch_size, cls.time_dim_3, cls.feature_dim_3)) # Shape: (B, T_3, F_3)
        cls.d_1 = torch.rand((cls.batch_size, cls.time_dim_1, cls.d_model)) # Shape: (B, T_1, d_model)
        cls.d_2 = torch.rand((cls.batch_size, cls.time_dim_2, cls.d_model)) # Shape: (B, T_2, d_model)
        cls.d_3 = torch.rand((cls.batch_size, cls.time_dim_3, cls.d_model)) # Shape: (B, T_2, d_model)


    def test_self_attention_transformer_encoder(self):
        encoder = TransformerEncoder({'d_model': self.d_model})
        output = encoder(self.d_1)
        self.assertEqual(output.shape, (self.batch_size, self.time_dim_1, self.d_model))


    def test_cross_attention_transformer_encoder(self):
        encoder = TransformerEncoder({'d_model': self.d_model})
        output = encoder(self.d_1, self.d_2, self.d_2)
        self.assertEqual(output.shape, (self.batch_size, self.time_dim_1, self.d_model))


    def test_2i(self):
        model = LinMulT(
            {
                'input_feature_dim': [self.feature_dim_1, self.feature_dim_2],
                'output_dim': [self.output_dim_1],
                'time_dim_reducer': 'gmp'
            }
        )
        output = model([self.x_1, self.x_2])
        self.assertEqual(output[0].shape, (self.batch_size, self.output_dim_1))


    def test_2o(self):
        model = LinMulT(
            {
                'input_feature_dim': [self.feature_dim_1, self.feature_dim_2],
                'output_dim': [self.output_dim_1, self.output_dim_2],
                'time_dim_reducer': 'gap'
            }
        )
        output = model([self.x_1, self.x_2])
        self.assertEqual(output[0].shape, (self.batch_size, self.output_dim_1))
        self.assertEqual(output[1].shape, (self.batch_size, self.output_dim_2))


    def test_3i(self):
        model = LinMulT(
            {
                'input_feature_dim': [self.feature_dim_1, self.feature_dim_2, self.feature_dim_3],
                'output_dim': [self.output_dim_1],
                'time_dim_reducer': 'gap'
            }
        )
        output = model([self.x_1, self.x_2, self.x_3])
        self.assertEqual(output[0].shape, (self.batch_size, self.output_dim_1))


    def test_lint(self):
        model = LinT(
            {
                'input_feature_dim': self.feature_dim_1,
                'output_dim': [self.output_dim_1]
            }
        )
        output = model(self.x_1)
        self.assertEqual(output[0].shape, (self.batch_size, self.time_dim_1, self.output_dim_1))


if __name__ == '__main__':
    unittest.main()