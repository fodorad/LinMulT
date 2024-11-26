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
        cls.mask_1 = (torch.arange(cls.x_1.size(1)).unsqueeze(0) < cls.x_1.size(1)-10).expand(cls.batch_size, -1).bool() # Shape: (B, T_1)
        cls.mask_2 = (torch.arange(cls.x_2.size(1)).unsqueeze(0) < cls.x_2.size(1)-10).expand(cls.batch_size, -1).bool() # Shape: (B, T_2)
        cls.mask_3 = (torch.arange(cls.x_3.size(1)).unsqueeze(0) < cls.x_3.size(1)-10).expand(cls.batch_size, -1).bool() # Shape: (B, T_3)
        cls.mask_3f = torch.zeros(size=cls.x_3.size()[:2], dtype=bool) # Shape: (B, T_3)


    def test_self_attention_transformer_encoder(self):
        encoder = TransformerEncoder({'d_model': self.d_model})
        output = encoder(self.d_1)
        self.assertEqual(output.shape, (self.batch_size, self.time_dim_1, self.d_model))


    def test_cross_attention_transformer_encoder(self):
        encoder = TransformerEncoder({'d_model': self.d_model})
        output = encoder(self.d_1, self.d_2, self.d_2)
        self.assertEqual(output.shape, (self.batch_size, self.time_dim_1, self.d_model))


    def test_self_attention_transformer_encoder_with_mask(self):
        encoder = TransformerEncoder({'d_model': self.d_model})
        output = encoder(self.d_1, query_mask=self.mask_1)
        self.assertEqual(output.shape, (self.batch_size, self.time_dim_1, self.d_model))
        
        masked_positions = ~self.mask_1.unsqueeze(-1).expand_as(output) # Expand mask to match (B, T_1, d_model)
        self.assertTrue(torch.all(output[masked_positions].abs() == 0.), "Masked positions are not zero.")


    def test_cross_attention_transformer_encoder_with_mask(self):
        encoder = TransformerEncoder({'d_model': self.d_model})
        output = encoder(self.d_1, self.d_2, self.d_2, query_mask=self.mask_1, key_mask=self.mask_2)
        self.assertEqual(output.shape, (self.batch_size, self.time_dim_1, self.d_model))

        masked_positions = ~self.mask_1.unsqueeze(-1).expand_as(output) # Expand mask to match (B, T_1, d_model)
        self.assertTrue(torch.all(output[masked_positions].abs() == 0.), "Masked positions are not zero.")


    def test_self_attention_transformer_encoder_with_full_query_mask(self):
        encoder = TransformerEncoder({'d_model': self.d_model})
        output = encoder(self.d_3, query_mask=self.mask_3f)
        self.assertEqual(output.shape, (self.batch_size, self.time_dim_3, self.d_model))
        self.assertTrue(torch.all(output == 0.), "Masked positions are not zero.")


    def test_cross_attention_transformer_encoder_with_full_query_mask(self):
        encoder = TransformerEncoder({'d_model': self.d_model})
        output = encoder(self.d_3, self.d_2, self.d_2, query_mask=self.mask_3f, key_mask=self.mask_2)
        self.assertEqual(output.shape, (self.batch_size, self.time_dim_3, self.d_model))
        self.assertTrue(torch.all(output == 0.), "Masked positions are not zero.")


    def test_cross_attention_transformer_encoder_with_full_key_mask(self):
        encoder = TransformerEncoder({'d_model': self.d_model})
        output = encoder(self.d_2, self.d_3, self.d_3, query_mask=self.mask_2, key_mask=self.mask_3f)
        self.assertEqual(output.shape, (self.batch_size, self.time_dim_2, self.d_model))
        self.assertTrue(torch.all(output == 0.), "Masked positions are not zero.")


    def test_2i(self):
        model = LinMulT(
            {
                'input_modality_channels': [self.feature_dim_1, self.feature_dim_2],
                'output_dim': [self.output_dim_1],
                'module_time_reduce': 'gmp'
            }
        )
        output = model([self.x_1, self.x_2])
        self.assertEqual(output[0].shape, (self.batch_size, self.output_dim_1))


    def test_2o(self):
        model = LinMulT(
            {
                'input_modality_channels': [self.feature_dim_1, self.feature_dim_2],
                'output_dim': [self.output_dim_1, self.output_dim_2],
                'module_time_reduce': 'gmp'
            }
        )
        output = model([self.x_1, self.x_2])
        self.assertEqual(output[0].shape, (self.batch_size, self.output_dim_1))
        self.assertEqual(output[1].shape, (self.batch_size, self.output_dim_2))


    def test_3i(self):
        model = LinMulT(
            {
                'input_modality_channels': [self.feature_dim_1, self.feature_dim_2, self.feature_dim_3],
                'output_dim': [self.output_dim_1],
                'module_time_reduce': 'gmp'
            }
        )
        output = model([self.x_1, self.x_2, self.x_3])
        self.assertEqual(output[0].shape, (self.batch_size, self.output_dim_1))


    def test_2i_with_masks(self):
        model = LinMulT(
            {
                'input_modality_channels': [self.feature_dim_1, self.feature_dim_2],
                'output_dim': [self.output_dim_1],
                'module_time_reduce': 'gap'
            }
        )
        output = model([self.x_1, self.x_2], masks=[self.mask_1, self.mask_2])
        self.assertEqual(output[0].shape, (self.batch_size, self.output_dim_1))


    def test_3i_with_masks(self):
        model = LinMulT(
            {
                'input_modality_channels': [self.feature_dim_1, self.feature_dim_2, self.feature_dim_3],
                'output_dim': [self.output_dim_1],
                'module_time_reduce': 'gap'
            }
        )
        output = model([self.x_1, self.x_2, self.x_3], masks=[self.mask_1, self.mask_2, self.mask_3])
        self.assertEqual(output[0].shape, (self.batch_size, self.output_dim_1))


    def test_modality_masking(self):
        model = LinMulT(
            {
                'input_modality_channels': [self.feature_dim_1, self.feature_dim_2, self.feature_dim_3],
                'output_dim': [self.output_dim_1],
                'module_time_reduce': 'gap'
            }
        )
        output = model([self.x_1, self.x_2, self.x_3], masks=[self.mask_1, self.mask_2, self.mask_3f])
        self.assertEqual(output[0].shape, (self.batch_size, self.output_dim_1))


    def test_modality_sequence_masking(self):
        model = LinMulT(
            {
                'input_modality_channels': [self.feature_dim_2, self.feature_dim_2, self.feature_dim_3],
                'output_dim': [self.output_dim_1],
            }
        )
        output_seq = model([self.x_2, self.x_2, self.x_3], masks=[self.mask_2, self.mask_2, self.mask_3f])
        output_cls = LinMulT.apply_logit_aggregation(output_seq, 'meanpooling')
        self.assertEqual(output_seq[0].shape, (self.batch_size, self.time_dim_2, self.output_dim_1))
        self.assertEqual(output_cls[0].shape, (self.batch_size, self.output_dim_1))


    def test_lint(self):
        model = LinT(
            {
                'input_modality_channels': self.feature_dim_1,
                'output_dim': [self.output_dim_1]
            }
        )
        output = model(self.x_1)
        self.assertEqual(output[0].shape, (self.batch_size, self.time_dim_1, self.output_dim_1))


if __name__ == '__main__':
    unittest.main()