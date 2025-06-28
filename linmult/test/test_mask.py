import unittest
import torch
from linmult import LinMulT
from linmult.models.transformer import TransformerEncoder
from linmult.models.attention import AttentionLayer, SoftmaxAttention, LinearAttention, BigBirdAttention
from linmult.models.utils import apply_logit_aggregation


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
        cls.n_heads = 8
        cls.block_size = 64
        cls.num_global_tokens = 16
        cls.num_random_tokens = 16
        cls.output_dim_1 = 5
        cls.output_dim_2 = 1
        cls.x_1 = torch.rand((cls.batch_size, cls.time_dim_1, cls.feature_dim_1))  # Shape: (B, T_1, F_1)
        cls.x_2 = torch.rand((cls.batch_size, cls.time_dim_2, cls.feature_dim_2))  # Shape: (B, T_2, F_2)
        cls.x_3 = torch.rand((cls.batch_size, cls.time_dim_3, cls.feature_dim_3))  # Shape: (B, T_3, F_3)
        cls.d_1 = torch.rand((cls.batch_size, cls.time_dim_1, cls.d_model))  # Shape: (B, T_1, d_model)
        cls.d_2 = torch.rand((cls.batch_size, cls.time_dim_2, cls.d_model))  # Shape: (B, T_2, d_model)
        cls.d_3 = torch.rand((cls.batch_size, cls.time_dim_3, cls.d_model))  # Shape: (B, T_3, d_model)
        cls.mask_1 = (torch.arange(cls.x_1.size(1)).unsqueeze(0) < cls.x_1.size(1) - 10).expand(cls.batch_size, -1).bool()  # Shape: (B, T_1)
        cls.mask_2 = (torch.arange(cls.x_2.size(1)).unsqueeze(0) < cls.x_2.size(1) - 10).expand(cls.batch_size, -1).bool()  # Shape: (B, T_2)
        cls.mask_3 = (torch.arange(cls.x_3.size(1)).unsqueeze(0) < cls.x_3.size(1) - 10).expand(cls.batch_size, -1).bool()  # Shape: (B, T_3)
        cls.mask_3f = torch.zeros(size=cls.x_3.size()[:2], dtype=bool)  # Shape: (B, T_3)


    def test_self_attention_transformer_encoder_with_mask(self):
        encoder = TransformerEncoder({'d_model': self.d_model})
        output = encoder(self.d_1, query_mask=self.mask_1)
        self.assertEqual(output.shape, (self.batch_size, self.time_dim_1, self.d_model))


    def test_cross_attention_transformer_encoder_with_mask(self):
        encoder = TransformerEncoder({'d_model': self.d_model})
        output = encoder(self.d_1, self.d_2, self.d_2, query_mask=self.mask_1, key_mask=self.mask_2)
        self.assertEqual(output.shape, (self.batch_size, self.time_dim_1, self.d_model))


    def test_self_attention_transformer_encoder_with_full_query_mask(self):
        encoder = TransformerEncoder({'d_model': self.d_model})
        output = encoder(self.d_3, query_mask=self.mask_3f)
        self.assertEqual(output.shape, (self.batch_size, self.time_dim_3, self.d_model))


    def test_cross_attention_transformer_encoder_with_full_query_mask(self):
        encoder = TransformerEncoder({'d_model': self.d_model})
        output = encoder(self.d_3, self.d_2, self.d_2, query_mask=self.mask_3f, key_mask=self.mask_2)
        self.assertEqual(output.shape, (self.batch_size, self.time_dim_3, self.d_model))


    def test_cross_attention_transformer_encoder_with_full_key_mask(self):
        encoder = TransformerEncoder({'d_model': self.d_model})
        output = encoder(self.d_2, self.d_3, self.d_3, query_mask=self.mask_2, key_mask=self.mask_3f)
        self.assertEqual(output.shape, (self.batch_size, self.time_dim_2, self.d_model))


    def test_self_attention_softmax_with_mask(self):
        softmax_attention = AttentionLayer(
            SoftmaxAttention(self.d_model, self.n_heads),
            d_model=self.d_model,
            n_heads=self.n_heads
        )
        output, _ = softmax_attention(self.d_1, self.d_1, self.d_1, query_mask=self.mask_1, key_mask=self.mask_1)
        self.assertEqual(output.shape, self.d_1.shape, "SoftmaxAttention output shape mismatch")


    def test_cross_attention_softmax_with_mask(self):
        softmax_attention = AttentionLayer(
            SoftmaxAttention(self.d_model, self.n_heads),
            d_model=self.d_model,
            n_heads=self.n_heads
        )
        output, _ = softmax_attention(self.d_1, self.d_2, self.d_2, query_mask=self.mask_1, key_mask=self.mask_2)
        self.assertEqual(output.shape, self.d_1.shape, "SoftmaxAttention output shape mismatch")


    def test_self_attention_linear_with_mask(self):
        linear_attention = AttentionLayer(
            LinearAttention(self.d_model, self.n_heads),
            d_model=self.d_model,
            n_heads=self.n_heads
        )
        output, _ = linear_attention(self.d_1, self.d_1, self.d_1, query_mask=self.mask_1, key_mask=self.mask_1)
        self.assertEqual(output.shape, self.d_1.shape, "LinearAttention output shape mismatch")


    def test_cross_attention_linear_with_mask(self):
        linear_attention = AttentionLayer(
            LinearAttention(self.d_model, self.n_heads),
            d_model=self.d_model,
            n_heads=self.n_heads
        )
        output, _ = linear_attention(self.d_1, self.d_2, self.d_2, query_mask=self.mask_1, key_mask=self.mask_2)
        self.assertEqual(output.shape, self.d_1.shape, "LinearAttention output shape mismatch")


    def test_self_attention_bigbird_with_mask(self):
        bigbird_attention = AttentionLayer(
            BigBirdAttention(
                self.d_model, self.n_heads, self.block_size, self.num_global_tokens, self.num_random_tokens
            ),
            d_model=self.d_model,
            n_heads=self.n_heads
        )
        output, _ = bigbird_attention(self.d_1, self.d_1, self.d_1, query_mask=self.mask_1, key_mask=self.mask_1)
        self.assertEqual(output.shape, self.d_1.shape, "BigBirdAttention output shape mismatch")


    def test_cross_attention_bigbird_with_mask(self):
        bigbird_attention = AttentionLayer(
            BigBirdAttention(
                self.d_model, self.n_heads, self.block_size, self.num_global_tokens, self.num_random_tokens
            ),
            d_model=self.d_model,
            n_heads=self.n_heads
        )
        output, _ = bigbird_attention(self.d_1, self.d_2, self.d_2, query_mask=self.mask_1, key_mask=self.mask_2)
        self.assertEqual(output.shape, self.d_1.shape, "BigBirdAttention output shape mismatch")

    
    def test_2i_with_masks(self):
        model = LinMulT(
            {
                'input_feature_dim': [self.feature_dim_1, self.feature_dim_2],
                'heads': [{'type': 'simple', 'output_dim': self.output_dim_1}],
                'time_dim_reducer': 'gap'
            }
        )
        output = list(model([self.x_1, self.x_2], masks=[self.mask_1, self.mask_2]).values())
        self.assertEqual(output[0].shape, (self.batch_size, self.output_dim_1))


    def test_3i_with_masks(self):
        model = LinMulT(
            {
                'input_feature_dim': [self.feature_dim_1, self.feature_dim_2, self.feature_dim_3],
                'heads': [{'type': 'simple', 'output_dim': self.output_dim_1}],
                'time_dim_reducer': 'gap'
            }
        )
        output = list(model([self.x_1, self.x_2, self.x_3], masks=[self.mask_1, self.mask_2, self.mask_3]).values())
        self.assertEqual(output[0].shape, (self.batch_size, self.output_dim_1))


    def test_modality_masking(self):
        model = LinMulT(
            {
                'input_feature_dim': [self.feature_dim_1, self.feature_dim_2, self.feature_dim_3],
                'heads': [{'type': 'simple', 'output_dim': self.output_dim_1}],
                'time_dim_reducer': 'gap'
            }
        )
        output = list(model([self.x_1, self.x_2, self.x_3], masks=[self.mask_1, self.mask_2, self.mask_3f]).values())
        self.assertEqual(output[0].shape, (self.batch_size, self.output_dim_1))


    def test_modality_sequence_masking(self):
        model = LinMulT(
            {
                'input_feature_dim': [self.feature_dim_2, self.feature_dim_2, self.feature_dim_3],
                'heads': [{'type': 'simple', 'output_dim': self.output_dim_1}]
            }
        )
        output_seq = list(model([self.x_2, self.x_2, self.x_3], masks=[self.mask_2, self.mask_2, self.mask_3f]).values())
        output_cls = apply_logit_aggregation(x=output_seq[0], mask=self.mask_2, method='meanpooling')
        self.assertEqual(output_seq[0].shape, (self.batch_size, self.time_dim_2, self.output_dim_1))
        self.assertEqual(output_cls.shape, (self.batch_size, self.output_dim_1))


if __name__ == "__main__":
    unittest.main()