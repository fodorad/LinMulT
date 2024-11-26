import unittest
import torch
from torch import nn
from linmult.models.attention import AttentionLayer, SoftmaxAttention, LinearAttention, BigBirdAttention


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


    def test_self_attention_mha(self):
        mha = nn.MultiheadAttention(embed_dim=self.d_model, num_heads=self.n_heads, batch_first=True)
        output, _ = mha(self.d_1, self.d_1, self.d_1)
        self.assertEqual(output.shape, self.d_1.shape, "MultiheadAttention output shape mismatch")


    def test_cross_attention_mha(self):
        mha = nn.MultiheadAttention(embed_dim=self.d_model, num_heads=self.n_heads, batch_first=True)
        output, _ = mha(self.d_1, self.d_2, self.d_2)
        self.assertEqual(output.shape, self.d_1.shape, "MultiheadAttention output shape mismatch")


    def test_self_attention_softmax(self):
        softmax_attention = AttentionLayer(
            SoftmaxAttention(self.d_model, self.n_heads),
            d_model=self.d_model,
            n_heads=self.n_heads
        )
        output, _ = softmax_attention(self.d_1, self.d_1, self.d_1)
        self.assertEqual(output.shape, self.d_1.shape, "SoftmaxAttention output shape mismatch")


    def test_cross_attention_softmax(self):
        softmax_attention = AttentionLayer(
            SoftmaxAttention(self.d_model, self.n_heads),
            d_model=self.d_model,
            n_heads=self.n_heads
        )
        output, _ = softmax_attention(self.d_1, self.d_2, self.d_2)
        self.assertEqual(output.shape, self.d_1.shape, "SoftmaxAttention output shape mismatch")


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


    def test_self_attention_linear(self):
        linear_attention = AttentionLayer(
            LinearAttention(self.d_model, self.n_heads),
            d_model=self.d_model,
            n_heads=self.n_heads
        )
        output, _ = linear_attention(self.d_1, self.d_1, self.d_1)
        self.assertEqual(output.shape, self.d_1.shape, "LinearAttention output shape mismatch")


    def test_cross_attention_linear(self):
        linear_attention = AttentionLayer(
            LinearAttention(self.d_model, self.n_heads),
            d_model=self.d_model,
            n_heads=self.n_heads
        )
        output, _ = linear_attention(self.d_1, self.d_2, self.d_2)
        self.assertEqual(output.shape, self.d_1.shape, "LinearAttention output shape mismatch")


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


    def test_self_attention_bigbird(self):
        bigbird_attention = AttentionLayer(
            BigBirdAttention(
                self.d_model, self.n_heads, self.block_size, self.num_global_tokens, self.num_random_tokens
            ),
            d_model=self.d_model,
            n_heads=self.n_heads
        )
        output, _ = bigbird_attention(self.d_1, self.d_1, self.d_1)
        self.assertEqual(output.shape, self.d_1.shape, "BigBirdAttention output shape mismatch")


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


    def test_cross_attention_bigbird(self):
        bigbird_attention = AttentionLayer(
            BigBirdAttention(
                self.d_model, self.n_heads, self.block_size, self.num_global_tokens, self.num_random_tokens
            ),
            d_model=self.d_model,
            n_heads=self.n_heads
        )
        output, _ = bigbird_attention(self.d_1, self.d_2, self.d_2)
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


if __name__ == "__main__":
    unittest.main()