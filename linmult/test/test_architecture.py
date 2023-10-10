import unittest
import torch
from linmult import LinT, LinMulT
from linmult.models.transformer import TransformerEncoder


class TestArchitecture(unittest.TestCase):

    batch_size: int = 16

    def test_transformer_encoder(self):
        encoder = TransformerEncoder(300, 4, 2)
        x_q = torch.rand(20, 2, 300)
        x_k = torch.rand(20, 2, 300)
        x_v = torch.rand(20, 2, 300)
        self.assertEqual(encoder(x_q, x_k, x_v).size(), (20, 2, 300))

    def test_same_time_dim(self):
        x_1 = torch.rand((self.batch_size, 30, 1024)).cuda()
        x_2 = torch.rand((self.batch_size, 30, 160)).cuda()
        model = LinMulT((1024, 160), 5).cuda()
        output_cls, output_seq = model([x_1, x_2])
        self.assertEqual(output_cls.detach().cpu().size(), (self.batch_size, 5))
        self.assertEqual(output_seq.detach().cpu().size(), (self.batch_size, 30, 5))

    def test_different_time_dim(self):
        x_1 = torch.rand((self.batch_size, 1500, 512)).cuda()
        x_2 = torch.rand((self.batch_size, 450, 256)).cuda()
        model = LinMulT((512, 256), 5, add_time_collapse=True, add_self_attention_fusion=False).cuda()
        output_cls = model([x_1, x_2])
        self.assertEqual(output_cls.detach().cpu().size(), (self.batch_size, 5))

    def test_lint(self):
        x = torch.rand((self.batch_size, 450, 256)).cuda()
        model = LinT(256, 5).cuda()
        output_seq = model(x)
        self.assertEqual(output_seq.detach().cpu().size(), (self.batch_size, 450, 5))


if __name__ == '__main__':
    unittest.main()