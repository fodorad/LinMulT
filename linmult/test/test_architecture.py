import unittest
import torch
from linmult import LinT, LinMulT
from linmult.models.transformer import TransformerEncoder


class TestArchitecture(unittest.TestCase):

    batch_size: int = 16

    def test_transformer_encoder(self):
        encoder = TransformerEncoder({'d_model': 300, 'n_heads': 10})
        x_q = torch.rand(20, 2, 300)
        x_k = torch.rand(20, 2, 300)
        x_v = torch.rand(20, 2, 300)
        self.assertEqual(encoder(x_q, x_k, x_v).size(), (20, 2, 300))

    def test_same_time_dim(self):
        x_1 = torch.rand((self.batch_size, 30, 1024))
        x_2 = torch.rand((self.batch_size, 30, 160))
        model = LinMulT(
            {
                'input_modality_channels': (1024, 160),
                'output_dim': (5,)
            }
        )
        output_cls, output_seq = model([x_1, x_2])
        self.assertEqual(output_cls[0].detach().cpu().size(), (self.batch_size, 5))
        self.assertEqual(output_seq[0].detach().cpu().size(), (self.batch_size, 30, 5))

    def test_different_time_dim(self):
        x_1 = torch.rand((self.batch_size, 1500, 512))
        x_2 = torch.rand((self.batch_size, 450, 256))
        model = LinMulT(
            {
                'input_modality_channels': (512, 256),
                'output_dim': (5,),
                'time_reduce_type': 'gap'
            }
        )
        output_cls = model([x_1, x_2])
        self.assertEqual(output_cls[0].detach().cpu().size(), (self.batch_size, 5))

    def test_lint(self):
        x = torch.rand((self.batch_size, 450, 256))
        model = LinT(
            {
                'input_modality_channels': 256,
                'output_dim': (5,)
            }
        )
        output_seq = model(x)
        self.assertEqual(output_seq[0].detach().cpu().size(), (self.batch_size, 450, 5))


if __name__ == '__main__':
    unittest.main()