import unittest

import torch

from linear_mult.models.MulT import MulT
from linear_mult.models.transformer import TransformerEncoder


class TestArchitecture(unittest.TestCase):

    batch_size = 16

    def test_transformer_encoder(self):
        encoder = TransformerEncoder(300, 4, 2)
        x = torch.rand(20, 2, 300)
        x_k = torch.rand(20, 2, 300)
        x_v = torch.rand(20, 2, 300)
        self.assertEqual(encoder(x, x_k, x_v).size(), (20, 2, 300)) 


    def test_num_inputs(self):
        
        i2 = [torch.rand((self.batch_size, 5, 4)),
              torch.rand((self.batch_size, 6, 5))]

        i3 = [torch.rand((self.batch_size, 5, 4)),
              torch.rand((self.batch_size, 6, 5)),
              torch.rand((self.batch_size, 7, 6))]

        i4 = [torch.rand((self.batch_size, 5, 4)),
              torch.rand((self.batch_size, 7, 5)),
              torch.rand((self.batch_size, 6, 6)),
              torch.rand((self.batch_size, 8, 7))]

        i5 = [torch.rand((self.batch_size, 4, 4)),
              torch.rand((self.batch_size, 5, 5)),
              torch.rand((self.batch_size, 6, 6)),
              torch.rand((self.batch_size, 7, 7)),
              torch.rand((self.batch_size, 8, 8))]

        for inputs in [i2, i3, i4, i5]:
            model = MulT([i.shape[-1] for i in inputs], 1)
            output = model(inputs)
            self.assertEqual(output.size(), (self.batch_size, 1))


    def test_multimodal(self):
        # simulate opensmile egemaps, openface action units, wav2vec, fabnet, roberta features as inputs
        opensmile_egemaps = torch.rand((self.batch_size, 1500, 25)).cuda()
        openface_action_units = torch.rand((self.batch_size, 450, 35)).cuda()
        wav2vec = torch.rand((self.batch_size, 1500, 512)).cuda()
        fabnet = torch.rand((self.batch_size, 450, 256)).cuda()
        roberta = torch.rand((self.batch_size, 105, 1024)).cuda()
        model = MulT((25, 35, 512, 256, 1024), 5).cuda()
        output = model([opensmile_egemaps, openface_action_units, wav2vec, fabnet, roberta])
        self.assertEqual(output.detach().cpu().size(), (self.batch_size, 5))


    def test_pretrained_weight(self):
        model = MulT((25, 35, 512, 256, 1024), 1, weights='fi-linmult-oob-wfr-0').cuda()
        self.assertEqual(next(model.parameters()).device.type, 'cuda')


if __name__ == '__main__':
    unittest.main()

