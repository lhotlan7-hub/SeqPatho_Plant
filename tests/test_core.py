"""Behavioral smoke tests using synthetic tensors, with no external archive."""
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'core'))
from models import DualHead, ProteinOnlyHead, train_fold, masked_softmax


class CoreTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.cfg = {'model': {'dna_hidden_size': 8, 'protein_hidden_size': 6,
                    'bigru_hidden_size': 4, 'fusion_dim': 8, 'attention_hidden_size': 4, 'dropout': .1},
                    'improvement': {'dna_aux_loss_weight': .1, 'protein_aux_loss_weight': .1},
                    'frozen_training': {'batch_size': 4, 'learning_rate': .001, 'weight_decay': .0002,
                    'epochs': 2, 'patience': 2, 'rc_consistency_weight': .1,
                    'gradient_clip_norm': 1., 'threshold_min': .05, 'threshold_max': .95,
                    'threshold_step': .05}}
        rng = np.random.default_rng(3)
        self.arrays = {'dna': rng.normal(size=(24, 2, 4, 8)).astype('float32'),
                       'dm': np.ones((24, 4), bool),
                       'prot': rng.normal(size=(24, 3, 6)).astype('float32'),
                       'pm': np.ones((24, 3), bool)}

    def test_masked_softmax(self):
        x = torch.tensor([[1., 100., 2.], [3., 4., 5.]])
        mask = torch.tensor([[True, False, True], [False, False, False]])
        result = masked_softmax(x, mask, 1)
        self.assertEqual(result[0, 1].item(), 0.)
        self.assertAlmostEqual(result[0].sum().item(), 1., places=6)
        self.assertTrue(torch.equal(result[1], torch.zeros(3)))

    def test_protein_head_ignores_dna_and_masked_protein_views(self):
        model = ProteinOnlyHead(self.cfg).eval()
        a = {k: torch.from_numpy(v[:2].copy()) for k, v in self.arrays.items()}
        a['pm'][:, -1] = False
        with torch.no_grad():
            first = model(a['dna'], a['dm'], a['prot'], a['pm'])['logits']
            a['prot'][:, -1] = 1000
            second = model(a['dna'] * 100, ~a['dm'], a['prot'], a['pm'])['logits']
        torch.testing.assert_close(first, second, rtol=0, atol=0)

    def test_both_heads_train_and_save_selected_checkpoint(self):
        labels = np.tile([0, 1], 12).astype('float32')
        with tempfile.TemporaryDirectory() as tmp:
            for cls in (DualHead, ProteinOnlyHead):
                with self.subTest(model=cls.__name__):
                    checkpoint = Path(tmp) / (cls.__name__ + '.pt')
                    y, score, threshold, epoch, auc = train_fold(
                        lambda: cls(self.cfg), self.arrays, labels,
                        np.arange(12), np.arange(12, 18), np.arange(18, 24),
                        42, self.cfg, torch.device('cpu'), checkpoint)
                    np.testing.assert_array_equal(y, labels[18:])
                    self.assertTrue(np.isfinite(score).all())
                    self.assertTrue(((score >= 0) & (score <= 1)).all())
                    self.assertTrue(.05 <= threshold <= .95 + 1e-8)
                    saved = torch.load(checkpoint, map_location='cpu', weights_only=True)
                    self.assertEqual(saved['best_epoch'], epoch)
                    self.assertEqual(saved['best_validation_auroc'], auc)
                    cls(self.cfg).load_state_dict(saved['state_dict'])


if __name__ == '__main__':
    unittest.main()
