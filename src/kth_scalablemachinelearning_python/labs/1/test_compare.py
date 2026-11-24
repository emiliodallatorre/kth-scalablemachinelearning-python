import unittest

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from compare import run_epoch


class ComparisonTests(unittest.TestCase):
    def test_epoch_limits_metrics_and_updates_only_during_training(self):
        images = torch.tensor([[3.0, 0.0], [0.0, 3.0], [0.0, 1.0], [1.0, 0.0]])
        labels = torch.tensor([0, 0, 1, 0])
        loader = DataLoader(TensorDataset(images, labels), batch_size=2)
        model = nn.Linear(2, 2, bias=False)
        with torch.no_grad():
            model.weight.copy_(torch.eye(2))
        original = model.weight.detach().clone()

        loss, accuracy, count = run_epoch(model, loader, max_batches=1)
        self.assertEqual(count, 2)
        self.assertEqual(accuracy, 0.5)
        self.assertGreater(loss, 0)
        self.assertFalse(model.training)
        self.assertTrue(torch.equal(original, model.weight))

        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        _, _, count = run_epoch(model, loader, optimizer=optimizer, max_batches=1)
        self.assertEqual(count, 2)
        self.assertTrue(model.training)
        self.assertFalse(torch.equal(original, model.weight))


if __name__ == "__main__":
    unittest.main()
