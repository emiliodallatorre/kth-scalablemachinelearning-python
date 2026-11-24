import importlib.util
import unittest
from pathlib import Path

import torch


spec = importlib.util.spec_from_file_location(
    "lab1_models", Path(__file__).with_name("models.py")
)
models = importlib.util.module_from_spec(spec)
spec.loader.exec_module(models)


class ModelTests(unittest.TestCase):
    def test_seeded_initialization_preserves_rng_and_repeats(self):
        torch.manual_seed(7)
        state = torch.get_rng_state()
        model = models.build_model("alexnet", seed=42)
        self.assertTrue(torch.equal(state, torch.get_rng_state()))
        repeated = models.build_model("alexnet", seed=42)
        different = models.build_model("alexnet", seed=43)
        self.assertTrue(
            all(
                torch.equal(first, second)
                for first, second in zip(model.parameters(), repeated.parameters())
            )
        )
        self.assertFalse(
            torch.equal(next(model.parameters()), next(different.parameters()))
        )
        with self.assertRaises(ValueError):
            models.build_model("unknown")

    def test_cifar_batch_produces_ten_logits_and_supports_backpropagation(self):
        model = models.build_model("alexnet")
        images = torch.rand(2, 3, 32, 32)
        labels = torch.tensor([0, 9])
        logits = model(images)
        self.assertEqual(tuple(logits.shape), (2, 10))
        torch.nn.functional.cross_entropy(logits, labels).backward()
        self.assertIsNotNone(model.features[0].weight.grad)


if __name__ == "__main__":
    unittest.main()
