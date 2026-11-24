import importlib.util
import unittest
from collections import Counter
from pathlib import Path

from torch.utils.data import Dataset


spec = importlib.util.spec_from_file_location("lab1", Path(__file__).with_name("lab1.py"))
lab1 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(lab1)


class ExampleDataset(Dataset):
    def __init__(self):
        self.targets = [index % 10 for index in range(100)]

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):
        return index, self.targets[index]


class DataPreparationTests(unittest.TestCase):
    def test_stratified_split_is_disjoint_and_reproducible(self):
        dataset = ExampleDataset()
        train, validation = lab1.split_training_data(dataset, seed=42)
        repeated_train, repeated_validation = lab1.split_training_data(dataset, seed=42)
        other_train, _ = lab1.split_training_data(dataset, seed=43)

        self.assertEqual(len(train), 90)
        self.assertEqual(len(validation), 10)
        self.assertEqual(set(train.indices) | set(validation.indices), set(range(100)))
        self.assertFalse(set(train.indices) & set(validation.indices))
        self.assertEqual(
            Counter(dataset.targets[i] for i in train.indices),
            dict.fromkeys(range(10), 9),
        )
        self.assertEqual(
            Counter(dataset.targets[i] for i in validation.indices),
            dict.fromkeys(range(10), 1),
        )
        self.assertEqual(train.indices, repeated_train.indices)
        self.assertEqual(validation.indices, repeated_validation.indices)
        self.assertNotEqual(train.indices, other_train.indices)

    def test_loader_order_is_reproducible_and_evaluation_is_unshuffled(self):
        dataset = ExampleDataset()
        train, validation = lab1.split_training_data(dataset)
        train_loader, validation_loader, test_loader = lab1.make_data_loaders(
            train, validation, dataset, batch_size=16
        )
        repeated_loader, _, _ = lab1.make_data_loaders(
            train, validation, dataset, batch_size=16
        )

        def image_order(loader):
            return [index for batch, _ in loader for index in batch.tolist()]

        self.assertEqual(image_order(validation_loader), validation.indices)
        self.assertEqual(image_order(test_loader), list(range(100)))
        self.assertEqual(image_order(train_loader), image_order(repeated_loader))
        self.assertNotEqual(image_order(train_loader), train.indices)


if __name__ == "__main__":
    unittest.main()
