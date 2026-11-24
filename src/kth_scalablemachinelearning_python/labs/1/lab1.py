"""Lab 1 guide: CIFAR-10 image classification.

1. Prepare data, splits, preprocessing, and reproducible random seeds.
2. Implement and compare AlexNet and ResNet under a defined compute budget.
3. Investigate capacity, L2 regularization, and dropout.
4. Evaluate augmentation on clean and corrupted images.
5. Analyze predictions, errors, and adversarial examples.
6. Measure costs and discuss accuracy and robustness tradeoffs.

See TASKS.md for the full requirements and optional challenges. The submission
requires an executed notebook with results, references, and discussion.
"""

import argparse
from collections import Counter
from pathlib import Path

import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import CIFAR10
from torchvision.transforms import ToTensor


def load_cifar10(data_dir: Path) -> tuple[CIFAR10, CIFAR10]:
    """Load the original training and test sets, downloading them if needed."""
    transform = ToTensor()
    training = CIFAR10(root=data_dir, train=True, download=True, transform=transform)
    test = CIFAR10(root=data_dir, train=False, download=True, transform=transform)
    return training, test


def split_training_data(
    training: CIFAR10, *, seed: int = 42
) -> tuple[Subset, Subset]:
    """Reserve 10% of the original training set for stratified validation."""
    train_indices, validation_indices = train_test_split(
        range(len(training)),
        test_size=0.1,
        random_state=seed,
        stratify=training.targets,
    )
    return Subset(training, train_indices), Subset(training, validation_indices)


def make_data_loaders(
    training: Subset,
    validation: Subset,
    test: CIFAR10,
    *,
    batch_size: int = 128,
    seed: int = 42,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Shuffle training reproducibly; keep validation and test in fixed order."""
    training_loader = DataLoader(
        training,
        batch_size=batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
    )
    validation_loader = DataLoader(
        validation,
        batch_size=batch_size,
        shuffle=False,
        generator=torch.Generator().manual_seed(seed),
    )
    test_loader = DataLoader(
        test,
        batch_size=batch_size,
        shuffle=False,
        generator=torch.Generator().manual_seed(seed),
    )
    return training_loader, validation_loader, test_loader


def main() -> None:
    parser = argparse.ArgumentParser(description="Load and inspect CIFAR-10.")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path.home() / ".cache" / "kth-scalablemachinelearning-python",
        help="Dataset download and cache directory",
    )
    parser.add_argument("--seed", type=int, default=42, help="Split and shuffle seed")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size")
    args = parser.parse_args()

    original_training, test = load_cifar10(args.data_dir)
    training, validation = split_training_data(original_training, seed=args.seed)
    train_loader, validation_loader, test_loader = make_data_loaders(
        training, validation, test, batch_size=args.batch_size, seed=args.seed
    )
    image, label = training[0]

    print(f"Training images: {len(training)}")
    print(f"Validation images: {len(validation)}")
    print(f"Test images reserved for final evaluation: {len(test)}")
    print(f"Seed: {args.seed}; batch size: {args.batch_size}")
    print(
        f"Batches (training/validation/test): "
        f"{len(train_loader)}/{len(validation_loader)}/{len(test_loader)}"
    )
    print(
        f"Example: shape={tuple(image.shape)}, class={original_training.classes[label]}"
    )
    train_counts = Counter(original_training.targets[index] for index in training.indices)
    validation_counts = Counter(
        original_training.targets[index] for index in validation.indices
    )
    print("Class distribution (training/validation):")
    for index, name in enumerate(original_training.classes):
        print(f"  {name}: {train_counts[index]}/{validation_counts[index]}")


if __name__ == "__main__":
    main()