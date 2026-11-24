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

from torchvision.datasets import CIFAR10
from torchvision.transforms import ToTensor


def load_cifar10(data_dir: Path) -> tuple[CIFAR10, CIFAR10]:
    """Load the original training and test sets, downloading them if needed."""
    transform = ToTensor()
    training = CIFAR10(root=data_dir, train=True, download=True, transform=transform)
    test = CIFAR10(root=data_dir, train=False, download=True, transform=transform)
    return training, test


def main() -> None:
    parser = argparse.ArgumentParser(description="Load and inspect CIFAR-10.")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path.home() / ".cache" / "kth-scalablemachinelearning-python",
        help="Dataset download and cache directory",
    )
    args = parser.parse_args()

    training, test = load_cifar10(args.data_dir)
    image, label = training[0]
    counts = Counter(training.targets)

    print(f"Training images: {len(training)}")
    print(f"Test images reserved for final evaluation: {len(test)}")
    print(f"Example: shape={tuple(image.shape)}, class={training.classes[label]}")
    print("Training class distribution:")
    for index, name in enumerate(training.classes):
        print(f"  {name}: {counts[index]}")


if __name__ == "__main__":
    main()