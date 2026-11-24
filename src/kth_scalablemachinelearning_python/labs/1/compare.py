"""Train and compare CIFAR-10 classifiers on a shared validation split."""

import argparse
import time
from itertools import islice
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from lab1 import load_cifar10, make_data_loaders, split_training_data
from models import build_model


def positive_int(value: str) -> int:
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return number


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    *,
    optimizer: torch.optim.Optimizer | None = None,
    max_batches: int | None = None,
) -> tuple[float, float, int]:
    """Return mean cross-entropy, accuracy, and image count."""
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    correct = 0
    examples = 0

    with torch.set_grad_enabled(training):
        for images, labels in islice(loader, max_batches):
            if optimizer is not None:
                optimizer.zero_grad()
            logits = model(images)
            loss = nn.functional.cross_entropy(logits, labels)
            if optimizer is not None:
                loss.backward()
                optimizer.step()

            count = labels.size(0)
            total_loss += loss.item() * count
            correct += (logits.argmax(dim=1) == labels).sum().item()
            examples += count

    if examples == 0:
        raise ValueError("The data loader did not produce any images")
    return total_loss / examples, correct / examples, examples


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare unpretrained CIFAR-10 models on the validation split."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path.home() / ".cache" / "kth-scalablemachinelearning-python",
        help="Dataset download and cache directory",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=positive_int, default=128)
    parser.add_argument("--epochs", type=positive_int, default=1)
    parser.add_argument("--max-train-batches", type=positive_int)
    parser.add_argument("--max-validation-batches", type=positive_int)
    args = parser.parse_args()

    torch.use_deterministic_algorithms(True)
    original_training, test = load_cifar10(args.data_dir)
    training, validation = split_training_data(original_training, seed=args.seed)
    print(
        f"CPU comparison: seed={args.seed}, batch_size={args.batch_size}, "
        f"epochs={args.epochs}, SGD lr=0.01, momentum=0.9",
        flush=True,
    )

    for name in ("alexnet", "resnet18"):
        torch.manual_seed(args.seed)
        model = build_model(name, seed=args.seed)
        train_loader, validation_loader, _ = make_data_loaders(
            training, validation, test, batch_size=args.batch_size, seed=args.seed
        )
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
        parameters = sum(parameter.numel() for parameter in model.parameters())
        print(f"\n{name}: {parameters:,} trainable parameters", flush=True)

        for epoch in range(1, args.epochs + 1):
            start = time.perf_counter()
            train_loss, _, train_count = run_epoch(
                model, train_loader, optimizer=optimizer, max_batches=args.max_train_batches
            )
            train_seconds = time.perf_counter() - start
            _, validation_accuracy, validation_count = run_epoch(
                model, validation_loader, max_batches=args.max_validation_batches
            )
            print(
                f"Epoch {epoch}: train loss={train_loss:.4f} "
                f"({train_count} images, {train_seconds:.1f}s); "
                f"validation accuracy={validation_accuracy:.2%} "
                f"({validation_count} images)",
                flush=True,
            )


if __name__ == "__main__":
    main()
