"""CIFAR-10 classifiers with a shared PyTorch model interface."""

import argparse

import torch
from torch import nn


class CifarAlexNet(nn.Module):
    """AlexNet-style classifier adapted to 32 x 32 RGB images."""

    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=0.5),
            nn.Linear(256 * 4 * 4, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(512, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 10),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(images))


def build_model(name: str, *, seed: int = 42) -> nn.Module:
    """Build a reproducibly initialized CPU model without changing the global RNG."""
    if name != "alexnet":
        raise ValueError(f"Unknown model: {name!r}")

    with torch.random.fork_rng(devices=[]), torch.device("cpu"):
        torch.random.default_generator.manual_seed(seed)
        return CifarAlexNet()


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect a CIFAR-10 classifier.")
    parser.add_argument("--model", choices=["alexnet"], default="alexnet")
    parser.add_argument("--seed", type=int, default=42, help="Model initialization seed")
    args = parser.parse_args()

    model = build_model(args.model, seed=args.seed)
    model.eval()
    with torch.inference_mode():
        logits = model(torch.zeros(1, 3, 32, 32))
    print(f"Model: {args.model} (untrained)")
    print(f"Initialization seed: {args.seed}")
    print(f"Parameters: {sum(parameter.numel() for parameter in model.parameters()):,}")
    print(f"Input shape: (1, 3, 32, 32); logits shape: {tuple(logits.shape)}")


if __name__ == "__main__":
    main()
