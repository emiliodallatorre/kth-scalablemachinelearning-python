# Scalable Machine Learning

This repository contains work for the Scalable Machine Learning course.

## Labs

- [Lab 1: Convolutional neural networks](src/kth_scalablemachinelearning_python/labs/1/)

### Lab 1: CIFAR-10 starter

In the devcontainer, download and inspect [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html) with:

```sh
uv run python src/kth_scalablemachinelearning_python/labs/1/lab1.py
```

The first run downloads the dataset; subsequent runs reuse the cached files. The default cache is `~/.cache/kth-scalablemachinelearning-python` inside the devcontainer. To use a different cache location, pass `--data-dir /path/to/data`.

The script loads the original 50,000 training images and 10,000 test images using torchvision. Each example is a 32 x 32 colour image with one of ten labels. Images become channel-first tensors of shape `(3, 32, 32)` with pixel values in `[0, 1]`.

It splits the original training set into 45,000 training and 5,000 validation images, stratified by class (4,500 and 500 images per class, respectively). The 10,000 test images remain separate for final evaluation. The script prints the split sizes, batch counts, an example, and class distributions.

The split and training-batch shuffle use seed `42` by default; `--seed` changes both, and `--batch-size` changes the default batch size of `128`. Validation and test loaders retain their fixed order. The loaders use the default of zero worker processes. No augmentation or model-specific normalization is applied yet.

#### AlexNet starter

`labs/1/models.py` provides `build_model("alexnet", seed=42)`, returning an untrained CPU PyTorch `nn.Module`. The same factory can later expose ResNet with the same input/output interface: batches of `(N, 3, 32, 32)` images in `[0, 1]` produce `(N, 10)` logits. The factory seeds initialization without changing the caller's CPU random state. This alone does not make training reproducible: dropout, optimization, device algorithms, and the data loader also need controlled randomness.

This provisional CIFAR-10 adaptation uses five convolutional layers and three 2 x 2 pooling layers. Compared with original AlexNet, its first convolution is 3 x 3 with stride 1 rather than 11 x 11 with stride 4, preserving detail in the smaller images. It uses torchvision-style channel widths, smaller pooling windows, and no grouped convolutions or local response normalization. The final feature maps are 256 x 4 x 4, followed by two 512-unit fully connected layers with 0.5 dropout and a ten-class output; these smaller layers reduce the classification head's cost. The referenced Part A9 is not in this repository, so its recommended adaptations still need to be checked before treating this as the required baseline.

The feature-map shapes after each convolution are `64 x 32 x 32`, `192 x 16 x 16`, `384 x 8 x 8`, `256 x 8 x 8`, and `256 x 8 x 8`; the last pooling layer produces `256 x 4 x 4`. The model has 4,813,642 trainable parameters. With the Lab 1 directory on Python's import path, create it using:

```python
from models import build_model

model = build_model("alexnet", seed=42)
```

Inspect the architecture's parameter count and output shape in the devcontainer:

```sh
uv run python src/kth_scalablemachinelearning_python/labs/1/models.py --model alexnet --seed 42
```

Run the data and model checks with `uv run python -m unittest discover -s src/kth_scalablemachinelearning_python/labs/1 -p "test_*.py"`.

The lab deliverable is an executed notebook; this script does not replace it.