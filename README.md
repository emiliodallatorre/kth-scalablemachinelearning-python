# Scalable Machine Learning

This repository contains work for the Scalable Machine Learning course.

## Local setup

On an Intel Mac with an AMD Radeon GPU, use Python 3.10, PyTorch 2.2.2, torchvision 0.17.2, and NumPy below 2, following the [tested MPS configuration](https://github.com/emiliodallatorre/torch-mpsradeon-python). Install the locked dependencies locally with `uv sync`. The comparison can explicitly use `--device mps`; it reports an error if MPS is unavailable rather than silently switching to the CPU.

## Labs

- [Lab 1: Convolutional neural networks](src/kth_scalablemachinelearning_python/labs/1/)

### Lab 1: CIFAR-10 starter

Download and inspect [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html) locally with:

```sh
uv run python src/kth_scalablemachinelearning_python/labs/1/lab1.py
```

The first run downloads the dataset; subsequent runs reuse the cached files. The default cache is `~/.cache/kth-scalablemachinelearning-python`. To use a different cache location, pass `--data-dir /path/to/data`.

The script loads the original 50,000 training images and 10,000 test images using torchvision. Each example is a 32 x 32 colour image with one of ten labels. Images become channel-first tensors of shape `(3, 32, 32)` with pixel values in `[0, 1]`.

It splits the original training set into 45,000 training and 5,000 validation images, stratified by class (4,500 and 500 images per class, respectively). The 10,000 test images remain separate for final evaluation. The script prints the split sizes, batch counts, an example, and class distributions.

The split and training-batch shuffle use seed `42` by default; `--seed` changes both, and `--batch-size` changes the default batch size of `128`. Validation and test loaders retain their fixed order. The loaders use the default of zero worker processes. No augmentation or model-specific normalization is applied yet.

#### Models and comparison

`labs/1/models.py` provides `build_model("alexnet", seed=42)` and `build_model("resnet18", seed=42)`, returning untrained CPU PyTorch `nn.Module` models. Both accept batches of `(N, 3, 32, 32)` images in `[0, 1]` and produce `(N, 10)` logits. The factory seeds initialization without changing the caller's CPU random state. This alone does not make training reproducible: dropout, optimization, device algorithms, and the data loader also need controlled randomness.

This provisional CIFAR-10 adaptation uses five convolutional layers and three 2 x 2 pooling layers. Compared with original AlexNet, its first convolution is 3 x 3 with stride 1 rather than 11 x 11 with stride 4, preserving detail in the smaller images. It uses torchvision-style channel widths, smaller pooling windows, and no grouped convolutions or local response normalization. The final feature maps are 256 x 4 x 4, followed by two 512-unit fully connected layers with 0.5 dropout and a ten-class output; these smaller layers reduce the classification head's cost. The referenced Part A9 is not in this repository, so its recommended adaptations still need to be checked before treating this as the required baseline.

The feature-map shapes after each convolution are `64 x 32 x 32`, `192 x 16 x 16`, `384 x 8 x 8`, `256 x 8 x 8`, and `256 x 8 x 8`; the last pooling layer produces `256 x 4 x 4`. The model has 4,813,642 trainable parameters. With the Lab 1 directory on Python's import path, create it using:

```python
from models import build_model

model = build_model("alexnet", seed=42)
```

ResNet-18 uses torchvision's implementation without pretrained weights, with its initial 7 x 7 stride-2 convolution replaced by a 3 x 3 stride-1 convolution and its initial max-pooling layer removed. This preserves spatial detail for CIFAR-10; its final layer outputs ten logits. Residual connections add each block's input to its transformed features, helping information and gradients flow through the network. The model has 11,173,962 trainable parameters. Like AlexNet, this adaptation remains provisional until checked against Part A9.

Inspect either architecture's parameter count and output shape:

```sh
uv run python src/kth_scalablemachinelearning_python/labs/1/models.py --model alexnet --seed 42
```

For a small end-to-end comparison, run:

```sh
uv run python src/kth_scalablemachinelearning_python/labs/1/compare.py --device mps --batch-size 8 --max-train-batches 20 --max-validation-batches 10
```

Both models use the same seeded split, batch order, and image budget, cross-entropy loss, and SGD with learning rate `0.01` and momentum `0.9`. The run seeds PyTorch for each model and enables deterministic algorithms. The default device is CPU; `--device mps` moves the models and batches to MPS. It reports per-epoch training loss, training time, validation accuracy, and the actual image counts; the test set is not evaluated. Timing varies by hardware and load, and matching seeds does not eliminate model-dependent randomness such as dropout. Results may also vary across software versions, hardware, and backends.

The command above trains for one epoch on only 160 images and validates on 80: it is a smoke comparison, not evidence for selecting a model. Omit the batch limits to use the full training and validation splits, and set `--epochs` to change the training budget. A full run may take a long time. No checkpoints, augmentation, or final test evaluation are implemented yet.

Run the data and model checks with `uv run python -m unittest discover -s src/kth_scalablemachinelearning_python/labs/1 -p "test_*.py"`.

The lab deliverable is an executed notebook; this script does not replace it.