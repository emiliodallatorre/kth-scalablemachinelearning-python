# Scalable Machine Learning

This repository contains work for the Scalable Machine Learning course.

## Labs

- [Lab 1: Convolutional neural networks](src/kth_scalablemachinelearning_python/labs/1/)

To download and inspect CIFAR-10 for Lab 1, run:

```sh
uv run python src/kth_scalablemachinelearning_python/labs/1/lab1.py
```

The dataset is cached outside the repository by default; use `--data-dir` to choose another location. Images are converted to tensors with values in `[0, 1]`. A validation split and model-specific preprocessing are still needed before training; reserve the test set for final evaluation.