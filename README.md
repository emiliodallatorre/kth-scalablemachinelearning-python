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

The script loads the original 50,000 training images and 10,000 test images using torchvision. Each example is a 32 x 32 colour image with one of ten labels. It converts images to channel-first tensors of shape `(3, 32, 32)` with pixel values in `[0, 1]`, prints the first training example's class, and reports the training class counts (5,000 per class).

This is only the loading and inspection starting point: no validation split, random sampling, augmentation, or model-specific normalization is applied yet. Create a reproducible training/validation split before training and reserve the test set for final evaluation. The lab deliverable is an executed notebook; this script does not replace it.