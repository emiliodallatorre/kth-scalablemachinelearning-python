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

Run the data-preparation checks with `uv run python -m unittest discover -s src/kth_scalablemachinelearning_python/labs/1 -p "test_*.py"`.

The lab deliverable is an executed notebook; this script does not replace it.