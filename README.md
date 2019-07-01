# Unsupervised Data Augmentation (UDA)
A PyTorch implementation for Unsupervised Data Augmentation.

## Disclaimer

* This is not an official implementation. The official TensorFlow implementation is at this [Github link](https://github.com/google-research/uda).
* Plan to implement CIFAR10 and ImageNet experiments.

## Updates

- **2019.06.28**: CIFAR-10 with 4,000 labeled set achieves top-1 accuracy **93.69%** without TSA. (on paper, 94.33% without TSA)

## TODO List

- [x] CIFAR-10 baseline & UDA validation
- [ ] ImageNet ResNet50 baseline validation
- [ ] ImageNet ResNet50 UDA validation

## MISC

- CIFAR10 labeled set is from AutoAugment policy search subset.
- ImageNet labeled set is randomly selected 10% for each class.
- ImageNet baseline settings are from [S4L: Self-Supervised Semi-Supervised Learning](https://arxiv.org/abs/1905.03670).
