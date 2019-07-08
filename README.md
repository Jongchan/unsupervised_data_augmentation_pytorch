# Unsupervised Data Augmentation (UDA)
A PyTorch implementation for [Unsupervised Data Augmentation](https://arxiv.org/abs/1904.12848).

## Disclaimer

* This is not an official implementation. The official TensorFlow implementation is at this [Github link](https://github.com/google-research/uda).
* Plan to implement CIFAR10 and ImageNet experiments.

## Updates

- **2019.06.28**: CIFAR-10 with 4,000 labeled set achieves top-1 accuracy **93.69%** without TSA. (on paper, 94.33% without TSA)

## Performance

### CIFAR-10

| Exp               | Top-1 acc(%) in paper | Top-1 acc(%) |
|-------------------|-----------------------|--------------|
| Baseline          | 79.74                 | 83.94        |
| UDA (without TSA) | 94.33                 | 93.69        |
| UDA               | 94.90                 | -            |

### ImageNet (10% labeled)

| Exp      | Top-1 (paper) | Top-5 (paper)         | Top-1  | Top-5  |
|----------|---------------|-----------------------|--------|--------|
| RN50     | 55.09         | 77.26 (80.43 in S4L)  | 54.184 | 79.116 |
| RN18     | -             | -                     | 50.594 | 76.138 |
| UDA(RN50)| 68.66         | 88.52                 | -      | -      |
| S4L(RN50)| -             | 91.23 (ResNet50v2 4x) | -      | -      |

## TODO List

- [x] CIFAR-10 baseline & UDA validation
- [x] ImageNet ResNet50 baseline validation
- [ ] ImageNet ResNet50 UDA validation

## MISC

- CIFAR10 baseline on paper is from [Realistic Evaluation of Deep Semi-Supervised Learning Algorithms](https://papers.nips.cc/paper/7585-realistic-evaluation-of-deep-semi-supervised-learning-algorithms), and it may be sub-optimal OR use different data split from the UDA paper. A naive baseline with weight decay 5e-4 and 100K iteration with cosine annealing LR can achieve higher performance as shown in the table.
- CIFAR10 labeled set is from AutoAugment policy search subset.
- CIFAR10 AutoAugment policy includes full set (95 policies), rather than 25 policies.
- ImageNet labeled set is randomly selected 10% for each class.
- ImageNet baseline settings are from [S4L: Self-Supervised Semi-Supervised Learning](https://arxiv.org/abs/1905.03670).
