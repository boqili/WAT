# Worst-class Adversarial Training
This repository provides codes for Worst-class Adversarial Training(WAT) 

**WAT: Improve the Worst-class Robustness in Adversarial Training** [[arxiv](https://arxiv.org/abs/2302.04025)] (AAAI 2023)

Boqi Li and Weiwei Liu.

## Requirements
- python 3.8.11
- numpy 1.21.2
- torch 1.12
- torchvision 0.13
- tensorboardX 2.4
- tqdm 4.62.3

We adapt and use some code snippets from:
- [Friendly-Adversarial-Training](https://github.com/zjfheart/Friendly-Adversarial-Training)
- [fair_robust](https://github.com/hannxu123/fair_robust)

## Example
you can train a resnet18 on CIFAR-10 with our algorithm as follow:

    CUDA_VISIBLE_DEVICES=0 python worst_class_adversarial_train.py --lr 0.1 --dataset cifar10 --net resnet18 --beta 6.0 --alg wat --eta 0.1

