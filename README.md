# Robustness and Accuracy Could Be Reconcilable by (Proper) Definition

Code for the ICML 2022 submission **Robustness and Accuracy Could Be Reconcilable by (Proper) Definition** (paper ID 1046).

## Environment settings and libraries we used in our experiments

This project is tested under the following environment settings:
- OS: Ubuntu 20.04.3
- GPU: NVIDIA A100
- Cuda: 11.1, Cudnn: v8.2
- Python: 3.9.5
- PyTorch: 1.8.0
- Torchvision: 0.9.0

## Acknowledgement
The codes are modifed based on the [PyTorch implementation](https://github.com/imrahulr/adversarial_robustness_pytorch) of [Rebuffi et al., 2021](https://arxiv.org/abs/2103.01946).

## Requirements

- Install or download [AutoAttack](https://github.com/fra31/auto-attack):
```
pip install git+https://github.com/fra31/auto-attack
```

- Download 1M DDPM generated data from the [official implementation](https://github.com/deepmind/deepmind-research/tree/master/adversarial_robustness) of [Rebuffi et al., 2021](https://arxiv.org/abs/2103.01946):

| dataset | model | size | link |
|---|---|:---:|:---:|
| CIFAR-10 | DDPM | 1M | [npz](https://storage.googleapis.com/dm-adversarial-robustness/cifar10_ddpm.npz) |
| CIFAR-100 | DDPM | 1M | [npz](https://storage.googleapis.com/dm-adversarial-robustness/cifar100_ddpm.npz) |
| SVHN | DDPM | 1M | [npz](https://storage.googleapis.com/dm-adversarial-robustness/svhn_ddpm.npz) |

