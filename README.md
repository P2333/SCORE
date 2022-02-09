# Robustness and Accuracy Could Be Reconcilable by (Proper) Definition

Code for the paper **Robustness and Accuracy Could Be Reconcilable by (Proper) Definition**.

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

- Download 1M DDPM generated data from the [official implementation](https://github.com/deepmind/deepmind-research/tree/master/adversarial_robustness) of Rebuffi et al., 2021:

| dataset | model | size | link |
|---|---|:---:|:---:|
| CIFAR-10 | DDPM | 1M | [npz](https://storage.googleapis.com/dm-adversarial-robustness/cifar10_ddpm.npz) |
| CIFAR-100 | DDPM | 1M | [npz](https://storage.googleapis.com/dm-adversarial-robustness/cifar100_ddpm.npz) |
| SVHN | DDPM | 1M | [npz](https://storage.googleapis.com/dm-adversarial-robustness/svhn_ddpm.npz) |

## Training Commands
To run the KL-based baselines (with 1M DDPM generated data), an example is:
```python
python train-wa.py --data-dir 'cifar-data' \
    --log-dir 'trained_models' \
    --desc 'WRN28-10Swish_cifar10s_lr0p2_TRADES5_epoch400_bs512_fraction0p7_ls0p1' \
    --data cifar10s \
    --batch-size 512 \
    --model wrn-28-10-swish \
    --num-adv-epochs 400 \
    --lr 0.2 \
    --beta 5.0 \
    --unsup-fraction 0.7 \
    --aux-data-filename 'cifar10_ddpm.npz' \
    --ls 0.1
```
Here `--ls 0.1` is inherent from the the code implementation of Rebuffi et al., 2021.


To run our methods (with 1M DDPM generated data), an example is:
```python
python train-wa.py --data-dir 'cifar-data' \
    --log-dir 'trained_models' \
    --desc 'WRN28-10Swish_cifar10s_lr0p2_TRADES4_epoch400_bs512_fraction0p7_LSE' \
    --data cifar10s \
    --batch-size 512 \
    --model wrn-28-10-swish \
    --num-adv-epochs 400 \
    --lr 0.2 \
    --beta 4.0 \
    --unsup-fraction 0.7 \
    --aux-data-filename 'cifar10_ddpm.npz' \
    --LSE --ls 0
```
Here we only need to activate the flag `--LSE` and set `--ls 0`.

## Evaluation Commands
For evaluation under AutoAttack, run the command (taking our method as an example):
```python
python eval-aa.py --data-dir 'cifar-data' \
    --log-dir 'trained_models' \
    --desc 'WRN28-10Swish_cifar10s_lr0p2_TRADES4_epoch400_bs512_fraction0p7_LSE'
```

## Toy demos
To re-implement the toy demos, we could run:
```python
python toy_demo_1d.py --demo PGDAT --divergence KL --divergence_C L1
```
The flag `--demo` refers to the objective used for training; `--divergence` refers to the used metric loss; `--divergence_C` refers to the metric calculating SCORE values.
