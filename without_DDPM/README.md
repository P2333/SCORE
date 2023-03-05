# WITHOUT DDPM genearted data
## Training commands of baselines
To run the **PGD-AT baseline**, an example is:
```python
python train_cifar.py --model PreActResNet18 \
                                              --epochs 110 --epsilon 8 \
                                              --attack-iters 10 --pgd-alpha 2 \
                                              --fname auto --weight_decay 5e-4 \
                                              --batch-size 128 --lr-max 0.1 \
                                              --attack PGD --divergence 'KL' \
                                              --optimizer 'SGDmom' \
                                              --seed 1
```
Here `--divergence 'KL'` refers to using KL divergence (i.e., cross-entropy).

To run the **TRADES baseline**, an example is:
```python
python train_cifar.py --model PreActResNet18 \
                                              --epochs 110 --epsilon 8 \
                                              --attack-iters 10 --pgd-alpha 2 \
                                              --fname auto --weight_decay 5e-4 \
                                              --batch-size 128 --lr-max 0.1 \
                                              --attack TRADES --TRADESlambda 6 \
                                              --divergence 'KL' \
                                              --optimizer 'SGDmom' \
                                              --seed 1
```
Here `--TRADESlambda` is the hyperparameter in TRADES.

## Training commands of our methods (with loss clipping)
To run the **PGD-AT (LSE)**, an example is:
```python
python train_cifar.py --model PreActResNet18 \
                                              --epochs 110 --epsilon 8 \
                                              --attack-iters 10 --pgd-alpha 2 \
                                              --fname auto --weight_decay 5e-4 \
                                              --batch-size 128 --lr-max 0.05 \
                                              --attack PGD --divergence 'LSE' \
                                              --optimizer 'SGDmom' --clip_loss 0.4 \
                                              --seed 1
```
Here `--lr-max 0.05` and `--clip_loss 0.4` as described in Section 5.2 of our paper.

To run the **TRADES (LSE)**, an example is:
```python
python train_cifar.py --model PreActResNet18 \
                                              --epochs 110 --epsilon 8 \
                                              --attack-iters 10 --pgd-alpha 2 \
                                              --fname auto --weight_decay 5e-4 \
                                              --batch-size 128 --lr-max 0.05 \
                                              --attack TRADES --TRADESlambda 6 \
                                              --divergence 'LSE' \
                                              --optimizer 'SGDmom' --clip_loss 0.3 \
                                              --seed 1
```
Here `--lr-max 0.05` and `--clip_loss 0.3` as described in Section 5.2 of our paper.

## Evaluation commands
```python
python -u eval_cifar.py --activation YOUR ACTIVATION --out-dir YOUR DIR
```
Here `YOUR ACTIVATION` is the activation function used in your model architecture, could be `'ReLU'` or `'Softplus'`. `YOUR DIR` is your file name (automatically generated if you use `--fname auto` during training).
