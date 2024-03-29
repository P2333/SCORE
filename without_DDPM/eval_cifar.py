import argparse
import copy
import logging
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from preactresnet import PreActResNet18
from wideresnet import WideResNet
from utils_plus import (upper_limit, lower_limit, std, clamp, get_loaders,
    attack_pgd, evaluate_pgd, evaluate_standard, normalize)
from autoattack import AutoAttack
# installing AutoAttack by: pip install git+https://github.com/fra31/auto-attack

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', default=128, type=int)
    parser.add_argument('--data-dir', default='../../cifar-data', type=str)
    parser.add_argument('--epsilon', default=8, type=int)
    parser.add_argument('--out-dir', default='train_fgsm_output', type=str, help='Output directory')
    parser.add_argument('--seed', default=0, type=int, help='Random seed')
    parser.add_argument('--activation', default='ReLU', type=str)
    return parser.parse_args()


def main():
    args = get_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    logger = logging.getLogger(__name__)

    logging.basicConfig(
        format='[%(asctime)s] - %(message)s',
        datefmt='%Y/%m/%d %H:%M:%S',
        level=logging.DEBUG,
        handlers=[
            logging.StreamHandler()
        ])

    logger.info(args)

    _, test_loader = get_loaders(args.data_dir, args.batch_size)

    best_state_dict = torch.load(os.path.join(args.out_dir, 'model_best.pth'))

    # Evaluation
    model_test = PreActResNet18(activation=args.activation, softplus_beta=10)
    #model_test = WideResNet(34, 10, widen_factor=10, dropRate=0.0, activation=args.activation)
    model_test = nn.DataParallel(model_test).cuda()
    if 'state_dict' in best_state_dict.keys():
        model_test.load_state_dict(best_state_dict['state_dict'])
    else:
        model_test.load_state_dict(best_state_dict)
    model_test.float()
    model_test.eval()


    ### Evaluate clean acc ###
    _, test_acc = evaluate_standard(test_loader, model_test)
    print('Clean acc: ', test_acc)




    # # ### Evaluate PGD (CE loss) acc ###
    _, pgd_acc_CE = evaluate_pgd(test_loader, model_test, attack_iters=10, restarts=1, eps=8, step=2, norm='linf', use_CWloss=False)
    print('PGD-10 (10 restarts, step 2, CE loss) acc: ', pgd_acc_CE)
    ### Evaluate PGD (CW loss) acc ###
    _, pgd_acc_CW = evaluate_pgd(test_loader, model_test, attack_iters=10, restarts=1, eps=8, step=2, norm='linf', use_CWloss=True)
    print('PGD-10 (10 restarts, step 2, CW loss) acc: ', pgd_acc_CW)



    # # ### Evaluate PGD (CE loss) acc ###
    # _, pgd_acc_CE = evaluate_pgd(test_loader, model_test, attack_iters=10, restarts=1, eps=128, step=16, norm='l2', use_CWloss=False)
    # print('PGD-10 (10 restarts, step 2, CE loss) acc: ', pgd_acc_CE)
    # ### Evaluate PGD (CW loss) acc ###
    # _, pgd_acc_CW = evaluate_pgd(test_loader, model_test, attack_iters=10, restarts=1, eps=128, step=16, norm='l2', use_CWloss=True)
    # print('PGD-10 (10 restarts, step 2, CW loss) acc: ', pgd_acc_CW)



    ### Evaluate AutoAttack ###
    l = [x for (x, y) in test_loader]
    x_test = torch.cat(l, 0)
    l = [y for (x, y) in test_loader]
    y_test = torch.cat(l, 0)
    class normalize_model():
        def __init__(self, model):
            self.model_test = model
        def __call__(self, x):
            return self.model_test(normalize(x))
    new_model = normalize_model(model_test)



    epsilon = 8 / 255.
    adversary = AutoAttack(new_model, norm='Linf', eps=epsilon, version='standard')
    # epsilon = 128 / 255.
    # adversary = AutoAttack(new_model, norm='L2', eps=epsilon, version='standard')



    
    X_adv = adversary.run_standard_evaluation(x_test, y_test, bs=128)

if __name__ == "__main__":
    main()