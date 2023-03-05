import argparse
import logging
import sys
import time
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.autograd import Variable
import os

from wideresnet import WideResNet
from preactresnet import PreActResNet18

from utils import *

criterion_kl = nn.KLDivLoss(reduction='none')

def distance_func(output, target, divergence): # both output and target are summed to 1
    M = (output + target) / 2
    if divergence == 'JSsqrt':
        return (0.5 * (criterion_kl(M.log(), output) + criterion_kl(M.log(), target))).sum(dim=-1).sqrt()
    elif divergence == 'JS':
        return (0.5 * (criterion_kl(M.log(), output) + criterion_kl(M.log(), target))).sum(dim=-1)
    elif divergence == 'LSE':
        return torch.sum((output - target) ** 2, dim=-1)
    elif divergence == 'L1square':
        return (output - target).abs().sum(dim=-1).square()
    elif divergence == 'KL':
        return criterion_kl(output.log(), target).sum(dim=-1)
    elif divergence == 'KLsqrt':
        return criterion_kl(output.log(), target).sum(dim=-1).sqrt()
    elif divergence == 'RKLsqrt':
        return criterion_kl(target.log(), output).sum(dim=-1).sqrt()
    else:
        return torch.norm(output - target, dim=-1, p=float(divergence))

def attack_trades_divergence(model, X, y, epsilon, alpha, attack_iters, restarts, norm, divergence='1'):
    model.eval()
    clean_output = model(normalize(X))
    clean_output = F.softmax(clean_output.detach(), dim=1)
    delta = 0.001 * torch.randn_like(X)
    delta = clamp(delta, lower_limit-X, upper_limit-X)
    delta.requires_grad = True
    for _ in range(attack_iters):
        output = model(normalize(X + delta))
        output = F.softmax(output, dim=1)
        loss = distance_func(output, clean_output, divergence)
        grad = torch.autograd.grad(loss.mean(), delta)[0]
        if norm == "linf":
            d = torch.clamp(delta + alpha * torch.sign(grad), min=-epsilon, max=epsilon)
        delta.data = clamp(d, lower_limit - X, upper_limit - X)
    model.train()
    return delta.detach()

def attack_pgd(model, X, y, epsilon, alpha, attack_iters, restarts, norm):
    bs = y.shape[0]
    max_loss = torch.zeros(bs).cuda()
    max_delta = torch.zeros_like(X).cuda()
    for _ in range(restarts):
        delta = torch.zeros_like(X).cuda()
        if norm == "linf":
            delta.uniform_(-epsilon, epsilon)
        elif norm == "l2":
            delta.normal_()
            d_flat = delta.view(delta.size(0),-1)
            n = d_flat.norm(p=2,dim=1).view(delta.size(0),1,1,1)
            r = torch.zeros_like(n).uniform_(0, 1)
            delta *= r/n*epsilon
        delta = clamp(delta, lower_limit-X, upper_limit-X)
        delta.requires_grad = True
        for _ in range(attack_iters):
            output = model(normalize(X + delta))
            loss = F.cross_entropy(output, y, reduction='none')
            loss.mean().backward()
            grad = delta.grad.detach()
            if norm == "linf":
                delta.data = torch.clamp(delta + alpha * torch.sign(grad), min=-epsilon, max=epsilon)
            elif norm == "l2":
                g_norm = torch.norm(grad.view(bs,-1),dim=1).view(bs,1,1,1)
                scaled_g = grad/(g_norm + 1e-10)
                delta.data = (delta + scaled_g*alpha).view(bs,-1).renorm(p=2,dim=0,maxnorm=epsilon).view_as(delta)
            delta.data = clamp(delta, lower_limit - X, upper_limit - X)
            delta.grad.zero_()
        all_loss = F.cross_entropy(model(normalize(X + delta)), y, reduction='none')
        max_delta[all_loss >= max_loss] = delta.detach()[all_loss >= max_loss]
        max_loss = torch.max(max_loss, all_loss)
    return max_delta.detach()

def attack_pgd_divergence(model, X, y, epsilon, alpha, attack_iters, restarts, norm, divergence='1', num_classes=10):
    bs = y.shape[0]
    y = F.one_hot(y, num_classes=num_classes)
    max_loss = torch.zeros(bs).cuda()
    max_delta = torch.zeros_like(X).cuda()
    for _ in range(restarts):
        delta = torch.zeros_like(X).cuda()
        if norm == "linf":
            delta.uniform_(-epsilon, epsilon)
        elif norm == "l2":
            delta.normal_()
            d_flat = delta.view(delta.size(0),-1)
            n = d_flat.norm(p=2,dim=1).view(delta.size(0),1,1,1)
            r = torch.zeros_like(n).uniform_(0, 1)
            delta *= r/n*epsilon
        delta = clamp(delta, lower_limit-X, upper_limit-X)
        delta.requires_grad = True
        for _ in range(attack_iters):
            output = model(normalize(X + delta))
            output = F.softmax(output, dim=1)
            loss = distance_func(output, y.float(), divergence)
            loss.mean().backward()
            grad = delta.grad.detach()
            if norm == "linf":
                delta.data = torch.clamp(delta + alpha * torch.sign(grad), min=-epsilon, max=epsilon)
            elif norm == "l2":
                g_norm = torch.norm(grad.view(bs,-1),dim=1).view(bs,1,1,1)
                scaled_g = grad/(g_norm + 1e-10)
                delta.data = (delta + scaled_g*alpha).view(bs,-1).renorm(p=2,dim=0,maxnorm=epsilon).view_as(delta)
            delta.data = clamp(delta, lower_limit - X, upper_limit - X)
            delta.grad.zero_()
    return delta.detach()



def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='PreActResNet18')
    parser.add_argument('--batch-size', default=128, type=int)
    parser.add_argument('--data-dir', default='../../cifar-data', type=str)
    parser.add_argument('--epochs', default=120, type=int)
    parser.add_argument('--lr-max', default=0.1, type=float)
    parser.add_argument('--attack', default='PGD', type=str, choices=['PGD', 'TRADES','Standard'])
    parser.add_argument('--epsilon', default=8, type=int)
    parser.add_argument('--test_epsilon', default=8, type=int)
    parser.add_argument('--attack-iters', default=10, type=int)
    parser.add_argument('--restarts', default=1, type=int)
    parser.add_argument('--pgd-alpha', default=2, type=float)
    parser.add_argument('--test-pgd-alpha', default=2, type=float)
    parser.add_argument('--norm', default='linf', type=str, choices=['linf', 'l2'])
    parser.add_argument('--fname', default='auto', type=str)
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--width-factor', default=10, type=int)
    parser.add_argument('--activation', default='ReLU', type=str)

    parser.add_argument('--TRADESlambda', default=6, type=float)
    parser.add_argument('--divergence', default='1', type=str)
    parser.add_argument('--weight_decay', default=5e-4, type=float)
    parser.add_argument('--optimizer', default='SGDmom', type=str)

    parser.add_argument('--clip_loss', default=0, type=float)
    parser.add_argument('--not_clip_clean', action='store_true')
    parser.add_argument('--label_smoothing', default=0, type=float)
    parser.add_argument('--dataset', default='CIFAR-10', type=str)
    return parser.parse_args()


def get_auto_fname(args):
    names = args.model
    if args.activation != 'ReLU': names += '_' + args.activation 
    names += '_' + args.attack
    if args.attack == 'TRADES': names += str(args.TRADESlambda)
    names += '_' + args.norm + '_epoch' + str(args.epochs) + '_eps' + str(args.epsilon) 
    names += '_wd' + str(args.weight_decay) + '_' + args.optimizer + 'lr' + str(args.lr_max)
    if args.divergence in ['JSsqrt', 'JS', 'LSE', 'L1square', 'KLsqrt', 'RKLsqrt', 'KL']:
        names += '_divergence' + args.divergence
    else:
        names += '_divergenceL' + args.divergence
    if args.clip_loss > 0: names += '_clip' + str(args.clip_loss)
    if args.not_clip_clean: names += '_notclean'
    if args.label_smoothing > 0: names += '_labelsm' + str(args.label_smoothing)
    if args.seed != 0: names += '_seed' + str(args.seed)
    print('File name: ', names)
    return names

def main():
    args = get_args()

    prefix = 'trained_models/Triangle/' + args.dataset + '/'

    if args.fname == 'auto':
        names = get_auto_fname(args)
        args.fname = prefix + names
    else:
        args.fname = prefix + args.fname

    if not os.path.exists(args.fname):
        os.makedirs(args.fname)

    logger = logging.getLogger(__name__)
    logging.basicConfig(
        format='[%(asctime)s] - %(message)s',
        datefmt='%Y/%m/%d %H:%M:%S',
        level=logging.DEBUG,
        handlers=[
            logging.FileHandler(os.path.join(args.fname, 'output.log')),
            logging.StreamHandler()
        ])

    logger.info(args)

    # Set seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    # Prepare data
    transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor()])
    transform_test = transforms.Compose([transforms.ToTensor()])

    if args.dataset == 'CIFAR-10':
        trainset = torchvision.datasets.CIFAR10(root='../../cifar-data', train=True, download=True, transform=transform_train)
        testset = torchvision.datasets.CIFAR10(root='../../cifar-data', train=False, download=True, transform=transform_test)
        num_cla = 10
    elif args.dataset == 'CIFAR-100':
        trainset = torchvision.datasets.CIFAR100(root='../../cifar-data', train=True, download=True, transform=transform_train)
        testset = torchvision.datasets.CIFAR100(root='../../cifar-data', train=False, download=True, transform=transform_test)
        num_cla = 100

    train_batches = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=False)
    test_batches = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=False)


    # Set perturbations
    epsilon = (args.epsilon / 255.)
    test_epsilon = (args.test_epsilon / 255.)
    pgd_alpha = (args.pgd_alpha / 255.)
    test_pgd_alpha = (args.test_pgd_alpha / 255.)

    # Compute clip bound
    clip_bound = torch.ones(num_cla) * args.clip_loss / (num_cla - 1)
    clip_bound[0] = 1 - args.clip_loss
    clip_bound_target = torch.zeros(num_cla)
    clip_bound_target[0] = 1
    clip_bound = (distance_func(clip_bound, clip_bound_target, args.divergence)).item()
    print('Clip loss bound: ', clip_bound)
    args.label_smoothing = args.label_smoothing / (num_cla - 1)

    # Set models
    if args.model == 'PreActResNet18':
        model = PreActResNet18(num_classes=num_cla, activation=args.activation, softplus_beta=10)
    elif args.model == 'WideResNet':
        model = WideResNet(34, num_cla, widen_factor=args.width_factor, dropRate=0.0, activation=args.activation)
    else:
        raise ValueError("Unknown model")

    model = nn.DataParallel(model).cuda()
    model.train()
    
    # Set training hyperparameters
    if args.optimizer == 'SGDmom' :
        opt = torch.optim.SGD(model.parameters(), lr=args.lr_max, momentum=0.9, weight_decay=args.weight_decay)
    elif args.optimizer == 'Cosine' :
        opt = torch.optim.SGD(model.parameters(), lr=args.lr_max, momentum=0.9, weight_decay=args.weight_decay, nesterov=True)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs, verbose=False)
    elif args.optimizer == 'Adam':
        opt = torch.optim.Adam(model.parameters(), lr=args.lr_max, weight_decay=args.weight_decay)
    
    epochs = args.epochs

    # Set lr schedule
    def lr_schedule(t):
        if t < 100:
            return args.lr_max
        elif t < 105:
            return args.lr_max / 10.
        else:
            return args.lr_max / 100.

    # def lr_schedule(t):
    #     if t < 100:
    #         return args.lr_max
    #     elif t < 150:
    #         return args.lr_max / 10.
    #     else:
    #         return args.lr_max / 100.
    
    best_test_robust_acc = 0
    start_epoch = 0

    logger.info('Epoch \t Test Acc \t Test Rob. Acc (KL / new div)')
    
    # Records per epoch for savetxt
    test_acc_record = []
    test_robust_acc_record_KL = []
    test_robust_acc_record_new = []

    for epoch in range(start_epoch, epochs):
        model.train()
        start_time = time.time()


        for i, (data, target) in enumerate(train_batches):
            X, y = data.cuda(), target.cuda()
            bs = X.size(0)
            if args.optimizer == 'SGDmom':
                epoch_now = epoch + (i + 1) / len(train_batches)
                lr = lr_schedule(epoch_now)
                opt.param_groups[0].update(lr=lr)

            opt.zero_grad()


            if args.attack == 'PGD':
                delta = attack_pgd_divergence(model, X, y, epsilon, pgd_alpha, 
                        args.attack_iters, args.restarts, args.norm, args.divergence, num_classes=num_cla)
            elif args.attack == 'TRADES':
                delta = attack_trades_divergence(model, X, y, epsilon, pgd_alpha, 
                        args.attack_iters, args.restarts, args.norm, args.divergence)
            elif args.attack == 'Standard':
                delta = torch.zeros_like(X)
            
            adv_input = torch.clamp(X + delta, min=lower_limit, max=upper_limit)
            robust_output = F.softmax(model(normalize(adv_input)), dim=1) # logits


            y = (1 - num_cla*args.label_smoothing) * F.one_hot(y, num_classes=num_cla) + args.label_smoothing
            if args.attack == 'TRADES':
                output = F.softmax(model(normalize(X)), dim=1)
                KL_term = distance_func(robust_output, output, args.divergence)
                if args.not_clip_clean:
                    KL_term = F.relu(KL_term - clip_bound)
                robust_loss = distance_func(output, y.float(), args.divergence)
                robust_loss += args.TRADESlambda * KL_term
                if not args.not_clip_clean:
                    robust_loss = F.relu(robust_loss - clip_bound)

            else:
                robust_loss = distance_func(robust_output, y.float(), args.divergence)
                robust_loss = F.relu(robust_loss - clip_bound)
            
            opt.zero_grad()
            robust_loss.mean().backward()
            opt.step()

            if args.optimizer == 'Cosine':
                scheduler.step()

                    
        # Evaluate on test data
        model.eval()
        test_acc = 0
        test_robust_acc_KL = 0
        test_robust_acc_new = 0
        test_n = 0
        for i, (data, target) in enumerate(test_batches):
            X, y = data.cuda(), target.cuda()

            delta_KL = attack_pgd(model, X, y, test_epsilon, test_pgd_alpha, args.attack_iters, args.restarts, args.norm)
            adv_input_KL = torch.clamp(X + delta_KL.detach(), min=lower_limit, max=upper_limit)
            robust_output_KL = model(normalize(adv_input_KL))

            delta_new = attack_pgd_divergence(model, X, y, test_epsilon, test_pgd_alpha, args.attack_iters, args.restarts, 
                args.norm, args.divergence, num_classes=num_cla)
            adv_input_new = torch.clamp(X + delta_new.detach(), min=lower_limit, max=upper_limit)
            robust_output_new = model(normalize(adv_input_new))

            output = model(normalize(X))

            test_robust_acc_KL += (robust_output_KL.max(1)[1] == y).sum().item()
            test_robust_acc_new += (robust_output_new.max(1)[1] == y).sum().item()
            test_acc += (output.max(1)[1] == y).sum().item()
            test_n += y.size(0)


        if True:
            logger.info('%d \t %.4f \t %.4f \t %.4f',
                epoch, test_acc/test_n, test_robust_acc_KL/test_n, test_robust_acc_new/test_n)

            # Save results
            test_acc_record.append(test_acc/test_n)
            test_robust_acc_record_KL.append(test_robust_acc_KL/test_n)
            test_robust_acc_record_new.append(test_robust_acc_new/test_n)

            np.savetxt(args.fname+'/test_acc_record.txt', np.array(test_acc_record))
            np.savetxt(args.fname+'/test_robust_acc_record_KL.txt', np.array(test_robust_acc_record_KL))
            np.savetxt(args.fname+'/test_robust_acc_record_new.txt', np.array(test_robust_acc_record_new))
            
            # save checkpoint
            if epoch > 99:
                torch.save({
                        'state_dict':model.state_dict(),
                        'test_robust_acc_KL':test_robust_acc_KL/test_n,
                        'test_robust_acc_new':test_robust_acc_new/test_n,
                        'test_acc':test_acc/test_n,
                    }, os.path.join(args.fname, f'model_{epoch}.pth'))

            # save best
            if min(test_robust_acc_KL, test_robust_acc_new)/test_n > best_test_robust_acc:
                torch.save({
                        'state_dict':model.state_dict(),
                        'test_robust_acc_KL':test_robust_acc_KL/test_n,
                        'test_robust_acc_new':test_robust_acc_new/test_n,
                        'test_acc':test_acc/test_n,
                    }, os.path.join(args.fname, f'model_best.pth'))
                best_test_robust_acc = min(test_robust_acc_KL, test_robust_acc_new)/test_n



if __name__ == "__main__":
    main()
