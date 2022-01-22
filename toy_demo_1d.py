# CUDA_VISIBLE_DEVICES=6 python toy_demo_1d.py --demo PGDAT --divergence L2
# CUDA_VISIBLE_DEVICES=5 python toy_demo_1d.py --demo PGDATconsistent --divergence KL
# CUDA_VISIBLE_DEVICES=4 python toy_demo_1d.py --demo Standard --divergence KL
# CUDA_VISIBLE_DEVICES=6 python toy_demo_1d.py --demo PGDAT --divergence KL --divergence_C L1


import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import argparse


iteration = 500 # iteration steps for optimization
num_features = 100 # num of feature for the model
lr = 1e-3 # learning rate (Adam optimizer)
half_bs_0 = 50000 # prior for class 0
half_bs_1 = 10000 # prior for class 1
# half_bs_0 = 5 # prior for class 0
# half_bs_1 = 1 # prior for class 1
epsilon = 1 # epsilon for perturbation
attack_iters = 10
alpha = epsilon/attack_iters

G_Mean_0 = -1
G_Mean_1 = 1
G_std_0 = 2
G_std_1 = 1

TRADES_lambda = 1.

np.random.seed(2)
torch.manual_seed(2)
torch.cuda.manual_seed(2)

criterion_kl = nn.KLDivLoss(reduction='batchmean')
criterion_kl_none = nn.KLDivLoss(reduction='none')
def distance_func(output, target, divergence): # both output and target are summed to 1
    M = (output + target) / 2
    if divergence == 'JSsqrt':
        return (0.5 * (criterion_kl(M.log(), output) + criterion_kl(M.log(), target))).sum(dim=-1).sqrt().mean(dim=0)
    elif divergence == 'LSE':
        return ((output - target) ** 2).sum(dim=-1).mean(dim=0)
    elif divergence == 'L2':
        return (torch.norm(output - target, p=2, dim=-1)).mean(dim=0)
    elif divergence == 'L1':
        return (torch.norm(output - target, p=1, dim=-1)).mean(dim=0)
    elif divergence == 'Linf':
        return (torch.norm(output - target, p=float('inf'), dim=-1)).mean(dim=0)
    elif divergence == 'KL':
        return criterion_kl(output.log(), target)

def Softmax(X):
    logit_0 = - (X - G_Mean_0)**2 / (2 * G_std_0**2) - math.log(G_std_0) + math.log(half_bs_0)
    logit_1 = - (X - G_Mean_1)**2 / (2 * G_std_1**2) - math.log(G_std_1) + math.log(half_bs_1)
    logit_all = torch.cat((logit_0, logit_1), dim=1) # 2bs x 2
    return F.softmax(logit_all, dim=1)

# compute C
def compute_C(X, divergence, epsilon=epsilon, alpha=alpha, attack_iters=attack_iters):
    delta = torch.zeros_like(X) # 2bs x 1
    delta.uniform_(-epsilon, epsilon)
    delta.requires_grad = True
    for _ in range(attack_iters):
        loss = distance_func(Softmax(X + delta), Softmax(X), divergence)
        loss.backward()
        grad = delta.grad.detach()
        delta.data = torch.clamp(delta + alpha * torch.sign(grad), min=-epsilon, max=epsilon)
        delta.grad.zero_()
    return delta.detach()

# solve R_{Madry}
def attack_pgd(model, X, y, divergence, epsilon=epsilon, alpha=alpha, attack_iters=attack_iters):
    delta = torch.zeros_like(X) # 2bs x 1
    delta.uniform_(-epsilon, epsilon)
    delta.requires_grad = True
    for _ in range(attack_iters):
        output = F.softmax(model(X + delta), dim=1)
        loss = distance_func(output, y, divergence)
        loss.backward()
        grad = delta.grad.detach()
        delta.data = torch.clamp(delta + alpha * torch.sign(grad), min=-epsilon, max=epsilon)
        delta.grad.zero_()
    return delta.detach()

# solve R_{SCORE}
def attack_pgd_consistent(model, X, divergence, epsilon=epsilon, alpha=alpha, attack_iters=attack_iters):
    delta = torch.zeros_like(X) # 2bs x 1
    delta.uniform_(-epsilon, epsilon)
    delta.requires_grad = True
    for _ in range(attack_iters):
        output = F.softmax(model(X + delta), dim=1)
        loss = distance_func(output, Softmax(X + delta), divergence)
        loss.backward()
        grad = delta.grad.detach()
        delta.data = torch.clamp(delta + alpha * torch.sign(grad), min=-epsilon, max=epsilon)
        delta.grad.zero_()
    return delta.detach()

def attack_trades(model, X, y, divergence, epsilon=epsilon, alpha=alpha, attack_iters=attack_iters):
    delta = torch.zeros_like(X) # 2bs x 1
    delta.uniform_(-epsilon, epsilon)
    delta.requires_grad = True
    clean_output = model(X)
    clean_output = F.softmax(clean_output.detach(), dim=1)
    for _ in range(attack_iters):
        output = F.softmax(model(X + delta), dim=1)
        loss = distance_func(output, clean_output, divergence)
        loss.backward()
        grad = delta.grad.detach()
        delta.data = torch.clamp(delta + alpha * torch.sign(grad), min=-epsilon, max=epsilon)
        delta.grad.zero_()
    return delta.detach()

class Small_NN(nn.Module):
    def __init__(self, num_features=100):
        super(Small_NN, self).__init__()
        self.model = nn.Sequential(
          nn.Linear(1, num_features),
          nn.Tanh(),
          nn.Linear(num_features, num_features),
          nn.Tanh(),
          nn.Linear(num_features, 2),
        )
    def forward(self, x):
        return self.model(x)

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--demo', default='PGDconsistent', type=str)
    parser.add_argument('--divergence', default='KL', type=str)
    parser.add_argument('--divergence_C', default='KL', type=str)
    return parser.parse_args()

def main():
    args = get_args()
    P_0 = torch.distributions.normal.Normal(G_Mean_0, G_std_0)
    P_1 = torch.distributions.normal.Normal(G_Mean_1, G_std_1)

    model = Small_NN(num_features=num_features).cuda()
    model.train()
    params = model.parameters()
    opt = torch.optim.Adam(params, lr=lr, weight_decay=0)

    # compute C^{L1}
    C_L = 0
    iter_C = 10
    for _ in range(iter_C):
        Sample_0 = P_0.sample(sample_shape=torch.Size([half_bs_0, 1])) # bs x 1
        Sample_1 = P_1.sample(sample_shape=torch.Size([half_bs_1, 1])) # bs x 1
        Sample_all = torch.cat((Sample_0, Sample_1), dim=0).cuda() # 2bs x 1
        delta = compute_C(Sample_all, args.divergence)
        C_L += distance_func(Softmax(Sample_all + delta), Softmax(Sample_all), args.divergence)
    C_L /= iter_C

    record_train_loss = []
    record_SCORE_loss = []
    record_Standard_loss = []
    # train model
    for ite in range(iteration):
        Sample_0 = P_0.sample(sample_shape=torch.Size([half_bs_0, 1])) # bs x 1
        Sample_1 = P_1.sample(sample_shape=torch.Size([half_bs_1, 1])) # bs x 1
        Sample_all = torch.cat((Sample_0, Sample_1), dim=0).cuda() # 2bs x 1

        opt.zero_grad()

        if args.demo == 'PGDAT':
            y_all = Softmax(Sample_all)
            delta = attack_pgd(model, Sample_all, y_all, args.divergence)
            output = F.softmax(model(Sample_all + delta), dim=1) # 2bs x 1
            robust_loss = distance_func(output, y_all, args.divergence)

        elif args.demo == 'PGDATconsistent':
            delta = attack_pgd_consistent(model, Sample_all, args.divergence)
            output = F.softmax(model(Sample_all + delta), dim=1) # 2bs x 1
            y_all = Softmax(Sample_all + delta)
            robust_loss = distance_func(output, y_all, args.divergence)

        elif args.demo == 'TRADES':
            y_all = Softmax(Sample_all)
            delta = attack_trades(model, Sample_all, y_all, args.divergence)
            clean_output = F.softmax(model(Sample_all), dim=1) # 2bs x 1
            output = F.softmax(model(Sample_all + delta), dim=1) # 2bs x 1
            robust_loss = distance_func(clean_output, y_all, args.divergence)
            robust_loss += TRADES_lambda * distance_func(output, y_all, args.divergence)

        elif args.demo == 'Standard':
            y_all = Softmax(Sample_all)
            output = F.softmax(model(Sample_all), dim=1) # 2bs x 1
            robust_loss = distance_func(output, y_all, args.divergence)

        opt.zero_grad()
        robust_loss.backward()
        opt.step()

        if ite % 1 == 0:   
            print('ite: ', ite)
            print('Train loss: ', robust_loss.item())
            record_train_loss += [robust_loss.cpu().item()]

            # compute R_{SCORE}
            opt.zero_grad()
            delta = attack_pgd_consistent(model, Sample_all, args.divergence_C)
            R_SCORE = distance_func(F.softmax(model(Sample_all + delta), dim=1), 
                Softmax(Sample_all + delta), args.divergence_C)
            print('R_SCORE loss: ', R_SCORE.item())
            record_SCORE_loss += [R_SCORE.cpu().item()]

            # compute R_{Standard}
            standard_loss = distance_func(F.softmax(model(Sample_all), dim=1), 
                Softmax(Sample_all), args.divergence_C)
            print('R_Standard loss: ', standard_loss.item())
            record_Standard_loss += [standard_loss.cpu().item()]

            print('diff: ', R_SCORE.item() - standard_loss.item())
            print('***')

        # # save intermediate states (for toy demo Figure 1)
        # if ite % 10 == 0 and ite < 301:
        #     model.eval()
        #     U = torch.distributions.uniform.Uniform(-10, 10)
        #     x_test = U.sample(sample_shape=torch.Size([10000, 1])).cuda()
        #     y_test = Softmax(x_test)
        #     pre_test = F.softmax(model(x_test).detach(), dim=1)
        #     np.savetxt('toy_results/' + args.demo + '_' + args.divergence + '/x_test_' + str(ite) + '.txt', x_test.cpu().numpy())
        #     np.savetxt('toy_results/' + args.demo + '_' + args.divergence + '/y_test_' + str(ite) + '.txt', y_test.cpu().numpy())
        #     np.savetxt('toy_results/' + args.demo + '_' + args.divergence + '/pre_test_' + str(ite) + '.txt', pre_test.cpu().numpy())
        #     model.train()

    model.eval()
    U = torch.distributions.uniform.Uniform(-10, 10)
    x_test = U.sample(sample_shape=torch.Size([10000, 1])).cuda()
    y_test = Softmax(x_test)
    pre_test = F.softmax(model(x_test).detach(), dim=1)
    np.savetxt('toy_results/' + args.demo + '_' + args.divergence + '/x_test.txt', x_test.cpu().numpy())
    np.savetxt('toy_results/' + args.demo + '_' + args.divergence + '/y_test.txt', y_test.cpu().numpy())
    np.savetxt('toy_results/' + args.demo + '_' + args.divergence + '/pre_test.txt', pre_test.cpu().numpy())
    np.savetxt('toy_results/' + args.demo + '_' + args.divergence + '/record_train_loss.txt', np.array(record_train_loss))
    np.savetxt('toy_results/' + args.demo + '_' + args.divergence + '/record_SCORE_loss.txt', np.array(record_SCORE_loss))
    np.savetxt('toy_results/' + args.demo + '_' + args.divergence + '/record_Standard_loss.txt', np.array(record_Standard_loss))
                
    print('C_L: ', C_L.item())
    #print('C_L: ', C_L.item()**2 / 2)

if __name__ == "__main__":
    main()