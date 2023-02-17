# Adapted from: https://github.com/zjfheart/Friendly-Adversarial-Training/blob/master/attack_generator.py

import numpy as np
from models import *
import random

def cwloss(output, target,confidence=50, num_classes=10):
    # Compute the probability of the label class versus the maximum other
    # The same implementation as in repo CAT https://github.com/sunblaze-ucb/curriculum-adversarial-training-CAT
    target = target.data
    target_onehot = torch.zeros(target.size() + (num_classes,))
    target_onehot = target_onehot.cuda()
    target_onehot.scatter_(1, target.unsqueeze(1), 1.)
    target_var = Variable(target_onehot, requires_grad=False)
    real = (target_var * output).sum(1)
    other = ((1. - target_var) * output - target_var * 10000.).max(1)[0]
    loss = -torch.clamp(real - other + confidence, min=0.)  # equiv to max(..., 0.)
    loss = torch.sum(loss)
    return loss

def pgd(model, data, target, epsilon, step_size, num_steps,loss_fn,category,rand_init, class_num):
    model.eval()
    if category == "trades":
        x_adv = data.detach() + 0.001 * torch.randn(data.shape).cuda().detach() if rand_init else data.detach()
    if category == "Madry":
        x_adv = data.detach() + torch.from_numpy(np.random.uniform(-epsilon, epsilon, data.shape)).float().cuda() if rand_init else data.detach()
        x_adv = torch.clamp(x_adv, 0.0, 1.0)
    for k in range(num_steps):
        x_adv.requires_grad_()
        output = model(x_adv)
        model.zero_grad()
        with torch.enable_grad():
            if loss_fn == "cent":
                loss_adv = nn.CrossEntropyLoss(reduction="mean")(output, target)
            if loss_fn == "cw":
                loss_adv = cwloss(output,target, num_classes=class_num)
        loss_adv.backward()
        eta = step_size * x_adv.grad.sign()
        x_adv = x_adv.detach() + eta
        x_adv = torch.min(torch.max(x_adv, data - epsilon), data + epsilon)
        x_adv = torch.clamp(x_adv, 0.0, 1.0)
    model.zero_grad()
    return x_adv.detach()

def trades(model, data, target, epsilon, step_size, num_steps,loss_fn,category,rand_init):
    model.eval()
    if category == "trades":
        x_adv = data.detach().clone() + 0.001 * torch.randn(data.shape).cuda() if rand_init else data.detach()
    if category == "Madry":
        x_adv = data.detach() + torch.from_numpy(np.random.uniform(-epsilon, epsilon, data.shape)).float().cuda() if rand_init else data.detach()
        x_adv = torch.clamp(x_adv, 0.0, 1.0)
    for k in range(num_steps):
        x_adv.requires_grad_()
        output = model(x_adv)
        model.zero_grad()
        with torch.enable_grad():
            if loss_fn == "cent":
                loss_adv = nn.CrossEntropyLoss(reduction="mean")(output, target)
            elif loss_fn == 'trades':
                loss_adv = nn.KLDivLoss(size_average=False)(F.log_softmax(output, dim=1),
                                       F.softmax(model(data.detach().clone()), dim=1))

        loss_adv.backward()
        eta = step_size * x_adv.grad.sign()
        x_adv = x_adv.detach() + eta
        x_adv = torch.min(torch.max(x_adv, data - epsilon), data + epsilon)
        x_adv = torch.clamp(x_adv, 0.0, 1.0)
    model.zero_grad()
    return x_adv.detach()

def eval_clean(model, test_loader, class_num=10):
    model.eval()
    test_loss = 0
    correct = 0
    class_wise_correct = []
    class_wise_num = []
    for i in range(class_num):
        class_wise_correct.append(0)
        class_wise_num.append(0)
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.cuda(), target.cuda()
            output = model(data)
            test_loss += nn.CrossEntropyLoss(reduction='mean')(output, target).item()
            pred = output.max(1, keepdim=True)[1]
            target = target.view_as(pred)
            eq_mat = pred.eq(target)
            for i in range(class_num):
                class_wise_num[i] += (target == i).int().sum().item()
                class_wise_correct[i] += eq_mat[target == i].sum().item()
            correct += pred.eq(target.view_as(pred)).sum().item()
    test_loss /= len(test_loader.dataset)
    test_accuracy = correct / len(test_loader.dataset)    
    for i in range(class_num): 
        class_wise_correct[i] /= class_wise_num[i]
    return test_loss, test_accuracy, class_wise_correct

def eval_robust(model, test_loader, perturb_steps, epsilon, step_size, loss_fn, category, rand_init, class_num=10):
    model.eval()
    test_loss = 0
    correct = 0
    class_wise_correct = []
    class_wise_num = []
    for i in range(class_num):
        class_wise_correct.append(0)
        class_wise_num.append(0)
    # with torch.no_grad():
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.cuda(), target.cuda()
            with torch.enable_grad():
                x_adv = pgd(model,data,target,epsilon,step_size,perturb_steps,loss_fn,category,rand_init=rand_init, class_num=class_num)
            output = model(x_adv)
            test_loss += nn.CrossEntropyLoss(reduction='mean')(output, target).item()
            pred = output.max(1, keepdim=True)[1]
            target = target.view_as(pred)
            eq_mat = pred.eq(target)
            for i in range(class_num):
                class_wise_num[i] += (target == i).int().sum().item()
                class_wise_correct[i] += eq_mat[target == i].sum().item()
            correct += pred.eq(target.view_as(pred)).sum().item()
    test_loss /= len(test_loader.dataset)
    test_accuracy = correct / len(test_loader.dataset)
    
    for i in range(class_num):
        class_wise_correct[i] /= class_wise_num[i]
    return test_loss, test_accuracy, class_wise_correct