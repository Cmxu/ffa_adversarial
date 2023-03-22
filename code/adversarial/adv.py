import torch
import torch.nn.functional as F
from matplotlib import pyplot as plt
import numpy as np
import sys
# from data_utils import toDeviceDataLoader, load_cifar, to_device
# from model_utils import VGG
from utils.utils import asr, accuracy, project_lp

# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# dataset_root = '<insert path>'
# cifar10_train, cifar10_val, cifar10_test = load_cifar(dataset_root)
# train_loader, val_loader, test_loader = toDeviceDataLoader(cifar10_train, cifar10_val, cifar10_test, device = device)

# mdl = to_device(VGG('VGG16'), device)
# mdl.load_state_dict(torch.load('../models/torch_cifar_vgg.pth'))
# mdl = mdl.eval()

def fgsm(x, y, model, norm = np.inf, xi = 1e-1, step_size = 1e-1, device = torch.device('cuda:0')):
    v = torch.zeros_like(x, requires_grad = True, device = device)

    loss = F.cross_entropy(model(x + v), y)
    loss.backward()

    return step_size * v.grad.sign()

def pgd(x, y, k, norm = np.inf, xi = 1e-1, step_size = 1e-2, epochs = 40, random_restart = 4, device = torch.device('cuda:0')):
    batch_size = x.shape[0]
    max_loss = F.cross_entropy(k(x), y)
    max_X = torch.zeros_like(x)
    random_delta = torch.rand(size = (batch_size * random_restart, *x.shape[1:]), device = device) - 0.5
    random_delta = project_lp(random_delta, norm = norm, xi = xi, exact = True, device = device)
    x = x.repeat(random_restart, 1)
    y = y.repeat(random_restart)
    for j in range(epochs):
        v = torch.zeros_like(random_delta, device = device, requires_grad = True)
        # print(x.shape, random_delta.shape, v.shape, y.shape)
        loss = F.cross_entropy(k(x + random_delta + v), y)
        loss.backward()
        pert = step_size * torch.sign(v.grad)#torch.mean(v.grad)
        random_delta = project_lp(random_delta + pert, norm = norm, xi = xi)
    _,idx = torch.max(F.cross_entropy(k(x + random_delta), y, reduction = 'none').reshape(random_restart, batch_size), axis = 0)
    return random_delta[idx * batch_size + torch.arange(batch_size, dtype = torch.int64, device = device)]

# #Initial Test on Small Batch
# x, y = next(iter(test_loader))
# print('Base Batch Accuracy {}'.format(accuracy(mdl(x), y))) # Varies with batch, mine ~ 0.875
# print('FGSM Batch Accuracy: {}'.format(accuracy(mdl(x + fgsm(x, y, mdl)), y))) # Varies with batch, mine ~ 0
# print('PGD Batch Accuracy: {}'.format(accuracy(mdl(x + pgd(x, y, mdl)), y))) # Varies with batch, mine ~ 0

# v = pgd(x, y, mdl)
# show_attack(x, v, mdl)

# #Test on Entire Dataset (this will take a few minutes depending on how many epochs of pgd you have)
# print('Base Accuracy: {}'.format(1 - asr(test_loader, mdl))) # ~ 0.9171
# print('FGSM Accuracy: {}'.format(1 - asr(test_loader, mdl, fgsm))) # ~ 0.0882
# print('PGD Accuracy: {}'.format(1 - asr(test_loader, mdl, pgd))) # ~ 0.0001


