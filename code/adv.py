import torch
import torch.nn.functional as F
# from matplotlib import pyplot as plt
import numpy as np
# import sys
from ffa.ffa import load_dataset, image_labeler
from ffa.ffa import FFANet
from utils.utils import asr, accuracy

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

train_loader, val_loader = load_dataset()

mdl = FFANet()
# mdl.load_state_dict(torch.load('./epoch2_acc0.86.pth'))
mdl.load_state_dict(torch.load('./epoch1_acc0.77.pth'))
# mdl.load_state_dict(torch.load('./models/epoch1_acc0.48.pth'))

def fgsm(x, y, model, norm = np.inf, xi = 1e-1, step_size = 1e-1, device = "cpu"):
    v = torch.zeros_like(x, requires_grad = True, device = device)

    loss = F.cross_entropy(model(x + v), y)
    loss.backward()

    return step_size * v.grad.sign()

#Initial Test on Small Batch
x, y = next(iter(val_loader))
# x, y = next(iter(train_loader))
x = x.view(x.shape[0], -1)
x = image_labeler(x, y, test=True)

print('Base Batch Accuracy {}'.format(accuracy(mdl.forward(x), y))) # Varies with batch, mine ~ 0.875
print('FGSM Batch Accuracy: {}'.format(accuracy(mdl.forward(x + fgsm(x, y, mdl)), y))) # Varies with batch, mine ~ 0