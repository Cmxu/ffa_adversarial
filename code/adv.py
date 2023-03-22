import os

import torch
import torch.nn.functional as F
import numpy as np

from adversarial.adv import fgsm, pgd

from ffa.ffa import load_mnist, image_labeler

from ffa.ffa_v2 import Net
from models.linear import LinearNet
from models.conv import ConvNet
from utils.utils import asr, accuracy, project_lp, scale_im
import yaml
import matplotlib.pyplot as plt


def load_weights(model, weights_path, device=torch.device("cuda")):
    for idx, layer in enumerate(model.layers):
        layer.load_state_dict(torch.load(f"{weights_path}/ffa_{idx}.pth", map_location=device))
    model.linear.load_state_dict(torch.load(f"{weights_path}/linear.pth", map_location=device))


def fgsm_pgd(net, name, device=torch.device("cpu")):
    fgsm_pert = fgsm(x, y, net, device = device)
    print(f'{name} FGSM Batch Accuracy: {accuracy(net.forward(x + fgsm_pert), y)}')

    pgd_pert = pgd(x, y, net, device = device)
    print(f'{name} PGD Batch Accuracy: {accuracy(net.forward(x + pgd_pert), y)}')

    return fgsm_pert, pgd_pert


def saliency_map(input, gt, model, file_name):
    input.requires_grad = True
    out = model.forward(input)
    loss = F.cross_entropy(out, gt)
    loss.backward()

    os.makedirs(f"saliency_maps/{file_name}", exist_ok=True)

    for i in range(10):
        img_idx = (gt == i).nonzero(as_tuple=False)[0].item()

        gradient = input.grad[img_idx,:].cpu().detach().reshape((28,28)).numpy()
        gradient = scale_im(gradient)
        
        plt.imsave(f"saliency_maps/{file_name}/{i}.png", gradient)


if __name__ == "__main__":

    config = yaml.load(open("config.yml"), Loader=yaml.FullLoader)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(f'Using {device}')

    # Load dataset
    train_loader,test_loader = load_mnist(config["batch_size"]["train"],
                                            config["batch_size"]["test"],
                                            device,
                                            dataset_path=config["dataset_path"])
    #Initial Test on Small Batch
    x, y = next(iter(test_loader))
    x = x.view(x.shape[0], -1)
    x = image_labeler(x, y, test=True)

    if device==torch.device("cuda"):
        x,y = x.cuda(), y.cuda()


    base_model_path = config["base_path"] + "/models"


    # FFANet Base
    ffa_net = Net([784, 100, 100, 100, 100]).to(device)
    weight_path = f"{base_model_path}/renewed"
    load_weights(ffa_net, weight_path, device)
    print('FFA Base Batch Accuracy {}'.format(accuracy(ffa_net.forward(x), y)))

    # # saliency_map
    saliency_map(x, y, ffa_net, "ffa")
    ffa_fgsm_pert, ffa_pgd_pert = fgsm_pgd(ffa_net, "FFA", device)


    # LinearNet Base
    epoch = 19
    linear_net = LinearNet().to(device)
    weight_path = f"{base_model_path}/linear/epoch{epoch}"
    load_weights(linear_net, weight_path, device)
    print('Linear Base Batch Accuracy {}'.format(accuracy(linear_net.forward(x), y)))

    x.grad.zero_()
    saliency_map(x, y, linear_net, "linear")
    linear_fgsm_pert, linear_pgd_pert = fgsm_pgd(linear_net, "Linear", device)


    # ConvNet Base
    epoch = 3
    conv_net = ConvNet().to(device)
    weight_path = f"{base_model_path}/conv/epoch{epoch}"
    conv_net.load_state_dict(torch.load(f"{weight_path}/conv.pth", map_location=device))
    print('Conv Base Batch Accuracy {}'.format(accuracy(conv_net.forward(x), y)))

    x.grad.zero_()
    saliency_map(x, y, conv_net, "conv")
    conv_fgsm_pert, conv_pgd_pert = fgsm_pgd(conv_net, "Conv", device)



    print('FFA with Linear FGSM Batch Accuracy: {}'.format(accuracy(ffa_net.forward(x + linear_fgsm_pert), y)))
    print('FFA with Conv FGSM Batch Accuracy: {}'.format(accuracy(ffa_net.forward(x + conv_fgsm_pert), y)))

    print('Linear with FFA FGSM Batch Accuracy: {}'.format(accuracy(linear_net.forward(x + ffa_fgsm_pert), y)))
    print('Linear with Conv FGSM Batch Accuracy: {}'.format(accuracy(linear_net.forward(x + conv_fgsm_pert), y)))

    print('Conv with FFA FGSM Batch Accuracy: {}'.format(accuracy(conv_net.forward(x + ffa_fgsm_pert), y)))
    print('Conv with Linear FGSM Batch Accuracy: {}'.format(accuracy(conv_net.forward(x + linear_fgsm_pert), y)))

    print('FFA with Linear PGD Batch Accuracy: {}'.format(accuracy(ffa_net.forward(x + linear_pgd_pert), y)))
    print('FFA with Conv PGD Batch Accuracy: {}'.format(accuracy(ffa_net.forward(x + conv_pgd_pert), y)))

    print('Linear with FFA PGD Batch Accuracy: {}'.format(accuracy(linear_net.forward(x + ffa_pgd_pert), y)))
    print('Linear with Conv PGD Batch Accuracy: {}'.format(accuracy(linear_net.forward(x + conv_pgd_pert), y)))

    print('Conv with FFA PGD Batch Accuracy: {}'.format(accuracy(conv_net.forward(x + ffa_pgd_pert), y)))
    print('Conv with Linear PGD Batch Accuracy: {}'.format(accuracy(conv_net.forward(x + linear_pgd_pert), y)))

    # # # fgsm_np = np.reshape((fgsm_pert.cpu().detach().numpy())[0], [28, 28])
    # # # image_np = np.reshape((x.cpu().detach().numpy())[0], [28, 28])
    # # # pert_np = np.reshape(((x + fgsm_pert).cpu().detach().numpy())[0], [28, 28])

    # # # plt.imsave("fgsm_np.png", fgsm_np, cmap='binary')
    # # # plt.imsave("image_np.png", image_np, cmap='binary')
    # # # plt.imsave("pert_np.png", pert_np, cmap='binary')