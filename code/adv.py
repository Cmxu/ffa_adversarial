import torch
import torch.nn.functional as F
import numpy as np
from ffa.ffa import load_mnist, image_labeler

from ffa.what import Net
from models.linear import LinearNet
from models.conv import ConvNet
from utils.utils import asr, accuracy, project_lp, scale_im
import yaml
import matplotlib.pyplot as plt


def load_weights(model, weights_path):
    for idx, layer in enumerate(model.layers):
        layer.load_state_dict(torch.load(f"{weights_path}/ffa_{idx}.pth"))
    model.linear.load_state_dict(torch.load(f"{weights_path}/linear.pth"))


def fgsm(x, y, model, norm = np.inf, xi = 1e-1, step_size = 1e-1, device = "cpu"):
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


def saliency_map(input, model, file_name):
    input.requires_grad = True
    input.grad.zero_()
    out = model.forward(input)
    loss = F.cross_entropy(out, y)
    loss.backward()

    gradient = input.grad[0,:].cpu().detach().reshape((28,28)).numpy()
    gradient = scale_im(gradient)
    plt.imsave(f"{file_name}.png", gradient)

if __name__ == "__main__":

    config = yaml.load(open("config.yml"), Loader=yaml.FullLoader)

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(f'Using {device}')


    train_loader,test_loader = load_mnist(config["batch_size"]["train"],
                                            config["batch_size"]["test"],
                                            device,
                                            dataset_path=config["dataset_path"])
    #Initial Test on Small Batch
    x, y = next(iter(test_loader))
    # x, y = next(iter(train_loader))
    x = x.view(x.shape[0], -1)
    x = image_labeler(x, y, test=True)
    x,y = x.cuda(), y.cuda()



    # FFANet Base
    ffa_net = Net([784, 100, 100, 100, 100]).to(device)
    weight_path = f"/home/shchoi4/ffa_adversarial/models/renewed"
    load_weights(ffa_net, weight_path)
    print('FFA Base Batch Accuracy {}'.format(accuracy(ffa_net.forward(x), y)))

    # saliency_map

    # FFANet FGSM
    ffa_fgsm_pert = fgsm(x, y, ffa_net, device = device)
    print('FFA FGSM Batch Accuracy: {}'.format(accuracy(ffa_net.forward(x + ffa_fgsm_pert), y)))

    ffa_pgd_pert = pgd(x, y, ffa_net, device = device)
    print('FFA PGD Batch Accuracy: {}'.format(accuracy(ffa_net.forward(x + ffa_pgd_pert), y)))


    x.requires_grad = True
    out = ffa_net.forward(x)
    loss = F.cross_entropy(out, y)
    loss.backward()

    gradient = x.grad[0,:].cpu().detach().reshape((28,28)).numpy()
    gradient = scale_im(gradient)
    plt.imsave("ffa_saliency.png", gradient)




    epoch = 19
    
    # # FFANet Base
    # ffa_net = FFANet().to(device)
    # weight_path = f"/home/shchoi4/ffa_adversarial/models/ffa/epoch{epoch}"
    # load_weights(ffa_net, weight_path)
    # print('FFA Base Batch Accuracy {}'.format(accuracy(ffa_net.forward(x), y)))
    # # FFANet FGSM
    # fgsm_pert = fgsm(x, y, ffa_net, device = device)
    # print('FFA FGSM Batch Accuracy: {}'.format(accuracy(ffa_net.forward(x + fgsm_pert), y)))

    # LinearNet Base
    linear_net = LinearNet().to(device)
    weight_path = f"/home/shchoi4/ffa_adversarial/models/linear/epoch{epoch}"
    load_weights(linear_net, weight_path)
    print('Linear Base Batch Accuracy {}'.format(accuracy(linear_net.forward(x), y)))
    # LinearNet FGSM
    linear_pert = fgsm(x, y, linear_net, device = device)
    print('Linear FGSM Batch Accuracy: {}'.format(accuracy(linear_net.forward(x + linear_pert), y)))

    linear_pgd_pert = pgd(x, y, linear_net, device = device)

    
    print('Linear PGD Batch Accuracy: {}'.format(accuracy(linear_net.forward(x + linear_pgd_pert), y)))

    x.grad.zero_()
    out = linear_net.forward(x)
    loss = F.cross_entropy(out, y)
    loss.backward()

    gradient = x.grad[0,:].cpu().detach().reshape((28,28)).numpy()
    gradient = scale_im(gradient)
    plt.imsave("linear_saliency.png", gradient)


    epoch = 3
    # ConvNet Base
    conv_net = ConvNet().to(device)
    weight_path = f"/home/shchoi4/ffa_adversarial/models/conv/epoch{epoch}"
    # load_weights(conv_net, weight_path)
    conv_net.load_state_dict(torch.load(f"{weight_path}/conv.pth"))
    print('Conv Base Batch Accuracy {}'.format(accuracy(conv_net.forward(x), y)))
    # ConvNet FGSM
    conv_pert = fgsm(x, y, conv_net, device = device)
    print('Conv FGSM Batch Accuracy: {}'.format(accuracy(conv_net.forward(x + conv_pert), y)))

    conv_pgd_pert = pgd(x, y, conv_net, device = device)

    
    print('Conv PGD Batch Accuracy: {}'.format(accuracy(conv_net.forward(x + conv_pgd_pert), y)))

    x.grad.zero_()
    out = conv_net.forward(x)
    loss = F.cross_entropy(out, y)
    loss.backward()

    gradient = x.grad[0,:].cpu().detach().reshape((28,28)).numpy()
    gradient = scale_im(gradient)
    plt.imsave("conv_saliency.png", gradient)


    # print('FFA with Linear FGSM Batch Accuracy: {}'.format(accuracy(ffa_net.forward(x + linear_pert), y)))
    # print('Linear with FFA FGSM Batch Accuracy: {}'.format(accuracy(linear_net.forward(x + fgsm_pert), y)))










    # fgsm_np = np.reshape((fgsm_pert.cpu().detach().numpy())[0], [28, 28])
    # image_np = np.reshape((x.cpu().detach().numpy())[0], [28, 28])
    # pert_np = np.reshape(((x + fgsm_pert).cpu().detach().numpy())[0], [28, 28])

    # plt.imsave("fgsm_np.png", fgsm_np, cmap='binary')
    # plt.imsave("image_np.png", image_np, cmap='binary')
    # plt.imsave("pert_np.png", pert_np, cmap='binary')