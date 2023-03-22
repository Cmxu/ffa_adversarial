import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.optim import Adam
from torchvision.datasets import MNIST
from torchvision.transforms import Compose, ToTensor, Normalize, Lambda
from torch.utils.data import DataLoader


def load_mnist(batch_size_train, batch_size_test, device, dataset_path="./"):
    transform = Compose([
        ToTensor(),
        Normalize((0.1307,), (0.3081,)),
        Lambda(lambda x: torch.flatten(x))])

    train_dataset = MNIST(dataset_path,
                            download=True,
                            train=True,
                            transform=transform)
    test_dataset = MNIST(dataset_path,
                            download=True,
                            train=False,
                            transform=transform)
    if device==torch.device("cuda"):
        train_loader = torch.utils.data.DataLoader(train_dataset,
                                                    batch_size=batch_size_train,
                                                    shuffle=True,
                                                    drop_last=True,
                                                    num_workers=4,
                                                    pin_memory=True)
        test_loader = torch.utils.data.DataLoader(test_dataset,
                                                    batch_size=batch_size_test,
                                                    shuffle=True,
                                                    num_workers=4,
                                                    pin_memory=True)
    else:
        train_loader = torch.utils.data.DataLoader(train_dataset,
                                                    batch_size=batch_size_train,
                                                    shuffle=True,
                                                    drop_last=True,)
        test_loader = torch.utils.data.DataLoader(test_dataset,
                                                    batch_size=batch_size_test,
                                                    shuffle=True,)

    return train_loader, test_loader


def image_labeler(img, label=1, test=False):
    img_ = img.clone()
    if test:
        img_[:,:10] = 0.1
        # img_[:,:10] = img.max()/10
    else:
        img_[:,:10] *= 0.0
        img_[range(img.shape[0]),(label%10)] = img.max()
    return img_


class Net(torch.nn.Module):
    def __init__(self, input_size=784, output_size=10, num_hidden = 4, hidden_size=100):
        super().__init__()
        self.layers = []
        for l in range(num_hidden):
            if (l==0):
                self.layers += [Layer(input_size, hidden_size)]
                # self.layers += [Layer(input_size, hidden_size).cuda()]
            else:
                self.layers += [Layer(hidden_size, hidden_size)]
                # self.layers += [Layer(hidden_size, hidden_size).cuda()]
        self.linear = torch.nn.Linear(hidden_size, output_size)

        self.loss = torch.nn.CrossEntropyLoss()
        self.opt = Adam(self.parameters(), lr=0.03, weight_decay=0.1)
        self.num_epochs = 1000

    def forward(self, x):
        layer_out = []
        output = image_labeler(x, test=True)

        for i, layer in enumerate(self.layers):
            output = layer.forward(output)
            if i==0:    continue
            layer_out.append(output)
        layer_out = torch.stack(layer_out)
        return self.linear(torch.mean(layer_out, dim=0))

    def train(self, x_pos, x_neg, label):
        linear_input = []
        h_pos, h_neg = x_pos, x_neg
        for i, layer in enumerate(self.layers):
            print('training layer', i, '...')
            h_pos, h_neg = layer.train(h_pos, h_neg, self.num_epochs)
            if i==0:    continue
            linear_input.append(h_pos)
        linear_input = torch.stack(linear_input)

        # train final linear layer
        linear_losses = []
        for _ in tqdm(range(self.num_epochs)):
            pred = self.linear(torch.mean(linear_input, dim=0))
            pos_loss = self.loss(pred, label)
            self.opt.zero_grad()
            pos_loss.backward()
            self.opt.step()

            linear_losses.append(pos_loss.cpu().detach().numpy())

        import numpy as np
        plt.plot()
        plt.title(f"Train Accuracy")
        plt.ylabel("Accuracy")
        plt.xlabel("Number of Epochs")
        plt.xticks(np.arange(1, self.num_epochs+1, 1.0))
        plt.plot(linear_losses)
        plt.savefig("softmax loss.png", dpi=200)

        return self.linear(torch.mean(linear_input, dim=0))


class Layer(torch.nn.Module):
    def __init__(self, in_features, out_features):
        super(Layer, self).__init__()
        self.linear = torch.nn.Linear(in_features, out_features)
        self.relu = torch.nn.ReLU()
        self.optim = Adam(self.parameters(), lr=0.03)
        self.threshold = 2.0

    def forward(self, x):
        x_direction = x / (x.norm(2, 1, keepdim=True) + 1e-4)
        return self.relu(self.linear(x_direction))

    def train(self, x_pos, x_neg, num_epochs):
        for _ in tqdm(range(num_epochs)):
            g_pos = self.forward(x_pos).pow(2).mean(1)
            g_neg = self.forward(x_neg).pow(2).mean(1)
            
            loss = torch.log(1 + torch.exp(torch.cat([
                -g_pos + self.threshold,
                g_neg - self.threshold]))).mean()

            self.optim.zero_grad()
            loss.backward()
            self.optim.step()
        return self.forward(x_pos).detach(), self.forward(x_neg).detach()

    
def visualize_sample(data, name='', idx=0):
    reshaped = data[idx].cpu().reshape(28, 28)
    plt.figure(figsize = (4, 4))
    plt.title(name)
    plt.imshow(reshaped, cmap="gray")
    plt.show()
    
    
if __name__ == "__main__":
    # torch.manual_seed(1234)
    torch.manual_seed(3407)

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(f'Using {device}')

    batch_size_train = 50000
    batch_size_test = 10000
    train_loader,test_loader = load_mnist(batch_size_train,
                                            batch_size_test,
                                            device,
                                            dataset_path="./MNIST")

    net = Net().to(device)

    # Training
    pred_cnt = 0
    pbar = tqdm(train_loader)
    for batch_idx, (data, target) in enumerate(pbar):
        x, y = data, target

        if device==torch.device("cuda"):
            x, y = x.cuda(), y.cuda()
        x_pos = image_labeler(x, y)
        rnd = torch.randperm(x.size(0))
        x_neg = image_labeler(x, y[rnd])

        net.train(x_pos, x_neg, y)

        pred = net.forward(x)
        pred = torch.nn.functional.softmax(pred, dim=1).argmax(1)
        batch_acc = pred.eq(y).float().mean().item()

        # Calculate accuracy
        pred_cnt += batch_acc
        pbar.set_postfix({'acc': pred_cnt/(batch_idx + 1), 'batch_acc': batch_acc})

    pred_acc = float(pred_cnt)/len(train_loader)
    print('train error:', 1.-pred_acc)


    # # save weights
    # import os
    # model_base_path = f"/home/shchoi4/ffa_adversarial/models/renewed"
    # print(model_base_path)
    # os.makedirs(model_base_path, exist_ok=True)

    # for idx, layer in enumerate(net.layers):
    #     torch.save(layer.state_dict(), f"{model_base_path}/ffa_{idx}.pth")
    # torch.save(net.linear.state_dict(), f"{model_base_path}/linear.pth")


    # test
    pred_cnt = 0
    pbar = tqdm(test_loader)
    for batch_idx, (data, target) in enumerate(pbar):
        if device==torch.device("cuda"):
            data, target = data.cuda(), target.cuda()
        data_test = image_labeler(data, test=True)

        pred = net.forward(data_test)
        pred = torch.nn.functional.softmax(pred, dim=1).argmax(1)
        
        # Calculate accuracy
        batch_acc = pred.eq(target).float().mean().item()
        pred_cnt += batch_acc
        pbar.set_postfix({'acc': pred_cnt/(batch_idx + 1), 'batch_acc': batch_acc})

    test_pred_acc = float(pred_cnt)/len(test_loader)
    print('test error:', 1.-test_pred_acc)