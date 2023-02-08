import torch
import torchvision
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from matplotlib import pyplot as plt
import math


input_size = 784
output_size = 10


class NeuralNet(torch.nn.Module):
    def __init__(self):
        # four hidden layers of 2000 ReLUs each for 100 epochs
        self.layers = []
        self.layers += [FFALayer(input_size, 2000)]
        self.layers += [FFALayer(2000, 2000)]
        self.layers += [FFALayer(2000, 2000)]
        self.layers += [FFALayer(2000, output_size)]

        self.opt = torch.optim.Adam(self.layers[3].parameters(), lr=0.0003)

    
    def forward(self, input):
        output = input
        for i, layer in enumerate(self.layers):
            output = layer(output)
        return torch.nn.functional.softmax(output, dim=1).argmax(1)


    def predict(self, x):
        goodness_per_label = []
        for label in range(10):
            h = image_labeler(x, label)
            goodness = []
            for layer in self.layers:
                h = layer(h)
                goodness += [h.pow(2).mean(1)]
            goodness_per_label += [sum(goodness).unsqueeze(1)]
        goodness_per_label = torch.cat(goodness_per_label, 1)
        return goodness_per_label.argmax(1)


    def train(self, x_pos, x_neg, label):
        pos_input, neg_input = x_pos, x_neg
        for i, layer in enumerate(self.layers):
            print('training layer', i, '...')
            # print(i==len(self.layers)-1)
            pos_input, neg_input = layer.train(pos_input, neg_input, last=(i==len(self.layers)-1))
            
        
        # # TODO: Look into loss for negative examples
        # # Loss for final softmax
        loss = torch.nn.CrossEntropyLoss()
        output_pos = loss(pos_input, label)

        # print("output_pos", output_pos)

        self.opt.zero_grad()
        output_pos.backward()
        self.opt.step()


class FFALayer(torch.nn.Linear):
    def __init__(self, in_features, out_features,
                 bias=True, device=None, dtype=None):
        super().__init__(in_features, out_features, bias, device, dtype)
        self.relu = torch.nn.ReLU()
        # self.opt = torch.optim.Adam(self.parameters(), lr=0.03)
        # self.opt = torch.optim.Adam(self.parameters(), lr=0.0003)
        self.opt = torch.optim.SGD(self.parameters(), lr=0.1, momentum=0.9)
        self.threshold = 2.0
        # self.num_epochs = 1000

    def forward(self, x):
        x = x / torch.sqrt(x.norm(2, 1, keepdim=True) + 1e-4)
        output = self.relu(
            torch.matmul(x, self.weight.T) + self.bias.unsqueeze(0)
        )
        return output
        # x = x / (x.norm(2, 1, keepdim=True) + 1e-4)
        # return self.relu(
        #     torch.mm(x, self.weight.T) +
        #     self.bias.unsqueeze(0))

    def train(self, x_pos, x_neg, last=False):
        # for i in tqdm(range(10)):
        output_pos = self.forward(x_pos)
        output_neg = self.forward(x_neg)

        # positive example
        p_pos = torch.sigmoid(output_pos.pow(2).sum(1) - self.threshold).mean()
        p_neg = torch.sigmoid(output_neg.pow(2).sum(1) - self.threshold).mean()
        loss = (1/p_pos) + p_neg

        # ---- Loss function on github ----
        # p_pos = (-output_pos.pow(2).mean(1) + self.threshold)
        # p_neg = (output_neg.pow(2).mean(1) - self.threshold)
        
        # loss = torch.log(1 + torch.exp(torch.cat([
        #         torch.exp(p_pos),
        #         torch.exp(p_neg)]))).mean()
        print("\tgoodness loss: ", loss)

        # print("\tp_pos", p_pos.mean())
        # print("\tp_neg", p_neg.mean())

        self.opt.zero_grad()
        loss.backward()
        self.opt.step()

        if last:
            return self.forward(x_pos), self.forward(x_neg)

        return self.forward(x_pos).detach(), self.forward(x_neg).detach()
    

def load_dataset():
    batch_size_train = 64
    batch_size_test = 1000
    random_seed = 1
    torch.manual_seed(random_seed)
    # torch.backends.cudnn.enabled = False

    transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                                torchvision.transforms.Normalize((0.1307,), (0.3081,))])

    train_dataset = torchvision.datasets.MNIST('./', download=True, train=True, transform=transform)
    val_dataset = torchvision.datasets.MNIST('./', download=True, train=False, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size_train, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size_test, shuffle=True)

    return train_loader, val_loader


def image_labeler(img, label):
    img_ = img.clone()
    # Why does multiplication make a difference?
    img_[:,:10] *= 0.0
    img_[range(img.shape[0]),(label%10)] = img.max()  #1
    return img_


def visualize_sample(data, name='', idx=0):
    reshaped = data[idx].cpu().reshape(28, 28)
    plt.figure(figsize = (4, 4))
    plt.title(name)
    plt.imshow(reshaped, cmap="gray")
    plt.show()


if __name__ == "__main__":
    net = NeuralNet()

    train_loader,val_loader = load_dataset()

    # for i in range(2):
    #     print('training epoch', i+1, '...')
    #     for batch_idx, (data, target) in enumerate(tqdm(train_loader)):
    #         data = data.view(data.shape[0], -1)
    #         pos_data = image_labeler(data, target)
    #         neg_data = image_labeler(data, target+4)

    #         # for data, name in zip([x, pos_data, neg_data], ['orig', 'pos', 'neg']):
    #         #     visualize_sample(data, name)
    #         # exit()

    #         # net.train(data, data, y)
    #         net.train(pos_data, neg_data, target)

    #         break

    x, y = next(iter(train_loader))
    x = x.view(x.shape[0], -1)

    # accuracy_lst = []
    pred_acc = []

    for i in range(100):
        pos_data = image_labeler(x, y)

        neg_data = image_labeler(x, y+torch.randint(9, (1,)) + 1)
        net.train(pos_data, neg_data, y)
        # net.train(pos_data, pos_data, y)

        # accuracy_lst.append((net.forward(x)==y).sum()/len(y))
        pred_acc.append((net.predict(x)==y).sum()/len(y))

    # plt.plot(accuracy_lst)
    plt.plot(pred_acc)
    plt.show()




    # # x, y = next(iter(train_loader))
    # # data = x.view(x.shape[0], -1)
    # data = image_labeler(x, y)
    # print(net.predict(data))
    # print(y)

    # print((net.predict(data)==y).sum()/len(y))


# idea, make negative samples from current adversarial attacks