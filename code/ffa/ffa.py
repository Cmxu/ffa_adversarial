import torch
import torchvision
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from matplotlib import pyplot as plt
import math


input_size = 784
output_size = 10

hidden_size = 100

batch_size_train = 64
batch_size_test = 1000
# random_seed = 1
# torch.manual_seed(random_seed)
# torch.backends.cudnn.enabled = False


class FFANet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # four hidden layers of 2000 ReLUs each for 100 epochs
        self.layers = []
        self.layers += [FFALayer(input_size, hidden_size)]
        self.layers += [FFALayer(hidden_size, hidden_size)]
        self.layers += [FFALayer(hidden_size, hidden_size)]
        self.layers += [FFALayer(hidden_size, hidden_size)]
        self.linear = torch.nn.Linear(hidden_size, output_size)

        self.opt = torch.optim.Adam(self.linear.parameters(), lr=0.0003)
        # self.opt = torch.optim.Adam(self.linear.parameters(), lr=0.1)


    def forward(self, input):
        layer_out = []
        output = input

        for i, layer in enumerate(self.layers):
            if i==1:    continue
            output = layer(output)
            layer_out.append(output)
        layer_out = torch.stack(layer_out)
        output = self.linear(torch.mean(layer_out, dim=0))
        # return torch.nn.functional.softmax(output, dim=1).argmax(1)
        return output

    def train(self, x_pos, x_neg, label):
        linear_input = []
        pos_input, neg_input = x_pos, x_neg
        for i, layer in enumerate(self.layers):
            # print('training layer', i, '...')
            # print(i==len(self.layers)-1)
            pos_input, neg_input = layer.train(pos_input, neg_input)
            linear_input.append(pos_input)

        linear_input = torch.stack(linear_input)
        pred = self.linear(torch.mean(linear_input, dim=0))

        # TODO: Look into loss for negative examples
        # Loss for final softmax
        loss = torch.nn.CrossEntropyLoss()
        output_pos = loss(pred, label)

        self.opt.zero_grad()
        output_pos.backward()
        self.opt.step()


class FFALayer(torch.nn.Module):
    def __init__(self, in_features, out_features):
        super(FFALayer, self).__init__()
        self.linear = torch.nn.Linear(in_features, out_features)
        self.relu = torch.nn.ReLU()
        self.opt = torch.optim.Adam(self.parameters(), lr=0.0003)
        self.threshold = 2.0

    def forward(self, input):
        output = self.relu(self.linear(input))
        output = output / torch.sqrt(output.norm(2, 1, keepdim=True) + 1e-4)
        return output

    def train(self, x_pos, x_neg):
        output_pos = self.relu(self.linear(x_pos))
        output_neg = self.relu(self.linear(x_neg))

        p_pos = torch.sigmoid(output_pos.pow(2).sum(1) - self.threshold).mean()
        p_neg = torch.sigmoid(output_neg.pow(2).sum(1) - self.threshold).mean()
        loss = (1/p_pos) + p_neg

        self.opt.zero_grad()
        loss.backward()
        self.opt.step()

        return self.forward(x_pos).detach(), self.forward(x_neg).detach()
        # return self.forward(x_pos), self.forward(x_neg)
    

def load_dataset():
    transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                                torchvision.transforms.Normalize((0.1307,), (0.3081,))])

    train_dataset = torchvision.datasets.MNIST('./', download=True, train=True, transform=transform)
    val_dataset = torchvision.datasets.MNIST('./', download=True, train=False, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size_train, shuffle=True, drop_last=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size_test, shuffle=True, drop_last=True)

    return train_loader, val_loader


def image_labeler(img, label, test=False):
    img_ = img.clone()
    if test:
        img_[:,:10] = 0.1
    else:
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


def save_model_weights(model, epoch, accuracy, model_name="ffa"):
    import os

    model_base_path = f"./models/{model_name}/epoch{epoch+1}"
    os.makedirs(model_base_path, exist_ok=True)

    # accuracy
    text_file = open(f"{model_base_path}/accuracy.txt", "w")
    text_file.write(str(round(accuracy, 3)))
    text_file.close()

    for idx, layer in enumerate(model.layers):
        torch.save(layer.state_dict(), f"{model_base_path}/ffa_{idx}.pth")
    torch.save(model.state_dict(), f"{model_base_path}/linear.pth")


if __name__ == "__main__":
    net = FFANet()

    train_loader,val_loader = load_dataset()

    pred_acc = []
    pred_cnt = 0

    for epoch in range(5):
        print('training epoch', epoch+1, '...')
        pbar = tqdm(train_loader)
        for batch_idx, (data, target) in enumerate(pbar):
            data = data.view(data.shape[0], -1)
            pos_data = image_labeler(data, target)
            neg_data = image_labeler(data, target+torch.randint(low=1, high=9, size=(batch_size_train,)))

            # for data, name in zip([x, pos_data, neg_data], ['orig', 'pos', 'neg']):
            #     visualize_sample(data, name)
            # exit()

            net.train(pos_data, neg_data, target)

            # Test
            test_data = image_labeler(data, target, test=True)

            predictions = torch.nn.functional.softmax(net.forward(test_data), dim=1).argmax(1)
            batch_acc = ((predictions==target).sum()/batch_size_train)
            
            pred_cnt += batch_acc
            pbar.set_postfix({'acc': pred_cnt/(batch_idx + 1), 'batch_acc': batch_acc})
        
        pred_acc.append(float(pred_cnt)/len(train_loader))
        print(f"epoch {epoch+1} train accuracy: {pred_acc[-1]}")
        pred_cnt = 0

        save_model_weights(net, epoch, pred_acc[-1])

    plt.plot(pred_acc)
    plt.show()

# idea, make negative samples from current adversarial attacks