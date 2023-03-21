import torch
import torchvision
import numpy as np
from tqdm import tqdm
from matplotlib import pyplot as plt


class FFANet(torch.nn.Module):
    def __init__(self, input_size=784, output_size=10, hidden_size=500):
        super().__init__()
        # four hidden layers of 2000 ReLUs each for 100 epochs
        self.layers = torch.nn.Sequential(FFALayer(input_size, hidden_size),
                                            FFALayer(hidden_size, hidden_size),
                                            FFALayer(hidden_size, hidden_size),
                                            FFALayer(hidden_size, hidden_size))
        self.linear = torch.nn.Linear(hidden_size, output_size)

        self.loss = torch.nn.CrossEntropyLoss()
        # self.opt = torch.optim.SGD(self.linear.parameters(), lr=0.1, weight_decay=0.1)
        # self.opt = torch.optim.AdamW(self.linear.parameters(), lr=0.1, weight_decay=0.1)
        self.opt = torch.optim.Adam(self.linear.parameters(), lr=0.01)

    # def forward(self, input):
    #     layer_out = []
    #     output = input

    #     for i, layer in enumerate(self.layers):
    #         output = layer(output)
    #         if i==0:    continue
    #         layer_out.append(output)
    #     layer_out = torch.stack(layer_out)
    #     output = self.linear(torch.mean(layer_out, dim=0))
    #     return output

    def predict(self, x):
        goodness_per_label = []
        for label in range(10):
            h = image_labeler(x, label)
            goodness = []
            for idx, layer in enumerate(self.layers):
                h = layer(h)
                if (idx == 0):
                    continue
                goodness += [h.pow(2).mean(1)]
            goodness_per_label += [sum(goodness).unsqueeze(1)]
        goodness_per_label = torch.cat(goodness_per_label, 1)
        return goodness_per_label.argmax(1)

    def train(self, x_pos, x_neg, label):
        # linear_input = []
        pos_input, neg_input = x_pos, x_neg
        for i, layer in enumerate(self.layers):
            pos_input, neg_input = layer.train(pos_input, neg_input)
        #     if i==0:    continue
        #     linear_input.append(pos_input)

        # linear_input = torch.stack(linear_input)
        # pred = self.linear(torch.mean(linear_input, dim=0))

        # # TODO: Look into loss for negative examples
        # # Loss for final softmax
        # pos_loss = self.loss(pred, label)
        
        # self.opt.zero_grad()
        # pos_loss.backward()
        # self.opt.step()
        # return pred

    def save_model_weights(self, base_path, epoch, model_name="ffa"):
        import os

        model_base_path = f"{base_path}/{model_name}/epoch{epoch+1}"
        print(model_base_path)
        os.makedirs(model_base_path, exist_ok=True)

        for idx, layer in enumerate(self.layers):
            torch.save(layer.state_dict(), f"{model_base_path}/ffa_{idx}.pth")
        torch.save(self.linear.state_dict(), f"{model_base_path}/linear.pth")


class FFALayer(torch.nn.Module):
    def __init__(self, in_features, out_features):
        super(FFALayer, self).__init__()
        self.linear = torch.nn.Linear(in_features, out_features)
        self.relu = torch.nn.ReLU()
        self.opt = torch.optim.SGD(self.parameters(), lr=0.1, weight_decay=0.1)
        self.threshold = 2.0

    def forward(self, input):
        output = input / torch.sqrt(input.norm(2, 1, keepdim=True) + 1e-4)
        output = self.relu(self.linear(output))
            
        return output

    def train(self, x_pos, x_neg):
        output_pos = self.forward(x_pos)
        output_neg = self.forward(x_neg)

        p_pos = torch.sigmoid(output_pos.pow(2).sum(1) - self.threshold).mean()
        p_neg = torch.sigmoid(output_neg.pow(2).sum(1) - self.threshold).mean()
        loss = (1/p_pos) + p_neg

        self.opt.zero_grad()
        loss.backward()
        self.opt.step()

        return output_pos.detach(), output_neg.detach()
    

def load_mnist(batch_size_train, batch_size_test, device, dataset_path="./"):
    transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                                torchvision.transforms.Normalize((0.1307,), (0.3081,))])

    train_dataset = torchvision.datasets.MNIST(dataset_path,
                                                download=True,
                                                train=True,
                                                transform=transform)
    test_dataset = torchvision.datasets.MNIST(dataset_path,
                                                download=True,
                                                train=False,
                                                transform=transform)
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

    return train_loader, test_loader


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


def run_model(mdl, dataset_loader, batch_size=64, train=True, device=torch.device("cpu")):
    pred_cnt = 0

    pbar = tqdm(dataset_loader)
    for batch_idx, (data, target) in enumerate(pbar):
        if device==torch.device("cuda"):
            data, target = data.cuda(), target.cuda()

        data = data.view(data.shape[0], -1)

        if train:
            pos_data = image_labeler(data, target)

            neg_data = image_labeler(data, target+torch.randint(low=1,
                                                                high=9,
                                                                size=(batch_size,),
                                                                device=device))
            # pred = mdl.train(pos_data, neg_data, target)
            # predictions = torch.nn.functional.softmax(pred, dim=1).argmax(1)
            mdl.train(pos_data, neg_data, target)
            predictions = mdl.predict(pos_data)
        else:
            test_data = image_labeler(data, target, test=True)
            # pred = mdl.forward(test_data)
            pred = mdl.predict(test_data)
            predictions = pred

        # Calculate accuracy
        # predictions = torch.nn.functional.softmax(pred, dim=1).argmax(1)
        batch_acc = ((predictions==target).sum()/batch_size)
        pred_cnt += batch_acc
        pbar.set_postfix({'acc': pred_cnt/(batch_idx + 1), 'batch_acc': batch_acc})

    pred_acc = float(pred_cnt)/len(dataset_loader)
    return pred_acc


def visualize_pred_acc(path, train_acc, valid_acc, test_acc, num_epochs):
    plt.title(f"Train/Test Accuracy")

    plt.ylabel("Accuracy")
    plt.xlabel("Number of Epochs")
    plt.xticks(np.arange(1, num_epochs+1, 1.0))

    plt.plot(train_acc)
    plt.plot(valid_acc)
    plt.plot(test_acc)

    location = 0
    legend_drawn_flag = True
    plt.legend(["Train", "Validation", "Test"], loc=0, frameon=legend_drawn_flag)

    plt.savefig(path, dpi=400)


if __name__ == "__main__":
    # device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    device = torch.device("cpu")
    print(f'Using {device}')

    net = FFANet().to(device)

    batch_size_train = 64
    batch_size_test = 1000
    train_loader,test_loader = load_mnist(batch_size_train,
                                            batch_size_test,
                                            device,
                                            dataset_path="/home/shchoi4/ffa_adversarial/code/MNIST")
        
    train_acc = []
    valid_acc = []
    test_acc = []

    num_epochs = 5

    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}: ")
        print('train...')
        train_acc.append(   run_model(net, train_loader, batch_size_train, train=True, device=device))
        print('validation...')
        valid_acc.append(   run_model(net, train_loader, batch_size_train, train=False, device=device))
        print('test...')
        test_acc.append(    run_model(net, test_loader, batch_size_test, train=False, device=device))

        net.save_model_weights("/home/shchoi4/ffa_adversarial/models", epoch)

    visualize_pred_acc("./a.png", train_acc, valid_acc, test_acc, num_epochs)
# idea, make negative samples from current adversarial attacks