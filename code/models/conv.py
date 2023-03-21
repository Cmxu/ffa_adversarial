import torch
import torch.nn.functional as F
import torchvision
import numpy as np
from tqdm import tqdm
from matplotlib import pyplot as plt


class ConvNet(torch.nn.Module):
    def __init__(self, input_size=784, output_size=10, hidden_size=100):
        super().__init__()
        # four hidden layers of 2000 ReLUs each for 100 epochs
        self.conv1 = torch.nn.Conv2d(1, 16, kernel_size=5)
        self.conv2 = torch.nn.Conv2d(16, 32, kernel_size=5)
        self.fc2 = torch.nn.Linear(512, 10)

        self.loss = torch.nn.CrossEntropyLoss()
        # self.opt = torch.optim.SGD(self.linear.parameters(), lr=0.1, weight_decay=0.1)
        self.opt = torch.optim.Adam(self.parameters(), lr=0.03)
        # self.opt = torch.optim.AdamW(self.linear.parameters(), lr=0.1, weight_decay=0.1)

    def forward(self, input):
        input = input.reshape([-1,1,28,28])
        output = F.relu(self.conv1(input))
        output = F.max_pool2d(output, 2)
        output = F.relu(self.conv2(output))
        output = F.max_pool2d(output, 2)
        output = output.view(-1, 512)
        output = self.fc2(output)
        return output

    def train(self, input, label):
        pred = self.forward(input)

        pos_loss = self.loss(pred, label)
        
        self.opt.zero_grad()
        pos_loss.backward()
        self.opt.step()
        return pred

    def save_model_weights(self, base_path, epoch, model_name="conv"):
        import os

        model_base_path = f"{base_path}/{model_name}/epoch{epoch+1}"
        print(model_base_path)
        os.makedirs(model_base_path, exist_ok=True)
        torch.save(self.state_dict(), f"{model_base_path}/conv.pth")
    

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

        # data = data.view(data.shape[0], -1)

        if (train):
            pred = mdl.train(data, target)
        else:
            with torch.no_grad():
                pred = mdl.forward(data)

        # Calculate accuracy
        predictions = torch.nn.functional.softmax(pred, dim=1).argmax(1)
        batch_acc = ((predictions==target).sum()/batch_size)
        pred_cnt += batch_acc
        pbar.set_postfix({'acc': pred_cnt/(batch_idx + 1), 'batch_acc': batch_acc})

    pred_acc = float(pred_cnt)/len(dataset_loader)
    return pred_acc


def visualize_pred_acc(path, train_acc, test_acc, num_epochs):
    plt.title(f"Train/Test Accuracy")

    plt.ylabel("Accuracy")
    plt.xlabel("Number of Epochs")
    plt.xticks(np.arange(1, num_epochs+1, 1.0))

    plt.plot(train_acc)
    plt.plot(test_acc)

    legend_drawn_flag = True
    plt.legend(["Train", "Test"], loc=0, frameon=legend_drawn_flag)

    plt.savefig(path, dpi=200)


if __name__ == "__main__":
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    # device = torch.device("cpu")
    print(f'Using {device}')

    net = ConvNet().to(device)

    batch_size_train = 64
    batch_size_test = 1000
    train_loader,test_loader = load_mnist(batch_size_train,
                                            batch_size_test,
                                            device,
                                            dataset_path="/home/shchoi4/ffa_adversarial/code/MNIST")
        
    train_acc = []
    test_acc = []

    num_epochs = 5

    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}: ")
        print('train...')
        train_acc.append(   run_model(net, train_loader, batch_size_train, train=True, device=device))
        print('test...')
        test_acc.append(    run_model(net, test_loader, batch_size_test, train=False, device=device))

        net.save_model_weights("/home/shchoi4/ffa_adversarial/models", epoch)

    # visualize_pred_acc("/home/shchoi4/ffa_adversarial/models/linear/fig.png",
    #                         train_acc,
    #                         test_acc, 
    #                         num_epochs)

# idea, make negative samples from current adversarial attacks