import torch
import torchvision
import numpy as np

import cv2


def mask_generator():
    def blur(I, vertical=False):
        for i in range(1,I.shape[0]-1):
            for j in range(1,I.shape[1]-1):
                if vertical:
                    I[i,j] = (2*I[i,j] + I[i-1,j] + I[i+1,j])/4.0
                else:
                    I[i,j] = (2*I[i,j] + I[i,j-1] + I[i,j+1])/4.0
        return I


    rand_bits = np.random.rand(28,28)
    # Blur image horizonally and vertically three times
    for _ in range(10):
        blurred = blur(blur(rand_bits), vertical=True)
    thresh = (blurred[:,:] >= 0.5)
    return thresh


def image_labeler(img, label):
    img[0,:10] = 0
    img[0,label] = 255
    return img


def load_dataset():
    batch_size_train = 64
    batch_size_test = 1000
    # random_seed = 1
    # torch.backends.cudnn.enabled = False
    # torch.manual_seed(random_seed)

    transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                                torchvision.transforms.Normalize((0.1307,), (0.3081,))])

    train_dataset = torchvision.datasets.MNIST('./', download=True, train=True, transform=transform)
    val_dataset = torchvision.datasets.MNIST('./', download=True, train=False, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size_train, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size_test, shuffle=True)

    return train_loader, val_loader


if __name__ == "__main__":
    train_loader, val_loader = load_dataset()

    dataiter = iter(train_loader)
    images, labels = dataiter.next()

    mask = mask_generator()

    images_numpy = images.numpy()
    images_numpy = np.moveaxis(images_numpy, 1, -1)
    base_img = images_numpy[0]*255

    # cv2.imwrite("mask.png", mask*255)
    # cv2.imwrite("base_img.png", base_img*255)

    img = np.zeros((28,28))
    img = image_labeler(img, 9)

    cv2.imwrite("labeled_dummy_img.png", img)
    exit()

    for idx, img in enumerate(images_numpy):
        img *= 255
        img = img.astype(int)

        base_masked = mask * base_img[:,:,0]
        img_masked = (1 - mask) * img[:,:,0]
        neg_img = base_masked + img_masked

        cv2.imwrite("img_masked.png", img_masked)
        cv2.imwrite("base_masked.png", base_masked)
        cv2.imwrite("neg_img.png", neg_img)

        # if idx == 1:
        #     cv2.imwrite("base_img.png", base_img)
        #     cv2.imwrite("add_img.png", img)
        #     cv2.imwrite("neg_img.png", neg_img)
        #     break