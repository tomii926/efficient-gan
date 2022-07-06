import os
from argparse import ArgumentParser

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST, FashionMNIST
from torchvision.transforms import ToTensor
from tqdm import tqdm

from dataset import PartialMNIST
from net import Discriminator, Encoder, Generator


def positive_rates(val_num, threshold: float, epoch: int, nz: int, device: str):
    """ returns positive rates of each class
    Args:
        input_nums(list[int]): which classes the model was trained with.
        val_nums(int): which class to use in determining threshold.
        threshold(float)
        epoch(int): which epoch the model is trained up to.
        vae(bool):
        nz(int): size of latent 
    """

    valset = PartialMNIST([val_num], train=True)
    valloader = DataLoader(valset, batch_size=256, shuffle=False, num_workers=2)

    D = Discriminator(nz).to(device)
    G = Generator(nz).to(device)
    E = Encoder(nz).to(device)
    G.eval()
    D.eval()
    E.eval()

    D.load_state_dict(torch.load(f'./trained_net/netD_epoch_{epoch}.pth'))
    G.load_state_dict(torch.load(f'./trained_net/netG_epoch_{epoch}.pth'))
    E.load_state_dict(torch.load(f'./trained_net/netE_epoch_{epoch}.pth'))

    all_losses = []
    for images, label in tqdm(valloader, desc='determining threshold'):
        images = images.to(device)

        z_out_real = E(images)
        images_reconst = G(z_out_real)
        
        loss = anomaly_score(images, images_reconst, z_out_real, D)
        all_losses += loss.tolist()

    all_losses.sort()
    threshold = all_losses[int(len(all_losses) * threshold)]

    testset = MNIST('./data', train=False, download=True, transform=ToTensor())
    testloader = DataLoader(testset, batch_size=256, shuffle=False, num_workers=2)

    positive_num = [0] * 10
    num_num = [0] * 10

    for images, labels in tqdm(testloader, desc='calculating positive rate of MNIST classes'):
        images = images.to(device)
        labels = labels.tolist()
        z_out_real = E(images)
        images_reconst = G(z_out_real)
        losses = anomaly_score(images, images_reconst, z_out_real, D)
        losses = losses.tolist()

        for loss, label in zip(losses, labels):
            if loss > threshold:
                positive_num[label] += 1
            num_num[label] += 1

    fashionset = FashionMNIST('./data', train=False, download=True, transform=ToTensor())
    fashionloader = DataLoader(fashionset, batch_size=256, shuffle=False, num_workers=2)

    positive = 0
    for images, _ in tqdm(fashionloader, desc='calculating positive rate of Fashion-MNIST'):
        images = images.to(device)
        z_out_real = E(images)
        images_reconst = G(z_out_real)
        losses = anomaly_score(images, images_reconst, z_out_real, D)
        positive += torch.count_nonzero(losses > threshold).item()

    positive_rates = np.array(positive_num)/np.array(num_num)

    return np.append(positive_rates, positive/len(fashionset))


def anomaly_score(x, fake_img, z_out_real, D, Lambda=0.1):

    residual_loss = torch.abs(x-fake_img)
    residual_loss = residual_loss.view(residual_loss.size()[0], -1)
    residual_loss = torch.sum(residual_loss, dim=1)

    _, x_feature = D(x, z_out_real)
    _, G_feature = D(fake_img, z_out_real)

    discrimination_loss = torch.abs(x_feature-G_feature)
    discrimination_loss = discrimination_loss.view(discrimination_loss.size()[0], -1)
    discrimination_loss = torch.sum(discrimination_loss, dim=1)

    loss_each = (1-Lambda)*residual_loss + Lambda*discrimination_loss

    return loss_each


if __name__ == "__main__":
    parser = ArgumentParser(description="Create a heatmap of positive rate")
    parser.add_argument('--nepoch', type=int, help="which epoch model to use", default=1000)
    parser.add_argument('--nz', type=int, help='size of the latent z vector', default=20)
    parser.add_argument('-t', '--threshold', type=float, help="threshold", default=0.99)
    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    positive_rate_list = [positive_rates(i, args.threshold, args.nepoch, args.nz, device) for i in range(10)]

    plt.figure(figsize = (10,8))
    sns.heatmap(positive_rate_list, annot=True, cmap='Blues_r', xticklabels=[str(i) for i in range(10)] + ['Fasion'])
    path = f'graphs/t{args.threshold:.3f}_epoch_{args.nepoch:4d}.png'
    plt.title(f"Positive rates (threshold: {args.threshold * 100:.1f}%)")
    plt.ylabel('The class used for determining threshold')
    plt.xlabel('class')
    plt.savefig(path, bbox_inches='tight')
    print(f'Image saved {path}')
    