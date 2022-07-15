from argparse import ArgumentParser

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import roc_curve
from torch.utils.data import DataLoader
from torchvision.datasets import KMNIST, MNIST, FashionMNIST
from torchvision.transforms import ToTensor
from tqdm import tqdm

from net import Discriminator, Encoder, Generator


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
    parser.add_argument('--ngpu', type=int, help="which gpu to use.", default=1)
    args = parser.parse_args()

    epoch = args.nepoch - 1

    device = torch.device(f"cuda:{args.ngpu}" if torch.cuda.is_available() else "cpu")

    D = Discriminator().to(device)
    G = Generator().to(device)
    E = Encoder().to(device)
    G.eval()
    D.eval()
    E.eval()

    D.load_state_dict(torch.load(f'./trained_net/netD_epoch_{epoch}.pth'))
    G.load_state_dict(torch.load(f'./trained_net/netG_epoch_{epoch}.pth'))
    E.load_state_dict(torch.load(f'./trained_net/netE_epoch_{epoch}.pth'))

    testset = MNIST('./data', train=False, download=True, transform=ToTensor())
    testloader = DataLoader(testset, batch_size=256, shuffle=False, num_workers=2)

    fashionset = FashionMNIST('./data', train=False, download=True, transform=ToTensor())
    fashionloader = DataLoader(fashionset, batch_size=256, shuffle=False, num_workers=2)

    kset = KMNIST('./data', train=False, download=True, transform=ToTensor())
    kloader = DataLoader(kset, batch_size=256, shuffle=False, num_workers=2)

    a_scores_seq = []
    
    with torch.no_grad():
        for images, labels in tqdm(testloader, desc='MNIST'):
            images = images.to(device)
            z_out_real = E(images)
            images_reconst = G(z_out_real)
            a_scores = anomaly_score(images, images_reconst, z_out_real, D)
            a_scores_seq += a_scores.tolist()

        for images, _ in tqdm(fashionloader, desc='calculating positive rate of Fashion-MNIST'):
            images = images.to(device)
            z_out_real = E(images)
            images_reconst = G(z_out_real)
            a_scores = anomaly_score(images, images_reconst, z_out_real, D)
            a_scores_seq += a_scores.tolist()

    roc = roc_curve([0] * len(testset) + [1] * len(fashionset), a_scores_seq)
    plt.figure(figsize = (5,5))
    path = f'graphs/epoch{args.nepoch:4d}.png'
    plt.plot(roc[0], roc[1])
    plt.title(f"ROC curve")
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.grid()
    plt.savefig(path, bbox_inches='tight')
    print(f'Image saved {path}')
    