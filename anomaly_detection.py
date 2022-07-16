from argparse import ArgumentParser

import matplotlib.pyplot as plt
import torch
from sklearn.metrics import roc_auc_score, roc_curve
from torch.utils.data import DataLoader
from torchvision.datasets import KMNIST, MNIST, FashionMNIST
from torchvision.utils import save_image
from tqdm import tqdm

from dataset import NoisyMNIST, OccludedMNIST, data_transform
from net import Discriminator, Encoder, Generator


def anomaly_score(x, fake_img, z_out_real, D, Lambda):

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


def plot_roc_curve(anomaly_dataset, file_name, E, G, Lambda):
    testset = MNIST('./data', train=False, download=True, transform=data_transform)
    testloader = DataLoader(testset, batch_size=256, shuffle=False, num_workers=2)
    anomaly_dataloader = DataLoader(anomaly_dataset, batch_size=256, shuffle=False, num_workers=2)
    a_scores_seq = []
    with torch.no_grad():
        first=True
        for images, _ in tqdm(testloader, desc=testset.__class__.__name__):
            images = images.to(device)
            z_out_real = E(images)
            images_reconst = G(z_out_real)
            if first:
                save_image(images[:64], f"graphs/original_mnist.png", pad_value=1, value_range=(-1, 1), padding=1)
                save_image(images_reconst[:64], f"graphs/reconst_mnist.png",  pad_value=1, value_range=(-1, 1), padding=1)
                first = False
            a_scores = anomaly_score(images, images_reconst, z_out_real, D, Lambda)
            a_scores_seq += a_scores.tolist()
        
        first = True
        for images, _ in tqdm(anomaly_dataloader, desc=anomaly_dataset.__class__.__name__):
            images = images.to(device)
            z_out_real = E(images)
            images_reconst = G(z_out_real)
            if first:
                save_image(images[:64], f"graphs/original_{file_name}",  pad_value=1, value_range=(-1, 1), padding=1)
                save_image(images_reconst[:64], f"graphs/reconst_{file_name}",  pad_value=1, value_range=(-1, 1), padding=1)
                first = False
            a_scores = anomaly_score(images, images_reconst, z_out_real, D, Lambda)
            a_scores_seq += a_scores.tolist()

    roc = roc_curve([0] * len(testset) + [1] * len(anomaly_dataset), a_scores_seq)
    plt.figure(figsize = (5,5))
    path = f'graphs/Lambda={Lambda:.2f}_{file_name}'
    plt.plot(roc[0], roc[1])
    auc_score = roc_auc_score([0] * len(testset) + [1] * len(anomaly_dataset), a_scores_seq)
    plt.title(f"ROC curve (AUC {auc_score:.3f})")
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.grid()
    plt.savefig(path, bbox_inches='tight')
    print(f'Image saved {path}')


if __name__ == "__main__":
    parser = ArgumentParser(description="Create a heatmap of positive rate")
    parser.add_argument('--nepoch', type=int, help="which epoch model to use", default=1000)
    parser.add_argument('--ngpu', type=int, help="which gpu to use.", default=1)
    parser.add_argument('-l', '--Lambda', type=float, help="lambda value for anomaly score", default=0.1)
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

    fashionset = FashionMNIST('./data', train=False, download=True, transform=data_transform)
    kset = KMNIST('./data', train=False, download=True, transform=data_transform)
    noisyset = NoisyMNIST('./data', train=False, download=True, transform=data_transform)
    occludedset = OccludedMNIST('./data', train=False, download=True, transform=data_transform)

    plot_roc_curve(fashionset, "fashion.png", E, G, args.Lambda)
    plot_roc_curve(kset, "kuzushiji.png", E, G, args.Lambda)
    plot_roc_curve(noisyset, "noisy.png", E, G, args.Lambda)
    plot_roc_curve(occludedset, "occluded.png", E, G, args.Lambda)
