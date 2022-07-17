import random

import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as data
from torchvision import datasets
from torchvision.utils import save_image
from tqdm import tqdm

from dataset import data_transform
from net import Discriminator, Encoder, Generator

torch.manual_seed(1234)
torch.cuda.manual_seed(1234)
np.random.seed(1234)
random.seed(1234)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

batch_size=256

dataset = datasets.MNIST('data', train=True, transform=data_transform, download=True)

dataloader = data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2)

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


G = Generator()
E = Encoder()
D = Discriminator()
G.to(device)
E.to(device)
D.to(device)

G.apply(weights_init)
E.apply(weights_init)
D.apply(weights_init)
    

lr_ge = 0.0002
lr_d = 0.00005
beta1, beta2 = 0.5, 0.999
g_optimizer = torch.optim.Adam(G.parameters(), lr_ge, [beta1, beta2])
e_optimizer = torch.optim.Adam(E.parameters(), lr_ge, [beta1, beta2])
d_optimizer = torch.optim.Adam(D.parameters(), lr_d, [beta1, beta2])

criterion = nn.BCEWithLogitsLoss(reduction='mean')

z_dim = 20

torch.backends.cudnn.benchmark = True

fixed_z = torch.randn((64, 20)).to(device)
fixed_real_images, _ = iter(dataloader).__next__()
fixed_real_images = fixed_real_images.to(device)[:64]

for epoch in range(1000):
    G.train()
    E.train()
    D.train()

    epoch_g_loss = []
    epoch_e_loss = []
    epoch_d_loss = []

    for images, _ in tqdm(dataloader, leave=False):
        ### Discriminator ###
        label_real = torch.full((images.size(0),), 1, dtype=images.dtype).to(device)
        label_fake = torch.full((images.size(0),), 0, dtype=images.dtype).to(device)

        images = images.to(device)

        # train by real images
        z_out_real = E(images)
        d_out_real, _ = D(images, z_out_real)

        # train by fake images
        input_z = torch.randn(images.size(0), z_dim).to(device)
        fake_images = G(input_z)
        d_out_fake, _ = D(fake_images, input_z)

        d_loss_real = criterion(d_out_real.view(-1), label_real)
        d_loss_fake = criterion(d_out_fake.view(-1), label_fake)
        d_loss = d_loss_real + d_loss_fake

        d_optimizer.zero_grad()
        d_loss.backward()
        d_optimizer.step()

        ### Generator ###
        input_z = torch.randn(images.size(0), z_dim).to(device)
        fake_images = G(input_z)
        d_out_fake, _ = D(fake_images, input_z)

        g_loss = criterion(d_out_fake.view(-1), label_real)

        g_optimizer.zero_grad()
        g_loss.backward()
        g_optimizer.step()

        ### Encoder ###
        z_out_real = E(images)
        d_out_real, _ = D(images, z_out_real)

        e_loss = criterion(d_out_real.view(-1), label_fake)

        e_optimizer.zero_grad()
        e_loss.backward()
        e_optimizer.step()

        epoch_d_loss.append(d_loss.item())
        epoch_g_loss.append(g_loss.item())
        epoch_e_loss.append(e_loss.item())
    
    print('epoch {} || Epoch_D_Loss:{:.4f} ||Epoch_G_Loss:{:.4f} ||Epoch_E_Loss:{:.4f}'.format(
            epoch, np.mean(epoch_d_loss), np.mean(epoch_g_loss), np.mean(epoch_e_loss)))

    with torch.no_grad():
        G.eval()
        E.eval()
        z_out_real = E(fixed_real_images)
        reconst_images = G(z_out_real)
        save_image(reconst_images, f'images/image_epoch_{epoch}_reconstructed.png', pad_value=1, value_range=(-1, 1), padding=1)
        save_image(G(fixed_z), f'images/image_epoch_{epoch}_generated.png', pad_value=1, value_range=(-1, 1), padding=1)

    torch.save(D.state_dict(), 'trained_net/netD_epoch_%d.pth' % (epoch))
    torch.save(G.state_dict(), 'trained_net/netG_epoch_%d.pth' % (epoch))
    torch.save(E.state_dict(), 'trained_net/netE_epoch_%d.pth' % (epoch))
    
