import torch
import torch.nn as nn


class Generator(nn.Module):

    def __init__(self, z_dim=20):
        super(Generator, self).__init__()

        self.layer1 = nn.Sequential(
            nn.Linear(z_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True))

        self.layer2 = nn.Sequential(
            nn.Linear(1024, 7*7*128),
            nn.BatchNorm1d(7*7*128),
            nn.ReLU(inplace=True))

        self.layer3 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=128, out_channels=64,
                               kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True))

        self.last = nn.Sequential(
            nn.ConvTranspose2d(in_channels=64, out_channels=1,
                               kernel_size=4, stride=2, padding=1),
            nn.Tanh())
        # 注意：白黒画像なので出力チャネルは1つだけ

    def forward(self, z):
        out = self.layer1(z)
        out = self.layer2(out)

        # 転置畳み込み層に入れるためにテンソルの形を整形
        out = out.view(z.shape[0], 128, 7, 7)
        out = self.layer3(out)
        out = self.last(out)

        return out


class Discriminator(nn.Module):
    def __init__(self, z_dim=20):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 64, 3, stride=1, padding=1),  # b, 64, 28, 28
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),            
            nn.MaxPool2d(2)  # b, 64, 14, 14
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, stride=1, padding=1),  # b, 128, 14, 14
            nn.BatchNorm2d(128),
            nn.LeakyReLU(),           
            nn.MaxPool2d(2)  # b, 128, 7, 7
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 256, 3, stride=1, padding=1),  # b, 256, 7, 7
            nn.BatchNorm2d(256),
            nn.LeakyReLU(),
            nn.MaxPool2d(2),  # b, 256, 3, 3
            nn.Dropout(0.2)
        )

        # z input
        self.z_layer1 = nn.Linear(z_dim, 256)  # b, z_dim ==> b, 512

        self.last1 = nn.Sequential(
            nn.Linear(256 * 3 * 3 + 256, 1024),
            nn.LeakyReLU(0.1, inplace=True),
        )

        self.last2 = nn.Linear(1024, 1)

    def forward(self, x, z):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)

        z = z.view(z.shape[0], -1)
        z = self.z_layer1(z)

        # x_outとz_outを結合し、全結合層で判定
        x = x.view(x.size(0), -1)
        out = torch.cat([x, z], dim=1)
        out = self.last1(out)

        feature = out  # 最後にチャネルを1つに集約する手前の情報
        feature = feature.view(feature.size()[0], -1)  # 2次元に変換

        out = self.last2(out)
        return out, feature


class Encoder(nn.Module):

    def __init__(self, z_dim=20):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 64, 3, stride=1, padding=1),  # b, 64, 28, 28
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),            
            nn.MaxPool2d(2)  # b, 64, 14, 14
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, stride=1, padding=1),  # b, 128, 14, 14
            nn.BatchNorm2d(128),
            nn.LeakyReLU(),           
            nn.MaxPool2d(2)  # b, 128, 7, 7
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 256, 3, stride=1, padding=1),  # b, 256, 7, 7
            nn.BatchNorm2d(256),
            nn.LeakyReLU(),
            nn.MaxPool2d(2),  # b, 256, 3, 3
            nn.Dropout(0.2)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(256, 512, 3, stride=1, padding=0),  # b, 512, 1, 1
            nn.BatchNorm2d(512),
            nn.LeakyReLU(),
        )
        self.z = nn.Linear(512, z_dim)  # b, 512 ==> b, latent_dim

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = x.view(-1,512)
        z = self.z(x)
        return z
