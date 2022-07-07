import torch
import torch.nn as nn


class Generator(nn.Module):

    def __init__(self, z_dim=20):
        super().__init__()
        self.decoder = nn.Sequential(
            nn.Linear(z_dim, 1024),# b, latent_dim ==> b, 512
            nn.BatchNorm1d(1024),
            nn.ReLU(),
        )
        self.decoder2 = nn.Sequential(
            nn.Linear(1024, 256 * 7 * 7),
            nn.BatchNorm1d(256 * 7 * 7),
            nn.ReLU(),
        )
        self.convTrans2 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 3, stride=1),  # b, 128, 9, 9
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )
        self.convTrans3 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 3, stride=2, padding = 2),  # b, 64, 15, 15
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        self.convTrans4 = nn.Sequential(
            nn.ConvTranspose2d(64, 1, 4, stride=2, padding = 2),  # b, 3, 28, 28
            nn.Tanh(),
        )

    def forward(self, z):
        out = self.decoder(z)
        out = self.decoder2(out)
        out = out.view(-1,256,7,7)
        #out = self.convTrans1(out)
        out = self.convTrans2(out)
        out = self.convTrans3(out)
        out = self.convTrans4(out)
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
        )

        # z input
        self.z_layer1 = nn.Linear(z_dim, 512)  # b, z_dim ==> b, 512

        self.last1 = nn.Sequential(
            nn.Linear(256 * 3 * 3 + 512, 1024),
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


if __name__ == "__main__":
    from torchinfo import summary
    summary(Generator(), (64, 20))
    summary(Encoder(), (64, 1, 28, 28))
    summary(Discriminator(), [(64, 1, 28, 28), (64, 20)])