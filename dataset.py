from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.datasets import MNIST
from torchvision.utils import save_image


class OccludedMNIST(Dataset):
    def __init__(self, *args, **kwargs):
        self.dataset = MNIST(*args, **kwargs)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        image, _ = self.dataset[index]
        for i in range(28):
            for j in range(28):
                if (i - 20) ** 2 + (j - 14) ** 2 <= 25:
                    image[0][i][j] = 0.0
        return image, _


class NoisyMNIST(Dataset):
    def __init__(self, *args, **kwargs):
        self.dataset = MNIST(*args, **kwargs)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        image, _ = self.dataset[index]
        for i in range(28):
            for j in range(28):
                if (i * 28 + j) % 10 == 0:
                    image[0][i][j] = 1.0
        return image, _


data_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(0.5, 0.5)
])


if __name__ == "__main__":
    o = OccludedMNIST('./data', train=False, transform=transforms.ToTensor(), download=True)
    n = NoisyMNIST('./data', train=False, transform=transforms.ToTensor(), download=True)
    ol = DataLoader(o, 64, shuffle=False, num_workers=2)
    nl = DataLoader(n, 64, shuffle=False, num_workers=2)
    o_images, _ = iter(ol).__next__()
    n_images, _ = iter(nl).__next__()
    save_image(o_images, 'occluded.png', pad_value=1, padding=1)
    save_image(n_images, 'noisy.png', pad_value=1, padding=1)
