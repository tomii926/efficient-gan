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


if __name__ == "__main__":
    o = OccludedMNIST('./data', train=False, transform=transforms.ToTensor(), download=True)
    ol = DataLoader(o, 64, shuffle=False, num_workers=2)
    images, _ = iter(ol).__next__()
    save_image(images, 'occluded.png', pad_value=1, padding=1)
