from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.datasets import MNIST
from tqdm import tqdm


class PartialMNIST(Dataset):
    def __init__(self, numbers, train=True, transform=transforms.ToTensor()):
        self.dataset = MNIST(root='./data', train=train, transform=transform, download=True)
        dataloader = DataLoader(self.dataset, batch_size=256, shuffle=False, num_workers=2)
        now = 0
        self.indices = []
        print('loading partial MNIST ...')
        for _, labels in tqdm(dataloader):
            self.indices += [i for i, label in enumerate(labels, now) if label in numbers]
            now += len(labels)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        return self.dataset[self.indices[index]]
