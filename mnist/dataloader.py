import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

def load_data(config):
        data_dir = config['data_dir']

        transform = transforms.Compose([
                transforms.Resize((28, 28)),
                transforms.Grayscale(),
                transforms.ToTensor(),
                transforms.Normalize((0,), (1,))])

        trainset = datasets.MNIST(data_dir, train=True, download=True, transform=transform)
        testset = datasets.MNIST(data_dir, train=False, download=True, transform=transform)

        return trainset, testset