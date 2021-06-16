import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda, Compose
import matplotlib.pyplot as plt

training_data = datasets.FashionMNIST(
        root="./MNIST/raw/train-images-idx3-ubyte.gz",
        train=True,
        download=False,
        transform=ToTensor()
)
