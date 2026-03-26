import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

    
def get_mnist_dataloader(root, batch_size=64, train=True):
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    dataset = datasets.MNIST(root=root, train=train, transform=transform, download=True)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=train, drop_last=train)
    return dataloader

def get_cifar10_dataloader(root, batch_size=64, train=True):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    dataset = datasets.CIFAR10(root=root, train=train, download=True, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=train, drop_last=train)
    return dataloader