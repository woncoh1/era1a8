import torch
import torchvision


# Pixel statistics of all (train + test) CIFAR-10 images
# https://github.com/kuangliu/pytorch-cifar/blob/master/main.py
AVG = (0.4914, 0.4822, 0.4465) # Mean
STD = (0.2023, 0.1994, 0.2010) # Standard deviation
CHW = (3, 32, 32) # Channel, height, width
CLASSES = [
    'airplane',							
    'automobile',
    'bird',
    'cat',										
    'deer',									
    'dog',									
    'frog',									
    'horse',									
    'ship',										
    'truck',		
]


def get_dataset(
    transform:dict[str, torchvision.transforms],
) -> dict[str, torchvision.datasets]:
    return {
        'train': torchvision.datasets.CIFAR10(
            '../data',
            train=True,
            download=True,
            transform=transform['train'],
        ),
        'test': torchvision.datasets.CIFAR10(
            '../data',
            train=False,
            download=False,
            transform=transform['test'],
        ),
    }


def get_dataloader(
    dataset:torchvision.datasets,
    params:dict[str, bool|int],
) -> dict[str, torch.utils.data.DataLoader]:
    return {
        'train': torch.utils.data.DataLoader(dataset['train'], **params),
        'test': torch.utils.data.DataLoader(dataset['test'], **params),
    }
