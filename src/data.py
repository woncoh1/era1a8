import torch
import torchvision


def get_dataset(
    transform:dict,
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
    params:dict,
) -> dict[str, torch.utils.data.DataLoader]:
    return {
        'train': torch.utils.data.DataLoader(dataset['train'], **params),
        'test': torch.utils.data.DataLoader(dataset['test'], **params),
    }
