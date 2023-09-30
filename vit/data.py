import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms


TRAIN_TRANSFORMS = transforms.Compose([
    transforms.ToTensor(),
    transforms.RandomCrop(32, 4),
    transforms.RandomHorizontalFlip(),
    transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
])


VAL_TRANSFORMS = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
])


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
    'truck'
]


def load_dataset(batch_size = 16):
    trainset = datasets.CIFAR10(root='data', train=True, download=True, transform=TRAIN_TRANSFORMS)
    valset = datasets.CIFAR10(root='data', train=False, download=True, transform=VAL_TRANSFORMS)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, pin_memory=True)
    valloader = torch.utils.data.DataLoader(valset, batch_size=batch_size, shuffle=False, pin_memory=True)
    return trainloader, valloader


def compute_accuracy(outputs, labels):
    _, max_indices = torch.max(outputs, dim=1)
    num_matches = torch.sum(max_indices == labels)
    return (num_matches.item() / labels.numel()) * 100.0