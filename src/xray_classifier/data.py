"""Dataset loading and preprocessing for chest X-ray classification."""

from torchvision import datasets, transforms
from torch.utils.data import DataLoader

IMG_SIZE = 224


def get_transforms():
    """Return the standard preprocessing pipeline for X-ray images."""
    return transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])


def get_dataloaders(train_dir, test_dir, batch_size=32):
    """Build train and test DataLoaders from ImageFolder-structured directories."""
    transform = get_transforms()

    train_dataset = datasets.ImageFolder(root=train_dir, transform=transform)
    test_dataset = datasets.ImageFolder(root=test_dir, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader, train_dataset.classes
