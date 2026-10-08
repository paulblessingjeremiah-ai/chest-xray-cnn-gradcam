"""Dataset loading and preprocessing for chest X-ray classification."""

from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

IMG_SIZE = 224


def get_transforms():
    """Return the standard preprocessing pipeline for X-ray images."""
    return transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])


def get_dataloaders(train_dir, test_dir, batch_size=32, val_fraction=0.15, seed=42):
    """Build train, validation and test DataLoaders from ImageFolder-structured directories.

    A stratified share of the training images is held back as a validation set.
    It is used to pick the best checkpoint. The test set is only for the final evaluation.
    """
    transform = get_transforms()
    full_train = datasets.ImageFolder(root=train_dir, transform=transform)
    test_dataset = datasets.ImageFolder(root=test_dir, transform=transform)

    train_idx, val_idx = train_test_split(
        list(range(len(full_train))),
        test_size=val_fraction,
        random_state=seed,
        stratify=full_train.targets,
    )

    train_loader = DataLoader(Subset(full_train, train_idx), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(Subset(full_train, val_idx), batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader, full_train.classes
