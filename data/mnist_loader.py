"""
Multi-modal MNIST dataset loader with transformations.
"""
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import datasets, transforms
import torchvision.transforms.functional as TF


class MultiModalMNIST(Dataset):
    """
    Multi-modal MNIST dataset with 3 modalities:
    - Modality 0: Original image (28x28, normalized [0,1])
    - Modality 1: Rotated 90° clockwise
    - Modality 2: Horizontally flipped
    """

    def __init__(self, root='./data', train=True, download=True):
        """
        Args:
            root: Root directory for MNIST data
            train: If True, use training set, otherwise test set
            download: If True, download dataset if not present
        """
        # Base transform: convert to tensor and normalize to [0, 1]
        base_transform = transforms.Compose([
            transforms.ToTensor(),
        ])

        self.mnist = datasets.MNIST(
            root=root,
            train=train,
            download=download,
            transform=base_transform
        )

    def __len__(self):
        return len(self.mnist)

    def __getitem__(self, idx):
        """
        Returns:
            x_m0: Original image (1, 28, 28)
            x_m1: Rotated 90° (1, 28, 28)
            x_m2: Horizontally flipped (1, 28, 28)
            label: Digit label (0-9)
        """
        image, label = self.mnist[idx]

        # Modality 0: Original
        x_m0 = image

        # Modality 1: Rotate 90° clockwise
        x_m1 = TF.rotate(image, angle=-90)

        # Modality 2: Horizontal flip
        x_m2 = TF.hflip(image)

        return x_m0, x_m1, x_m2, label


def get_dataloaders(batch_size=128, num_workers=4, val_split=0.1, root='./data'):
    """
    Create train, validation, and test dataloaders.

    Args:
        batch_size: Batch size for dataloaders
        num_workers: Number of worker processes for data loading
        val_split: Fraction of training data to use for validation
        root: Root directory for MNIST data

    Returns:
        train_loader, val_loader, test_loader
    """
    # Create datasets
    train_dataset_full = MultiModalMNIST(root=root, train=True, download=True)
    test_dataset = MultiModalMNIST(root=root, train=False, download=True)

    # Split training into train and validation
    train_size = int((1 - val_split) * len(train_dataset_full))
    val_size = len(train_dataset_full) - train_size

    train_dataset, val_dataset = random_split(
        train_dataset_full,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)  # Reproducibility
    )

    # Determine if CUDA is available for pin_memory
    pin_memory = torch.cuda.is_available()

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    print(f"Dataset sizes - Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")

    return train_loader, val_loader, test_loader


def get_single_modality_loader(modality_idx, split='train', batch_size=128,
                                num_workers=4, val_split=0.1, root='./data'):
    """
    Create a dataloader for a single modality (for VAE training).

    Args:
        modality_idx: Which modality to extract (0, 1, or 2)
        split: 'train', 'val', or 'test'
        batch_size: Batch size
        num_workers: Number of workers
        val_split: Validation split ratio
        root: Data root directory

    Returns:
        DataLoader yielding (images, labels) for the specified modality
    """
    train_loader, val_loader, test_loader = get_dataloaders(
        batch_size=batch_size,
        num_workers=num_workers,
        val_split=val_split,
        root=root
    )

    # Select the appropriate loader
    if split == 'train':
        base_loader = train_loader
    elif split == 'val':
        base_loader = val_loader
    elif split == 'test':
        base_loader = test_loader
    else:
        raise ValueError(f"Invalid split: {split}. Must be 'train', 'val', or 'test'.")

    # Create wrapper that extracts only the specified modality
    class SingleModalityLoader:
        def __init__(self, base_loader, modality_idx):
            self.base_loader = base_loader
            self.modality_idx = modality_idx

        def __iter__(self):
            for x_m0, x_m1, x_m2, labels in self.base_loader:
                modalities = [x_m0, x_m1, x_m2]
                yield modalities[self.modality_idx], labels

        def __len__(self):
            return len(self.base_loader)

        @property
        def dataset(self):
            return self.base_loader.dataset

    return SingleModalityLoader(base_loader, modality_idx)


if __name__ == '__main__':
    # Test the dataloader
    print("Testing Multi-Modal MNIST Dataloader...")

    train_loader, val_loader, test_loader = get_dataloaders(batch_size=8, num_workers=0)

    # Get one batch
    x_m0, x_m1, x_m2, labels = next(iter(train_loader))

    print(f"\nBatch shapes:")
    print(f"  Modality 0 (original): {x_m0.shape}")
    print(f"  Modality 1 (rot 90°):  {x_m1.shape}")
    print(f"  Modality 2 (flip H):   {x_m2.shape}")
    print(f"  Labels: {labels.shape}")
    print(f"\nSample labels: {labels.tolist()}")

    # Test single modality loader
    print("\n\nTesting Single Modality Loader (modality 0)...")
    single_loader = get_single_modality_loader(modality_idx=0, split='train', batch_size=8, num_workers=0)
    images, labels = next(iter(single_loader))
    print(f"Images shape: {images.shape}")
    print(f"Labels: {labels.tolist()}")
