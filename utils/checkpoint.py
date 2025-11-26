"""
Checkpoint management utilities for saving and loading model states.
"""
import os
import torch
from datetime import datetime


def save_checkpoint(model, optimizer, epoch, loss, filepath, **kwargs):
    """
    Save a complete checkpoint including model, optimizer, and training state.

    Args:
        model: PyTorch model
        optimizer: PyTorch optimizer
        epoch: Current epoch number
        loss: Current loss value
        filepath: Path to save checkpoint
        **kwargs: Additional data to save in checkpoint
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'timestamp': datetime.now().isoformat(),
        **kwargs
    }

    torch.save(checkpoint, filepath)
    print(f"Checkpoint saved to {filepath}")


def load_checkpoint(filepath, model, optimizer=None, device='cpu'):
    """
    Load a checkpoint and restore model and optimizer states.

    Args:
        filepath: Path to checkpoint file
        model: PyTorch model to load state into
        optimizer: Optional PyTorch optimizer to load state into
        device: Device to map checkpoint to

    Returns:
        Tuple of (epoch, loss, additional_data_dict)
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Checkpoint not found at {filepath}")

    checkpoint = torch.load(filepath, map_location=device)

    model.load_state_dict(checkpoint['model_state_dict'])

    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    epoch = checkpoint.get('epoch', 0)
    loss = checkpoint.get('loss', float('inf'))

    # Extract additional data (excluding standard keys)
    standard_keys = {'epoch', 'model_state_dict', 'optimizer_state_dict', 'loss', 'timestamp'}
    additional_data = {k: v for k, v in checkpoint.items() if k not in standard_keys}

    print(f"Checkpoint loaded from {filepath} (epoch {epoch}, loss {loss:.4f})")

    return epoch, loss, additional_data


def get_latest_checkpoint(checkpoint_dir, prefix=''):
    """
    Find the most recent checkpoint in a directory.

    Args:
        checkpoint_dir: Directory containing checkpoints
        prefix: Optional prefix to filter checkpoints

    Returns:
        Path to latest checkpoint or None if not found
    """
    if not os.path.exists(checkpoint_dir):
        return None

    checkpoints = [
        os.path.join(checkpoint_dir, f)
        for f in os.listdir(checkpoint_dir)
        if f.startswith(prefix) and f.endswith('.pth')
    ]

    if not checkpoints:
        return None

    # Sort by modification time
    latest = max(checkpoints, key=os.path.getmtime)
    return latest
