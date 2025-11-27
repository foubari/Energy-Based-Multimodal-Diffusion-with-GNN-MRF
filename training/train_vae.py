"""
Training functions and utilities for VAE.
"""
import torch
import torch.nn.functional as F
from torch.cuda.amp import autocast
from tqdm import tqdm


class EarlyStopping:
    """Early stopping handler to prevent overfitting."""

    def __init__(self, patience=15, min_delta=1e-4):
        """
        Args:
            patience: Number of epochs to wait before stopping
            min_delta: Minimum change in loss to qualify as improvement
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.should_stop = False

    def __call__(self, val_loss):
        """
        Check if training should stop.

        Args:
            val_loss: Current validation loss

        Returns:
            True if training should stop, False otherwise
        """
        if self.best_loss is None:
            self.best_loss = val_loss
            return False

        if val_loss < self.best_loss - self.min_delta:
            # Improvement
            self.best_loss = val_loss
            self.counter = 0
            return False
        else:
            # No improvement
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
                return True
            return False


def compute_beta(epoch, config):
    """
    Compute β for KL annealing in β-VAE.

    Args:
        epoch: Current epoch number
        config: Configuration object

    Returns:
        Beta value for current epoch
    """
    if not config.training.beta_kl_schedule.enabled:
        return 1.0

    start = config.training.beta_kl_schedule.start
    end = config.training.beta_kl_schedule.end
    warmup_epochs = config.training.beta_kl_schedule.warmup_epochs

    if epoch >= warmup_epochs:
        return end

    # Linear warmup
    beta = start + (end - start) * (epoch / warmup_epochs)
    return beta


def train_one_epoch(model, dataloader, optimizer, scaler, beta, device, epoch):
    """
    Train VAE for one epoch.

    Args:
        model: VAE model
        dataloader: Training dataloader
        optimizer: Optimizer
        scaler: GradScaler for mixed precision (or None)
        beta: β coefficient for KL divergence
        device: Device to train on
        epoch: Current epoch number

    Returns:
        Dictionary with average losses: {'total', 'recon', 'kl'}
    """
    model.train()

    total_loss_sum = 0
    recon_loss_sum = 0
    kl_loss_sum = 0
    num_samples = 0

    # Progress bar
    pbar = tqdm(dataloader, desc=f'Epoch {epoch} [Train]', leave=False)

    for batch in pbar:
        # Extract images and move to device
        if isinstance(batch[0], torch.Tensor) and len(batch) == 2:
            x, _ = batch
        else:
            x, _, _, _ = batch  # Multimodal case, take first element
        x = x.to(device)

        batch_size = x.shape[0]

        # Forward pass with mixed precision
        with autocast(enabled=(scaler is not None)):
            recon_x, mu, logvar = model(x)

            # Compute loss (using binary_cross_entropy_with_logits for AMP stability)
            recon_loss = F.binary_cross_entropy_with_logits(recon_x, x, reduction='sum')
            kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
            total_loss = recon_loss + beta * kl_loss

        # Backward pass
        optimizer.zero_grad()

        if scaler is not None:
            scaler.scale(total_loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            total_loss.backward()
            optimizer.step()

        # Accumulate losses
        total_loss_sum += total_loss.item()
        recon_loss_sum += recon_loss.item()
        kl_loss_sum += kl_loss.item()
        num_samples += batch_size

        # Update progress bar
        pbar.set_postfix({
            'loss': total_loss.item() / batch_size,
            'recon': recon_loss.item() / batch_size,
            'kl': kl_loss.item() / batch_size
        })

    # Compute averages per sample
    avg_losses = {
        'total': total_loss_sum / num_samples,
        'recon': recon_loss_sum / num_samples,
        'kl': kl_loss_sum / num_samples
    }

    return avg_losses


def validate(model, dataloader, beta, device, epoch):
    """
    Validate VAE on validation set.

    Args:
        model: VAE model
        dataloader: Validation dataloader
        beta: β coefficient for KL divergence
        device: Device
        epoch: Current epoch number

    Returns:
        Dictionary with average losses: {'total', 'recon', 'kl'}
    """
    model.eval()

    total_loss_sum = 0
    recon_loss_sum = 0
    kl_loss_sum = 0
    num_samples = 0

    pbar = tqdm(dataloader, desc=f'Epoch {epoch} [Val]', leave=False)

    with torch.no_grad():
        for batch in pbar:
            # Extract images
            if isinstance(batch[0], torch.Tensor) and len(batch) == 2:
                x, _ = batch
            else:
                x, _, _, _ = batch
            x = x.to(device)

            batch_size = x.shape[0]

            # Forward pass
            recon_x, mu, logvar = model(x)

            # Compute loss (using binary_cross_entropy_with_logits for AMP stability)
            recon_loss = F.binary_cross_entropy_with_logits(recon_x, x, reduction='sum')
            kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
            total_loss = recon_loss + beta * kl_loss

            # Accumulate
            total_loss_sum += total_loss.item()
            recon_loss_sum += recon_loss.item()
            kl_loss_sum += kl_loss.item()
            num_samples += batch_size

            # Update progress bar
            pbar.set_postfix({
                'loss': total_loss.item() / batch_size,
                'recon': recon_loss.item() / batch_size,
                'kl': kl_loss.item() / batch_size
            })

    # Compute averages
    avg_losses = {
        'total': total_loss_sum / num_samples,
        'recon': recon_loss_sum / num_samples,
        'kl': kl_loss_sum / num_samples
    }

    return avg_losses


if __name__ == '__main__':
    # Test training functions
    print("Testing VAE training functions...")

    import sys
    sys.path.append('..')

    from models.vae import VAE
    from data.mnist_loader import get_single_modality_loader

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Create model
    model = VAE(latent_dim=16).to(device)

    # Create optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # Create dataloader
    train_loader = get_single_modality_loader(
        modality_idx=0,
        split='train',
        batch_size=32,
        num_workers=0
    )

    # Test one epoch
    print("\nTesting one epoch...")
    from torch.cuda.amp import GradScaler
    scaler = GradScaler() if torch.cuda.is_available() else None

    losses = train_one_epoch(model, train_loader, optimizer, scaler, beta=1.0, device=device, epoch=1)

    print(f"\nEpoch losses:")
    print(f"  Total: {losses['total']:.4f}")
    print(f"  Reconstruction: {losses['recon']:.4f}")
    print(f"  KL: {losses['kl']:.4f}")

    # Test early stopping
    print("\nTesting early stopping...")
    early_stopping = EarlyStopping(patience=3, min_delta=1e-3)

    test_losses = [10.0, 9.5, 9.4, 9.35, 9.34, 9.33, 9.32]
    for i, loss in enumerate(test_losses):
        should_stop = early_stopping(loss)
        print(f"  Epoch {i+1}, Loss: {loss:.2f}, Should stop: {should_stop}")

    print("\n✓ Training functions test passed!")
