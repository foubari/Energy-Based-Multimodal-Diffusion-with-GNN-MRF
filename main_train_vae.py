"""
Main script to train VAE models for all 3 modalities.
"""
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'  # Fix OMP duplicate library issue on Windows

import argparse
import yaml
import torch
from torch.optim import Adam
from torch.cuda.amp import GradScaler
from torch.utils.tensorboard import SummaryWriter

from models.vae import VAE
from data.mnist_loader import get_single_modality_loader
from training.train_vae import train_one_epoch, validate, EarlyStopping, compute_beta
from utils.checkpoint import save_checkpoint
from utils.visualization import visualize_reconstructions, visualize_samples, plot_vae_losses


def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Convert to namespace for easy access
    class ConfigNamespace:
        def __init__(self, d):
            for key, value in d.items():
                if isinstance(value, dict):
                    setattr(self, key, ConfigNamespace(value))
                else:
                    setattr(self, key, value)

    return ConfigNamespace(config)


def train_vae_for_modality(modality_idx, config, device):
    """
    Train a single VAE for one modality.

    Args:
        modality_idx: Index of modality (0, 1, or 2)
        config: Configuration object
        device: Device to train on
    """
    print(f"\n{'='*60}")
    print(f"Training VAE for Modality {modality_idx}")
    print(f"{'='*60}\n")

    # Create dataloaders
    print("Loading data...")
    train_loader = get_single_modality_loader(
        modality_idx=modality_idx,
        split='train',
        batch_size=config.training.batch_size,
        num_workers=config.data.num_workers,
        val_split=config.data.val_split
    )

    val_loader = get_single_modality_loader(
        modality_idx=modality_idx,
        split='val',
        batch_size=config.training.batch_size,
        num_workers=config.data.num_workers,
        val_split=config.data.val_split
    )

    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

    # Create model
    print(f"\nCreating VAE model (latent_dim={config.model.latent_dim})...")
    model = VAE(latent_dim=config.model.latent_dim).to(device)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params:,}")

    # Optimizer
    optimizer = Adam(
        model.parameters(),
        lr=config.training.lr,
        betas=tuple(config.training.betas)
    )

    # Mixed precision scaler
    scaler = GradScaler() if config.training.use_amp else None
    if scaler:
        print("Using mixed precision training (AMP)")

    # TensorBoard writer
    log_dir = os.path.join(config.logging.tensorboard_dir, f'vae_m{modality_idx}')
    writer = SummaryWriter(log_dir)
    print(f"TensorBoard logs: {log_dir}")

    # Early stopping
    early_stopping = EarlyStopping(
        patience=config.training.early_stopping.patience,
        min_delta=config.training.early_stopping.min_delta
    )

    # Training history
    history = {
        'train_total': [],
        'train_recon': [],
        'train_kl': [],
        'val_total': [],
        'val_recon': [],
        'val_kl': [],
        'beta': []
    }

    best_val_loss = float('inf')

    # Training loop
    print(f"\nStarting training for {config.training.epochs} epochs...")
    print(f"Early stopping: patience={config.training.early_stopping.patience}\n")

    for epoch in range(1, config.training.epochs + 1):
        # Compute beta for KL annealing
        beta = compute_beta(epoch - 1, config)  # epoch-1 for 0-indexing

        # Train
        train_losses = train_one_epoch(
            model, train_loader, optimizer, scaler, beta, device, epoch
        )

        # Validate
        val_losses = validate(model, val_loader, beta, device, epoch)

        # Record history
        history['train_total'].append(train_losses['total'])
        history['train_recon'].append(train_losses['recon'])
        history['train_kl'].append(train_losses['kl'])
        history['val_total'].append(val_losses['total'])
        history['val_recon'].append(val_losses['recon'])
        history['val_kl'].append(val_losses['kl'])
        history['beta'].append(beta)

        # Log to TensorBoard
        writer.add_scalar('Loss/train_total', train_losses['total'], epoch)
        writer.add_scalar('Loss/train_recon', train_losses['recon'], epoch)
        writer.add_scalar('Loss/train_kl', train_losses['kl'], epoch)
        writer.add_scalar('Loss/val_total', val_losses['total'], epoch)
        writer.add_scalar('Loss/val_recon', val_losses['recon'], epoch)
        writer.add_scalar('Loss/val_kl', val_losses['kl'], epoch)
        writer.add_scalar('Training/beta', beta, epoch)

        # Print progress
        print(f"Epoch {epoch}/{config.training.epochs} | "
              f"β={beta:.3f} | "
              f"Train Loss: {train_losses['total']:.4f} | "
              f"Val Loss: {val_losses['total']:.4f} | "
              f"Recon: {val_losses['recon']:.4f} | "
              f"KL: {val_losses['kl']:.4f}")

        # Visualizations
        if epoch % config.logging.plot_every == 0:
            print(f"  → Generating visualizations...")
            visualize_reconstructions(
                model, val_loader, epoch, modality_idx,
                num_images=8, save_dir='outputs/plots', device=device
            )
            visualize_samples(
                model, epoch, modality_idx,
                num_samples=config.logging.num_samples_viz,
                save_dir='outputs/plots', device=device
            )

        # Checkpointing
        if epoch % config.training.save_every == 0:
            checkpoint_path = f'outputs/vae_checkpoints/vae_m{modality_idx}_epoch_{epoch}.pth'
            save_checkpoint(model, optimizer, epoch, val_losses['total'], checkpoint_path)

        # Save best model
        if val_losses['total'] < best_val_loss and config.training.save_best:
            best_val_loss = val_losses['total']
            best_checkpoint_path = f'outputs/vae_checkpoints/vae_modality_{modality_idx}.pth'
            save_checkpoint(model, optimizer, epoch, val_losses['total'], best_checkpoint_path)
            print(f"  → New best model saved (val_loss={best_val_loss:.4f})")

        # Early stopping
        if early_stopping(val_losses['total']):
            print(f"\n⚠ Early stopping triggered at epoch {epoch}")
            print(f"  Best val loss: {early_stopping.best_loss:.4f}")
            break

    # Final checkpoint
    final_checkpoint_path = f'outputs/vae_checkpoints/vae_m{modality_idx}_final.pth'
    save_checkpoint(model, optimizer, epoch, val_losses['total'], final_checkpoint_path)

    # Plot loss curves
    plot_path = f'outputs/plots/vae_m{modality_idx}_losses.png'
    plot_vae_losses(history, save_path=plot_path)

    writer.close()

    print(f"\n✓ Training complete for modality {modality_idx}")
    print(f"  Final val loss: {val_losses['total']:.4f}")
    print(f"  Best val loss: {best_val_loss:.4f}")


def main():
    parser = argparse.ArgumentParser(description='Train VAE models for all modalities')
    parser.add_argument('--config', type=str, default='configs/vae_config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--modality', type=int, default=None,
                       help='Train only specific modality (0, 1, or 2). If None, train all.')
    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)
    print(f"Loaded configuration from {args.config}")

    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

    # Create output directories
    os.makedirs('outputs/vae_checkpoints', exist_ok=True)
    os.makedirs('outputs/plots', exist_ok=True)
    os.makedirs('outputs/logs', exist_ok=True)

    # Train VAEs
    if args.modality is not None:
        # Train single modality
        if args.modality not in [0, 1, 2]:
            raise ValueError("Modality must be 0, 1, or 2")
        train_vae_for_modality(args.modality, config, device)
    else:
        # Train all 3 modalities
        for modality_idx in range(3):
            train_vae_for_modality(modality_idx, config, device)

    print("\n" + "="*60)
    print("All VAE training complete!")
    print("="*60)


if __name__ == '__main__':
    main()
