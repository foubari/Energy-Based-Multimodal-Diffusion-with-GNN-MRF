"""
Main script to train EBM (GNN Energy Network) with frozen VAEs.
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
from models.gnn_energy import GNNEnergyNetwork
from models.langevin_sampler import LangevinSampler
from data.mnist_loader import get_dataloaders
from training.train_ebm import train_one_epoch, ReplayBuffer
from utils.checkpoint import save_checkpoint, load_checkpoint
from utils.visualization import (
    visualize_energy_distribution,
    visualize_generated_samples_multimodal,
    visualize_conditional_generation
)


def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    class ConfigNamespace:
        def __init__(self, d):
            for key, value in d.items():
                if isinstance(value, dict):
                    setattr(self, key, ConfigNamespace(value))
                elif isinstance(value, list):
                    setattr(self, key, value)
                else:
                    setattr(self, key, value)

    return ConfigNamespace(config)


def load_frozen_vaes(config, device):
    """
    Load pre-trained VAE models and freeze them.

    Args:
        config: Configuration object
        device: Device to load models on

    Returns:
        List of frozen VAE models [vae_m0, vae_m1, vae_m2]
    """
    vaes = []
    vae_dir = config.data.vae_checkpoints_dir

    print("Loading pre-trained VAEs...")
    for m in range(3):
        vae_path = os.path.join(vae_dir, f'vae_modality_{m}.pth')

        if not os.path.exists(vae_path):
            raise FileNotFoundError(
                f"VAE checkpoint not found: {vae_path}\n"
                f"Please train VAEs first using: python main_train_vae.py"
            )

        vae = VAE(latent_dim=config.model.latent_dim).to(device)
        load_checkpoint(vae_path, vae, device=device)

        # Freeze parameters
        vae.eval()
        for param in vae.parameters():
            param.requires_grad = False

        vaes.append(vae)
        print(f"  ✓ Loaded and froze VAE for modality {m}")

    return vaes


def main():
    parser = argparse.ArgumentParser(description='Train EBM (GNN Energy Network)')
    parser.add_argument('--config', type=str, default='configs/ebm_config.yaml',
                       help='Path to configuration file')
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
    os.makedirs('outputs/ebm_checkpoints', exist_ok=True)
    os.makedirs('outputs/plots', exist_ok=True)
    os.makedirs('outputs/logs', exist_ok=True)

    print("\n" + "="*60)
    print("Initializing EBM Training")
    print("="*60)

    # 1. Load frozen VAEs
    vaes = load_frozen_vaes(config, device)

    # 2. Create GNN Energy Network
    print(f"\nCreating GNN Energy Network...")
    edge_index = torch.tensor(config.model.graph.edge_index, dtype=torch.long)
    print(f"  Graph structure: {edge_index.t().tolist()}")

    gnn_energy = GNNEnergyNetwork(
        num_modalities=config.model.num_modalities,
        latent_dim=config.model.latent_dim,
        modality_emb_dim=config.model.modality_emb_dim,
        hidden_dim=config.model.hidden_dim_gnn,
        num_layers=config.model.num_gnn_layers,
        edge_index=edge_index
    ).to(device)

    num_params = sum(p.numel() for p in gnn_energy.parameters())
    print(f"  GNN parameters: {num_params:,}")

    # 3. Setup optimizer
    optimizer = Adam(
        gnn_energy.parameters(),
        lr=config.training.lr,
        betas=tuple(config.training.betas)
    )

    # Mixed precision scaler
    scaler = GradScaler() if config.training.use_amp else None
    if scaler:
        print("  Using mixed precision training (AMP)")

    # 4. Create Langevin sampler
    print(f"\nSetting up Langevin sampler...")
    langevin_sampler = LangevinSampler(
        energy_fn=gnn_energy,
        num_steps=config.training.langevin.num_steps_train,
        step_size=config.training.langevin.step_size,
        clip_grad=config.training.langevin.clip_grad,
        device=device
    )
    print(f"  Training steps: {config.training.langevin.num_steps_train}")
    print(f"  Step size: {config.training.langevin.step_size}")

    # 5. Create replay buffer
    print(f"\nInitializing replay buffer...")
    replay_buffer = ReplayBuffer(
        buffer_size=config.training.buffer_size,
        num_modalities=config.model.num_modalities,
        latent_dim=config.model.latent_dim,
        device=device
    )
    print(f"  Buffer size: {config.training.buffer_size}")
    print(f"  Reinit ratio: {config.training.reinit_ratio}")

    # 6. Create dataloader
    print(f"\nLoading data...")
    train_loader, _, test_loader = get_dataloaders(
        batch_size=config.training.batch_size,
        num_workers=config.data.num_workers
    )
    print(f"  Train batches: {len(train_loader)}")

    # 7. TensorBoard
    log_dir = os.path.join(config.logging.tensorboard_dir, 'ebm')
    writer = SummaryWriter(log_dir)
    print(f"  TensorBoard logs: {log_dir}")

    # 8. Training loop
    print(f"\n{'='*60}")
    print(f"Starting EBM training for {config.training.epochs} epochs")
    print(f"{'='*60}\n")

    best_loss = float('inf')

    for epoch in range(1, config.training.epochs + 1):
        # Train
        metrics = train_one_epoch(
            gnn_energy, vaes, train_loader, optimizer, scaler,
            langevin_sampler, replay_buffer, config, device, epoch
        )

        # Log to TensorBoard
        writer.add_scalar('Loss/train', metrics['loss'], epoch)
        writer.add_scalar('Energy/positive', metrics['energy_pos'], epoch)
        writer.add_scalar('Energy/negative', metrics['energy_neg'], epoch)
        writer.add_scalar('Energy/margin', metrics['margin'], epoch)

        # Print progress
        print(f"Epoch {epoch}/{config.training.epochs} | "
              f"Loss: {metrics['loss']:.4f} | "
              f"E_pos: {metrics['energy_pos']:.4f} | "
              f"E_neg: {metrics['energy_neg']:.4f} | "
              f"Margin: {metrics['margin']:.4f}")

        # Visualizations
        if epoch % config.logging.plot_every == 0:
            print(f"  → Generating visualizations...")

            # Energy distribution
            visualize_energy_distribution(
                gnn_energy, vaes, test_loader, langevin_sampler,
                epoch, config, save_dir='outputs/plots', device=device
            )

            # Generated samples
            visualize_generated_samples_multimodal(
                gnn_energy, vaes, langevin_sampler,
                config, epoch, save_dir='outputs/plots', device=device
            )

            # Conditional generation
            visualize_conditional_generation(
                gnn_energy, vaes, test_loader, langevin_sampler,
                epoch, save_dir='outputs/plots', device=device
            )

        # Checkpointing
        if epoch % config.training.save_every == 0:
            checkpoint_path = f'outputs/ebm_checkpoints/gnn_energy_epoch_{epoch}.pth'
            save_checkpoint(gnn_energy, optimizer, epoch, metrics['loss'], checkpoint_path)

        # Save best model
        if metrics['loss'] < best_loss and config.training.save_best:
            best_loss = metrics['loss']
            best_checkpoint_path = 'outputs/ebm_checkpoints/gnn_energy_best.pth'
            save_checkpoint(gnn_energy, optimizer, epoch, metrics['loss'], best_checkpoint_path)
            print(f"  → New best model saved (loss={best_loss:.4f})")

    # Final checkpoint
    final_checkpoint_path = 'outputs/ebm_checkpoints/gnn_energy_final.pth'
    save_checkpoint(gnn_energy, optimizer, epoch, metrics['loss'], final_checkpoint_path)

    writer.close()

    print(f"\n{'='*60}")
    print("EBM Training Complete!")
    print(f"{'='*60}")
    print(f"Final loss: {metrics['loss']:.4f}")
    print(f"Best loss: {best_loss:.4f}")
    print(f"\nCheckpoints saved to: outputs/ebm_checkpoints/")
    print(f"Visualizations saved to: outputs/plots/")


if __name__ == '__main__':
    main()
