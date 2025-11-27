"""
Visualization utilities for VAE and EBM training.
"""
import os
import torch
import matplotlib.pyplot as plt
import numpy as np
from torchvision.utils import make_grid


def visualize_reconstructions(model, dataloader, epoch, modality_idx=None,
                               num_images=8, save_dir='outputs/plots', device='cpu'):
    """
    Visualize original vs reconstructed images from VAE.

    Args:
        model: Trained VAE model
        dataloader: DataLoader providing images
        epoch: Current epoch number (for filename)
        modality_idx: Optional modality index (for multimodal data)
        num_images: Number of image pairs to display
        save_dir: Directory to save plots
        device: Device to run model on
    """
    os.makedirs(save_dir, exist_ok=True)

    model.eval()
    with torch.no_grad():
        # Get one batch
        batch = next(iter(dataloader))
        if isinstance(batch[0], torch.Tensor) and len(batch) == 2:
            # Single modality: (images, labels)
            x, _ = batch
        else:
            # Multi-modal: extract modality
            x = batch[modality_idx] if modality_idx is not None else batch[0]

        x = x[:num_images].to(device)
        recon_x_logits, _, _ = model(x)

        # Apply sigmoid to convert logits to probabilities
        recon_x = torch.sigmoid(recon_x_logits)

        # Create figure
        fig, axes = plt.subplots(2, num_images, figsize=(num_images * 2, 4))

        for i in range(num_images):
            # Original
            axes[0, i].imshow(x[i].cpu().squeeze(), cmap='gray', vmin=0, vmax=1)
            axes[0, i].axis('off')
            if i == 0:
                axes[0, i].set_title('Original', fontsize=10)

            # Reconstructed
            axes[1, i].imshow(recon_x[i].cpu().squeeze(), cmap='gray', vmin=0, vmax=1)
            axes[1, i].axis('off')
            if i == 0:
                axes[1, i].set_title('Reconstructed', fontsize=10)

        plt.tight_layout()

        # Save
        if modality_idx is not None:
            filename = f'vae_m{modality_idx}_recon_epoch_{epoch}.png'
        else:
            filename = f'vae_recon_epoch_{epoch}.png'

        filepath = os.path.join(save_dir, filename)
        plt.savefig(filepath, dpi=100, bbox_inches='tight')
        plt.close()

        print(f"Saved reconstruction visualization to {filepath}")


def visualize_samples(model, epoch, modality_idx=None, num_samples=64,
                     save_dir='outputs/plots', device='cpu'):
    """
    Visualize samples generated from VAE prior.

    Args:
        model: Trained VAE model
        epoch: Current epoch number
        modality_idx: Optional modality index
        num_samples: Number of samples to generate
        save_dir: Directory to save plots
        device: Device to run model on
    """
    os.makedirs(save_dir, exist_ok=True)

    model.eval()
    with torch.no_grad():
        # Sample from prior N(0, I)
        z = torch.randn(num_samples, model.latent_dim).to(device)
        samples_logits = model.decode(z)

        # Apply sigmoid to convert logits to probabilities
        samples = torch.sigmoid(samples_logits)

        # Create grid
        grid = make_grid(samples.cpu(), nrow=8, normalize=True, value_range=(0, 1), padding=2)

        # Plot
        plt.figure(figsize=(10, 10))
        plt.imshow(grid.permute(1, 2, 0).squeeze(), cmap='gray')
        plt.axis('off')
        plt.title(f'Generated Samples (Epoch {epoch})', fontsize=14)

        # Save
        if modality_idx is not None:
            filename = f'vae_m{modality_idx}_samples_epoch_{epoch}.png'
        else:
            filename = f'vae_samples_epoch_{epoch}.png'

        filepath = os.path.join(save_dir, filename)
        plt.savefig(filepath, dpi=100, bbox_inches='tight')
        plt.close()

        print(f"Saved samples visualization to {filepath}")


def plot_vae_losses(losses_dict, save_path='outputs/plots/vae_losses.png'):
    """
    Plot VAE training and validation losses.

    Args:
        losses_dict: Dictionary containing loss histories
            {
                'train_total': [...],
                'train_recon': [...],
                'train_kl': [...],
                'val_total': [...],
                'val_recon': [...],
                'val_kl': [...]
            }
        save_path: Path to save plot
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    epochs = range(1, len(losses_dict['train_total']) + 1)

    # Total loss
    axes[0].plot(epochs, losses_dict['train_total'], label='Train', linewidth=2)
    axes[0].plot(epochs, losses_dict['val_total'], label='Val', linewidth=2)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Total Loss')
    axes[0].set_title('Total ELBO Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Reconstruction loss
    axes[1].plot(epochs, losses_dict['train_recon'], label='Train', linewidth=2)
    axes[1].plot(epochs, losses_dict['val_recon'], label='Val', linewidth=2)
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Reconstruction Loss')
    axes[1].set_title('Reconstruction Loss (BCE)')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # KL loss
    axes[2].plot(epochs, losses_dict['train_kl'], label='Train', linewidth=2)
    axes[2].plot(epochs, losses_dict['val_kl'], label='Val', linewidth=2)
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('KL Divergence')
    axes[2].set_title('KL Divergence')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Saved loss plot to {save_path}")


def visualize_energy_distribution(gnn_energy, vaes, dataloader, langevin_sampler,
                                  epoch, config, save_dir='outputs/plots', device='cpu'):
    """
    Visualize histogram of positive vs negative energies (EBM).

    Args:
        gnn_energy: Trained GNN energy network
        vaes: List of VAE models [vae_m0, vae_m1, vae_m2]
        dataloader: DataLoader for multimodal data
        langevin_sampler: LangevinSampler instance
        epoch: Current epoch
        config: Configuration dict
        save_dir: Directory to save plots
        device: Device
    """
    os.makedirs(save_dir, exist_ok=True)

    gnn_energy.eval()
    for vae in vaes:
        vae.eval()

    energies_pos = []
    energies_neg = []

    with torch.no_grad():
        for i, (x_m0, x_m1, x_m2, _) in enumerate(dataloader):
            if i >= 10:  # Limit to 10 batches
                break

            x_m0, x_m1, x_m2 = x_m0.to(device), x_m1.to(device), x_m2.to(device)

            # Positive: encode data
            mu0, _ = vaes[0].encode(x_m0)
            mu1, _ = vaes[1].encode(x_m1)
            mu2, _ = vaes[2].encode(x_m2)
            z_pos = torch.stack([mu0, mu1, mu2], dim=1)  # (B, 3, latent_dim)

            energy_pos = gnn_energy(z_pos)
            energies_pos.append(energy_pos.cpu())

            # Negative: sample from Langevin
            z_neg, _ = langevin_sampler.sample_unconditional(
                batch_size=x_m0.shape[0],
                num_modalities=3,
                latent_dim=config.model.latent_dim
            )
            energy_neg = gnn_energy(z_neg)
            energies_neg.append(energy_neg.cpu())

    energies_pos = torch.cat(energies_pos).numpy()
    energies_neg = torch.cat(energies_neg).numpy()

    # Plot histogram
    plt.figure(figsize=(10, 6))
    plt.hist(energies_pos, bins=50, alpha=0.6, label=f'E_pos (data) μ={energies_pos.mean():.2f}', color='blue', edgecolor='black')
    plt.hist(energies_neg, bins=50, alpha=0.6, label=f'E_neg (model) μ={energies_neg.mean():.2f}', color='red', edgecolor='black')
    plt.xlabel('Energy', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.title(f'Energy Distribution (Epoch {epoch})', fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)

    filepath = os.path.join(save_dir, f'ebm_energy_dist_epoch_{epoch}.png')
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Saved energy distribution to {filepath}")


def visualize_generated_samples_multimodal(gnn_energy, vaes, langevin_sampler,
                                           config, epoch, save_dir='outputs/plots', device='cpu'):
    """
    Generate and visualize samples for all 3 modalities (EBM).

    Args:
        gnn_energy: GNN energy network
        vaes: List of VAE decoders
        langevin_sampler: Langevin sampler
        config: Configuration
        epoch: Current epoch
        save_dir: Save directory
        device: Device
    """
    os.makedirs(save_dir, exist_ok=True)

    num_samples = config.logging.num_samples_viz

    gnn_energy.eval()
    for vae in vaes:
        vae.eval()

    # Generate latents
    with torch.no_grad():
        z_samples, _ = langevin_sampler.sample_unconditional(
            batch_size=num_samples,
            num_modalities=3,
            latent_dim=config.model.latent_dim
        )

        # Decode each modality (decode returns logits, apply sigmoid)
        images_m0 = torch.sigmoid(vaes[0].decode(z_samples[:, 0, :]))
        images_m1 = torch.sigmoid(vaes[1].decode(z_samples[:, 1, :]))
        images_m2 = torch.sigmoid(vaes[2].decode(z_samples[:, 2, :]))

    # Create figure: each row is one sample, 3 columns for modalities
    fig, axes = plt.subplots(num_samples, 3, figsize=(6, num_samples * 2))

    if num_samples == 1:
        axes = axes.reshape(1, -1)

    for i in range(num_samples):
        axes[i, 0].imshow(images_m0[i].cpu().squeeze(), cmap='gray', vmin=0, vmax=1)
        axes[i, 0].axis('off')
        if i == 0:
            axes[i, 0].set_title('M0 (Original)', fontsize=10)

        axes[i, 1].imshow(images_m1[i].cpu().squeeze(), cmap='gray', vmin=0, vmax=1)
        axes[i, 1].axis('off')
        if i == 0:
            axes[i, 1].set_title('M1 (Rot 90°)', fontsize=10)

        axes[i, 2].imshow(images_m2[i].cpu().squeeze(), cmap='gray', vmin=0, vmax=1)
        axes[i, 2].axis('off')
        if i == 0:
            axes[i, 2].set_title('M2 (Flip H)', fontsize=10)

    plt.tight_layout()

    filepath = os.path.join(save_dir, f'ebm_samples_epoch_{epoch}.png')
    plt.savefig(filepath, dpi=100, bbox_inches='tight')
    plt.close()

    print(f"Saved multimodal samples to {filepath}")


def visualize_conditional_generation(gnn_energy, vaes, dataloader, langevin_sampler,
                                     epoch, save_dir='outputs/plots', device='cpu'):
    """
    Conditional generation: observe M0, generate M1 and M2 (EBM).

    Args:
        gnn_energy: GNN energy network
        vaes: List of VAE models
        dataloader: Multimodal dataloader
        langevin_sampler: Langevin sampler
        epoch: Current epoch
        save_dir: Save directory
        device: Device
    """
    os.makedirs(save_dir, exist_ok=True)

    gnn_energy.eval()
    for vae in vaes:
        vae.eval()

    # Get one batch
    x_m0, x_m1, x_m2, _ = next(iter(dataloader))
    num_samples = min(8, x_m0.shape[0])

    x_m0 = x_m0[:num_samples].to(device)
    x_m1_true = x_m1[:num_samples].to(device)
    x_m2_true = x_m2[:num_samples].to(device)

    with torch.no_grad():
        # Encode observed modality
        z_m0, _ = vaes[0].encode(x_m0)

        # Initialize unobserved modalities
        z_m1_init = torch.randn_like(z_m0)
        z_m2_init = torch.randn_like(z_m0)
        z_init = torch.stack([z_m0, z_m1_init, z_m2_init], dim=1)

        # Conditional Langevin: fix M0, refine M1 and M2
        observed_mask = torch.tensor([[True, False, False]], device=z_init.device)
        observed_mask = observed_mask.expand(z_init.shape[0], -1)
        observed_values = z_init.clone()

        z_final, _ = langevin_sampler.sample(z_init, observed_mask, observed_values)

        # Decode (decode returns logits, apply sigmoid)
        x_m1_gen = torch.sigmoid(vaes[1].decode(z_final[:, 1, :]))
        x_m2_gen = torch.sigmoid(vaes[2].decode(z_final[:, 2, :]))

    # Plot: observed M0 | true M1, M2 | generated M1, M2
    fig, axes = plt.subplots(num_samples, 5, figsize=(10, num_samples * 2))

    if num_samples == 1:
        axes = axes.reshape(1, -1)

    for i in range(num_samples):
        axes[i, 0].imshow(x_m0[i].cpu().squeeze(), cmap='gray', vmin=0, vmax=1)
        axes[i, 0].axis('off')
        if i == 0:
            axes[i, 0].set_title('M0 (obs)', fontsize=10)

        axes[i, 1].imshow(x_m1_true[i].cpu().squeeze(), cmap='gray', vmin=0, vmax=1)
        axes[i, 1].axis('off')
        if i == 0:
            axes[i, 1].set_title('M1 true', fontsize=10)

        axes[i, 2].imshow(x_m1_gen[i].cpu().squeeze(), cmap='gray', vmin=0, vmax=1)
        axes[i, 2].axis('off')
        if i == 0:
            axes[i, 2].set_title('M1 gen', fontsize=10)

        axes[i, 3].imshow(x_m2_true[i].cpu().squeeze(), cmap='gray', vmin=0, vmax=1)
        axes[i, 3].axis('off')
        if i == 0:
            axes[i, 3].set_title('M2 true', fontsize=10)

        axes[i, 4].imshow(x_m2_gen[i].cpu().squeeze(), cmap='gray', vmin=0, vmax=1)
        axes[i, 4].axis('off')
        if i == 0:
            axes[i, 4].set_title('M2 gen', fontsize=10)

    plt.tight_layout()

    filepath = os.path.join(save_dir, f'ebm_conditional_epoch_{epoch}.png')
    plt.savefig(filepath, dpi=100, bbox_inches='tight')
    plt.close()

    print(f"Saved conditional generation to {filepath}")
