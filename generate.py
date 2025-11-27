"""
Generation script for unconditional and conditional multimodal sampling.
"""
import argparse
import os
import torch
import matplotlib.pyplot as plt
from PIL import Image
import torchvision.transforms as transforms

from models.vae import VAE
from models.gnn_energy import GNNEnergyNetwork
from models.langevin_sampler import LangevinSampler
from utils.checkpoint import load_checkpoint


def load_models(vae_dir, ebm_checkpoint, latent_dim, device):
    """
    Load pre-trained VAEs and EBM.

    Args:
        vae_dir: Directory containing VAE checkpoints
        ebm_checkpoint: Path to EBM checkpoint
        latent_dim: Latent dimension
        device: Device

    Returns:
        vaes: List of VAE models
        gnn_energy: GNN energy network
    """
    # Load VAEs
    vaes = []
    for m in range(3):
        vae_path = os.path.join(vae_dir, f'vae_modality_{m}.pth')
        vae = VAE(latent_dim=latent_dim).to(device)
        load_checkpoint(vae_path, vae, device=device)
        vae.eval()
        vaes.append(vae)

    # Load EBM
    edge_index = torch.tensor([
        [0, 0, 1, 1, 2, 2],
        [1, 2, 0, 2, 0, 1]
    ], dtype=torch.long)

    gnn_energy = GNNEnergyNetwork(
        num_modalities=3,
        latent_dim=latent_dim,
        modality_emb_dim=8,
        hidden_dim=64,
        num_layers=3,
        edge_index=edge_index
    ).to(device)

    load_checkpoint(ebm_checkpoint, gnn_energy, device=device)
    gnn_energy.eval()

    return vaes, gnn_energy


def generate_unconditional(vaes, gnn_energy, langevin_sampler,
                           num_samples, latent_dim, save_path, device):
    """
    Generate unconditional samples.

    Args:
        vaes: List of VAE models
        gnn_energy: GNN energy network
        langevin_sampler: Langevin sampler
        num_samples: Number of samples to generate
        latent_dim: Latent dimension
        save_path: Path to save visualization
        device: Device
    """
    print(f"\nGenerating {num_samples} unconditional samples...")

    with torch.no_grad():
        # Generate latents via Langevin
        z_samples, _ = langevin_sampler.sample_unconditional(
            batch_size=num_samples,
            num_modalities=3,
            latent_dim=latent_dim
        )

        # Decode each modality (decode returns logits, apply sigmoid)
        images_m0 = torch.sigmoid(vaes[0].decode(z_samples[:, 0, :]))
        images_m1 = torch.sigmoid(vaes[1].decode(z_samples[:, 1, :]))
        images_m2 = torch.sigmoid(vaes[2].decode(z_samples[:, 2, :]))

    # Visualize
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
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"✓ Saved unconditional samples to {save_path}")


def generate_conditional(vaes, gnn_energy, langevin_sampler,
                         input_image_path, observed_modality, latent_dim, save_path, device):
    """
    Generate conditional samples (impute missing modalities).

    Args:
        vaes: List of VAE models
        gnn_energy: GNN energy network
        langevin_sampler: Langevin sampler
        input_image_path: Path to input image
        observed_modality: Index of observed modality (0, 1, or 2)
        latent_dim: Latent dimension
        save_path: Path to save visualization
        device: Device
    """
    print(f"\nConditional generation from modality {observed_modality}...")
    print(f"Input image: {input_image_path}")

    # Load and preprocess image
    transform = transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize((28, 28)),
        transforms.ToTensor()
    ])

    image = Image.open(input_image_path)
    x_obs = transform(image).unsqueeze(0).to(device)  # (1, 1, 28, 28)

    with torch.no_grad():
        # Encode observed modality
        z_obs, _ = vaes[observed_modality].encode(x_obs)

        # Generate missing modalities
        z_full, _ = langevin_sampler.sample_conditional(
            z_observed=z_obs,
            observed_modality_idx=observed_modality,
            num_modalities=3,
            latent_dim=latent_dim
        )

        # Decode all modalities (decode returns logits, apply sigmoid)
        images = []
        for m in range(3):
            img_logits = vaes[m].decode(z_full[:, m, :])
            img = torch.sigmoid(img_logits)
            images.append(img)

    # Visualize
    fig, axes = plt.subplots(1, 3, figsize=(9, 3))

    titles = ['M0 (Original)', 'M1 (Rot 90°)', 'M2 (Flip H)']

    for m in range(3):
        axes[m].imshow(images[m][0].cpu().squeeze(), cmap='gray', vmin=0, vmax=1)
        axes[m].axis('off')
        if m == observed_modality:
            axes[m].set_title(f'{titles[m]}\n(Observed)', fontsize=10, color='green', fontweight='bold')
        else:
            axes[m].set_title(f'{titles[m]}\n(Generated)', fontsize=10)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"✓ Saved conditional samples to {save_path}")


def main():
    parser = argparse.ArgumentParser(description='Generate multimodal samples')
    parser.add_argument('--mode', choices=['unconditional', 'conditional'], required=True,
                       help='Generation mode')
    parser.add_argument('--ebm_checkpoint', type=str, required=True,
                       help='Path to EBM checkpoint')
    parser.add_argument('--vae_dir', type=str, default='outputs/vae_checkpoints',
                       help='Directory containing VAE checkpoints')
    parser.add_argument('--num_samples', type=int, default=16,
                       help='Number of samples (for unconditional)')
    parser.add_argument('--langevin_steps', type=int, default=50,
                       help='Number of Langevin steps')
    parser.add_argument('--step_size', type=float, default=0.01,
                       help='Langevin step size')
    parser.add_argument('--input_image', type=str, default=None,
                       help='Input image path (for conditional)')
    parser.add_argument('--observed_modality', type=int, default=0,
                       help='Which modality is observed (0, 1, or 2)')
    parser.add_argument('--output', type=str, default='outputs/plots/generated.png',
                       help='Output path for generated images')
    parser.add_argument('--latent_dim', type=int, default=16,
                       help='Latent dimension')

    args = parser.parse_args()

    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Create output directory
    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    # Load models
    print("\nLoading models...")
    vaes, gnn_energy = load_models(args.vae_dir, args.ebm_checkpoint, args.latent_dim, device)
    print("✓ Models loaded")

    # Create Langevin sampler
    langevin_sampler = LangevinSampler(
        energy_fn=gnn_energy,
        num_steps=args.langevin_steps,
        step_size=args.step_size,
        clip_grad=1.0,
        device=device
    )
    print(f"✓ Langevin sampler created ({args.langevin_steps} steps, step_size={args.step_size})")

    # Generate
    if args.mode == 'unconditional':
        generate_unconditional(
            vaes, gnn_energy, langevin_sampler,
            args.num_samples, args.latent_dim, args.output, device
        )

    elif args.mode == 'conditional':
        if args.input_image is None:
            raise ValueError("--input_image required for conditional generation")

        generate_conditional(
            vaes, gnn_energy, langevin_sampler,
            args.input_image, args.observed_modality, args.latent_dim, args.output, device
        )

    print(f"\n✓ Generation complete!")
    print(f"Output saved to: {args.output}")


if __name__ == '__main__':
    main()
