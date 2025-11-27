"""
Metrics and evaluation utilities for VAE and EBM.
"""
import torch
import torch.nn.functional as F
import numpy as np
from sklearn.manifold import TSNE


def compute_elbo(recon_x, x, mu, logvar):
    """
    Compute ELBO (Evidence Lower Bound) for VAE.

    Args:
        recon_x: Reconstructed images
        x: Original images
        mu: Latent mean
        logvar: Latent log variance

    Returns:
        ELBO per sample (negative loss, higher is better)
    """
    batch_size = x.shape[0]
    recon_loss = F.binary_cross_entropy(recon_x, x, reduction='sum')
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    elbo = -(recon_loss + kl_loss) / batch_size
    return elbo.item()


def compute_reconstruction_error(model, dataloader, device):
    """
    Compute average reconstruction error over dataset.

    Args:
        model: VAE model
        dataloader: Data loader
        device: Device

    Returns:
        Average MSE reconstruction error
    """
    model.eval()
    total_mse = 0
    num_samples = 0

    with torch.no_grad():
        for batch in dataloader:
            if isinstance(batch[0], torch.Tensor) and len(batch) == 2:
                x, _ = batch
            else:
                x, _, _, _ = batch

            x = x.to(device)
            recon_x_logits, _, _ = model(x)
            recon_x = torch.sigmoid(recon_x_logits)

            mse = F.mse_loss(recon_x, x, reduction='sum')
            total_mse += mse.item()
            num_samples += x.shape[0]

    return total_mse / num_samples


def compute_energy_statistics(energies_pos, energies_neg):
    """
    Compute statistics for energy distributions.

    Args:
        energies_pos: Positive energies (data) - torch.Tensor or np.array
        energies_neg: Negative energies (model) - torch.Tensor or np.array

    Returns:
        Dictionary with statistics
    """
    if isinstance(energies_pos, torch.Tensor):
        energies_pos = energies_pos.cpu().numpy()
    if isinstance(energies_neg, torch.Tensor):
        energies_neg = energies_neg.cpu().numpy()

    stats = {
        'pos_mean': energies_pos.mean(),
        'pos_std': energies_pos.std(),
        'neg_mean': energies_neg.mean(),
        'neg_std': energies_neg.std(),
        'margin': energies_pos.mean() - energies_neg.mean(),
        'overlap': compute_distribution_overlap(energies_pos, energies_neg)
    }

    return stats


def compute_distribution_overlap(dist1, dist2, num_bins=50):
    """
    Compute overlap between two distributions (histogram intersection).

    Args:
        dist1: First distribution
        dist2: Second distribution
        num_bins: Number of bins for histogram

    Returns:
        Overlap ratio [0, 1] (0 = no overlap, 1 = complete overlap)
    """
    # Create histograms
    min_val = min(dist1.min(), dist2.min())
    max_val = max(dist1.max(), dist2.max())
    bins = np.linspace(min_val, max_val, num_bins + 1)

    hist1, _ = np.histogram(dist1, bins=bins, density=True)
    hist2, _ = np.histogram(dist2, bins=bins, density=True)

    # Normalize
    hist1 = hist1 / hist1.sum() if hist1.sum() > 0 else hist1
    hist2 = hist2 / hist2.sum() if hist2.sum() > 0 else hist2

    # Compute intersection
    overlap = np.minimum(hist1, hist2).sum()

    return overlap


def compute_latent_statistics(model, dataloader, device):
    """
    Compute statistics of latent codes.

    Args:
        model: VAE model
        dataloader: Data loader
        device: Device

    Returns:
        Dictionary with latent statistics
    """
    model.eval()

    all_mu = []
    all_logvar = []

    with torch.no_grad():
        for batch in dataloader:
            if isinstance(batch[0], torch.Tensor) and len(batch) == 2:
                x, _ = batch
            else:
                x, _, _, _ = batch

            x = x.to(device)
            mu, logvar = model.encode(x)

            all_mu.append(mu.cpu())
            all_logvar.append(logvar.cpu())

    all_mu = torch.cat(all_mu, dim=0).numpy()
    all_logvar = torch.cat(all_logvar, dim=0).numpy()

    stats = {
        'mu_mean': all_mu.mean(axis=0),
        'mu_std': all_mu.std(axis=0),
        'logvar_mean': all_logvar.mean(axis=0),
        'logvar_std': all_logvar.std(axis=0),
        'mu_norm_mean': np.linalg.norm(all_mu, axis=1).mean(),
        'kl_per_dim': -0.5 * (1 + all_logvar - all_mu**2 - np.exp(all_logvar)).mean(axis=0)
    }

    return stats


def visualize_latent_space_2d(model, dataloader, device, method='tsne', num_samples=1000):
    """
    Visualize latent space in 2D using t-SNE or PCA.

    Args:
        model: VAE model
        dataloader: Data loader
        device: Device
        method: 'tsne' or 'pca'
        num_samples: Number of samples to visualize

    Returns:
        2D coordinates and labels
    """
    model.eval()

    latents = []
    labels = []

    with torch.no_grad():
        for batch in dataloader:
            if isinstance(batch[0], torch.Tensor) and len(batch) == 2:
                x, label = batch
            else:
                x, _, _, label = batch

            x = x.to(device)
            mu, _ = model.encode(x)

            latents.append(mu.cpu())
            labels.append(label)

            if len(latents) * x.shape[0] >= num_samples:
                break

    latents = torch.cat(latents, dim=0)[:num_samples].numpy()
    labels = torch.cat(labels, dim=0)[:num_samples].numpy()

    # Dimensionality reduction
    if method == 'tsne':
        reducer = TSNE(n_components=2, random_state=42)
        latents_2d = reducer.fit_transform(latents)
    elif method == 'pca':
        from sklearn.decomposition import PCA
        reducer = PCA(n_components=2, random_state=42)
        latents_2d = reducer.fit_transform(latents)
    else:
        raise ValueError(f"Unknown method: {method}")

    return latents_2d, labels


def multimodal_consistency_score(vaes, gnn_energy, langevin_sampler,
                                 num_samples=100, device='cpu'):
    """
    Check consistency of generated samples across modalities.
    This is a simple heuristic: check if generated samples are similar when
    rotated/flipped back to original orientation.

    Args:
        vaes: List of VAE models
        gnn_energy: GNN energy model
        langevin_sampler: Langevin sampler
        num_samples: Number of samples
        device: Device

    Returns:
        Consistency score (MSE between aligned modalities)
    """
    import torchvision.transforms.functional as TF

    # Generate samples
    with torch.no_grad():
        z_samples, _ = langevin_sampler.sample_unconditional(
            batch_size=num_samples,
            num_modalities=3,
            latent_dim=16
        )

        # Decode each modality (decode returns logits, apply sigmoid)
        img_m0 = torch.sigmoid(vaes[0].decode(z_samples[:, 0, :]))  # Original
        img_m1 = torch.sigmoid(vaes[1].decode(z_samples[:, 1, :]))  # Rotated 90°
        img_m2 = torch.sigmoid(vaes[2].decode(z_samples[:, 2, :]))  # Flipped H

        # Align back to original orientation
        img_m1_aligned = TF.rotate(img_m1, angle=90)  # Rotate back
        img_m2_aligned = TF.hflip(img_m2)  # Flip back

        # Compute MSE between aligned images
        mse_m0_m1 = F.mse_loss(img_m0, img_m1_aligned).item()
        mse_m0_m2 = F.mse_loss(img_m0, img_m2_aligned).item()
        mse_m1_m2 = F.mse_loss(img_m1_aligned, img_m2_aligned).item()

    consistency_score = (mse_m0_m1 + mse_m0_m2 + mse_m1_m2) / 3

    return {
        'consistency_score': consistency_score,
        'mse_m0_m1': mse_m0_m1,
        'mse_m0_m2': mse_m0_m2,
        'mse_m1_m2': mse_m1_m2
    }


if __name__ == '__main__':
    # Test metrics
    print("Testing metrics utilities...")

    # Test ELBO
    batch_size = 8
    recon_x = torch.rand(batch_size, 1, 28, 28)
    x = torch.rand(batch_size, 1, 28, 28)
    mu = torch.randn(batch_size, 16)
    logvar = torch.randn(batch_size, 16)

    elbo = compute_elbo(recon_x, x, mu, logvar)
    print(f"\nELBO: {elbo:.4f}")

    # Test energy statistics
    energies_pos = torch.randn(100) + 2  # Higher energy
    energies_neg = torch.randn(100)      # Lower energy

    stats = compute_energy_statistics(energies_pos, energies_neg)
    print(f"\nEnergy statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value:.4f}")

    # Test overlap
    dist1 = np.random.randn(1000)
    dist2 = np.random.randn(1000) + 1

    overlap = compute_distribution_overlap(dist1, dist2)
    print(f"\nDistribution overlap: {overlap:.4f}")

    print("\n✓ Metrics test passed!")
