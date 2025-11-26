"""
Variational Autoencoder (VAE) for MNIST images.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class VAE(nn.Module):
    """
    Variational Autoencoder with convolutional encoder and decoder.

    Architecture:
    - Encoder: 2 conv layers → latent space (μ, log σ²)
    - Decoder: 2 transposed conv layers → reconstructed image
    """

    def __init__(self, latent_dim=16):
        super(VAE, self).__init__()

        self.latent_dim = latent_dim

        # Encoder
        self.encoder = nn.Sequential(
            # Input: 1 x 28 x 28
            nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1),  # → 32 x 14 x 14
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # → 64 x 7 x 7
            nn.ReLU(),
            nn.Flatten(),  # → 64 * 7 * 7 = 3136
            nn.Linear(64 * 7 * 7, 256),
            nn.ReLU()
        )

        # Latent space parameters
        self.fc_mu = nn.Linear(256, latent_dim)
        self.fc_logvar = nn.Linear(256, latent_dim)

        # Decoder
        self.fc_decode = nn.Linear(latent_dim, 256)

        self.decoder = nn.Sequential(
            nn.Linear(256, 64 * 7 * 7),
            nn.ReLU(),
            nn.Unflatten(dim=1, unflattened_size=(64, 7, 7)),  # → 64 x 7 x 7
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),  # → 32 x 14 x 14
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, kernel_size=4, stride=2, padding=1),  # → 1 x 28 x 28
            nn.Sigmoid()  # Output in [0, 1]
        )

    def encode(self, x):
        """
        Encode input to latent parameters.

        Args:
            x: Input images (batch_size, 1, 28, 28)

        Returns:
            mu: Mean of latent distribution (batch_size, latent_dim)
            logvar: Log variance of latent distribution (batch_size, latent_dim)
        """
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        """
        Reparameterization trick: sample z from N(μ, σ²).

        Args:
            mu: Mean (batch_size, latent_dim)
            logvar: Log variance (batch_size, latent_dim)

        Returns:
            z: Sampled latent code (batch_size, latent_dim)
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z

    def decode(self, z):
        """
        Decode latent code to reconstructed image.

        Args:
            z: Latent code (batch_size, latent_dim)

        Returns:
            Reconstructed image (batch_size, 1, 28, 28)
        """
        h = F.relu(self.fc_decode(z))
        return self.decoder(h)

    def forward(self, x):
        """
        Full forward pass: encode, sample, decode.

        Args:
            x: Input images (batch_size, 1, 28, 28)

        Returns:
            recon_x: Reconstructed images (batch_size, 1, 28, 28)
            mu: Latent mean (batch_size, latent_dim)
            logvar: Latent log variance (batch_size, latent_dim)
        """
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decode(z)
        return recon_x, mu, logvar


def vae_loss(recon_x, x, mu, logvar, beta=1.0):
    """
    Compute VAE loss (ELBO).

    Loss = Reconstruction Loss + β * KL Divergence

    Args:
        recon_x: Reconstructed images (batch_size, 1, 28, 28)
        x: Original images (batch_size, 1, 28, 28)
        mu: Latent mean (batch_size, latent_dim)
        logvar: Latent log variance (batch_size, latent_dim)
        beta: Weight for KL divergence (β-VAE)

    Returns:
        total_loss: Total loss (scalar)
        recon_loss: Reconstruction loss (scalar)
        kl_loss: KL divergence loss (scalar)
    """
    # Reconstruction loss (binary cross-entropy)
    recon_loss = F.binary_cross_entropy(recon_x, x, reduction='sum')

    # KL divergence: -0.5 * sum(1 + log(σ²) - μ² - σ²)
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    # Total loss
    total_loss = recon_loss + beta * kl_loss

    return total_loss, recon_loss, kl_loss


if __name__ == '__main__':
    # Test VAE architecture
    print("Testing VAE architecture...")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Create model
    vae = VAE(latent_dim=16).to(device)
    print(f"\nModel parameters: {sum(p.numel() for p in vae.parameters()):,}")

    # Create dummy batch
    batch_size = 8
    x = torch.randn(batch_size, 1, 28, 28).to(device)

    # Forward pass
    recon_x, mu, logvar = vae(x)

    print(f"\nInput shape: {x.shape}")
    print(f"Reconstructed shape: {recon_x.shape}")
    print(f"Latent mu shape: {mu.shape}")
    print(f"Latent logvar shape: {logvar.shape}")

    # Test loss
    total_loss, recon_loss, kl_loss = vae_loss(recon_x, x, mu, logvar, beta=1.0)

    print(f"\nLoss values:")
    print(f"  Total loss: {total_loss.item():.4f}")
    print(f"  Reconstruction loss: {recon_loss.item():.4f}")
    print(f"  KL loss: {kl_loss.item():.4f}")

    # Test encode/decode separately
    mu_test, logvar_test = vae.encode(x)
    z_test = vae.reparameterize(mu_test, logvar_test)
    recon_test = vae.decode(z_test)

    print(f"\nSeparate encode/decode:")
    print(f"  Latent z shape: {z_test.shape}")
    print(f"  Reconstructed shape: {recon_test.shape}")

    print("\n✓ VAE architecture test passed!")
