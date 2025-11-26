"""
Training functions and utilities for Energy-Based Model (EBM) with GNN.
"""
import torch
from torch.cuda.amp import autocast
from tqdm import tqdm


class ReplayBuffer:
    """
    FIFO replay buffer for storing negative samples in contrastive divergence.
    """

    def __init__(self, buffer_size, num_modalities, latent_dim, device='cpu'):
        """
        Args:
            buffer_size: Maximum number of samples to store
            num_modalities: Number of modalities
            latent_dim: Latent dimension
            device: Device to store buffer on
        """
        self.buffer_size = buffer_size
        self.buffer = torch.randn(buffer_size, num_modalities, latent_dim).to(device)
        self.ptr = 0
        self.device = device

    def sample(self, batch_size, reinit_ratio=0.05):
        """
        Sample from buffer with some fresh random samples.

        Args:
            batch_size: Number of samples to return
            reinit_ratio: Fraction of samples to reinitialize from N(0,I)

        Returns:
            Sampled latent codes (batch_size, num_modalities, latent_dim)
        """
        num_buffer = int(batch_size * (1 - reinit_ratio))
        num_fresh = batch_size - num_buffer

        # Sample from buffer (random indices)
        indices = torch.randint(0, self.buffer_size, (num_buffer,), device=self.device)
        buffer_samples = self.buffer[indices]

        # Fresh samples from prior
        fresh_samples = torch.randn(
            num_fresh, self.buffer.shape[1], self.buffer.shape[2],
            device=self.device
        )

        return torch.cat([buffer_samples, fresh_samples], dim=0)

    def update(self, samples):
        """
        Update buffer with new samples (FIFO replacement).

        Args:
            samples: New samples to add (batch_size, num_modalities, latent_dim)
        """
        batch_size = samples.shape[0]

        if self.ptr + batch_size <= self.buffer_size:
            # No wrap-around
            self.buffer[self.ptr:self.ptr + batch_size] = samples.detach()
            self.ptr = (self.ptr + batch_size) % self.buffer_size
        else:
            # Wrap around
            overflow = (self.ptr + batch_size) - self.buffer_size
            self.buffer[self.ptr:] = samples[:self.buffer_size - self.ptr].detach()
            self.buffer[:overflow] = samples[self.buffer_size - self.ptr:].detach()
            self.ptr = overflow


def train_one_epoch(gnn_energy, vaes, dataloader, optimizer, scaler,
                   langevin_sampler, replay_buffer, config, device, epoch):
    """
    Train EBM for one epoch using contrastive divergence.

    Args:
        gnn_energy: GNN energy network
        vaes: List of VAE models (frozen)
        dataloader: Training dataloader (multimodal)
        optimizer: Optimizer for GNN
        scaler: GradScaler for mixed precision
        langevin_sampler: LangevinSampler instance
        replay_buffer: ReplayBuffer instance
        config: Configuration object
        device: Device
        epoch: Current epoch number

    Returns:
        Dictionary with metrics: {'loss', 'energy_pos', 'energy_neg', 'margin'}
    """
    gnn_energy.train()
    for vae in vaes:
        vae.eval()

    total_loss_sum = 0
    energy_pos_sum = 0
    energy_neg_sum = 0
    num_batches = 0

    pbar = tqdm(dataloader, desc=f'Epoch {epoch} [Train]', leave=False)

    for batch_idx, (x_m0, x_m1, x_m2, _) in enumerate(pbar):
        x_m0 = x_m0.to(device)
        x_m1 = x_m1.to(device)
        x_m2 = x_m2.to(device)
        batch_size = x_m0.shape[0]

        # 1. Encode to get positive latents (data distribution)
        with torch.no_grad():
            mu0, _ = vaes[0].encode(x_m0)
            mu1, _ = vaes[1].encode(x_m1)
            mu2, _ = vaes[2].encode(x_m2)
            z_pos = torch.stack([mu0, mu1, mu2], dim=1)  # (B, 3, latent_dim)

        # 2. Sample negative latents from buffer + Langevin refinement
        z_neg_init = replay_buffer.sample(batch_size, config.training.reinit_ratio)
        z_neg, _ = langevin_sampler.sample(z_neg_init)

        # 3. Compute energies
        with autocast(enabled=(scaler is not None)):
            energy_pos = gnn_energy(z_pos)
            energy_neg = gnn_energy(z_neg)

            # Contrastive divergence loss: E[E_θ(z_pos)] - E[E_θ(z_neg)]
            cd_loss = energy_pos.mean() - energy_neg.mean()

            # Optional: gradient penalty for stability
            loss = cd_loss
            if config.training.lambda_reg > 0:
                z_pos_reg = z_pos.detach().requires_grad_(True)
                energy_pos_reg = gnn_energy(z_pos_reg)
                grad = torch.autograd.grad(
                    energy_pos_reg.sum(), z_pos_reg,
                    create_graph=True, retain_graph=True
                )[0]
                grad_penalty = (grad ** 2).sum(dim=[1, 2]).mean()
                loss = cd_loss + config.training.lambda_reg * grad_penalty

        # 4. Backward pass
        optimizer.zero_grad()

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(gnn_energy.parameters(),
                                          config.training.grad_clip_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(gnn_energy.parameters(),
                                          config.training.grad_clip_norm)
            optimizer.step()

        # 5. Update replay buffer with refined negatives
        replay_buffer.update(z_neg)

        # Accumulate metrics
        total_loss_sum += loss.item()
        energy_pos_sum += energy_pos.mean().item()
        energy_neg_sum += energy_neg.mean().item()
        num_batches += 1

        # Update progress bar
        pbar.set_postfix({
            'loss': loss.item(),
            'E_pos': energy_pos.mean().item(),
            'E_neg': energy_neg.mean().item(),
            'margin': energy_pos.mean().item() - energy_neg.mean().item()
        })

    # Compute averages
    metrics = {
        'loss': total_loss_sum / num_batches,
        'energy_pos': energy_pos_sum / num_batches,
        'energy_neg': energy_neg_sum / num_batches,
        'margin': (energy_pos_sum - energy_neg_sum) / num_batches
    }

    return metrics


if __name__ == '__main__':
    # Test EBM training functions
    print("Testing EBM training functions...")

    import sys
    sys.path.append('..')

    from models.gnn_energy import GNNEnergyNetwork
    from models.vae import VAE
    from models.langevin_sampler import LangevinSampler
    from data.mnist_loader import get_dataloaders

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Create models
    print("\n1. Creating models...")
    vaes = [VAE(latent_dim=16).to(device) for _ in range(3)]
    for vae in vaes:
        vae.eval()

    gnn_energy = GNNEnergyNetwork(
        num_modalities=3,
        latent_dim=16,
        modality_emb_dim=8,
        hidden_dim=64,
        num_layers=3
    ).to(device)

    print(f"   GNN parameters: {sum(p.numel() for p in gnn_energy.parameters()):,}")

    # Create optimizer
    optimizer = torch.optim.Adam(gnn_energy.parameters(), lr=1e-4)

    # Create Langevin sampler
    langevin_sampler = LangevinSampler(
        energy_fn=gnn_energy,
        num_steps=10,  # Short for testing
        step_size=0.01,
        clip_grad=1.0,
        device=device
    )

    # Create replay buffer
    replay_buffer = ReplayBuffer(
        buffer_size=1000,
        num_modalities=3,
        latent_dim=16,
        device=device
    )

    print("   ✓ Models created")

    # Test replay buffer
    print("\n2. Testing replay buffer...")
    samples = replay_buffer.sample(batch_size=32, reinit_ratio=0.1)
    print(f"   Sampled shape: {samples.shape}")
    print(f"   Buffer pointer before update: {replay_buffer.ptr}")

    new_samples = torch.randn(32, 3, 16, device=device)
    replay_buffer.update(new_samples)
    print(f"   Buffer pointer after update: {replay_buffer.ptr}")
    print("   ✓ Replay buffer works")

    # Test training loop (mock config)
    print("\n3. Testing training loop...")

    class MockConfig:
        class training:
            reinit_ratio = 0.05
            lambda_reg = 0.1
            grad_clip_norm = 10.0

    config = MockConfig()

    # Get small dataloader
    train_loader, _, _ = get_dataloaders(batch_size=16, num_workers=0, val_split=0.1)

    # Limit to 2 batches for testing
    from itertools import islice
    small_loader = list(islice(train_loader, 2))

    # Run one "epoch"
    from torch.cuda.amp import GradScaler
    scaler = GradScaler() if torch.cuda.is_available() else None

    metrics = train_one_epoch(
        gnn_energy, vaes, small_loader, optimizer, scaler,
        langevin_sampler, replay_buffer, config, device, epoch=1
    )

    print(f"\n   Metrics:")
    print(f"     Loss: {metrics['loss']:.4f}")
    print(f"     Energy (pos): {metrics['energy_pos']:.4f}")
    print(f"     Energy (neg): {metrics['energy_neg']:.4f}")
    print(f"     Margin: {metrics['margin']:.4f}")

    print("\n✓ EBM training functions test passed!")
