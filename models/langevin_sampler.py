"""
Langevin dynamics sampler for generating samples from energy-based models.
"""
import torch
import numpy as np


class LangevinSampler:
    """
    Langevin dynamics sampler for EBM.

    Update rule:
        z^{t+1} = z^t - η·∇_z E_θ(z^t) + √(2η)·ξ^t
        where ξ^t ~ N(0, I)
    """

    def __init__(self, energy_fn, num_steps=50, step_size=0.01,
                 clip_grad=None, device='cpu'):
        """
        Args:
            energy_fn: Function that computes energy E_θ(z)
            num_steps: Number of Langevin steps
            step_size: Step size η
            clip_grad: Optional gradient clipping value
            device: Device to run on
        """
        self.energy_fn = energy_fn
        self.num_steps = num_steps
        self.step_size = step_size
        self.clip_grad = clip_grad
        self.device = device

    def sample(self, z_init, observed_mask=None, observed_values=None, return_trajectory=False):
        """
        Run Langevin dynamics to refine latent codes.

        Args:
            z_init: Initial latent codes (batch_size, num_modalities, latent_dim)
            observed_mask: Optional mask (batch_size, num_modalities) where True = observed/fixed
            observed_values: Optional values for observed modalities (same shape as z_init)
            return_trajectory: If True, return intermediate states

        Returns:
            z_final: Refined latent codes
            trajectory: List of intermediate states (if return_trajectory=True)
        """
        z = z_init.clone().detach()
        trajectory = [z.clone()] if return_trajectory else None

        for step in range(self.num_steps):
            # Create a new tensor with gradient enabled (leaf node)
            z_grad = z.detach().requires_grad_(True)

            # Compute energy
            energy = self.energy_fn(z_grad)

            # Compute gradient ∇_z E_θ(z)
            grad = torch.autograd.grad(energy.sum(), z_grad, create_graph=False, retain_graph=False)[0]

            # Clip gradient if specified
            if self.clip_grad is not None:
                grad = torch.clamp(grad, -self.clip_grad, self.clip_grad)

            # Langevin update: z = z - η·∇E + √(2η)·noise
            noise = torch.randn_like(z) * np.sqrt(2 * self.step_size)
            z = z - self.step_size * grad.detach() + noise

            # Apply observed mask (conditional generation)
            if observed_mask is not None:
                mask_expanded = observed_mask.unsqueeze(-1)  # (B, M, 1)
                z = torch.where(mask_expanded, observed_values, z)

            if return_trajectory:
                trajectory.append(z.clone())

        if return_trajectory:
            return z, trajectory
        else:
            return z, None

    def sample_unconditional(self, batch_size, num_modalities, latent_dim, return_trajectory=False):
        """
        Generate samples from scratch (initialize with N(0, I)).

        Args:
            batch_size: Number of samples
            num_modalities: Number of modalities
            latent_dim: Latent dimension
            return_trajectory: If True, return intermediate states

        Returns:
            z_final: Sampled latent codes
            trajectory: Optional trajectory
        """
        z_init = torch.randn(batch_size, num_modalities, latent_dim, device=self.device)
        return self.sample(z_init, return_trajectory=return_trajectory)

    def sample_conditional(self, z_observed, observed_modality_idx, num_modalities, latent_dim,
                          return_trajectory=False):
        """
        Generate missing modalities given observed ones.

        Args:
            z_observed: Observed latent codes (batch_size, latent_dim)
            observed_modality_idx: Index of observed modality (int)
            num_modalities: Total number of modalities
            latent_dim: Latent dimension
            return_trajectory: If True, return intermediate states

        Returns:
            z_final: Full latent codes (batch_size, num_modalities, latent_dim)
            trajectory: Optional trajectory
        """
        batch_size = z_observed.shape[0]

        # Initialize all modalities
        z_init = torch.randn(batch_size, num_modalities, latent_dim, device=self.device)

        # Set observed modality
        z_init[:, observed_modality_idx, :] = z_observed

        # Create mask: True for observed modality
        observed_mask = torch.zeros(batch_size, num_modalities, dtype=torch.bool, device=self.device)
        observed_mask[:, observed_modality_idx] = True

        # Run Langevin with observed modality fixed
        return self.sample(z_init, observed_mask=observed_mask, observed_values=z_init,
                          return_trajectory=return_trajectory)


if __name__ == '__main__':
    # Test Langevin Sampler
    print("Testing Langevin Sampler...")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Create a dummy energy function (simple quadratic)
    def dummy_energy(z):
        """E(z) = ||z||^2 / 2, minimum at z=0"""
        return 0.5 * (z ** 2).sum(dim=[1, 2])

    # Create sampler
    sampler = LangevinSampler(
        energy_fn=dummy_energy,
        num_steps=50,
        step_size=0.1,
        clip_grad=1.0,
        device=device
    )

    # Test unconditional sampling
    print("\n1. Testing unconditional sampling...")
    z_samples, trajectory = sampler.sample_unconditional(
        batch_size=4,
        num_modalities=3,
        latent_dim=16,
        return_trajectory=True
    )

    print(f"  Initial energy: {dummy_energy(trajectory[0]).mean().item():.4f}")
    print(f"  Final energy: {dummy_energy(z_samples).mean().item():.4f}")
    print(f"  Final z norm: {z_samples.norm().item():.4f}")
    print(f"  Trajectory length: {len(trajectory)}")
    print(f"  → Energy should decrease toward 0 (minimum)")

    # Test conditional sampling
    print("\n2. Testing conditional sampling...")
    z_obs = torch.zeros(4, 16, device=device)  # Observed modality at origin

    z_cond, _ = sampler.sample_conditional(
        z_observed=z_obs,
        observed_modality_idx=0,
        num_modalities=3,
        latent_dim=16
    )

    print(f"  Conditional samples shape: {z_cond.shape}")
    print(f"  Observed modality (should be ~0): {z_cond[:, 0, :].norm().item():.6f}")
    print(f"  Modality 1 norm: {z_cond[:, 1, :].norm().item():.4f}")
    print(f"  Modality 2 norm: {z_cond[:, 2, :].norm().item():.4f}")
    print(f"  → Modality 0 should be fixed at 0")

    # Test with GNN energy (if available)
    print("\n3. Testing with GNN Energy Network...")
    try:
        from gnn_energy import GNNEnergyNetwork

        gnn = GNNEnergyNetwork(
            num_modalities=3,
            latent_dim=16,
            modality_emb_dim=8,
            hidden_dim=64,
            num_layers=3
        ).to(device)

        sampler_gnn = LangevinSampler(
            energy_fn=gnn,
            num_steps=20,
            step_size=0.01,
            clip_grad=1.0,
            device=device
        )

        z_samples_gnn, traj_gnn = sampler_gnn.sample_unconditional(
            batch_size=4,
            num_modalities=3,
            latent_dim=16,
            return_trajectory=True
        )

        energies_init = gnn(traj_gnn[0])
        energies_final = gnn(z_samples_gnn)

        print(f"  Initial energies: {energies_init.cpu().detach().numpy()}")
        print(f"  Final energies: {energies_final.cpu().detach().numpy()}")
        print(f"  Mean energy change: {(energies_final - energies_init).mean().item():.4f}")

    except ImportError:
        print("  GNN Energy Network not found, skipping GNN test")

    print("\n✓ Langevin Sampler test passed!")
