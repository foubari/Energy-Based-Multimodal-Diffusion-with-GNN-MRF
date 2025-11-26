"""
Graph Neural Network for computing energy over multimodal latent codes.
"""
import torch
import torch.nn as nn


class GNNLayer(nn.Module):
    """
    Single GNN layer with message passing and residual connections.
    """

    def __init__(self, hidden_dim):
        super(GNNLayer, self).__init__()

        # Edge MLP (for computing messages)
        self.edge_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU()
        )

        # Node update MLP
        self.node_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU()
        )

    def forward(self, h, edge_index):
        """
        Args:
            h: Node features (batch_size, num_nodes, hidden_dim)
            edge_index: Edge connectivity (2, num_edges)

        Returns:
            h_new: Updated node features (batch_size, num_nodes, hidden_dim)
        """
        batch_size, num_nodes, hidden_dim = h.shape
        src, dst = edge_index[0], edge_index[1]
        num_edges = edge_index.shape[1]

        # Compute messages for each edge
        h_src = h[:, src, :]  # (B, E, hidden_dim)
        h_dst = h[:, dst, :]  # (B, E, hidden_dim)
        edge_input = torch.cat([h_src, h_dst], dim=-1)  # (B, E, 2*hidden_dim)
        messages = self.edge_mlp(edge_input)  # (B, E, hidden_dim)

        # Aggregate messages per node (sum over incoming edges)
        aggregated = torch.zeros(batch_size, num_nodes, hidden_dim,
                                device=h.device, dtype=h.dtype)

        for i in range(num_edges):
            dst_node = dst[i].item()
            aggregated[:, dst_node, :] += messages[:, i, :]

        # Update nodes
        node_input = torch.cat([h, aggregated], dim=-1)  # (B, N, 2*hidden_dim)
        h_new = self.node_mlp(node_input)  # (B, N, hidden_dim)

        # Residual connection
        h_new = h_new + h

        return h_new


class GNNEnergyNetwork(nn.Module):
    """
    Graph Neural Network for computing energy E_θ(z, G) over multimodal latents.

    Architecture:
    1. Node initialization: latent codes + modality embeddings → hidden features
    2. Message passing: L layers of GNN with residual connections
    3. Potential computation:
       - Unary potentials (per node)
       - Binary potentials (per edge)
    4. Total energy: sum of all potentials
    """

    def __init__(self,
                 num_modalities=3,
                 latent_dim=16,
                 modality_emb_dim=8,
                 hidden_dim=64,
                 num_layers=3,
                 edge_index=None):
        """
        Args:
            num_modalities: Number of modalities
            latent_dim: Dimension of latent codes
            modality_emb_dim: Dimension of modality embeddings
            hidden_dim: Hidden dimension for GNN layers
            num_layers: Number of GNN layers
            edge_index: Edge connectivity tensor (2, num_edges). If None, use complete graph.
        """
        super(GNNEnergyNetwork, self).__init__()

        self.num_modalities = num_modalities
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim

        # Graph structure (adjacency list format)
        if edge_index is None:
            # Default: complete graph for 3 modalities
            edge_index = torch.tensor([
                [0, 0, 1, 1, 2, 2],  # source nodes
                [1, 2, 0, 2, 0, 1]   # target nodes
            ], dtype=torch.long)

        self.register_buffer('edge_index', edge_index)

        # Modality embeddings (learnable)
        self.modality_emb = nn.Embedding(num_modalities, modality_emb_dim)

        # Node initialization MLP
        self.init_mlp = nn.Sequential(
            nn.Linear(latent_dim + modality_emb_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU()
        )

        # Message passing layers
        self.gnn_layers = nn.ModuleList([
            GNNLayer(hidden_dim) for _ in range(num_layers)
        ])

        # Unary potential MLP (per node)
        self.unary_mlp = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

        # Binary potential MLP (per edge)
        self.binary_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, z):
        """
        Compute energy for a batch of latent codes.

        Args:
            z: Latent codes (batch_size, num_modalities, latent_dim)

        Returns:
            energy: Energy values (batch_size,)
        """
        batch_size, num_modalities, _ = z.shape
        device = z.device

        # 1. Initialize node features
        mod_ids = torch.arange(num_modalities, device=device)
        mod_emb = self.modality_emb(mod_ids)  # (num_modalities, emb_dim)
        mod_emb = mod_emb.unsqueeze(0).expand(batch_size, -1, -1)  # (B, M, emb_dim)

        h = torch.cat([z, mod_emb], dim=-1)  # (B, M, latent_dim + emb_dim)
        h = self.init_mlp(h)  # (B, M, hidden_dim)

        # 2. Message passing
        for layer in self.gnn_layers:
            h = layer(h, self.edge_index)  # (B, M, hidden_dim)

        # 3. Compute potentials
        # Unary potentials (per node)
        psi_unary = self.unary_mlp(h).squeeze(-1)  # (B, M)
        unary_energy = psi_unary.sum(dim=1)  # (B,)

        # Binary potentials (per edge)
        src, dst = self.edge_index[0], self.edge_index[1]
        h_src = h[:, src, :]  # (B, num_edges, hidden_dim)
        h_dst = h[:, dst, :]  # (B, num_edges, hidden_dim)
        edge_features = torch.cat([h_src, h_dst], dim=-1)  # (B, num_edges, 2*hidden_dim)
        psi_binary = self.binary_mlp(edge_features).squeeze(-1)  # (B, num_edges)
        binary_energy = psi_binary.sum(dim=1)  # (B,)

        # 4. Total energy
        energy = unary_energy + binary_energy  # (B,)

        return energy


if __name__ == '__main__':
    # Test GNN Energy Network
    print("Testing GNN Energy Network...")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Create model
    model = GNNEnergyNetwork(
        num_modalities=3,
        latent_dim=16,
        modality_emb_dim=8,
        hidden_dim=64,
        num_layers=3
    ).to(device)

    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params:,}")

    # Create dummy batch
    batch_size = 8
    z = torch.randn(batch_size, 3, 16).to(device)

    print(f"\nInput shape: {z.shape}")

    # Forward pass
    energy = model(z)

    print(f"Energy shape: {energy.shape}")
    print(f"Energy values: {energy.cpu().detach().numpy()}")

    # Test gradient computation (needed for Langevin)
    z_test = torch.randn(4, 3, 16, requires_grad=True).to(device)
    energy_test = model(z_test)
    grad = torch.autograd.grad(energy_test.sum(), z_test)[0]

    print(f"\nGradient test:")
    print(f"  Input shape: {z_test.shape}")
    print(f"  Energy shape: {energy_test.shape}")
    print(f"  Gradient shape: {grad.shape}")
    print(f"  Gradient norm: {grad.norm().item():.4f}")

    # Test custom edge index
    print("\nTesting custom edge index (chain graph)...")
    chain_edges = torch.tensor([
        [0, 1],  # 0 -> 1
        [1, 2]   # 1 -> 2
    ], dtype=torch.long).t()

    model_chain = GNNEnergyNetwork(
        num_modalities=3,
        latent_dim=16,
        modality_emb_dim=8,
        hidden_dim=64,
        num_layers=3,
        edge_index=chain_edges
    ).to(device)

    energy_chain = model_chain(z)
    print(f"Chain graph energy shape: {energy_chain.shape}")
    print(f"Chain graph energy values: {energy_chain.cpu().detach().numpy()}")

    print("\n✓ GNN Energy Network test passed!")
