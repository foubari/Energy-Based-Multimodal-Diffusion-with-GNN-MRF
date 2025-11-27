# Energy-Based Multimodal Diffusion with GNN-MRF

A PyTorch implementation of a multimodal generative model combining Variational Autoencoders (VAEs) with a Graph Neural Network (GNN) energy-based prior for modeling dependencies between modalities.

## Overview

This project implements a two-stage generative model:

1. **Phase 1**: Train separate VAEs for each modality to learn latent representations
2. **Phase 2**: Train a GNN-based Energy-Based Model (EBM) to model dependencies between modality latents
3. **Generation**: Use Langevin dynamics to sample coherent multimodal data

### Architecture

```
Input Images → VAEs (frozen) → Latent Codes → GNN Energy → Langevin Sampling → Generated Samples
                                       ↓
                                 E_θ(z, G) : Markov Random Field Energy
```

## Features

- **Multimodal MNIST**: 3 modalities (original, rotated 90°, horizontally flipped)
- **VAE**: Convolutional encoder-decoder with β-VAE annealing
- **GNN Energy Network**: Custom GNN implementation with configurable graph structure
- **Langevin Dynamics**: Efficient sampling from learned energy landscape
- **Mixed Precision Training**: AMP support for faster training
- **Visualization**: Comprehensive plotting of losses, reconstructions, and samples
- **TensorBoard Integration**: Real-time training monitoring

## Installation

### Requirements

- Python 3.8+
- PyTorch 2.0+
- CUDA (optional, for GPU acceleration)

### Setup

```bash
# Clone or navigate to project directory
cd energy_diffusion

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

### 1. Train VAEs (Phase 1)

Train 3 separate VAEs, one for each modality:

```bash
python main_train_vae.py --config configs/vae_config.yaml
```

This will:
- Train VAE for each of the 3 modalities
- Save checkpoints to `outputs/vae_checkpoints/`
- Generate visualizations in `outputs/plots/`
- Log metrics to TensorBoard in `outputs/logs/`

**Training time**: ~10-30 minutes per VAE (depending on GPU)

To train a single modality:
```bash
python main_train_vae.py --modality 0  # Train only modality 0
```

### 2. Train EBM (Phase 2)

Train the GNN energy network with frozen VAEs:

```bash
python main_train_ebm.py --config configs/ebm_config.yaml
```

This will:
- Load pre-trained VAEs (frozen)
- Train GNN energy network using contrastive divergence
- Save checkpoints to `outputs/ebm_checkpoints/`
- Generate energy distributions, samples, and conditional generations

**Training time**: ~1-3 hours (depending on GPU and epochs)

### 3. Generate Samples

**Unconditional Generation** (sample from scratch):

```bash
python generate.py \
    --mode unconditional \
    --ebm_checkpoint outputs/ebm_checkpoints/gnn_energy_best.pth \
    --num_samples 16 \
    --output outputs/plots/generated_samples.png
```

**Conditional Generation** (impute missing modalities):

```bash
python generate.py \
    --mode conditional \
    --ebm_checkpoint outputs/ebm_checkpoints/gnn_energy_best.pth \
    --input_image path/to/mnist_digit.png \
    --observed_modality 0 \
    --output outputs/plots/conditional_samples.png
```

## Project Structure

```
energy_diffusion/
├── data/
│   └── mnist_loader.py          # Multimodal MNIST dataset
├── models/
│   ├── vae.py                   # VAE architecture
│   ├── gnn_energy.py            # GNN energy network
│   └── langevin_sampler.py      # Langevin dynamics sampler
├── training/
│   ├── train_vae.py             # VAE training functions
│   └── train_ebm.py             # EBM training functions
├── utils/
│   ├── visualization.py         # Plotting utilities
│   ├── metrics.py               # Evaluation metrics
│   └── checkpoint.py            # Model checkpointing
├── configs/
│   ├── vae_config.yaml          # VAE hyperparameters
│   └── ebm_config.yaml          # EBM hyperparameters
├── outputs/                     # Generated outputs
├── main_train_vae.py            # VAE training script
├── main_train_ebm.py            # EBM training script
├── generate.py                  # Generation script
├── requirements.txt
└── README.md
```

## Configuration

### VAE Configuration (`configs/vae_config.yaml`)

Key parameters:
- `latent_dim`: Dimension of latent space (default: 16)
- `batch_size`: Training batch size (default: 128)
- `epochs`: Number of training epochs (default: 100)
- `lr`: Learning rate (default: 1e-3)
- `beta_kl_schedule`: KL annealing settings for β-VAE
- `early_stopping`: Early stopping patience and threshold

### EBM Configuration (`configs/ebm_config.yaml`)

Key parameters:
- `hidden_dim_gnn`: GNN hidden dimension (default: 64)
- `num_gnn_layers`: Number of GNN layers (default: 3)
- `graph.edge_index`: Graph structure (default: complete graph)
- `langevin.num_steps_train`: Langevin steps during training (default: 20)
- `buffer_size`: Replay buffer size (default: 10000)
- `lr`: Learning rate (default: 1e-4)

## Model Details

### VAE Architecture

**Encoder** (per modality):
```
Input (28×28) → Conv(32) → Conv(64) → FC(256) → [μ, log σ²] ∈ ℝ^16
```

**Decoder** (per modality):
```
z ∈ ℝ^16 → FC(256) → FC(3136) → Reshape → ConvT(32) → ConvT(1) → Output (28×28)
```

**Loss**: ELBO = Reconstruction Loss (BCE) + β × KL Divergence

### GNN Energy Network

**Architecture**:
1. Node initialization: latent codes + modality embeddings → hidden features
2. Message passing: 3 GNN layers with residual connections
3. Potentials:
   - Unary: ψ_i for each node
   - Binary: ψ_ij for each edge
4. Total energy: E_θ(z, G) = Σ ψ_i + Σ ψ_ij

**Graph**: Configurable via YAML (default: complete graph for 3 modalities)

### Langevin Sampling

Update rule:
```
z^{t+1} = z^t - η·∇_z E_θ(z^t) + √(2η)·ξ^t
```
where ξ^t ~ N(0, I)

- Training: 20 steps
- Generation: 50 steps
- Step size: 0.01 (tunable)

## Monitoring Training

### TensorBoard

```bash
tensorboard --logdir outputs/logs
```

View at `http://localhost:6006`

**VAE Metrics**:
- Total loss (ELBO)
- Reconstruction loss
- KL divergence
- Beta schedule

**EBM Metrics**:
- Training loss
- Energy (positive/negative)
- Energy margin
- Gradient norms

### Visualizations

Generated automatically during training:

**VAE** (every 5 epochs):
- Reconstructions: Original vs reconstructed images
- Samples: Digits sampled from prior N(0, I)
- Loss curves

**EBM** (every 10 epochs):
- Energy distributions: Histogram of positive vs negative energies
- Generated samples: All 3 modalities
- Conditional generation: Imputing missing modalities

## Advanced Usage

### Custom Graph Structure

Edit `configs/ebm_config.yaml` to define custom edge connectivity:

```yaml
model:
  graph:
    edge_index:
      - [0, 1, 2]  # Source nodes
      - [1, 2, 0]  # Target nodes (chain graph: 0→1→2→0)
```

### Hyperparameter Tuning

Key hyperparameters to tune:

**VAE**:
- `latent_dim`: Higher = more expressive, harder to train
- `beta_kl_schedule.warmup_epochs`: Longer = better reconstructions, worse samples

**EBM**:
- `langevin.num_steps_train`: More steps = better negatives, slower training
- `langevin.step_size`: Smaller = more accurate, slower convergence
- `buffer_size`: Larger = more diverse negatives, more memory

## Troubleshooting

### Common Issues

**VAE not reconstructing well**:
- Increase training epochs
- Reduce KL β during warmup
- Check learning rate (try 5e-4 or 5e-3)

**EBM energy collapse** (E_pos ≈ E_neg):
- Reduce learning rate (try 5e-5)
- Increase gradient clipping
- More Langevin steps
- Check that VAEs are properly frozen

**Out of memory**:
- Reduce batch size
- Enable gradient checkpointing (set `use_checkpoint: true` in config)
- Reduce buffer size

**OMP Error on Windows** (`libomp.dll` / `libiomp5md.dll` conflict):
- Already fixed in all scripts via `os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'`
- If error persists in notebooks/custom scripts, add this line before importing torch:
  ```python
  import os
  os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
  ```

## Results

Expected results after training:

**VAE**:
- Reconstruction error < 0.05 MSE
- Clear digit reconstructions
- Diverse samples from prior

**EBM**:
- Energy margin > 5.0 (E_pos - E_neg)
- Low overlap between positive/negative distributions
- Coherent multimodal samples (same digit across modalities)

## Extensions

Potential improvements (not implemented):

1. **PyTorch Geometric Integration**: Replace custom GNN with PyG MessagePassing
2. **MALA Sampling**: Metropolis-adjusted Langevin for better convergence
3. **Spectral Normalization**: Stabilize EBM training
4. **Flexible Conditioning**: Observe arbitrary subsets of modalities
5. **Annealed Langevin**: Temperature scheduling for sampling

## Citation

If you use this code, please cite:

```bibtex
@software{energy_diffusion_gnn_mrf,
  title={Energy-Based Multimodal Diffusion with GNN-MRF},
  author={Energy Diffusion Project},
  year={2025},
  url={https://github.com/yourusername/energy_diffusion}
}
```

## License

MIT License - see LICENSE file for details

## Acknowledgments

- VAE implementation inspired by PyTorch examples
- GNN architecture based on message passing neural networks
- Energy-based models following contrastive divergence training
