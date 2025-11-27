"""
Extract sample images from MNIST test set for conditional generation testing.
"""
import os
import sys

# Add parent directory to path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

import torch
import torchvision.transforms as transforms
from PIL import Image

from data.mnist_loader import get_dataloaders


def extract_test_images(output_dir='test_images', num_images=5):
    """
    Extract sample images from MNIST test set.

    Args:
        output_dir: Directory to save test images
        num_images: Number of images to extract per modality
    """
    os.makedirs(output_dir, exist_ok=True)

    # Load data
    _, _, test_loader = get_dataloaders(batch_size=num_images, num_workers=0)

    # Get one batch
    x_m0, x_m1, x_m2, labels = next(iter(test_loader))

    # Convert to PIL and save
    to_pil = transforms.ToPILImage()

    print(f"Extracting {num_images} test images to '{output_dir}/'...")

    for i in range(num_images):
        # Save all 3 modalities for each sample
        label = labels[i].item()

        # Modality 0 (original)
        img_m0 = to_pil(x_m0[i])
        path_m0 = os.path.join(output_dir, f'digit_{label}_m0_original.png')
        img_m0.save(path_m0)

        # Modality 1 (rotated 90°)
        img_m1 = to_pil(x_m1[i])
        path_m1 = os.path.join(output_dir, f'digit_{label}_m1_rot90.png')
        img_m1.save(path_m1)

        # Modality 2 (flipped H)
        img_m2 = to_pil(x_m2[i])
        path_m2 = os.path.join(output_dir, f'digit_{label}_m2_flipH.png')
        img_m2.save(path_m2)

        print(f"  ✓ Digit {label}: saved 3 modalities")

    # Create README
    readme_path = os.path.join(output_dir, 'README.md')
    with open(readme_path, 'w') as f:
        f.write(f"""# Test Images for Conditional Generation

This directory contains {num_images} sample MNIST digits in 3 modalities:

- **m0_original**: Original MNIST digit
- **m1_rot90**: Rotated 90° clockwise
- **m2_flipH**: Horizontally flipped

## Usage

Use these images to test conditional generation:

```bash
# Generate from modality 0 (original)
python generate.py --mode conditional \\
    --input_image test_images/digit_7_m0_original.png \\
    --observed_modality 0 \\
    --ebm_checkpoint outputs/ebm_checkpoints/gnn_energy_best.pth \\
    --output outputs/plots/conditional_from_m0.png

# Generate from modality 1 (rotated)
python generate.py --mode conditional \\
    --input_image test_images/digit_7_m1_rot90.png \\
    --observed_modality 1 \\
    --ebm_checkpoint outputs/ebm_checkpoints/gnn_energy_best.pth \\
    --output outputs/plots/conditional_from_m1.png

# Generate from modality 2 (flipped)
python generate.py --mode conditional \\
    --input_image test_images/digit_7_m2_flipH.png \\
    --observed_modality 2 \\
    --ebm_checkpoint outputs/ebm_checkpoints/gnn_energy_best.pth \\
    --output outputs/plots/conditional_from_m2.png
```

The model will generate the two missing modalities based on the observed one.
""")

    print(f"\n✓ Extracted {num_images * 3} images total")
    print(f"✓ Created README with usage examples")
    print(f"\nYou can now test conditional generation with:")
    print(f"  python generate.py --mode conditional \\")
    print(f"      --input_image {output_dir}/digit_7_m0_original.png \\")
    print(f"      --observed_modality 0 \\")
    print(f"      --ebm_checkpoint outputs/ebm_checkpoints/gnn_energy_best.pth \\")
    print(f"      --output outputs/plots/conditional_test.png")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Extract MNIST test images')
    parser.add_argument('--output_dir', type=str, default='test_images',
                       help='Directory to save test images')
    parser.add_argument('--num_images', type=int, default=5,
                       help='Number of images to extract')

    args = parser.parse_args()

    extract_test_images(args.output_dir, args.num_images)
