"""
Test script to visualize geometric augmentations.

This script loads a few samples from the dataset with the new geometric
augmentations enabled and saves visualization to verify they're working correctly.

Usage:
    python test_augmentation.py
"""

import sys
from pathlib import Path
import matplotlib.pyplot as plt # type: ignore
import torch

from config import get_config
from dataset import IRColorPairDataset, denormalize
from utils import tensor_to_image


def test_augmentations():
    """Test and visualize geometric augmentations."""
    
    # Get config
    config = get_config()
    
    # Ensure augmentations are enabled
    config.data.use_augmentation = True
    config.data.random_rotation = True
    config.data.random_perspective = True
    
    # Find test images
    test_dirs = [
        "./data/coco/val2017",
        "./data/coco/train2017",
        "./data/cityscapes/leftImg8bit/val"
    ]
    
    test_dir = None
    for d in test_dirs:
        if Path(d).exists():
            test_dir = d
            break
    
    if test_dir is None:
        print("No dataset found. Please run training first to download COCO dataset.")
        print("Or manually place some images in one of these directories:")
        for d in test_dirs:
            print(f"  - {d}")
        return
    
    print(f"Using dataset from: {test_dir}")
    
    # Create dataset
    dataset = IRColorPairDataset(
        image_source=test_dir,
        config=config.data,
        is_training=True
    )
    
    if len(dataset) == 0:
        print(f"No images found in {test_dir}")
        return
    
    print(f"Found {len(dataset)} images")
    
    # Create figure with multiple augmentations of the same image
    fig, axes = plt.subplots(4, 4, figsize=(16, 16))
    
    print("Generating augmented samples...")
    
    # Use the same image index but get different augmentations
    test_idx = 0
    
    for i in range(4):
        # Get augmented sample (each call will produce different augmentation)
        sample = dataset[test_idx]
        
        # Denormalize for visualization
        ir = denormalize(sample['ir_image'], gray=True)
        ref = denormalize(sample['ref_image'])
        target = denormalize(sample['target_image'])
        
        # Convert to numpy images
        ir_img = tensor_to_image(ir)
        ref_img = tensor_to_image(ref)
        target_img = tensor_to_image(target)
        
        # Plot row
        axes[i, 0].imshow(ir_img)
        if i == 0:
            axes[i, 0].set_title('IR Input', fontsize=12, fontweight='bold')
        axes[i, 0].axis('off')
        
        axes[i, 1].imshow(ref_img)
        if i == 0:
            axes[i, 1].set_title('Reference', fontsize=12, fontweight='bold')
        axes[i, 1].axis('off')
        
        axes[i, 2].imshow(target_img)
        if i == 0:
            axes[i, 2].set_title('Target', fontsize=12, fontweight='bold')
        axes[i, 2].axis('off')
        
        # Add text label for the row
        axes[i, 3].text(
            0.5, 0.5, 
            f'Sample {i+1}\n(Same image,\ndifferent\naugmentation)',
            ha='center', va='center',
            fontsize=10,
            transform=axes[i, 3].transAxes
        )
        axes[i, 3].axis('off')
    
    fig.suptitle(
        'Geometric Augmentation Test\n'
        'Each row shows the same source image with different random rotation and perspective warping',
        fontsize=14, fontweight='bold'
    )
    
    plt.tight_layout()
    
    # Save
    output_path = Path('./outputs/augmentation_test.png')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\nVisualization saved to: {output_path}")
    print("\nAugmentation parameters:")
    print(f"  Rotation: {config.data.random_rotation} (±{config.data.max_rotation_angle}°)")
    print(f"  Perspective: {config.data.random_perspective} (distortion={config.data.perspective_distortion})")
    print("\nNote: Each row shows the same source image with different random augmentations.")
    print("You should see variations in rotation and perspective warping across the rows.")


if __name__ == "__main__":
    test_augmentations()
