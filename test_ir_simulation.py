"""
Quick test script to visualize the new randomized IR simulation.

This script creates side-by-side comparisons showing:
- Original color image
- Simulated IR with different random channel subtractions
- Effect of noise parameter

Run with: python test_ir_simulation.py
"""
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from pathlib import Path
import random

from config import get_config
from dataset import IRColorPairDataset


def test_ir_variations():
    """Generate multiple IR simulations from the same image to show variability."""
    
    config = get_config()
    
    # Get a test image
    test_images = list(Path('data/cityscapes/leftImg8bit/val/frankfurt').glob('*.png'))
    if not test_images:
        print("No test images found. Please download Cityscapes dataset.")
        return
    
    # Create dataset
    dataset = IRColorPairDataset(test_images[:1], config.data, is_training=True)
    
    # Get the same crop multiple times to see variation
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    fig.suptitle('Randomized IR Simulation - Multiple Samples from Same Image', fontsize=16)
    
    for i in range(4):
        # Get a sample
        sample = dataset[0]
        
        # Denormalize for display
        ir_img = sample['ir_image'][0].numpy()  # Take first channel (they're all the same)
        ir_img = (ir_img - ir_img.min()) / (ir_img.max() - ir_img.min())
        
        target_img = sample['target_image'].numpy().transpose(1, 2, 0)
        target_img = (target_img - target_img.min()) / (target_img.max() - target_img.min())
        
        # Display
        axes[0, i].imshow(target_img)
        axes[0, i].set_title(f'Original Crop #{i+1}')
        axes[0, i].axis('off')
        
        axes[1, i].imshow(ir_img, cmap='gray')
        axes[1, i].set_title(f'Simulated IR #{i+1}')
        axes[1, i].axis('off')
    
    plt.tight_layout()
    output_path = 'ir_simulation_comparison.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ Visualization saved to {output_path}")
    print(f"  Each IR image has different random channel subtractions and noise")
    print(f"  Config: max_channel_subtract={config.data.max_channel_subtract}")
    print(f"  Config: ir_noise_std={config.data.ir_noise_std}")
    plt.close()


def test_parameter_effects():
    """Show the effect of different parameter settings."""
    
    config = get_config()
    test_images = list(Path('data/cityscapes/leftImg8bit/val/frankfurt').glob('*.png'))
    
    if not test_images:
        print("No test images found.")
        return
    
    fig, axes = plt.subplots(3, 3, figsize=(12, 12))
    fig.suptitle('Effect of IR Simulation Parameters', fontsize=16)
    
    # Test different parameter combinations
    params = [
        (0.0, 0.0, "No subtract, No noise"),
        (0.0, 5.0, "No subtract, Noise=5"),
        (0.0, 10.0, "No subtract, Noise=10"),
        (0.15, 0.0, "Subtract=0.15, No noise"),
        (0.15, 5.0, "Subtract=0.15, Noise=5"),
        (0.15, 10.0, "Subtract=0.15, Noise=10"),
        (0.25, 0.0, "Subtract=0.25, No noise"),
        (0.25, 5.0, "Subtract=0.25, Noise=5"),
        (0.25, 10.0, "Subtract=0.25, Noise=10"),
    ]
    
    for idx, (max_sub, noise_std, label) in enumerate(params):
        row = idx // 3
        col = idx % 3
        
        # Temporarily modify config
        config.data.max_channel_subtract = max_sub
        config.data.ir_noise_std = noise_std
        
        # Create dataset with this config
        dataset = IRColorPairDataset(test_images[:1], config.data, is_training=True)
        sample = dataset[0]
        
        # Display IR
        ir_img = sample['ir_image'][0].numpy()
        ir_img = (ir_img - ir_img.min()) / (ir_img.max() - ir_img.min())
        
        axes[row, col].imshow(ir_img, cmap='gray')
        axes[row, col].set_title(label, fontsize=10)
        axes[row, col].axis('off')
    
    plt.tight_layout()
    output_path = 'ir_parameter_effects.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ Parameter comparison saved to {output_path}")
    plt.close()


if __name__ == "__main__":
    print("Testing Randomized IR Simulation")
    print("=" * 60)
    
    test_ir_variations()
    test_parameter_effects()
    
    print("\n✓ All tests completed successfully!")
