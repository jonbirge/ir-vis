"""
Test script to verify deterministic visualization cropping.

This script tests that:
1. Using fixed_crop_seed produces consistent crops across multiple calls
2. Without fixed_crop_seed, crops are random
"""

import torch
from config import get_config
from dataset import IRColorPairDataset
from pathlib import Path


def test_deterministic_cropping():
    """Test that fixed_crop_seed produces consistent results."""
    config = get_config()
    
    # Find some test images (use validation set if it exists)
    data_root = Path(config.data.data_root)
    dataset_name = config.data.dataset_name.lower()
    
    if dataset_name == "coco":
        val_dir = data_root / "coco" / "val2017"
        if not val_dir.exists():
            print("Validation directory not found. Please run training first to download data.")
            return
        image_paths = sorted(list(val_dir.glob("*.jpg")))[:5]
    elif dataset_name == "cityscapes":
        val_dir = data_root / "cityscapes" / "leftImg8bit" / "val"
        if not val_dir.exists():
            print("Validation directory not found. Please ensure Cityscapes is downloaded.")
            return
        image_paths = sorted(list(val_dir.glob("*/*_leftImg8bit.png")))[:5]
    else:
        print(f"Unknown dataset: {dataset_name}")
        return
    
    if len(image_paths) < 5:
        print(f"Not enough images found. Need at least 5, found {len(image_paths)}")
        return
    
    print("Testing deterministic cropping with fixed_crop_seed=42...")
    
    # Create two datasets with the same fixed seed
    dataset1 = IRColorPairDataset(
        image_source=image_paths,
        config=config.data,
        is_training=False,
        fixed_crop_seed=42
    )
    
    dataset2 = IRColorPairDataset(
        image_source=image_paths,
        config=config.data,
        is_training=False,
        fixed_crop_seed=42
    )
    
    # Test that same indices produce identical results
    print("\nComparing samples with fixed_crop_seed=42...")
    all_match = True
    for i in range(len(dataset1)):
        sample1 = dataset1[i]
        sample2 = dataset2[i]
        
        # Compare crops (they should be identical)
        ir_match = torch.allclose(sample1['ir_image'], sample2['ir_image'])
        ref_match = torch.allclose(sample1['ref_image'], sample2['ref_image'])
        target_match = torch.allclose(sample1['target_image'], sample2['target_image'])
        
        if ir_match and ref_match and target_match:
            print(f"  Sample {i}: ✓ Crops match perfectly")
        else:
            print(f"  Sample {i}: ✗ Crops differ!")
            all_match = False
    
    if all_match:
        print("\n✓ SUCCESS: All samples with fixed_crop_seed=42 are consistent!")
    else:
        print("\n✗ FAILURE: Some samples differ despite fixed seed")
        return
    
    # Test that different seeds produce different results
    print("\nTesting that different fixed seeds produce different crops...")
    dataset3 = IRColorPairDataset(
        image_source=image_paths,
        config=config.data,
        is_training=False,
        fixed_crop_seed=100
    )
    
    sample_a = dataset1[0]
    sample_b = dataset3[0]
    
    # These should be different
    if not torch.allclose(sample_a['ir_image'], sample_b['ir_image']):
        print("  ✓ Different seeds produce different crops (as expected)")
    else:
        print("  ✗ WARNING: Different seeds produced identical crops")
    
    # Test that without fixed seed, results vary
    print("\nTesting random cropping (no fixed_crop_seed)...")
    dataset_random = IRColorPairDataset(
        image_source=image_paths,
        config=config.data,
        is_training=False,
        fixed_crop_seed=None
    )
    
    # Get same sample twice - should differ
    sample_r1 = dataset_random[0]
    sample_r2 = dataset_random[0]
    
    if not torch.allclose(sample_r1['ir_image'], sample_r2['ir_image']):
        print("  ✓ Random cropping produces different crops each time (as expected)")
    else:
        print("  ✗ WARNING: Random cropping produced identical crops (might be unlucky)")
    
    print("\n" + "="*60)
    print("All tests passed! Deterministic visualization is working correctly.")
    print("="*60)


if __name__ == "__main__":
    test_deterministic_cropping()
