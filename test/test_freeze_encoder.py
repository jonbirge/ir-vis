"""
Test script to verify content encoder freezing functionality.
"""

from config import get_config
from model import create_model


def test_freeze_encoder():
    """Test that freeze_content_encoder works correctly."""
    
    print("=" * 60)
    print("Testing Content Encoder Freezing")
    print("=" * 60)
    
    # Test 1: Default behavior (not frozen)
    print("\n1. Testing with freeze_content_encoder=False (default):")
    print("-" * 60)
    config = get_config()
    config.model.freeze_content_encoder = False
    
    model = create_model(config.model)
    
    # Count trainable params in content encoder
    content_trainable = sum(p.numel() for p in model.content_encoder.parameters() if p.requires_grad)
    content_total = sum(p.numel() for p in model.content_encoder.parameters())
    
    print(f"\nContent encoder:")
    print(f"  Total params: {content_total:,}")
    print(f"  Trainable params: {content_trainable:,}")
    print(f"  All trainable: {content_trainable == content_total}")
    
    # Test 2: Frozen behavior
    print("\n2. Testing with freeze_content_encoder=True:")
    print("-" * 60)
    config = get_config()
    config.model.freeze_content_encoder = True
    
    model = create_model(config.model)
    
    # Count trainable params in content encoder
    content_trainable = sum(p.numel() for p in model.content_encoder.parameters() if p.requires_grad)
    content_total = sum(p.numel() for p in model.content_encoder.parameters())
    
    # Count trainable params in reference encoder (should still be trainable)
    ref_trainable = sum(p.numel() for p in model.reference_encoder.parameters() if p.requires_grad)
    ref_total = sum(p.numel() for p in model.reference_encoder.parameters())
    
    print(f"\nContent encoder (frozen):")
    print(f"  Total params: {content_total:,}")
    print(f"  Trainable params: {content_trainable:,}")
    print(f"  Frozen: {content_trainable == 0}")
    
    print(f"\nReference encoder (should still be trainable):")
    print(f"  Total params: {ref_total:,}")
    print(f"  Trainable params: {ref_trainable:,}")
    print(f"  All trainable: {ref_trainable == ref_total}")
    
    # Verify decoder and feature matching are trainable
    decoder_trainable = sum(p.numel() for p in model.decoder.parameters() if p.requires_grad)
    decoder_total = sum(p.numel() for p in model.decoder.parameters())
    
    fm_trainable = sum(p.numel() for p in model.feature_matching.parameters() if p.requires_grad)
    fm_total = sum(p.numel() for p in model.feature_matching.parameters())
    
    print(f"\nDecoder:")
    print(f"  Total params: {decoder_total:,}")
    print(f"  Trainable params: {decoder_trainable:,}")
    print(f"  All trainable: {decoder_trainable == decoder_total}")
    
    print(f"\nFeature matching:")
    print(f"  Total params: {fm_total:,}")
    print(f"  Trainable params: {fm_trainable:,}")
    print(f"  All trainable: {fm_trainable == fm_total}")
    
    # Summary
    print("\n" + "=" * 60)
    if content_trainable == 0 and ref_trainable == ref_total:
        print("✓ SUCCESS: Content encoder frozen, other components trainable!")
    else:
        print("✗ FAILURE: Freezing behavior incorrect")
    print("=" * 60)


if __name__ == "__main__":
    test_freeze_encoder()
