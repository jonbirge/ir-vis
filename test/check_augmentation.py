"""
Simple check to see how often geometric augmentation actually happens
"""
import torch
from config import get_config
from dataset import IRColorPairDataset
from pathlib import Path

config = get_config()
config.data.use_augmentation = True
config.data.random_rotation = True
config.data.random_perspective = True

# Find dataset
test_dir = None
for d in ["./data/coco/val2017", "./data/coco/train2017"]:
    if Path(d).exists():
        test_dir = d
        break

if not test_dir:
    print("No dataset found")
    exit(1)

dataset = IRColorPairDataset(
    image_source=test_dir,
    config=config.data,
    is_training=True
)

print("Loading same image 100 times to check augmentation frequency...")

ref_images = []
for i in range(100):
    sample = dataset[0]  # Same image every time
    ref_images.append(sample['ref_image'])

# Check uniqueness
unique_count = 0
for i in range(1, len(ref_images)):
    if not torch.equal(ref_images[0], ref_images[i]):
        unique_count += 1

print(f"\nResults:")
print(f"  {unique_count}/99 images were different from the first")
print(f"  Augmentation rate: {unique_count/99*100:.1f}%")

if unique_count > 70:
    print("  ✓ Augmentation is working well")
elif unique_count > 40:
    print("  ⚠ Augmentation is working but at reduced frequency")
    print("    (Expected: ~75% due to 0.5 probability on rotation AND perspective)")
else:
    print("  ✗ Augmentation may not be working properly")

# Check variance
ref_stack = torch.stack(ref_images)
variance = ref_stack.var(dim=0).mean().item()
print(f"\n  Average pixel variance: {variance:.6f}")

if variance < 0.0001:
    print("  ✗ Very low variance - images look identical!")
