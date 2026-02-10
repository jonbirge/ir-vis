# IR-to-Color Image Translation Network

A PyTorch implementation of a reference-based colorization network for translating infrared (IR) images to higher resolution and color using a visible-light (VIS) reference image from a different viewpoint/time. An attention mechanism between the IR image and the reference VIS image is used to inform the output image and avoid halucination. While it doesn't always work great, it appears to "fail well" in that if it doesn't work it just fails to improve the IR image rather than imagining things that aren't there. This is trained with a "curriculum" approach where first we handle relatively easy images with close up objects before training on more complex outdoor scenes. For some reason this was the only way I was able to get it to work on the complex scenes. The training is done by synthesizing IR from VIS by mixing the color channels into one using random weightings but biased towards the red channel. Not a great model for thermal IR, but the hope is that the extra noise thrown in will make it robust enough. In the limited testing I've done with actual IR images, this seems to be true.

## Overview

This project implements a deep learning approach to colorize IR images by leveraging color information from a reference visible image. The architecture uses:

- **Dual encoders**: Separate encoders for IR (content) and visible (reference) images based on pretrained ResNet
- **Attention-based feature matching**: Cross-attention mechanism to align features across different viewpoints
- **U-Net style decoder**: With skip connections to preserve fine structural details
- **Multi-component loss**: Combining L1, perceptual, style, and histogram losses for high-quality results

## Architecture

```
IR Image (grayscale) ────► Content Encoder ────┐
                                               ├─► Feature Matching ─► Decoder ─► Colorized Output
Reference Image (color) ─► Reference Encoder ──┘
                                 │
                          Cross-Attention
                        (handles viewpoint
                           differences)
```

## Installation

### Prerequisites

- Python 3.8+
- CUDA-capable GPU with 8GB+ VRAM (16GB+ recommended)
- ~25GB disk space for COCO dataset

### Setup

1. Clone or download this repository:
```bash
cd ir_colorization
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

For GPU support, ensure you have the correct CUDA version installed and install PyTorch accordingly:
```bash
# Example for CUDA 11.8
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

## Training

### Quick Start

#### Option 1: Cityscapes (Default - Recommended for Urban Scenes)

Cityscapes provides high-quality 2048x1024 urban street scenes. It requires free registration but can auto-download:

**Automatic Download:**
1. Register (free) at: https://www.cityscapes-dataset.com/register/
2. Set environment variables with your credentials:
   
   **Windows (PowerShell):**
   ```powershell
   $env:CITYSCAPES_USERNAME = "your_email@example.com"
   $env:CITYSCAPES_PASSWORD = "your_password"
   ```
   
   **Windows (CMD):**
   ```cmd
   set CITYSCAPES_USERNAME=your_email@example.com
   set CITYSCAPES_PASSWORD=your_password
   ```
   
   **Linux/Mac:**
   ```bash
   export CITYSCAPES_USERNAME="your_email@example.com"
   export CITYSCAPES_PASSWORD="your_password"
   ```

3. Run training (downloads automatically ~11GB):
   ```bash
   python train.py
   ```

**Manual Download (Alternative):**
1. Download `leftImg8bit_trainvaltest.zip` from https://www.cityscapes-dataset.com/downloads/
2. Extract to `./data/cityscapes/`
3. Run: `python train.py`

#### Option 2: COCO (Auto-downloads, No Registration)

COCO provides diverse outdoor scenes and downloads automatically without any registration:

```bash
python train.py --dataset coco
```

This will download COCO 2017 (~18GB for training, ~1GB for validation) on first run.

### What Happens During Training

The training process:
1. Loads images from the selected dataset
2. Creates simulated IR/visible image pairs (see Training Data Simulation below)
3. Trains the model for 100 epochs (configurable)
4. Saves checkpoints, visualizations, and logs to `./outputs/`

### Training Data Simulation

Since paired IR/visible datasets are rare, we simulate training data:
1. Load a color outdoor image from COCO
2. Take a **random crop** (40-70% of original) to simulate different FOV
3. Extract the **red channel** from the crop as simulated IR
4. Use the **full original image** as the color reference
5. The **cropped region in color** is the ground truth target

This approach works because:
- Red channel correlates with near-IR reflectance for many natural materials
- The crop simulates the typical FOV difference between IR and visible cameras
- COCO provides diverse outdoor scenes

### Configuration

Edit `config.py` to adjust hyperparameters:

```python
# Key settings to modify:

# Data settings
dataset_name = "cityscapes"    # Options: 'cityscapes', 'coco'
crop_ratio_range = (0.4, 0.7)  # FOV difference simulation
ir_image_size = (256, 256)     # Network input size

# Model settings
encoder_backbone = "resnet34"   # Options: resnet18, resnet34, resnet50
num_attention_heads = 8         # More heads = more diverse matching

# Training settings
batch_size = 8                  # Reduce if out of GPU memory
num_epochs = 100
learning_rate = 1e-4

# Loss weights (tune these for your use case)
l1_weight = 1.0                 # Pixel accuracy
perceptual_weight = 0.5         # Perceptual quality
style_weight = 10.0             # Color transfer from reference
histogram_weight = 0.1          # Global color distribution
```

### Command Line Options

```bash
# Use COCO instead of Cityscapes
python train.py --dataset coco

# Resume from checkpoint
python train.py --resume outputs/checkpoints/epoch_050.pt

# Override settings
python train.py --epochs 200 --batch-size 16 --lr 5e-5

# Custom output directory
python train.py --output ./my_experiment

# Combine options
python train.py --dataset coco --epochs 50 --batch-size 4
```

### Monitoring Training

Training progress is saved to `./outputs/`:
- `checkpoints/`: Model weights (best and periodic)
- `visualizations/`: Sample predictions each epoch
- `logs/`: Loss curves and metrics JSON

The visualization images show:
```
[IR Input] [Color Reference] [Model Prediction] [Ground Truth]
```

### Expected Training Time

On a single RTX 3090 (batch_size=8):
- ~3-4 hours per epoch on full COCO
- ~50-100 epochs for good results
- Total: 1-2 days for full training

For faster iteration, use `max_samples` in the dataset config to limit training data.

## Inference

### Colorize a Single Image

```bash
python inference.py \
    --checkpoint outputs/checkpoints/best_model.pt \
    --ir path/to/ir_image.png \
    --ref path/to/color_reference.jpg \
    --output results/colorized.png
```

### Process a Directory

```bash
python inference.py \
    --checkpoint outputs/checkpoints/best_model.pt \
    --ir-dir ./ir_images/ \
    --ref-dir ./reference_images/ \
    --output ./results/ \
    --ref-mode match  # Options: match, single, random
```

Reference matching modes:
- `match`: Match IR and reference by filename
- `single`: Use the same reference for all IR images
- `random`: Randomly select reference from directory

### Using in Python

```python
from inference import IRColorizer

# Initialize
colorizer = IRColorizer(
    checkpoint_path='outputs/checkpoints/best_model.pt',
    device='cuda',
    image_size=(256, 256)
)

# Colorize
from PIL import Image
ir_image = Image.open('ir_image.png')
ref_image = Image.open('reference.jpg')
result = colorizer.colorize(ir_image, ref_image)
result.save('colorized.png')
```

## Working with Real IR Images

The model is trained on simulated IR (red channel), but can be applied to real IR images:

### Near-IR (NIR) Images
- Work well since NIR and red channel have similar characteristics
- May need brightness/contrast adjustment to match training data

### Thermal IR (Long-wave IR)
- Less correlation with visible appearance
- Consider fine-tuning on real paired data if available
- Results will be more "hallucinated" color than accurate reconstruction

### Tips for Best Results

1. **Reference image quality**: Use sharp, well-exposed reference images
2. **Scene similarity**: Best results when reference shows similar scene content
3. **Preprocessing**: Normalize IR images to similar intensity range as training
4. **Fine-tuning**: For domain-specific applications, fine-tune on your own data

## Project Structure

```
ir_colorization/
├── config.py          # All hyperparameters and settings
├── dataset.py         # Data loading and augmentation
├── model.py           # Neural network architecture
├── losses.py          # Loss functions (perceptual, style, etc.)
├── train.py           # Training script
├── inference.py       # Inference/testing script
├── utils.py           # Utility functions
├── requirements.txt   # Python dependencies
└── README.md          # This file
```

## Customization

### Using Your Own Dataset

Create a custom dataset class that returns dictionaries with:
```python
{
    'ir_image': torch.Tensor,      # [3, H, W] grayscale repeated to 3 channels
    'ref_image': torch.Tensor,     # [3, H, W] color reference
    'target_image': torch.Tensor,  # [3, H, W] ground truth color
}
```

### Modifying the Architecture

Key components in `model.py`:
- `ResNetEncoder`: Change backbone or add more layers
- `CrossAttention`: Modify attention mechanism
- `Decoder`: Adjust upsampling strategy

### Adding New Losses

Extend `losses.py` with additional loss terms:
```python
class MyCustomLoss(nn.Module):
    def forward(self, pred, target):
        # Your loss computation
        return loss
```

Then add to `CombinedLoss` class.

## Troubleshooting

### Out of Memory

- Reduce `batch_size` in config
- Use smaller `image_size`
- Enable gradient checkpointing (not currently implemented)

### Training Not Converging

- Check that images are properly normalized
- Try lower learning rate
- Increase `l1_weight` for more stable training initially

### Poor Color Quality

- Increase `perceptual_weight` and `style_weight`
- Train for more epochs
- Use more diverse reference images

### Download Issues

If COCO download fails:
1. Download manually from https://cocodataset.org/#download
2. Extract to `./data/coco/train2017/` and `./data/coco/val2017/`

## References

```bibtex
@article{johnson2016perceptual,
  title={Perceptual losses for real-time style transfer and super-resolution},
  author={Johnson, Justin and Alahi, Alexandre and Fei-Fei, Li},
  journal={ECCV},
  year={2016}
}

@article{zhang2019deep,
  title={Deep exemplar-based colorization},
  author={Zhang, Bo and others},
  journal={ACM TOG},
  year={2019}
}
```

## License

This project is provided for research and educational purposes. The COCO dataset has its own license terms.

## Acknowledgments

- COCO dataset team for the training data
- PyTorch team for the deep learning framework
- Authors of VGG, ResNet, and perceptual loss papers
