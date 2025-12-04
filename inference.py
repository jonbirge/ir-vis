"""
Inference Script for IR-to-Color Image Translation

This script applies a trained model to colorize IR images using reference
visible images. It can process:
- Single image pairs
- Directories of images
- Real IR images (not just simulated)

Usage:
    # Single image pair
    python inference.py --checkpoint path/to/model.pt --ir image_ir.png --ref image_ref.png
    
    # Directory of images
    python inference.py --checkpoint path/to/model.pt --ir-dir ./ir_images --ref-dir ./ref_images
    
    # With custom output directory
    python inference.py --checkpoint path/to/model.pt --ir image.png --ref ref.png --output ./results
"""

import os
import argparse
from pathlib import Path
from typing import Optional, List, Tuple

import torch
import torch.nn.functional as F
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from PIL import Image
import numpy as np
from tqdm import tqdm # type: ignore

from config import get_config, ModelConfig
from model import IRColorNet, create_model
from utils import load_checkpoint, save_comparison_image


class IRColorizer:
    """
    Inference wrapper for the IR-to-Color translation model.
    
    This class handles:
    - Model loading from checkpoint
    - Image preprocessing
    - Inference
    - Output postprocessing
    
    The class is designed to work with both simulated IR (red channel)
    and real IR images (thermal or near-IR sensors).
    """
    
    def __init__(
        self,
        checkpoint_path: str,
        device: str = 'cuda',
        image_size: Tuple[int, int] = (256, 256)
    ):
        """
        Initialize the colorizer.
        
        Args:
            checkpoint_path: Path to the model checkpoint
            device: Device to run inference on ('cuda' or 'cpu')
            image_size: Size to resize images to (height, width)
        """
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.image_size = image_size
        
        # Load model
        self._load_model(checkpoint_path)
        
        # Setup transforms
        self._setup_transforms()
        
    def _load_model(self, checkpoint_path: str) -> None:
        """
        Load the model from checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint file
        """
        print(f"Loading model from {checkpoint_path}")
        
        # Load checkpoint to get config
        # weights_only=False is needed to load Config dataclass
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        
        # Create model with saved config
        if 'config' in checkpoint:
            config = checkpoint['config']
            model_config = config.model if hasattr(config, 'model') else ModelConfig()
        else:
            model_config = ModelConfig()
        
        self.model = IRColorNet(model_config)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model = self.model.to(self.device)
        self.model.eval()
        
        print(f"Model loaded successfully")
        
    def _setup_transforms(self) -> None:
        """Setup image preprocessing transforms."""
        # ImageNet normalization
        self.normalize_rgb = T.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
        
        self.normalize_gray = T.Normalize(
            mean=[0.485, 0.485, 0.485],
            std=[0.229, 0.229, 0.229]
        )
    
    def preprocess_ir(self, image: Image.Image) -> torch.Tensor:
        """
        Preprocess an IR image for model input.
        
        Handles both grayscale and RGB input (extracts luminance from RGB).
        
        Args:
            image: PIL Image (grayscale or RGB)
            
        Returns:
            Preprocessed tensor [1, 3, H, W]
        """
        # Convert to grayscale if RGB
        if image.mode == 'RGB':
            image = image.convert('L')
        elif image.mode == 'RGBA':
            image = image.convert('L')
        elif image.mode != 'L':
            image = image.convert('L')
        
        # Resize
        image = TF.resize(image, self.image_size)
        
        # Convert to tensor [1, H, W]
        tensor = TF.to_tensor(image)
        
        # Repeat to 3 channels [3, H, W]
        tensor = tensor.repeat(3, 1, 1)
        
        # Normalize
        tensor = self.normalize_gray(tensor)
        
        # Add batch dimension [1, 3, H, W]
        tensor = tensor.unsqueeze(0)
        
        return tensor.to(self.device)
    
    def preprocess_ref(self, image: Image.Image) -> torch.Tensor:
        """
        Preprocess a reference color image for model input.
        
        Args:
            image: PIL Image (RGB)
            
        Returns:
            Preprocessed tensor [1, 3, H, W]
        """
        # Ensure RGB
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Resize
        image = TF.resize(image, self.image_size)
        
        # Convert to tensor
        tensor = TF.to_tensor(image)
        
        # Normalize
        tensor = self.normalize_rgb(tensor)
        
        # Add batch dimension
        tensor = tensor.unsqueeze(0)
        
        return tensor.to(self.device)
    
    def postprocess(self, output: torch.Tensor) -> Image.Image:
        """
        Postprocess model output to PIL Image.
        
        Args:
            output: Model output tensor [1, 3, H, W] in [-1, 1] range
            
        Returns:
            PIL Image
        """
        # Remove batch dimension and move to CPU
        output = output.squeeze(0).cpu()
        
        # Convert from [-1, 1] to [0, 1]
        output = (output + 1) / 2
        output = torch.clamp(output, 0, 1)
        
        # Convert to numpy
        output = output.numpy().transpose(1, 2, 0)
        
        # Convert to uint8
        output = (output * 255).astype(np.uint8)
        
        return Image.fromarray(output)
    
    @torch.no_grad()
    def colorize(
        self,
        ir_image: Image.Image,
        ref_image: Image.Image,
        output_size: Optional[Tuple[int, int]] = None
    ) -> Image.Image:
        """
        Colorize an IR image using a reference color image.
        
        Args:
            ir_image: Grayscale IR image (PIL Image)
            ref_image: Color reference image (PIL Image)
            output_size: Optional output size (default: same as ir_image)
            
        Returns:
            Colorized PIL Image
        """
        # Store original size for output
        if output_size is None:
            output_size = ir_image.size[::-1]  # PIL uses (W, H), we want (H, W)
        
        # Preprocess
        ir_tensor = self.preprocess_ir(ir_image)
        ref_tensor = self.preprocess_ref(ref_image)
        
        # Forward pass
        outputs = self.model(ir_tensor, ref_tensor)
        pred = outputs['output']
        
        # Postprocess
        result = self.postprocess(pred)
        
        # Resize to original size if needed
        if result.size[::-1] != output_size:
            result = result.resize((output_size[1], output_size[0]), Image.BILINEAR)
        
        return result
    
    def colorize_files(
        self,
        ir_path: str,
        ref_path: str,
        output_path: Optional[str] = None,
        save_comparison: bool = True
    ) -> Image.Image:
        """
        Colorize an IR image file using a reference image file.
        
        Args:
            ir_path: Path to IR image
            ref_path: Path to reference image
            output_path: Optional path to save result
            save_comparison: Whether to save a comparison image
            
        Returns:
            Colorized PIL Image
        """
        # Load images
        ir_image = Image.open(ir_path)
        ref_image = Image.open(ref_path)
        
        # Colorize
        result = self.colorize(ir_image, ref_image)
        
        # Save if path provided
        if output_path:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            result.save(output_path)
            print(f"Saved colorized image to {output_path}")
            
            if save_comparison:
                # Create comparison image
                comp_path = output_path.replace('.', '_comparison.')
                create_comparison_figure(
                    ir_image, ref_image, result, comp_path
                )
                print(f"Saved comparison to {comp_path}")
        
        return result


def create_comparison_figure(
    ir_image: Image.Image,
    ref_image: Image.Image,
    result_image: Image.Image,
    output_path: str
) -> None:
    """
    Create a side-by-side comparison figure.
    
    Args:
        ir_image: Original IR image
        ref_image: Reference color image
        result_image: Colorized result
        output_path: Path to save the figure
    """
    import matplotlib.pyplot as plt # type: ignore
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Ensure all images are RGB for display
    if ir_image.mode == 'L':
        ir_display = ir_image.convert('RGB')
    else:
        ir_display = ir_image
    
    if ref_image.mode != 'RGB':
        ref_display = ref_image.convert('RGB')
    else:
        ref_display = ref_image
    
    axes[0].imshow(ir_display)
    axes[0].set_title('IR Input', fontsize=14)
    axes[0].axis('off')
    
    axes[1].imshow(ref_display)
    axes[1].set_title('Color Reference', fontsize=14)
    axes[1].axis('off')
    
    axes[2].imshow(result_image)
    axes[2].set_title('Colorized Output', fontsize=14)
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def process_directory(
    colorizer: IRColorizer,
    ir_dir: str,
    ref_dir: str,
    output_dir: str,
    ref_mode: str = 'match'
) -> None:
    """
    Process a directory of IR images.
    
    Args:
        colorizer: The IRColorizer instance
        ir_dir: Directory containing IR images
        ref_dir: Directory containing reference images
        output_dir: Directory to save results
        ref_mode: How to match references:
                  'match': Match by filename
                  'single': Use single reference for all
                  'random': Random reference from directory
    """
    ir_dir = Path(ir_dir)
    ref_dir = Path(ref_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get IR images
    valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
    ir_files = [f for f in ir_dir.iterdir() if f.suffix.lower() in valid_extensions]
    ir_files.sort()
    
    # Get reference images
    ref_files = [f for f in ref_dir.iterdir() if f.suffix.lower() in valid_extensions]
    ref_files.sort()
    
    print(f"Found {len(ir_files)} IR images and {len(ref_files)} reference images")
    
    # Process each IR image
    for ir_file in tqdm(ir_files, desc="Processing images"):
        # Determine reference image
        if ref_mode == 'match':
            # Try to find matching filename
            ref_file = ref_dir / ir_file.name
            if not ref_file.exists():
                # Try different extensions
                for ext in valid_extensions:
                    ref_file = ref_dir / f"{ir_file.stem}{ext}"
                    if ref_file.exists():
                        break
            if not ref_file.exists() and ref_files:
                ref_file = ref_files[0]  # Fallback to first
        elif ref_mode == 'single':
            ref_file = ref_files[0] if ref_files else None
        elif ref_mode == 'random':
            import random
            ref_file = random.choice(ref_files) if ref_files else None
        else:
            ref_file = ref_files[0] if ref_files else None
        
        if ref_file is None:
            print(f"Warning: No reference found for {ir_file.name}, skipping")
            continue
        
        # Output path
        output_path = output_dir / f"{ir_file.stem}_colorized.png"
        
        # Colorize
        try:
            colorizer.colorize_files(
                str(ir_file),
                str(ref_file),
                str(output_path),
                save_comparison=True
            )
        except Exception as e:
            print(f"Error processing {ir_file.name}: {e}")


def main():
    """Main entry point for inference script."""
    parser = argparse.ArgumentParser(
        description="Colorize IR images using trained model"
    )
    
    # Required arguments
    parser.add_argument(
        '--checkpoint', type=str, required=True,
        help="Path to model checkpoint"
    )
    
    # Input options (either single files or directories)
    parser.add_argument(
        '--ir', type=str, default=None,
        help="Path to single IR image"
    )
    parser.add_argument(
        '--ref', type=str, default=None,
        help="Path to single reference image"
    )
    parser.add_argument(
        '--ir-dir', type=str, default=None,
        help="Directory of IR images"
    )
    parser.add_argument(
        '--ref-dir', type=str, default=None,
        help="Directory of reference images"
    )
    
    # Output options
    parser.add_argument(
        '--output', type=str, default='./inference_results',
        help="Output path (file for single image, directory for batch)"
    )
    parser.add_argument(
        '--no-comparison', action='store_true',
        help="Don't save comparison images"
    )
    
    # Processing options
    parser.add_argument(
        '--device', type=str, default='cuda',
        help="Device to run on (cuda/cpu)"
    )
    parser.add_argument(
        '--image-size', type=int, default=256,
        help="Image size for processing"
    )
    parser.add_argument(
        '--ref-mode', type=str, default='match',
        choices=['match', 'single', 'random'],
        help="How to match references in batch mode"
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    single_mode = args.ir is not None and args.ref is not None
    batch_mode = args.ir_dir is not None and args.ref_dir is not None
    
    if not single_mode and not batch_mode:
        parser.error("Must provide either (--ir and --ref) or (--ir-dir and --ref-dir)")
    
    # Create colorizer
    colorizer = IRColorizer(
        checkpoint_path=args.checkpoint,
        device=args.device,
        image_size=(args.image_size, args.image_size)
    )
    
    if single_mode:
        # Process single image pair
        output_path = args.output
        if not output_path.endswith(('.png', '.jpg', '.jpeg')):
            output_path = os.path.join(output_path, 'colorized.png')
        
        colorizer.colorize_files(
            args.ir,
            args.ref,
            output_path,
            save_comparison=not args.no_comparison
        )
    else:
        # Process directories
        process_directory(
            colorizer,
            args.ir_dir,
            args.ref_dir,
            args.output,
            ref_mode=args.ref_mode
        )
    
    print("Inference complete!")


if __name__ == "__main__":
    main()
