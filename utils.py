"""
Utility functions for IR-to-Color Image Translation

This module provides helper functions for:
- Checkpoint saving and loading
- Visualization of results
- Logging and metrics tracking
- Image format conversions
- Random seed management for reproducibility
"""

import os
import random
from pathlib import Path
from typing import Dict, Optional, Any, List, Tuple
import json
from datetime import datetime

import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt # type: ignore
import matplotlib.gridspec as gridspec # type: ignore


def set_seed(seed: int) -> None:
    """
    Set random seeds for reproducibility.
    
    Sets seeds for Python random, NumPy, and PyTorch (both CPU and CUDA).
    Also configures PyTorch for deterministic behavior where possible.
    
    Args:
        seed: Random seed to use
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # For multi-GPU
        
        # Deterministic behavior (may slow down training slightly)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[Any],
    epoch: int,
    loss: float,
    config: Any,
    filepath: str
) -> None:
    """
    Save a training checkpoint.
    
    Saves model weights, optimizer state, scheduler state, and training
    metadata to enable resuming training from any point.
    
    Args:
        model: The neural network model
        optimizer: The optimizer
        scheduler: Learning rate scheduler (can be None)
        epoch: Current epoch number
        loss: Current loss value
        config: Configuration object
        filepath: Path to save the checkpoint
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'config': config,
        'timestamp': datetime.now().isoformat()
    }
    
    if scheduler is not None:
        checkpoint['scheduler_state_dict'] = scheduler.state_dict()
    
    # Create directory if needed
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    
    torch.save(checkpoint, filepath)
    print(f"Saved checkpoint to {filepath}")


def load_checkpoint(
    filepath: str,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    device: str = 'cuda'
) -> Dict[str, Any]:
    """
    Load a training checkpoint.
    
    Restores model weights and optionally optimizer/scheduler states.
    
    Args:
        filepath: Path to the checkpoint file
        model: Model to load weights into
        optimizer: Optional optimizer to restore state
        scheduler: Optional scheduler to restore state
        device: Device to map checkpoint tensors to
        
    Returns:
        Dictionary with checkpoint metadata (epoch, loss, etc.)
    """
    print(f"Loading checkpoint from {filepath}")
    
    checkpoint = torch.load(filepath, map_location=device)
    
    # Load model weights
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Load optimizer state if provided
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    # Load scheduler state if provided
    if scheduler is not None and 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    metadata = {
        'epoch': checkpoint.get('epoch', 0),
        'loss': checkpoint.get('loss', float('inf')),
        'timestamp': checkpoint.get('timestamp', 'unknown')
    }
    
    print(f"Loaded checkpoint from epoch {metadata['epoch']} (loss: {metadata['loss']:.6f})")
    
    return metadata


def denormalize(
    tensor: torch.Tensor, 
    mean: Tuple[float, ...] = (0.485, 0.456, 0.406),
    std: Tuple[float, ...] = (0.229, 0.224, 0.225)
) -> torch.Tensor:
    """
    Denormalize a tensor from ImageNet normalization.
    
    Reverses the normalization applied during preprocessing to convert
    tensors back to [0, 1] range for visualization.
    
    Args:
        tensor: Normalized tensor [B, C, H, W] or [C, H, W]
        mean: Normalization mean
        std: Normalization std
        
    Returns:
        Denormalized tensor in [0, 1] range
    """
    device = tensor.device
    
    mean = torch.tensor(mean, device=device).view(1, 3, 1, 1)
    std = torch.tensor(std, device=device).view(1, 3, 1, 1)
    
    if tensor.dim() == 3:
        mean = mean.squeeze(0)
        std = std.squeeze(0)
    
    denorm = tensor * std + mean
    return torch.clamp(denorm, 0, 1)


def denormalize_gray(tensor: torch.Tensor) -> torch.Tensor:
    """
    Denormalize a grayscale tensor (repeated to 3 channels).
    
    Uses the same mean/std for all channels as used during preprocessing.
    
    Args:
        tensor: Normalized grayscale tensor [B, 3, H, W] or [3, H, W]
        
    Returns:
        Denormalized tensor in [0, 1] range
    """
    return denormalize(
        tensor, 
        mean=(0.485, 0.485, 0.485),
        std=(0.229, 0.229, 0.229)
    )


def tensor_to_image(tensor: torch.Tensor) -> np.ndarray:
    """
    Convert a tensor to a numpy image for visualization.
    
    Args:
        tensor: Image tensor [C, H, W] or [B, C, H, W] in [0, 1] range
        
    Returns:
        NumPy array [H, W, C] in [0, 255] uint8 range
    """
    if tensor.dim() == 4:
        tensor = tensor[0]  # Take first image if batched
    
    # Move to CPU and convert to numpy
    image = tensor.detach().cpu().numpy()
    
    # Transpose from [C, H, W] to [H, W, C]
    image = np.transpose(image, (1, 2, 0))
    
    # Convert to uint8
    image = (image * 255).clip(0, 255).astype(np.uint8)
    
    return image


def save_comparison_image(
    ir_image: torch.Tensor,
    ref_image: torch.Tensor,
    pred_image: torch.Tensor,
    target_image: torch.Tensor,
    filepath: str,
    title: Optional[str] = None
) -> None:
    """
    Save a comparison visualization of IR, reference, prediction, and target.
    
    Creates a 2x2 grid showing:
    - Top-left: IR input (grayscale)
    - Top-right: Color reference
    - Bottom-left: Model prediction
    - Bottom-right: Ground truth target
    
    Args:
        ir_image: IR input tensor [3, H, W] (normalized grayscale)
        ref_image: Reference image tensor [3, H, W] (normalized)
        pred_image: Predicted colorized image [3, H, W] (model output in [-1, 1])
        target_image: Ground truth target [3, H, W] (normalized)
        filepath: Path to save the visualization
        title: Optional title for the figure
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    
    # Denormalize tensors
    ir_denorm = denormalize_gray(ir_image)
    ref_denorm = denormalize(ref_image)
    target_denorm = denormalize(target_image)
    
    # Model output is in [-1, 1], convert to [0, 1]
    pred_denorm = (pred_image + 1) / 2
    pred_denorm = torch.clamp(pred_denorm, 0, 1)
    
    # Convert to images
    ir_img = tensor_to_image(ir_denorm)
    ref_img = tensor_to_image(ref_denorm)
    pred_img = tensor_to_image(pred_denorm)
    target_img = tensor_to_image(target_denorm)
    
    # Plot
    axes[0, 0].imshow(ir_img)
    axes[0, 0].set_title('IR Input (Grayscale)', fontsize=12)
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(ref_img)
    axes[0, 1].set_title('Color Reference', fontsize=12)
    axes[0, 1].axis('off')
    
    axes[1, 0].imshow(pred_img)
    axes[1, 0].set_title('Model Prediction', fontsize=12)
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(target_img)
    axes[1, 1].set_title('Ground Truth', fontsize=12)
    axes[1, 1].axis('off')
    
    if title:
        fig.suptitle(title, fontsize=14)
    
    plt.tight_layout()
    
    # Create directory if needed
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close(fig)


def save_batch_visualization(
    ir_images: torch.Tensor,
    ref_images: torch.Tensor,
    pred_images: torch.Tensor,
    target_images: torch.Tensor,
    filepath: str,
    max_samples: int = 4
) -> None:
    """
    Save visualization for a batch of samples.
    
    Creates a grid with each row showing one sample:
    [IR | Reference | Prediction | Target]
    
    Args:
        ir_images: Batch of IR inputs [B, 3, H, W]
        ref_images: Batch of references [B, 3, H, W]
        pred_images: Batch of predictions [B, 3, H, W]
        target_images: Batch of targets [B, 3, H, W]
        filepath: Path to save visualization
        max_samples: Maximum number of samples to visualize
    """
    batch_size = min(ir_images.shape[0], max_samples)
    
    fig, axes = plt.subplots(batch_size, 4, figsize=(16, 4 * batch_size))
    
    # Handle single sample case
    if batch_size == 1:
        axes = axes.reshape(1, -1)
    
    column_titles = ['IR Input', 'Reference', 'Prediction', 'Ground Truth']
    
    for i in range(batch_size):
        # Get single samples
        ir = ir_images[i]
        ref = ref_images[i]
        pred = pred_images[i]
        target = target_images[i]
        
        # Denormalize
        ir_denorm = denormalize_gray(ir)
        ref_denorm = denormalize(ref)
        target_denorm = denormalize(target)
        pred_denorm = (pred + 1) / 2
        pred_denorm = torch.clamp(pred_denorm, 0, 1)
        
        # Convert to images
        images = [
            tensor_to_image(ir_denorm),
            tensor_to_image(ref_denorm),
            tensor_to_image(pred_denorm),
            tensor_to_image(target_denorm)
        ]
        
        # Plot row
        for j, (img, title) in enumerate(zip(images, column_titles)):
            axes[i, j].imshow(img)
            if i == 0:
                axes[i, j].set_title(title, fontsize=12)
            axes[i, j].axis('off')
    
    plt.tight_layout()
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close(fig)


class MetricsLogger:
    """
    Logger for training metrics.
    
    Tracks loss values and other metrics during training, saves to JSON,
    and provides methods for computing running averages.
    """
    
    def __init__(self, log_dir: str):
        """
        Initialize the metrics logger.
        
        Args:
            log_dir: Directory to save log files
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.metrics = {
            'train': [],
            'val': []
        }
        
        self.current_epoch_metrics = {}
        
    def log_iteration(
        self, 
        metrics: Dict[str, float], 
        phase: str = 'train'
    ) -> None:
        """
        Log metrics for a single iteration.
        
        Args:
            metrics: Dictionary of metric names and values
            phase: 'train' or 'val'
        """
        for name, value in metrics.items():
            key = f"{phase}_{name}"
            if key not in self.current_epoch_metrics:
                self.current_epoch_metrics[key] = []
            self.current_epoch_metrics[key].append(value)
    
    def end_epoch(self, epoch: int) -> Dict[str, float]:
        """
        Finalize metrics for an epoch.
        
        Computes averages and saves to history.
        
        Args:
            epoch: Epoch number
            
        Returns:
            Dictionary of average metrics for the epoch
        """
        epoch_averages = {'epoch': epoch}
        
        for key, values in self.current_epoch_metrics.items():
            avg = sum(values) / len(values) if values else 0
            epoch_averages[key] = avg
        
        self.metrics['train'].append(epoch_averages)
        self.current_epoch_metrics = {}
        
        return epoch_averages
    
    def save(self) -> None:
        """Save metrics to JSON file."""
        filepath = self.log_dir / 'metrics.json'
        with open(filepath, 'w') as f:
            json.dump(self.metrics, f, indent=2)
    
    def plot_losses(self, filepath: Optional[str] = None) -> None:
        """
        Plot training loss curves.
        
        Args:
            filepath: Optional path to save the plot
        """
        if not self.metrics['train']:
            return
        
        epochs = [m['epoch'] for m in self.metrics['train']]
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Total loss
        if 'train_total' in self.metrics['train'][0]:
            total_losses = [m.get('train_total', 0) for m in self.metrics['train']]
            axes[0, 0].plot(epochs, total_losses, 'b-', label='Total Loss')
            axes[0, 0].set_xlabel('Epoch')
            axes[0, 0].set_ylabel('Loss')
            axes[0, 0].set_title('Total Loss')
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)
        
        # L1 loss
        if 'train_l1' in self.metrics['train'][0]:
            l1_losses = [m.get('train_l1', 0) for m in self.metrics['train']]
            axes[0, 1].plot(epochs, l1_losses, 'r-', label='L1 Loss')
            axes[0, 1].set_xlabel('Epoch')
            axes[0, 1].set_ylabel('Loss')
            axes[0, 1].set_title('L1 (Pixel) Loss')
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)
        
        # Perceptual loss
        if 'train_perceptual' in self.metrics['train'][0]:
            perc_losses = [m.get('train_perceptual', 0) for m in self.metrics['train']]
            axes[1, 0].plot(epochs, perc_losses, 'g-', label='Perceptual Loss')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('Loss')
            axes[1, 0].set_title('Perceptual Loss')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
        
        # Style loss
        if 'train_style' in self.metrics['train'][0]:
            style_losses = [m.get('train_style', 0) for m in self.metrics['train']]
            axes[1, 1].plot(epochs, style_losses, 'm-', label='Style Loss')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Loss')
            axes[1, 1].set_title('Style Loss')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if filepath:
            Path(filepath).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(filepath, dpi=150)
        
        plt.close(fig)


def count_parameters(model: nn.Module) -> Dict[str, int]:
    """
    Count model parameters.
    
    Args:
        model: PyTorch model
        
    Returns:
        Dictionary with total and trainable parameter counts
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return {
        'total': total,
        'trainable': trainable,
        'frozen': total - trainable
    }


def format_time(seconds: float) -> str:
    """
    Format seconds into a human-readable string.
    
    Args:
        seconds: Number of seconds
        
    Returns:
        Formatted string like "1h 23m 45s"
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    
    if hours > 0:
        return f"{hours}h {minutes}m {secs}s"
    elif minutes > 0:
        return f"{minutes}m {secs}s"
    else:
        return f"{secs}s"


# Test utilities when run directly
if __name__ == "__main__":
    print("Testing utility functions...")
    
    # Test seed setting
    set_seed(42)
    print(f"Random test (should be deterministic): {random.random():.6f}")
    
    # Test denormalization
    test_tensor = torch.randn(1, 3, 64, 64)
    denorm = denormalize(test_tensor)
    print(f"Denormalized range: [{denorm.min():.3f}, {denorm.max():.3f}]")
    
    # Test metrics logger
    logger = MetricsLogger('./test_logs')
    for epoch in range(3):
        for i in range(10):
            logger.log_iteration({'total': random.random(), 'l1': random.random()})
        avg = logger.end_epoch(epoch)
        print(f"Epoch {epoch} averages: {avg}")
    
    print("\nUtilities test complete!")
