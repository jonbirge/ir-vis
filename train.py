"""
Training Script for IR-to-Color Image Translation Network

This script handles the complete training pipeline:
1. Configuration loading
2. Dataset preparation (including download if needed)
3. Model initialization
4. Training loop with:
   - Forward/backward passes
   - Gradient accumulation and clipping
   - Learning rate scheduling
   - Mixed precision training (optional)
5. Validation
6. Checkpointing and logging
7. Visualization of results

Usage:
    python train.py

The script uses configuration from config.py. To modify hyperparameters,
edit that file or extend this script with command-line argument parsing.
"""

import os
import time
from pathlib import Path
from typing import Dict, Optional, Tuple
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm # type: ignore

from config import Config, get_config
from dataset import get_dataloaders, denormalize
from model import IRColorNet, create_model
from losses import CombinedLoss
from utils import (
    set_seed, save_checkpoint, load_checkpoint,
    save_batch_visualization, MetricsLogger,
    count_parameters, format_time
)


def create_optimizer(
    model: nn.Module, 
    config: Config
) -> torch.optim.Optimizer:
    """
    Create the optimizer for training.
    
    We use AdamW (Adam with decoupled weight decay) which typically
    works well for vision tasks with pretrained features.
    
    Args:
        model: The neural network model
        config: Configuration object
        
    Returns:
        Configured optimizer
    """
    # Separate parameters into groups (for potentially different learning rates)
    # Here we use the same LR for all, but this structure allows easy modification
    params = [
        {
            'params': model.content_encoder.parameters(),
            'lr': config.training.learning_rate
        },
        {
            'params': model.reference_encoder.parameters(),
            'lr': config.training.learning_rate
        },
        {
            'params': model.feature_matching.parameters(),
            'lr': config.training.learning_rate
        },
        {
            'params': model.decoder.parameters(),
            'lr': config.training.learning_rate
        }
    ]
    
    optimizer = optim.AdamW(
        params,
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
        betas=(0.9, 0.999)
    )
    
    return optimizer


def create_scheduler(
    optimizer: torch.optim.Optimizer,
    config: Config
) -> Optional[torch.optim.lr_scheduler._LRScheduler]:
    """
    Create learning rate scheduler.
    
    Options:
    - 'cosine': Cosine annealing (smooth decay to minimum LR)
    - 'step': Step decay at specified milestones
    - 'plateau': Reduce on validation loss plateau
    - 'none': No scheduling
    
    Args:
        optimizer: The optimizer
        config: Configuration object
        
    Returns:
        Configured scheduler or None
    """
    scheduler_type = config.training.lr_scheduler.lower()
    
    if scheduler_type == 'cosine':
        # Cosine annealing with warm restarts
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=config.training.num_epochs,
            eta_min=config.training.lr_min
        )
    elif scheduler_type == 'step':
        # Step decay at milestones
        scheduler = optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=config.training.lr_milestones,
            gamma=config.training.lr_decay_factor
        )
    elif scheduler_type == 'plateau':
        # Reduce on plateau
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=config.training.lr_decay_factor,
            patience=5,
            verbose=True
        )
    elif scheduler_type == 'none':
        scheduler = None
    else:
        raise ValueError(f"Unknown scheduler type: {scheduler_type}")
    
    return scheduler


def train_epoch(
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    config: Config,
    scaler: Optional[GradScaler] = None,
    logger: Optional[MetricsLogger] = None,
    epoch: int = 0
) -> Dict[str, float]:
    """
    Train for one epoch.
    
    Performs forward pass, loss computation, backward pass, and optimizer step
    for each batch. Optionally uses mixed precision training for speed.
    
    Args:
        model: The neural network model
        train_loader: Training data loader
        criterion: Loss function
        optimizer: Optimizer
        device: Device to train on
        config: Configuration object
        scaler: GradScaler for mixed precision (or None)
        logger: Metrics logger (or None)
        epoch: Current epoch number
        
    Returns:
        Dictionary of average loss values for the epoch
    """
    model.train()
    
    total_loss = 0.0
    loss_components = {}
    num_batches = 0
    
    # Progress bar
    pbar = tqdm(
        train_loader, 
        desc=f"Epoch {epoch + 1} [Train]",
        leave=False
    )
    
    for batch_idx, batch in enumerate(pbar):
        # Move data to device
        ir_image = batch['ir_image'].to(device)
        ref_image = batch['ref_image'].to(device)
        target_image = batch['target_image'].to(device)
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass with optional mixed precision
        if scaler is not None:
            with autocast():
                # Forward pass
                outputs = model(ir_image, ref_image)
                pred_image = outputs['output']
                
                # Compute loss
                # Note: target_image is normalized, pred_image is in [-1, 1]
                # We need to convert target to [-1, 1] for loss computation
                # target_norm = (target_image - 0.485) / 0.229  # Approximate inverse
                # Actually, our loss functions handle this internally
                
                # Convert target from normalized to [-1, 1] range to match model output
                target_for_loss = denormalize_for_loss(target_image)
                
                loss, losses = criterion(pred_image, target_for_loss)
            
            # Backward pass with scaling
            scaler.scale(loss).backward()
            
            # Gradient clipping
            if config.training.gradient_clip_norm > 0:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(
                    model.parameters(), 
                    config.training.gradient_clip_norm
                )
            
            # Optimizer step
            scaler.step(optimizer)
            scaler.update()
        else:
            # Standard training without mixed precision
            outputs = model(ir_image, ref_image)
            pred_image = outputs['output']
            
            target_for_loss = denormalize_for_loss(target_image)
            loss, losses = criterion(pred_image, target_for_loss)
            
            loss.backward()
            
            if config.training.gradient_clip_norm > 0:
                nn.utils.clip_grad_norm_(
                    model.parameters(),
                    config.training.gradient_clip_norm
                )
            
            optimizer.step()
        
        # Track losses
        total_loss += loss.item()
        for name, value in losses.items():
            if name not in loss_components:
                loss_components[name] = 0.0
            loss_components[name] += value.item()
        num_batches += 1
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f"{loss.item():.4f}",
            'lr': f"{optimizer.param_groups[0]['lr']:.2e}"
        })
        
        # Log to metrics logger
        if logger is not None:
            log_dict = {name: value.item() for name, value in losses.items()}
            logger.log_iteration(log_dict, phase='train')
        
        # Periodic logging
        if batch_idx % config.training.log_every == 0 and batch_idx > 0:
            avg_loss = total_loss / num_batches
            tqdm.write(
                f"  Batch {batch_idx}/{len(train_loader)}: "
                f"Loss = {avg_loss:.4f}"
            )
    
    # Compute averages
    avg_losses = {'total': total_loss / num_batches}
    for name, value in loss_components.items():
        avg_losses[name] = value / num_batches
    
    return avg_losses


def denormalize_for_loss(normalized_tensor: torch.Tensor) -> torch.Tensor:
    """
    Convert normalized tensor to [-1, 1] range for loss computation.
    
    Our model outputs in [-1, 1] range (tanh activation), but target images
    are normalized with ImageNet stats. This function converts targets
    to the same range for fair loss computation.
    
    Args:
        normalized_tensor: Tensor normalized with ImageNet stats
        
    Returns:
        Tensor in [-1, 1] range
    """
    # ImageNet normalization parameters
    mean = torch.tensor([0.485, 0.456, 0.406], device=normalized_tensor.device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=normalized_tensor.device).view(1, 3, 1, 1)
    
    # Denormalize to [0, 1]
    denorm = normalized_tensor * std + mean
    denorm = torch.clamp(denorm, 0, 1)
    
    # Convert to [-1, 1]
    return denorm * 2 - 1


@torch.no_grad()
def validate(
    model: nn.Module,
    val_loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
    config: Config
) -> Dict[str, float]:
    """
    Validate the model on the validation set.
    
    Args:
        model: The neural network model
        val_loader: Validation data loader
        criterion: Loss function
        device: Device to run on
        config: Configuration object
        
    Returns:
        Dictionary of average validation losses
    """
    model.eval()
    
    total_loss = 0.0
    loss_components = {}
    num_batches = 0
    
    pbar = tqdm(val_loader, desc="Validation", leave=False)
    
    for batch in pbar:
        ir_image = batch['ir_image'].to(device)
        ref_image = batch['ref_image'].to(device)
        target_image = batch['target_image'].to(device)
        
        # Forward pass
        outputs = model(ir_image, ref_image)
        pred_image = outputs['output']
        
        target_for_loss = denormalize_for_loss(target_image)
        loss, losses = criterion(pred_image, target_for_loss)
        
        total_loss += loss.item()
        for name, value in losses.items():
            if name not in loss_components:
                loss_components[name] = 0.0
            loss_components[name] += value.item()
        num_batches += 1
        
        pbar.set_postfix({'loss': f"{loss.item():.4f}"})
    
    avg_losses = {'total': total_loss / num_batches}
    for name, value in loss_components.items():
        avg_losses[name] = value / num_batches
    
    return avg_losses


@torch.no_grad()
def visualize_samples(
    model: nn.Module,
    val_loader: torch.utils.data.DataLoader,
    device: torch.device,
    output_path: str,
    num_samples: int = 4
) -> None:
    """
    Generate and save visualization of model predictions.
    
    Args:
        model: The neural network model
        val_loader: Validation data loader
        device: Device to run on
        output_path: Path to save the visualization
        num_samples: Number of samples to visualize
    """
    model.eval()
    
    # Get a batch
    batch = next(iter(val_loader))
    
    ir_image = batch['ir_image'][:num_samples].to(device)
    ref_image = batch['ref_image'][:num_samples].to(device)
    target_image = batch['target_image'][:num_samples].to(device)
    
    # Forward pass
    outputs = model(ir_image, ref_image)
    pred_image = outputs['output']
    
    # Save visualization
    save_batch_visualization(
        ir_image.cpu(),
        ref_image.cpu(),
        pred_image.cpu(),
        target_image.cpu(),
        output_path,
        max_samples=num_samples
    )


def train(config: Config) -> None:
    """
    Main training function.
    
    Orchestrates the complete training process including:
    - Data loading
    - Model creation
    - Training loop
    - Validation
    - Checkpointing
    - Visualization
    
    Args:
        config: Configuration object
    """
    print("=" * 60)
    print("IR-to-Color Image Translation Training")
    print("=" * 60)
    
    # Set random seed for reproducibility
    set_seed(config.training.seed)
    print(f"\nRandom seed: {config.training.seed}")
    
    # Setup device
    device = torch.device(config.training.device if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    if device.type == 'cuda':
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Create data loaders (this may download the dataset)
    print("\nPreparing datasets...")
    train_loader, val_loader = get_dataloaders(config)
    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Validation samples: {len(val_loader.dataset)}")
    print(f"Batches per epoch: {len(train_loader)}")
    
    # Create model
    print("\nCreating model...")
    model = create_model(config.model)
    model = model.to(device)
    
    param_counts = count_parameters(model)
    print(f"  Total parameters: {param_counts['total']:,}")
    print(f"  Trainable parameters: {param_counts['trainable']:,}")
    
    # Create loss function
    print("\nSetting up loss functions...")
    criterion = CombinedLoss(config.loss).to(device)
    print(f"  L1 weight: {config.loss.l1_weight}")
    print(f"  Perceptual weight: {config.loss.perceptual_weight}")
    print(f"  Style weight: {config.loss.style_weight}")
    print(f"  Histogram weight: {config.loss.histogram_weight}")
    
    # Create optimizer and scheduler
    print("\nConfiguring optimizer...")
    optimizer = create_optimizer(model, config)
    scheduler = create_scheduler(optimizer, config)
    print(f"  Optimizer: AdamW")
    print(f"  Learning rate: {config.training.learning_rate}")
    print(f"  Scheduler: {config.training.lr_scheduler}")
    
    # Setup mixed precision
    scaler = GradScaler() if config.training.use_amp and device.type == 'cuda' else None
    if scaler:
        print("  Mixed precision: Enabled")
    
    # Setup logging
    logger = MetricsLogger(os.path.join(config.training.output_dir, "logs"))
    
    # Resume from checkpoint if specified
    start_epoch = 0
    best_val_loss = float('inf')
    
    # Auto-detect latest checkpoint if no explicit resume path
    if config.training.resume_checkpoint is None:
        checkpoint_dir = os.path.join(config.training.output_dir, "checkpoints")
        if os.path.exists(checkpoint_dir):
            checkpoints = [f for f in os.listdir(checkpoint_dir) if f.startswith("epoch_") and f.endswith(".pt")]
            if checkpoints:
                # Sort and get latest
                checkpoints.sort()
                latest = os.path.join(checkpoint_dir, checkpoints[-1])
                print(f"Found existing checkpoint: {latest}")
                config.training.resume_checkpoint = latest
    
    if config.training.resume_checkpoint:
        metadata = load_checkpoint(
            config.training.resume_checkpoint,
            model,
            optimizer,
            scheduler,
            device
        )
        start_epoch = metadata['epoch'] + 1
        best_val_loss = metadata.get('loss', float('inf'))
    
    # Training loop
    print("\n" + "=" * 60)
    print("Starting training...")
    print("=" * 60)
    
    training_start_time = time.time()
    
    for epoch in range(start_epoch, config.training.num_epochs):
        epoch_start_time = time.time()
        
        print(f"\nEpoch {epoch + 1}/{config.training.num_epochs}")
        print("-" * 40)
        
        # Train for one epoch
        train_losses = train_epoch(
            model, train_loader, criterion, optimizer, 
            device, config, scaler, logger, epoch
        )
        
        # Validate
        val_losses = validate(model, val_loader, criterion, device, config)
        
        # Update learning rate scheduler
        if scheduler is not None:
            if config.training.lr_scheduler == 'plateau':
                scheduler.step(val_losses['total'])
            else:
                scheduler.step()
        
        # Log epoch metrics
        epoch_metrics = logger.end_epoch(epoch)
        
        # Print epoch summary
        epoch_time = time.time() - epoch_start_time
        print(f"\nEpoch {epoch + 1} Summary:")
        print(f"  Train Loss: {train_losses['total']:.4f}")
        print(f"  Val Loss: {val_losses['total']:.4f}")
        print(f"  Learning Rate: {optimizer.param_groups[0]['lr']:.2e}")
        print(f"  Time: {format_time(epoch_time)}")
        
        # Save best model
        if val_losses['total'] < best_val_loss:
            best_val_loss = val_losses['total']
            checkpoint_path = os.path.join(
                config.training.output_dir, "checkpoints", "best_model.pt"
            )
            save_checkpoint(
                model, optimizer, scheduler, epoch,
                val_losses['total'], config, checkpoint_path
            )
            print(f"  New best model saved! (Val Loss: {best_val_loss:.4f})")
        
        # Save periodic checkpoint
        if (epoch + 1) % config.training.save_every == 0:
            checkpoint_path = os.path.join(
                config.training.output_dir, "checkpoints", f"epoch_{epoch + 1:03d}.pt"
            )
            save_checkpoint(
                model, optimizer, scheduler, epoch,
                val_losses['total'], config, checkpoint_path
            )
        
        # Visualize samples
        if (epoch + 1) % config.training.visualize_every == 0:
            viz_path = os.path.join(
                config.training.output_dir, "visualizations", f"epoch_{epoch + 1:03d}.png"
            )
            visualize_samples(
                model, val_loader, device, viz_path,
                num_samples=config.training.num_visualize_samples
            )
            print(f"  Saved visualization to {viz_path}")
        
        # Save metrics and plot
        logger.save()
        logger.plot_losses(
            os.path.join(config.training.output_dir, "logs", "loss_curves.png")
        )
    
    # Training complete
    total_time = time.time() - training_start_time
    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)
    print(f"Total training time: {format_time(total_time)}")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Outputs saved to: {config.training.output_dir}")


def main():
    """Main entry point with command-line argument parsing."""
    parser = argparse.ArgumentParser(
        description="Train IR-to-Color Image Translation Network"
    )
    parser.add_argument(
        '--config', type=str, default=None,
        help="Path to configuration file (not yet implemented)"
    )
    parser.add_argument(
        '--resume', type=str, default=None,
        help="Path to checkpoint to resume from"
    )
    parser.add_argument(
        '--output', type=str, default=None,
        help="Output directory override"
    )
    parser.add_argument(
        '--epochs', type=int, default=None,
        help="Number of epochs override"
    )
    parser.add_argument(
        '--batch-size', type=int, default=None,
        help="Batch size override"
    )
    parser.add_argument(
        '--lr', type=float, default=None,
        help="Learning rate override"
    )
    parser.add_argument(
        '--dataset', type=str, default=None,
        choices=['cityscapes', 'coco'],
        help="Dataset to use: 'cityscapes' (reqs manual download) or 'coco' (default, auto-downloads)"
    )
    
    args = parser.parse_args()
    
    # Get configuration
    config = get_config()
    
    # Apply command-line overrides
    if args.resume:
        config.training.resume_checkpoint = args.resume
    if args.output:
        config.training.output_dir = args.output
    if args.epochs:
        config.training.num_epochs = args.epochs
    if args.batch_size:
        config.training.batch_size = args.batch_size
    if args.lr:
        config.training.learning_rate = args.lr
    if args.dataset:
        config.data.dataset_name = args.dataset
    
    # Run training
    train(config)


if __name__ == "__main__":
    main()
