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

The script uses configuration from config.py.
"""

import os
import time
from pathlib import Path
from typing import Dict, Optional, Tuple, Union, List, Any
import argparse

import torch # type: ignore
import torch.nn as nn # type: ignore
import torch.optim as optim # type: ignore
from torch.amp import GradScaler, autocast # type: ignore
from tqdm import tqdm # type: ignore

import random
import numpy as np # type: ignore

from config import Config, Curriculum, get_config, get_curriculum
from config import load_experiment_config
from dataset import get_dataloaders, denormalize
from model import create_model
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
    config: Config,
    total_epochs: Optional[int] = None
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
    if total_epochs is None:
        total_epochs = config.training.num_epochs
    
    if scheduler_type == 'cosine':
        # Cosine annealing with warm restarts
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=total_epochs,
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
    
    # Progress bar iterator
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
            with autocast('cuda'):
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
    num_samples: int = 5,
    epoch: int = 0
) -> None:
    """
    Generate and save visualization of model predictions with augmentations.
    
    Uses training mode (is_training=True) with a fixed random seed to show how
    the model handles augmented inputs consistently across epochs. This allows
    visualizing the model's robustness to color jitter, flips, rotations, etc.

    Args:
        model: The neural network model
        val_loader: Validation data loader
        device: Device to run on
        output_path: Path to save the visualization
        num_samples: Number of samples to visualize
        epoch: Current epoch (for naming only, not used for seeding)
    """
    model.eval()

    # Get original dataset info
    original_dataset = val_loader.dataset
    max_index = len(original_dataset)
    num_samples = min(num_samples, max_index)

    # Import dataset class
    from dataset import IRColorPairDataset
    
    # Create a dataset with training augmentations but fixed seed
    # This shows the model handling augmented data consistently across epochs
    vis_dataset = IRColorPairDataset(
        image_source=original_dataset.image_paths[:num_samples],
        config=original_dataset.config,
        is_training=True,  # Enable augmentations
        fixed_crop_seed=42  # Fixed seed ensures same augmentations every epoch
    )

    ir_list = []
    ref_list = []
    target_list = []

    for i in range(num_samples):
        sample = vis_dataset[i]
        ir_list.append(sample['ir_image'])
        ref_list.append(sample['ref_image'])
        target_list.append(sample['target_image'])

    # Stack into batched tensors and move to device
    ir_batch = torch.stack(ir_list, dim=0).to(device)
    ref_batch = torch.stack(ref_list, dim=0).to(device)
    target_batch = torch.stack(target_list, dim=0).to(device)

    # Forward pass
    outputs = model(ir_batch, ref_batch)
    pred_batch = outputs['output']

    # Save visualization (reuse existing utility)
    save_batch_visualization(
        ir_batch.cpu(),
        ref_batch.cpu(),
        pred_batch.cpu(),
        target_batch.cpu(),
        output_path,
        max_samples=num_samples
    )


def train(
    config: Config,
    start_epoch: int = 0,
    end_epoch: Optional[int] = None,
    stage_idx: Optional[int] = None
) -> int:
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
        start_epoch: Global epoch to start from (default: 0)
        end_epoch: Global epoch to end at (default: config.training.num_epochs)
        
    Returns:
        Last completed global epoch number
    """

    if end_epoch is None:
        end_epoch = start_epoch + config.training.num_epochs
    
    # Set random seed for reproducibility
    set_seed(config.training.seed)
    print(f"\nRandom seed: {config.training.seed}")
    
    # Setup device
    device = torch.device(config.training.device if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    if device.type == 'cuda':
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Persist the active configuration so each run records its settings
    config.ensure_output_dirs()
    config_path = os.path.join(config.training.output_dir, "config.yaml")
    config.save_yaml(config_path)
    print(f"Saved configuration to {config_path}")
    
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
    total_epochs = end_epoch - start_epoch
    scheduler = create_scheduler(optimizer, config, total_epochs=total_epochs)
    print(f"  Optimizer: AdamW")
    print(f"  Learning rate: {config.training.learning_rate}")
    print(f"  Scheduler: {config.training.lr_scheduler}")
    
    # Setup mixed precision
    scaler = GradScaler('cuda') if config.training.use_amp and device.type == 'cuda' else None
    if scaler:
        print("  Mixed precision: Enabled")
    
    # Setup logging
    logger = MetricsLogger(os.path.join(config.training.output_dir, "logs"))
    
    # Resume from checkpoint if specified or auto-detect
    best_val_loss = float('inf')
    
    # Auto-detect latest checkpoint if no explicit resume path
    if config.training.resume_checkpoint is None and start_epoch > 0:
        checkpoint_dir = os.path.join(config.training.output_dir, "checkpoints")
        latest_path = os.path.join(checkpoint_dir, "latest.pt")
        if os.path.exists(latest_path):
            print(f"Found latest checkpoint: {latest_path}")
            config.training.resume_checkpoint = latest_path
    
    if config.training.resume_checkpoint:
        metadata = load_checkpoint(
            config.training.resume_checkpoint,
            model,
            optimizer,
            scheduler,
            device
        )
        # Verify we're resuming from the expected epoch
        checkpoint_epoch = metadata.get('epoch', -1)
        if checkpoint_epoch >= 0 and checkpoint_epoch < start_epoch - 1:
            print(f"Warning: Checkpoint is from epoch {checkpoint_epoch}, "
                  f"but starting from epoch {start_epoch}")
        best_val_loss = metadata.get('loss', float('inf'))
    
    # Training loop
    print("\n" + "=" * 60)
    print(f"Starting training from epoch {start_epoch + 1} to {end_epoch}...")
    print("=" * 60)
    
    training_start_time = time.time()
    last_completed_epoch = start_epoch - 1
    
    for epoch in range(start_epoch, end_epoch):
        epoch_start_time = time.time()
        stage_epoch = epoch - start_epoch
        
        print(f"\nEpoch {epoch + 1}/{end_epoch}")
        if stage_idx is not None:
            print(f"  (Stage {stage_idx + 1}, Stage Epoch {stage_epoch + 1}/{total_epochs})")
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
        
        checkpoint_extra = {
            'global_epoch': epoch,
        }
        if stage_idx is not None:
            checkpoint_extra['stage_idx'] = stage_idx
            checkpoint_extra['stage_epoch'] = stage_epoch
        
        # Always save latest checkpoint
        latest_path = os.path.join(
            config.training.output_dir, "checkpoints", "latest.pt"
        )
        save_checkpoint(
            model, optimizer, scheduler, epoch,
            val_losses['total'], config, latest_path,
            extra_metadata=checkpoint_extra
        )
        
        # Save best model
        if val_losses['total'] < best_val_loss:
            best_val_loss = val_losses['total']
            checkpoint_path = os.path.join(
                config.training.output_dir, "checkpoints", "best_model.pt"
            )
            save_checkpoint(
                model, optimizer, scheduler, epoch,
                val_losses['total'], config, checkpoint_path,
                extra_metadata=checkpoint_extra
            )
            print(f"  New best model saved! (Val Loss: {best_val_loss:.4f})")
        
        # Save periodic checkpoint
        if (epoch + 1) % config.training.save_every == 0:
            checkpoint_path = os.path.join(
                config.training.output_dir, "checkpoints", f"epoch_{epoch + 1:03d}.pt"
            )
            save_checkpoint(
                model, optimizer, scheduler, epoch,
                val_losses['total'], config, checkpoint_path,
                extra_metadata=checkpoint_extra
            )
        
        # Visualize samples
        if (epoch + 1) % config.training.visualize_every == 0:
            viz_path = os.path.join(
                config.training.output_dir, "visualizations", f"epoch_{epoch + 1:03d}.png"
            )
            visualize_samples(
                model, val_loader, device, viz_path,
                num_samples=config.training.num_visualize_samples,
                epoch=epoch
            )
            print(f"  Saved visualization to {viz_path}")
        
        # Save metrics and plot
        logger.save()
        logger.plot_losses(
            os.path.join(config.training.output_dir, "logs", "loss_curves.png")
        )
        
        last_completed_epoch = epoch
    
    # Training complete
    total_time = time.time() - training_start_time
    print("\n" + "=" * 60)
    print("Stage Complete!" if stage_idx is not None else "Training Complete!")
    print("=" * 60)
    print(f"Total training time: {format_time(total_time)}")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Outputs saved to: {config.training.output_dir}")
    
    return last_completed_epoch


def _infer_curriculum_position(curriculum: Curriculum, global_epoch: int) -> Tuple[int, int, int]:
    """Infer (stage_idx, stage_epoch, global_epoch) given a global epoch index."""
    if global_epoch < 0:
        return 0, 0, 0

    remaining = global_epoch
    for stage_idx, stage in enumerate(curriculum.stages):
        stage_len = int(stage.training.num_epochs)
        if remaining < stage_len:
            return stage_idx, remaining, global_epoch
        remaining -= stage_len

    # If we're beyond the end (e.g., resume after completion), clamp to last stage end.
    last_idx = len(curriculum.stages) - 1
    return last_idx, int(curriculum.stages[last_idx].training.num_epochs), global_epoch


def _apply_overrides_to_curriculum(curriculum: Curriculum, args: argparse.Namespace) -> None:
    """Apply safe CLI overrides to all stages of a curriculum (in-place)."""
    if args.output:
        # TrainingConfig may be shared across stages; mutate once is enough.
        curriculum.stages[0].training.output_dir = args.output
        for stage in curriculum.stages[1:]:
            stage.training.output_dir = args.output

    if args.resume:
        curriculum.stages[0].training.resume_checkpoint = args.resume

    if args.batch_size:
        curriculum.stages[0].training.batch_size = args.batch_size
    if args.lr:
        curriculum.stages[0].training.learning_rate = args.lr


def train_with_curriculum(curriculum: Curriculum) -> None:
    """Train using a Curriculum (multi-stage) with simplified stage iteration."""
    print("=" * 60)
    print("IR-to-Color Image Translation Training (Curriculum)")
    print("=" * 60)
    
    # Use stage 0 for global training knobs
    base_config = curriculum.stages[0]
    total_epochs = curriculum.total_epochs
    
    set_seed(base_config.training.seed)
    print(f"\nRandom seed: {base_config.training.seed}")
    
    device = torch.device(base_config.training.device if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    if device.type == 'cuda':
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    curriculum.ensure_output_dirs()
    config_path = os.path.join(curriculum.output_dir, "config.yaml")
    curriculum.save_yaml(config_path)
    print(f"Saved curriculum to {config_path}")
    
    # Determine starting point (resume if checkpoint exists)
    start_global_epoch = 0
    if base_config.training.resume_checkpoint is None:
        checkpoint_dir = os.path.join(curriculum.output_dir, "checkpoints")
        latest_path = os.path.join(checkpoint_dir, "latest.pt")
        if os.path.exists(latest_path):
            print(f"Found existing checkpoint: {latest_path}")
            base_config.training.resume_checkpoint = latest_path
    
    if base_config.training.resume_checkpoint:
        # Load to get the epoch number
        checkpoint = torch.load(
            base_config.training.resume_checkpoint,
            map_location='cpu',
            weights_only=False
        )
        start_global_epoch = checkpoint.get('epoch', 0) + 1
        print(f"Resuming from global epoch {start_global_epoch}")
    
    print(f"\nTotal curriculum epochs: {total_epochs}")
    print(f"Number of stages: {len(curriculum.stages)}")
    
    # Training loop across stages
    print("\n" + "=" * 60)
    print("Starting curriculum training...")
    print("=" * 60)
    
    training_start_time = time.time()
    current_global_epoch = start_global_epoch
    
    while current_global_epoch < total_epochs:
        # Get the appropriate config for current global epoch
        stage_config, stage_idx, stage_epoch = curriculum.get_config_for_epoch(current_global_epoch)
        stage_epochs = int(stage_config.training.num_epochs)
        
        # Calculate how many epochs to run in this stage
        stage_start_global = current_global_epoch
        stage_end_global = min(
            stage_start_global + (stage_epochs - stage_epoch),
            total_epochs
        )
        
        print(f"\n{'='*60}")
        print(f"Stage {stage_idx + 1}/{len(curriculum.stages)}")
        print(f"Global epochs {stage_start_global + 1} to {stage_end_global}")
        print(f"{'='*60}")
        
        # Train this stage
        last_epoch = train(
            config=stage_config,
            start_epoch=stage_start_global,
            end_epoch=stage_end_global,
            stage_idx=stage_idx
        )
        
        current_global_epoch = last_epoch + 1
    
    total_time = time.time() - training_start_time
    print("\n" + "=" * 60)
    print("Curriculum Training Complete!")
    print("=" * 60)
    print(f"Total training time: {format_time(total_time)}")
    print(f"Total epochs completed: {current_global_epoch}")
    print(f"Outputs saved to: {curriculum.output_dir}")


def main():
    """Main entry point with command-line argument parsing."""
    parser = argparse.ArgumentParser(
        description="Train IR-to-Color Image Translation Network"
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
    parser.add_argument(
        '--config', type=str, default=None,
        help="Optional YAML configuration file to load"
    )
    
    args = parser.parse_args()
    
    # Get curriculum, optionally from YAML
    if args.config:
        curriculum = load_experiment_config(args.config)
    else:
        curriculum = get_curriculum()
    #   curriculum = get_config()
    
    # Apply command-line overrides, if any, and begin training based on Curriculum/Config
    if isinstance(curriculum, Curriculum):
        _apply_overrides_to_curriculum(curriculum, args)
        if args.epochs is not None:
            print("Warning: --epochs override is ignored for curriculum runs; edit stage num_epochs in the curriculum YAML instead.")
        if args.dataset is not None:
            print("Warning: --dataset override is ignored for curriculum runs; edit per-stage dataset_name in the curriculum YAML instead.")

        curriculum.ensure_output_dirs()
        train_with_curriculum(curriculum)
    else:
        config = curriculum

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

        config.ensure_output_dirs()
        train(config)


if __name__ == "__main__":
    main()
