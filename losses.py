"""
Loss Functions for IR-to-Color Image Translation

This module implements various loss functions for training the colorization network:

1. L1 Loss (Pixel-wise): Basic reconstruction loss that encourages the output
   to match the ground truth pixel values. Important for overall color accuracy
   but can lead to blurry results if used alone.

2. Perceptual Loss: Compares features extracted by a pretrained VGG network
   rather than raw pixels. This encourages perceptually similar results even
   if individual pixels differ, leading to sharper, more realistic outputs.

3. Style Loss: Compares Gram matrices of VGG features, which capture texture
   and color statistics. Helps transfer the color palette and texture patterns
   from the reference to the output.

4. Color Histogram Loss: Matches the color distribution of the output to the
   ground truth. Useful for ensuring globally correct color balance.

The total loss is a weighted combination of these components, with weights
specified in the configuration.
"""

import torch # type: ignore
import torch.nn as nn # type: ignore
import torch.nn.functional as F # type: ignore
from torchvision import models # type: ignore
from typing import Dict, List, Optional, Tuple
from collections import OrderedDict

from config import LossConfig


class VGGFeatures(nn.Module):
    """
    VGG-19 feature extractor for perceptual and style losses.
    
    This module extracts features from specified layers of a pretrained VGG-19
    network. The features are used to compute perceptual similarity (comparing
    feature activations) and style similarity (comparing Gram matrices).
    
    VGG is commonly used because:
    - It was trained on ImageNet with diverse natural images
    - Its features capture a hierarchy from edges to textures to objects
    - It's been empirically shown to correlate well with human perception
    """
    
    # Mapping from layer names to VGG-19 layer indices
    # These are the output layers of each conv block before pooling
    LAYER_NAME_MAPPING = {
        'relu1_1': '1',   'relu1_2': '3',   # Block 1: 64 channels
        'relu2_1': '6',   'relu2_2': '8',   # Block 2: 128 channels
        'relu3_1': '11',  'relu3_2': '13',  'relu3_3': '15',  'relu3_4': '17',  # Block 3: 256 channels
        'relu4_1': '20',  'relu4_2': '22',  'relu4_3': '24',  'relu4_4': '26',  # Block 4: 512 channels
        'relu5_1': '29',  'relu5_2': '31',  'relu5_3': '33',  'relu5_4': '35',  # Block 5: 512 channels
    }
    
    def __init__(self, layers: List[str], requires_grad: bool = False):
        """
        Initialize VGG feature extractor.
        
        Args:
            layers: List of layer names to extract features from
                   e.g., ['relu1_2', 'relu2_2', 'relu3_4', 'relu4_4', 'relu5_4']
            requires_grad: Whether to compute gradients for VGG weights
                          (should be False - we don't want to update VGG)
        """
        super().__init__()
        
        # Load pretrained VGG-19
        vgg = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1)
        
        # Get the layer indices we need
        self.layer_indices = []
        self.layer_names = []
        for layer_name in layers:
            if layer_name in self.LAYER_NAME_MAPPING:
                self.layer_indices.append(int(self.LAYER_NAME_MAPPING[layer_name]))
                self.layer_names.append(layer_name)
            else:
                raise ValueError(f"Unknown layer name: {layer_name}")
        
        # Only keep layers up to the deepest one we need
        max_idx = max(self.layer_indices)
        self.features = nn.Sequential(*list(vgg.features.children())[:max_idx + 1])
        
        # Freeze VGG weights
        if not requires_grad:
            for param in self.features.parameters():
                param.requires_grad = False
        
        # VGG normalization (ImageNet stats)
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        
    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        """
        Normalize input for VGG.
        
        Note: Our model output is in [-1, 1] range (tanh activation),
        so we first convert to [0, 1] then apply ImageNet normalization.
        
        Args:
            x: Input tensor in [-1, 1] range
            
        Returns:
            Normalized tensor
        """
        # Clamp to expected range first
        x = torch.clamp(x, -1.0, 1.0)
        # Convert from [-1, 1] to [0, 1]
        x = (x + 1) / 2
        # Clamp again to be safe
        x = torch.clamp(x, 0.0, 1.0)
        # Apply ImageNet normalization
        return (x - self.mean) / self.std
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Extract features from specified VGG layers.
        
        Args:
            x: Input tensor [B, 3, H, W] in [-1, 1] range
            
        Returns:
            Dictionary mapping layer names to feature tensors
        """
        # Normalize for VGG
        x = self.normalize(x)
        
        # Extract features at each layer
        features = {}
        for i, layer in enumerate(self.features):
            x = layer(x)
            if i in self.layer_indices:
                layer_name = self.layer_names[self.layer_indices.index(i)]
                features[layer_name] = x
                
        return features


def gram_matrix(features: torch.Tensor) -> torch.Tensor:
    """
    Compute the Gram matrix of feature maps.
    
    The Gram matrix captures correlations between feature channels,
    encoding texture and style information independent of spatial arrangement.
    
    For features F of shape [B, C, H, W], the Gram matrix G has shape [B, C, C],
    where G[b, i, j] = sum over spatial positions of F[b, i, :, :] * F[b, j, :, :]
    
    Args:
        features: Feature tensor [B, C, H, W]
        
    Returns:
        Gram matrix [B, C, C]
    """
    B, C, H, W = features.shape
    
    # Reshape to [B, C, H*W]
    features = features.view(B, C, H * W)
    
    # Normalize features to prevent explosion in Gram matrix
    # This is crucial for numerical stability
    features = features / (H * W) ** 0.5
    
    # Compute Gram matrix: [B, C, H*W] @ [B, H*W, C] -> [B, C, C]
    gram = torch.bmm(features, features.transpose(1, 2))
    
    # Normalize by number of channels
    gram = gram / C
    
    return gram


class PerceptualLoss(nn.Module):
    """
    Perceptual loss using VGG features.
    
    Computes L1 distance between VGG features of predicted and target images.
    This encourages perceptually similar results by comparing high-level
    representations rather than raw pixels.
    
    Different VGG layers capture different aspects:
    - Early layers (relu1_x, relu2_x): Low-level features like edges and colors
    - Middle layers (relu3_x): Textures and patterns
    - Late layers (relu4_x, relu5_x): High-level semantics and objects
    """
    
    def __init__(self, layers: List[str]):
        """
        Initialize perceptual loss.
        
        Args:
            layers: VGG layers to use for comparison
        """
        super().__init__()
        self.vgg = VGGFeatures(layers)
        
    def forward(
        self, 
        pred: torch.Tensor, 
        target: torch.Tensor,
        pred_features: Optional[Dict[str, torch.Tensor]] = None,
        target_features: Optional[Dict[str, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute perceptual loss.
        
        Args:
            pred: Predicted image [B, 3, H, W] in [-1, 1] range
            target: Target image [B, 3, H, W] in [-1, 1] range
            pred_features: Optional precomputed features for pred
            target_features: Optional precomputed features for target
            
        Returns:
            Tuple of:
            - Total perceptual loss (scalar)
            - Dictionary of per-layer losses
        """
        # Extract features if not provided
        if pred_features is None:
            pred_features = self.vgg(pred)
        if target_features is None:
            target_features = self.vgg(target)
        
        # Compute loss for each layer
        layer_losses = {}
        total_loss = 0.0
        
        for layer_name in pred_features:
            layer_loss = F.l1_loss(
                pred_features[layer_name], 
                target_features[layer_name]
            )
            layer_losses[f"perceptual_{layer_name}"] = layer_loss
            total_loss = total_loss + layer_loss
        
        # Average over layers
        total_loss = total_loss / len(pred_features)
        
        return total_loss, layer_losses


class StyleLoss(nn.Module):
    """
    Style loss using Gram matrices of VGG features.
    
    Computes L1 distance between Gram matrices of predicted and target images.
    The Gram matrix captures texture and color statistics, so matching it
    encourages similar style/appearance independent of exact spatial arrangement.
    
    This is particularly useful for our task because the reference image
    has different viewpoint/content, but we want to transfer its color style.
    """
    
    def __init__(self, layers: List[str]):
        """
        Initialize style loss.
        
        Args:
            layers: VGG layers to use for Gram matrix comparison
        """
        super().__init__()
        self.vgg = VGGFeatures(layers)
        
    def forward(
        self, 
        pred: torch.Tensor, 
        target: torch.Tensor,
        pred_features: Optional[Dict[str, torch.Tensor]] = None,
        target_features: Optional[Dict[str, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute style loss.
        
        Args:
            pred: Predicted image [B, 3, H, W] in [-1, 1] range
            target: Target image [B, 3, H, W] in [-1, 1] range
            pred_features: Optional precomputed features for pred
            target_features: Optional precomputed features for target
            
        Returns:
            Tuple of:
            - Total style loss (scalar)
            - Dictionary of per-layer losses
        """
        # Extract features if not provided
        if pred_features is None:
            pred_features = self.vgg(pred)
        if target_features is None:
            target_features = self.vgg(target)
        
        # Compute Gram matrix loss for each layer
        layer_losses = {}
        total_loss = 0.0
        
        for layer_name in pred_features:
            pred_gram = gram_matrix(pred_features[layer_name])
            target_gram = gram_matrix(target_features[layer_name])
            
            layer_loss = F.l1_loss(pred_gram, target_gram)
            layer_losses[f"style_{layer_name}"] = layer_loss
            total_loss = total_loss + layer_loss
        
        # Average over layers
        total_loss = total_loss / len(pred_features)
        
        return total_loss, layer_losses


class ColorHistogramLoss(nn.Module):
    """
    Color histogram matching loss.
    
    Encourages the output image to have a similar color distribution to the
    target. This is computed by:
    1. Computing soft histograms for each color channel
    2. Comparing histograms using L1 distance
    
    Soft histograms use differentiable binning, allowing gradient flow.
    """
    
    def __init__(self, num_bins: int = 64, sigma: float = 0.1):
        """
        Initialize color histogram loss.
        
        Args:
            num_bins: Number of histogram bins per channel
            sigma: Softness of binning (larger = softer edges between bins)
                   Increased from 0.02 to 0.1 for numerical stability
        """
        super().__init__()
        self.num_bins = num_bins
        self.sigma = sigma
        
        # Create bin centers
        # For [-1, 1] range, bins span this interval
        bins = torch.linspace(-1, 1, num_bins)
        self.register_buffer('bins', bins)
        
    def soft_histogram(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute soft (differentiable) histogram.
        
        Uses Gaussian kernel to assign soft bin memberships.
        
        Args:
            x: Input tensor [B, C, H, W]
            
        Returns:
            Histogram tensor [B, C, num_bins]
        """
        B, C, H, W = x.shape
        
        # Clamp input to valid range to prevent extreme values
        x = torch.clamp(x, -1.0, 1.0)
        
        # Flatten spatial dimensions: [B, C, N] where N = H*W
        x_flat = x.view(B, C, -1)
        
        # Expand for broadcasting: [B, C, N, 1]
        x_flat = x_flat.unsqueeze(-1)
        
        # Bins: [1, 1, 1, num_bins]
        bins = self.bins.view(1, 1, 1, -1)
        
        # Soft binning using Gaussian kernel
        # Compute squared distance first to avoid extreme values
        diff = (x_flat - bins) / self.sigma
        # Clamp to prevent overflow in exp
        diff_sq = torch.clamp(diff ** 2, max=50.0)
        weights = torch.exp(-0.5 * diff_sq)
        
        # Sum over spatial dimension to get histogram: [B, C, num_bins]
        hist = weights.sum(dim=2)
        
        # Normalize to sum to 1
        hist = hist / (hist.sum(dim=-1, keepdim=True) + 1e-6)
        
        return hist
    
    def forward(
        self, 
        pred: torch.Tensor, 
        target: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute color histogram loss.
        
        Args:
            pred: Predicted image [B, 3, H, W] in [-1, 1] range
            target: Target image [B, 3, H, W] in [-1, 1] range
            
        Returns:
            Histogram matching loss (scalar)
        """
        pred_hist = self.soft_histogram(pred)
        target_hist = self.soft_histogram(target)
        
        # L1 distance between histograms
        loss = F.l1_loss(pred_hist, target_hist)
        
        return loss


class CombinedLoss(nn.Module):
    """
    Combined loss function for training.
    
    Aggregates all loss components with configurable weights:
    - L1 (pixel-wise) loss
    - Perceptual loss
    - Style loss
    - Color histogram loss
    
    The weights balance different objectives:
    - Higher L1 weight: More pixel-accurate but potentially blurry
    - Higher perceptual weight: Sharper but may have artifacts
    - Higher style weight: Better color transfer from reference
    - Higher histogram weight: More globally accurate colors
    
    Optimization: If both perceptual_weight and style_weight are zero,
    VGG feature extraction is skipped entirely for efficiency.
    """
    
    def __init__(self, config: LossConfig):
        """
        Initialize combined loss.
        
        Args:
            config: Loss configuration with weights and layer specifications
        """
        super().__init__()
        
        self.config = config
        
        # Check if we need VGG-based losses
        self.use_vgg = (config.perceptual_weight > 0 or config.style_weight > 0)
        
        # Individual loss components - only create VGG-based ones if needed
        if self.use_vgg:
            print("  VGG-based losses enabled")
            # Combine all required layers for efficient single-pass feature extraction
            all_layers = list(set(config.vgg_layers + config.style_layers))
            self.vgg_features = VGGFeatures(all_layers)
            
            if config.perceptual_weight > 0:
                self.perceptual_loss = PerceptualLoss(config.vgg_layers)
            
            if config.style_weight > 0:
                self.style_loss = StyleLoss(config.style_layers)
        else:
            print("  VGG-based losses disabled")
        
        if config.histogram_weight > 0:
            self.histogram_loss = ColorHistogramLoss()
        
    def forward(
        self, 
        pred: torch.Tensor, 
        target: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute total loss.
        
        Args:
            pred: Predicted image [B, 3, H, W] in [-1, 1] range
            target: Target image [B, 3, H, W] in [-1, 1] range
            
        Returns:
            Tuple of:
            - Total weighted loss (scalar)
            - Dictionary of individual loss components
        """
        losses = {}
        
        # Clamp inputs to expected range for stability
        pred = torch.clamp(pred, -1.0, 1.0)
        target = torch.clamp(target, -1.0, 1.0)
        
        # L1 (pixel-wise) loss
        l1_loss = F.l1_loss(pred, target)
        losses['l1'] = l1_loss
        
        total_loss = self.config.l1_weight * l1_loss
        
        # VGG-based losses (perceptual and style)
        # Extract features once and reuse if both losses are needed
        if self.use_vgg:
            pred_features = self.vgg_features(pred)
            target_features = self.vgg_features(target)
            
            # Perceptual loss
            if self.config.perceptual_weight > 0:
                # Filter features to only perceptual layers
                pred_perc = {k: v for k, v in pred_features.items() if k in self.config.vgg_layers}
                target_perc = {k: v for k, v in target_features.items() if k in self.config.vgg_layers}
                perceptual_loss, perceptual_details = self.perceptual_loss(
                    pred, target, pred_perc, target_perc
                )
                losses['perceptual'] = perceptual_loss
                losses.update(perceptual_details)
                total_loss = total_loss + self.config.perceptual_weight * perceptual_loss
            else:
                losses['perceptual'] = torch.tensor(0.0, device=pred.device)
            
            # Style loss
            if self.config.style_weight > 0:
                # Filter features to only style layers
                pred_style = {k: v for k, v in pred_features.items() if k in self.config.style_layers}
                target_style = {k: v for k, v in target_features.items() if k in self.config.style_layers}
                style_loss, style_details = self.style_loss(
                    pred, target, pred_style, target_style
                )
                losses['style'] = style_loss
                losses.update(style_details)
                total_loss = total_loss + self.config.style_weight * style_loss
            else:
                losses['style'] = torch.tensor(0.0, device=pred.device)
        else:
            # No VGG computation needed
            losses['perceptual'] = torch.tensor(0.0, device=pred.device)
            losses['style'] = torch.tensor(0.0, device=pred.device)
        
        # Color histogram loss
        if self.config.histogram_weight > 0:
            histogram_loss = self.histogram_loss(pred, target)
            losses['histogram'] = histogram_loss
            total_loss = total_loss + self.config.histogram_weight * histogram_loss
        else:
            losses['histogram'] = torch.tensor(0.0, device=pred.device)
        
        # Check for NaN and replace with L1 loss only if needed
        if torch.isnan(total_loss):
            print("WARNING: NaN detected in loss, falling back to L1 only")
            total_loss = l1_loss
            # Mark which losses were NaN for debugging
            for name, value in losses.items():
                if torch.isnan(value):
                    print(f"  NaN in: {name}")
        
        losses['total'] = total_loss
        
        return total_loss, losses


# Test losses when run directly
if __name__ == "__main__":
    from config import get_config
    
    config = get_config()
    
    # Create loss function
    criterion = CombinedLoss(config.loss)
    
    # Test with random tensors
    print("Testing loss functions...")
    print(f"VGG computation enabled: {criterion.use_vgg}")
    
    pred = torch.randn(2, 3, 256, 256).clamp(-1, 1)
    target = torch.randn(2, 3, 256, 256).clamp(-1, 1)
    
    # Move to GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    criterion = criterion.to(device)
    pred = pred.to(device)
    target = target.to(device)
    
    # Compute loss
    total_loss, loss_dict = criterion(pred, target)
    
    print(f"\nLoss breakdown:")
    for name, value in loss_dict.items():
        print(f"  {name}: {value.item():.6f}")
