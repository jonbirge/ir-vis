"""
Neural Network Architecture for IR-to-Color Image Translation

This module implements a reference-based colorization network with:
1. Content Encoder: Extracts structural features from the IR image
2. Reference Encoder: Extracts color/style features from the visible reference
3. Feature Matching Module: Attention-based alignment between encoders
4. Decoder: Generates colorized output from combined features

The key innovation is the attention-based feature matching, which allows
the network to find correspondences between the IR and reference images
despite their different viewpoints/FOVs. This is similar to the approach
used in exemplar-based colorization, adapted for our cross-modal setting.

Architecture Overview:
    IR Image (grayscale) ─────► Content Encoder ────► 
                                                      ├─► Feature Matching ─► Decoder ─► Colorized Output
    Reference Image (color) ──► Reference Encoder ──►

The encoders share architecture but not weights, as they process
different modalities (grayscale vs color) with different characteristics.
"""

import torch # type: ignore
import torch.nn as nn # type: ignore
import torch.nn.functional as F # type: ignore
from torchvision import models # type: ignore
from typing import List, Tuple, Optional, Dict

from config import ModelConfig


class ResNetEncoder(nn.Module):
    """
    Encoder based on ResNet architecture.
    
    Uses pretrained ResNet features, extracting intermediate layer outputs
    for skip connections (U-Net style). Features can be extracted at different
    layers (layer2, layer3, or layer4) for attention matching.
    """
    
    def __init__(
        self, 
        backbone: str = "resnet34",
        pretrained: bool = True
    ):
        """
        Initialize the ResNet encoder.
        
        Args:
            backbone: Which ResNet variant ('resnet18', 'resnet34', 'resnet50')
            pretrained: Whether to use ImageNet pretrained weights
        """
        super().__init__()
        
        # Load pretrained ResNet
        if backbone == "resnet18":
            weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
            resnet = models.resnet18(weights=weights)
            self.feature_channels = [64, 64, 128, 256, 512]
        elif backbone == "resnet34":
            weights = models.ResNet34_Weights.IMAGENET1K_V1 if pretrained else None
            resnet = models.resnet34(weights=weights)
            self.feature_channels = [64, 64, 128, 256, 512]
        elif backbone == "resnet50":
            weights = models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
            resnet = models.resnet50(weights=weights)
            self.feature_channels = [64, 256, 512, 1024, 2048]
        else:
            raise ValueError(f"Unknown backbone: {backbone}")
        
        # Extract layers for multi-scale features
        # Layer outputs at different spatial resolutions:
        # conv1: H/2, layer1: H/4, layer2: H/8, layer3: H/16, layer4: H/32
        self.conv1 = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
        )
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
        
    def forward(
        self, 
        x: torch.Tensor, 
        return_layer: str = "layer4"
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Forward pass through the encoder.
        
        Args:
            x: Input tensor [B, 3, H, W]
            return_layer: Which layer to return as main features
                         ('layer2', 'layer3', or 'layer4')
            
        Returns:
            Tuple of:
            - Features from specified layer
            - List of intermediate features for skip connections (up to return_layer)
        """
        # Track features at each scale for skip connections
        skip_features = []
        
        # Initial convolution: H -> H/2
        x = self.conv1(x)
        skip_features.append(x)  # [B, 64, H/2, W/2]
        
        # Pooling: H/2 -> H/4
        x = self.maxpool(x)
        
        # ResNet blocks
        x = self.layer1(x)  # H/4
        skip_features.append(x)
        
        x = self.layer2(x)  # H/8
        skip_features.append(x)
        if return_layer == "layer2":
            return x, skip_features
        
        x = self.layer3(x)  # H/16
        skip_features.append(x)
        if return_layer == "layer3":
            return x, skip_features
        
        x = self.layer4(x)  # H/32
        skip_features.append(x)
        return x, skip_features


class CrossAttention(nn.Module):
    """
    Cross-attention module for feature matching between IR and reference.
    
    Given IR features as queries and reference features as keys/values,
    this module computes attention-weighted combinations of reference features.
    This allows the network to find corresponding regions in the reference
    image and transfer their color information.
    
    Multi-head attention is used to capture diverse correspondences
    (e.g., different heads might attend to texture vs. semantic content).
    """
    
    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 8,
        dropout: float = 0.1
    ):
        """
        Initialize cross-attention module.
        
        Args:
            embed_dim: Dimension of input features
            num_heads: Number of attention heads
            dropout: Dropout rate for attention weights
        """
        super().__init__()
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        
        # Linear projections for queries, keys, values
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        
        # Output projection
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Scale factor for attention scores
        self.scale = self.head_dim ** -0.5
        
    def forward(
        self, 
        query: torch.Tensor, 
        key: torch.Tensor, 
        value: torch.Tensor,
        return_attention: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Compute cross-attention.
        
        Args:
            query: Query features from IR encoder [B, N_q, C]
            key: Key features from reference encoder [B, N_k, C]
            value: Value features from reference encoder [B, N_k, C]
            return_attention: Whether to return attention weights
            
        Returns:
            Tuple of:
            - Output features [B, N_q, C]
            - Attention weights [B, num_heads, N_q, N_k] (if return_attention)
        """
        B, N_q, C = query.shape
        N_k = key.shape[1]
        
        # Project to queries, keys, values
        q = self.q_proj(query)  # [B, N_q, C]
        k = self.k_proj(key)    # [B, N_k, C]
        v = self.v_proj(value)  # [B, N_k, C]
        
        # Reshape for multi-head attention
        # [B, N, C] -> [B, N, num_heads, head_dim] -> [B, num_heads, N, head_dim]
        q = q.view(B, N_q, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, N_k, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, N_k, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Compute attention scores
        # [B, num_heads, N_q, head_dim] @ [B, num_heads, head_dim, N_k]
        # -> [B, num_heads, N_q, N_k]
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        # Clamp attention scores to prevent overflow in softmax
        attn_scores = torch.clamp(attn_scores, min=-50.0, max=50.0)
        
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        # [B, num_heads, N_q, N_k] @ [B, num_heads, N_k, head_dim]
        # -> [B, num_heads, N_q, head_dim]
        out = torch.matmul(attn_weights, v)
        
        # Reshape back
        # [B, num_heads, N_q, head_dim] -> [B, N_q, num_heads, head_dim] -> [B, N_q, C]
        out = out.transpose(1, 2).contiguous().view(B, N_q, C)
        
        # Output projection
        out = self.out_proj(out)
        
        if return_attention:
            return out, attn_weights
        return out, None


class FeatureMatchingModule(nn.Module):
    """
    Feature matching module combining cross-attention with residual connections.
    
    This module takes features from both IR and reference encoders and produces
    combined features that incorporate color information from the reference
    aligned to the spatial structure of the IR image.
    
    The module includes:
    1. Cross-attention: IR queries, reference keys/values
    2. Self-attention: Refine combined features
    3. Feed-forward network: Non-linear transformation
    4. Layer normalization and residual connections throughout
    """
    
    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 8,
        dropout: float = 0.1,
        ff_dim: Optional[int] = None
    ):
        """
        Initialize the feature matching module.
        
        Args:
            embed_dim: Feature dimension
            num_heads: Number of attention heads
            dropout: Dropout rate
            ff_dim: Feed-forward hidden dimension (default: 4 * embed_dim)
        """
        super().__init__()
        
        ff_dim = ff_dim or embed_dim * 4
        
        # Layer normalizations
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.norm3 = nn.LayerNorm(embed_dim)
        
        # Cross-attention (IR attends to reference)
        self.cross_attn = CrossAttention(embed_dim, num_heads, dropout)
        
        # Self-attention (refine combined features)
        self.self_attn = CrossAttention(embed_dim, num_heads, dropout)
        
        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, embed_dim),
            nn.Dropout(dropout)
        )
        
    def forward(
        self, 
        ir_features: torch.Tensor, 
        ref_features: torch.Tensor
    ) -> torch.Tensor:
        """
        Match and combine IR and reference features.
        
        Args:
            ir_features: Features from IR encoder [B, C, H, W]
            ref_features: Features from reference encoder [B, C, H', W']
            
        Returns:
            Combined features [B, C, H, W] with color info from reference
        """
        B, C, H, W = ir_features.shape
        B, C_ref, H_ref, W_ref = ref_features.shape
        
        # Flatten spatial dimensions for attention
        # [B, C, H, W] -> [B, H*W, C]
        ir_flat = ir_features.flatten(2).transpose(1, 2)
        ref_flat = ref_features.flatten(2).transpose(1, 2)
        
        # Cross-attention: IR queries attend to reference
        ir_normed = self.norm1(ir_flat)
        ref_normed = self.norm1(ref_flat)
        cross_out, _ = self.cross_attn(ir_normed, ref_normed, ref_normed)
        ir_flat = ir_flat + cross_out
        
        # Self-attention: Refine combined features
        ir_normed = self.norm2(ir_flat)
        self_out, _ = self.self_attn(ir_normed, ir_normed, ir_normed)
        ir_flat = ir_flat + self_out
        
        # Feed-forward
        ir_normed = self.norm3(ir_flat)
        ff_out = self.ffn(ir_normed)
        ir_flat = ir_flat + ff_out
        
        # Reshape back to spatial
        # [B, H*W, C] -> [B, C, H, W]
        out = ir_flat.transpose(1, 2).view(B, C, H, W)
        
        return out


class DecoderBlock(nn.Module):
    """
    Single block of the decoder with upsampling.
    
    Each block:
    1. Upsamples by 2x using bilinear interpolation
    2. Concatenates skip connection features (if provided)
    3. Applies two conv-norm-relu sequences to refine
    """
    
    def __init__(
        self,
        in_channels: int,
        skip_channels: int,
        out_channels: int,
        use_instance_norm: bool = True
    ):
        """
        Initialize decoder block.
        
        Args:
            in_channels: Number of input channels (from previous decoder block)
            skip_channels: Number of skip connection channels (0 if no skip)
            out_channels: Number of output channels
            use_instance_norm: Use instance norm instead of batch norm
        """
        super().__init__()
        
        # Normalization choice
        norm_layer = nn.InstanceNorm2d if use_instance_norm else nn.BatchNorm2d
        
        # First conv after concatenation
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels + skip_channels, out_channels, 3, padding=1),
            norm_layer(out_channels),
            nn.ReLU(inplace=True)
        )
        
        # Second conv for refinement
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            norm_layer(out_channels),
            nn.ReLU(inplace=True)
        )
        
    def forward(
        self, 
        x: torch.Tensor, 
        skip: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass through decoder block.
        
        Args:
            x: Input features [B, C, H, W]
            skip: Optional skip connection features [B, C_skip, 2H, 2W]
            
        Returns:
            Upsampled features [B, out_channels, 2H, 2W]
        """
        # Upsample by 2x
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        
        # Concatenate skip connection if provided
        if skip is not None:
            # Handle size mismatch due to odd dimensions
            if x.shape[2:] != skip.shape[2:]:
                x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=False)
            x = torch.cat([x, skip], dim=1)
        
        # Apply convolutions
        x = self.conv1(x)
        x = self.conv2(x)
        
        return x


class Decoder(nn.Module):
    """
    Decoder network that generates the colorized output.
    
    Uses progressive upsampling with skip connections from the IR encoder
    to preserve fine structural details. The decoder takes the combined
    features from the feature matching module and generates RGB output.
    
    The encoder (ResNet) downsamples by 32x total:
    - conv1: 2x downsample
    - maxpool: 2x downsample  
    - layer1-4: each maintains or downsamples by 2x
    
    So for 256x256 input -> 8x8 bottleneck, we need 5 upsampling stages
    (each 2x) to get back to 256x256.
    """
    
    def __init__(
        self,
        bottleneck_channels: int,
        skip_channels: List[int],
        output_channels: int = 3,
        use_skip_connections: bool = True,
        use_instance_norm: bool = True,
        attention_layer: str = "layer4"
    ):
        """
        Initialize the decoder.
        
        Args:
            bottleneck_channels: Number of channels in bottleneck features
            skip_channels: List of channel counts for skip connections [conv1, layer1, layer2, layer3, layer4]
            output_channels: Number of output channels (3 for RGB)
            use_skip_connections: Whether to use skip connections
            use_instance_norm: Use instance norm instead of batch norm
            attention_layer: Which layer attention was applied at ('layer2', 'layer3', or 'layer4')
        """
        super().__init__()
        
        self.use_skip_connections = use_skip_connections
        self.attention_layer = attention_layer
        
        # Determine starting resolution based on attention layer
        # layer2: H/8, layer3: H/16, layer4: H/32
        layer_downsample = {'layer2': 8, 'layer3': 16, 'layer4': 32}
        start_downsample = layer_downsample[attention_layer]
        
        # Calculate number of upsampling stages needed
        # layer2 (H/8) needs 3 stages: H/8 -> H/4 -> H/2 -> H
        # layer3 (H/16) needs 4 stages: H/16 -> H/8 -> H/4 -> H/2 -> H
        # layer4 (H/32) needs 5 stages: H/32 -> H/16 -> H/8 -> H/4 -> H/2 -> H
        import math
        num_stages = int(math.log2(start_downsample))
        
        # Channel progression
        channel_sequence = [bottleneck_channels]
        for i in range(num_stages):
            next_ch = max(32, bottleneck_channels // (2 ** (i + 1)))
            channel_sequence.append(next_ch)
        
        self.blocks = nn.ModuleList()
        
        # Build decoder blocks
        for i in range(len(channel_sequence) - 1):
            in_ch = channel_sequence[i]
            out_ch = channel_sequence[i + 1]
            
            # Determine skip connection for this stage
            skip_ch = 0
            if use_skip_connections:
                # Map decoder stage to encoder layer
                # For layer2 attention (starting at H/8):
                #   stage 0: H/8->H/4, use layer1 (H/4)
                #   stage 1: H/4->H/2, use conv1 (H/2)
                #   stage 2: H/2->H, no skip
                # For layer4 attention (starting at H/32):
                #   stage 0: H/32->H/16, use layer3 (H/16)
                #   stage 1: H/16->H/8, use layer2 (H/8)
                #   ... etc
                
                # Encoder layer index where we started
                layer_start_idx = {'layer2': 2, 'layer3': 3, 'layer4': 4}[attention_layer]
                
                # Skip index for this decoder stage
                skip_idx = layer_start_idx - 1 - i
                
                if 0 <= skip_idx < len(skip_channels):
                    skip_ch = skip_channels[skip_idx]
            
            self.blocks.append(
                DecoderBlock(in_ch, skip_ch, out_ch, use_instance_norm)
            )
        
        # Final output convolution
        self.output_conv = nn.Sequential(
            nn.Conv2d(channel_sequence[-1], output_channels, 3, padding=1),
            nn.Tanh()  # Output in [-1, 1] range
        )
    
    def forward(
        self, 
        x: torch.Tensor, 
        skip_features: Optional[List[torch.Tensor]] = None
    ) -> torch.Tensor:
        """
        Generate colorized output from combined features.
        
        Args:
            x: Bottleneck features [B, C, H_start, W_start]
            skip_features: List of skip connection features from encoder
                          Order: [conv1 (H/2), layer1 (H/4), layer2 (H/8), ...]
                          
        Returns:
            Colorized image [B, 3, H, W]
        """
        # Apply decoder blocks with skip connections
        for i, block in enumerate(self.blocks):
            skip = None
            if self.use_skip_connections and skip_features is not None:
                # Determine which skip feature to use
                layer_start_idx = {'layer2': 2, 'layer3': 3, 'layer4': 4}[self.attention_layer]
                skip_idx = layer_start_idx - 1 - i
                
                if 0 <= skip_idx < len(skip_features):
                    skip = skip_features[skip_idx]
            
            x = block(x, skip)
        
        # Final output
        x = self.output_conv(x)
        
        return x


class IRColorNet(nn.Module):
    """
    Complete IR-to-Color translation network.
    
    This is the main model class that combines all components:
    - Content encoder for IR image
    - Reference encoder for visible image
    - Feature matching module for cross-modal alignment
    - Decoder for generating colorized output
    
    The forward pass:
    1. Encode IR image to get content features + skip connections
    2. Encode reference image to get color/style features
    3. Match features using attention to align color to content
    4. Decode to generate final colorized image
    """
    
    def __init__(self, config: ModelConfig):
        """
        Initialize the complete network.
        
        Args:
            config: Model configuration object
        """
        super().__init__()
        
        self.config = config
        
        # Content encoder (for IR image)
        self.content_encoder = ResNetEncoder(
            backbone=config.encoder_backbone,
            pretrained=config.pretrained_encoder
        )
        
        # Reference encoder (for visible image)
        # Separate encoder allows learning modality-specific features
        self.reference_encoder = ResNetEncoder(
            backbone=config.encoder_backbone,
            pretrained=config.pretrained_encoder
        )
        
        # Determine bottleneck channels based on attention layer
        layer_map = {'layer2': 2, 'layer3': 3, 'layer4': 4}
        layer_idx = layer_map.get(config.attention_layer, 4)
        bottleneck_channels = self.content_encoder.feature_channels[layer_idx]
        
        # Feature matching module
        self.feature_matching = FeatureMatchingModule(
            embed_dim=bottleneck_channels,
            num_heads=config.num_attention_heads,
            dropout=config.attention_dropout
        )
        
        # Decoder
        self.decoder = Decoder(
            bottleneck_channels=bottleneck_channels,
            skip_channels=self.content_encoder.feature_channels,
            output_channels=config.output_channels,
            use_skip_connections=config.use_skip_connections,
            use_instance_norm=config.use_instance_norm,
            attention_layer=config.attention_layer
        )
        
        # Initialize weights for non-pretrained layers
        self._init_weights()
        
        # Freeze content encoder if specified
        if self.config.freeze_content_encoder:
            self._freeze_content_encoder()
        
    def _init_weights(self):
        """Initialize weights for decoder and attention layers."""
        for module in [self.feature_matching, self.decoder]:
            for m in module.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.Linear):
                    nn.init.trunc_normal_(m.weight, std=0.02)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
                elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d, nn.LayerNorm)):
                    if m.weight is not None:
                        nn.init.constant_(m.weight, 1)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
    
    def _freeze_content_encoder(self):
        """
        Freeze the content (IR) encoder weights.
        
        This prevents the content encoder from being updated during training,
        which can be useful for:
        - Leveraging pretrained features without fine-tuning
        - Reducing memory usage during training
        - Focusing training on the reference encoder and decoder
        """
        for param in self.content_encoder.parameters():
            param.requires_grad = False
        
        print("  Content encoder weights frozen (not trainable)")
        
    def forward(
        self, 
        ir_image: torch.Tensor, 
        ref_image: torch.Tensor,
        return_attention: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the complete network.
        
        Args:
            ir_image: IR input image [B, 3, H, W]
            ref_image: Color reference image [B, 3, H', W']
            return_attention: Whether to return attention maps
            
        Returns:
            Dictionary with 'output', 'content_features', 'ref_features'
        """
        # Encode IR image at specified layer
        content_features, content_skips = self.content_encoder(
            ir_image, 
            return_layer=self.config.attention_layer
        )
        
        # Encode reference image at same layer
        ref_features, _ = self.reference_encoder(
            ref_image,
            return_layer=self.config.attention_layer
        )
        
        # Match features
        combined_features = self.feature_matching(content_features, ref_features)
        
        # Decode
        output = self.decoder(combined_features, content_skips)
        
        results = {
            'output': output,
            'content_features': content_features,
            'ref_features': ref_features
        }
        
        return results
    
    def get_num_params(self) -> int:
        """Get total number of parameters in the model."""
        return sum(p.numel() for p in self.parameters())
    
    def get_num_trainable_params(self) -> int:
        """Get number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def create_model(config: ModelConfig) -> IRColorNet:
    """
    Factory function to create the model.
    
    Args:
        config: Model configuration
        
    Returns:
        Initialized IRColorNet model
    """
    model = IRColorNet(config)
    
    print(f"Created IRColorNet with {model.get_num_params():,} parameters")
    print(f"  Trainable parameters: {model.get_num_trainable_params():,}")
    print(f"  Encoder backbone: {config.encoder_backbone}")
    print(f"  Attention heads: {config.num_attention_heads}")
    print(f"  Skip connections: {config.use_skip_connections}")
    print(f"  Content encoder frozen: {config.freeze_content_encoder}")
    
    return model


# Test the model when run directly
if __name__ == "__main__":
    from config import get_config
    
    config = get_config()
    model = create_model(config.model)
    
    # Test forward pass
    print("\nTesting forward pass...")
    ir_image = torch.randn(2, 3, 256, 256)
    ref_image = torch.randn(2, 3, 256, 256)
    
    results = model(ir_image, ref_image)
    
    print(f"Output shape: {results['output'].shape}")
    print(f"Content features shape: {results['content_features'].shape}")
    print(f"Reference features shape: {results['ref_features'].shape}")
    print(f"Output range: [{results['output'].min():.3f}, {results['output'].max():.3f}]")
