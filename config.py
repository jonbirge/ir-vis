"""
Configuration file for IR-to-Color Image Translation Network

This module contains all hyperparameters and configuration settings for the
reference-based colorization network. The architecture is designed to take
an IR image (simulated from the red channel of a cropped visible image) and
a reference visible image, producing a colorized version of the IR image.

The design philosophy here is to separate configuration from code, making it
easy to experiment with different hyperparameters without modifying the
training logic.
"""

import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple, Union

import yaml


@dataclass
class DataConfig:
    """
    Configuration for data loading and preprocessing.
    
    The training data simulation works as follows:
    1. Load a full-resolution color outdoor image
    2. Create "IR" image by: random crop -> extract red channel -> grayscale
    3. The original full image serves as the color reference
    4. Ground truth is the cropped region in full color
    
    This simulates the real-world scenario where IR and visible images
    have different fields of view and perspectives.
    """
    
    # Root directory for downloaded datasets
    data_root: str = "./data"
    
    # Dataset to use: 'cityscapes' (European cities), 'coco' (COCO 2017)
    # Cityscapes is recommended for urban/outdoor scenes with consistent quality
    # Note: Cityscapes requires manual download after free registration at:
    #       https://www.cityscapes-dataset.com/
    dataset_name: str = "coco"
    
    # Image dimensions for network input
    # IR image size (the image we want to colorize)
    ir_image_size: Tuple[int, int] = (256, 256)
    
    # Reference image size (the color image providing style/color info)
    # Can be different from IR size - the network handles this via adaptive pooling
    ref_image_size: Tuple[int, int] = (256, 256)
    
    # Crop ratio range for simulating different FOVs between IR and visible
    # e.g., (0.4, 0.7) means the IR crop will be 40-70% of the original image
    # This creates perspective/FOV mismatch that the network must learn to handle
    crop_ratio_range: Tuple[float, float] = (0.25, 1.0)
    
    # Number of data loading workers (adjust based on your CPU cores)
    num_workers: int = 8
    
    # Image statistics augmentation parameters
    use_augmentation: bool = True
    random_horizontal_flip: bool = True
    color_jitter_brightness: float = 0.1
    color_jitter_contrast: float = 0.1
    color_jitter_saturation: float = 0.1
    color_jitter_hue: float = 0.05
    
    # Geometric augmentation parameters (applied before cropping)
    random_rotation: bool = True
    max_rotation_angle: float = 15.0  # degrees
    random_perspective: bool = True
    perspective_distortion: float = 0.1  # 0.0 to 0.5, higher = more distortion
    
    # IR simulation parameters
    # Enable advanced IR simulation (channel subtraction + noise)
    # If False, uses simple red channel extraction (legacy behavior)
    use_ir_augmentation: bool = True
    
    # Maximum weight for channel subtraction when simulating IR
    # IR = R - random(0, max_channel_subtract) * (G + B)
    max_channel_subtract: float = 0.5
    
    # Maximum pixel noise standard deviation for simulated IR (0-255 scale)
    # Actual noise std is randomly selected from [0, ir_noise_std] per image
    ir_noise_std: float = 10.0
    
    # Maximum number of training samples to use (None = use all)
    # Useful for debugging or quick experiments with smaller subsets
    max_train_samples: Optional[int] = 32000


@dataclass
class ModelConfig:
    """
    Configuration for the neural network architecture.
    
    The architecture consists of:
    1. Content Encoder: Processes the IR image to extract structural features
    2. Reference Encoder: Processes the visible image to extract color/style features  
    3. Feature Matching Module: Attention-based alignment of reference features to content
    4. Decoder: Generates the colorized output from combined features
    
    For better small object color replication, attention is applied at H/8 resolution
    (32x32 for 256x256 input) instead of H/32 (8x8), preserving fine spatial details.
    """
    
    # Backbone for encoders: 'resnet18', 'resnet34', 'resnet50'
    encoder_backbone: str = "resnet18"
    
    # Whether to use pretrained ImageNet weights for encoder initialization
    pretrained_encoder: bool = True
    
    # Whether to freeze the content encoder (IR input) weights
    # If True, only the reference encoder, feature matching, and decoder are trained
    # Useful for leveraging pretrained features without fine-tuning
    freeze_content_encoder: bool = False
    
    # Which encoder layer to extract features from for attention
    # 'layer2': H/8 (32x32 for 256px input) - better for small objects, more memory
    # 'layer3': H/16 (16x16) - balanced
    # 'layer4': H/32 (8x8) - coarse, less memory (original)
    attention_layer: str = "layer3"
    
    # Number of attention heads in the feature matching module
    # Increase to 16 to capture more diverse spatial correspondences
    num_attention_heads: int = 16
    
    # Dropout rate in attention layers
    attention_dropout: float = 0.1
    
    # Decoder configuration
    decoder_blocks: int = 4
    use_skip_connections: bool = True
    use_instance_norm: bool = True
    output_channels: int = 3


@dataclass
class LossConfig:
    """
    Configuration for loss functions.
    
    We use a combination of losses to train the network:
    1. L1 Loss: Pixel-wise reconstruction (encourages correct overall color)
    2. Perceptual Loss: Feature-space loss using VGG (encourages realistic textures)
    3. Style Loss: Gram matrix matching (encourages similar color statistics)
    4. Color Histogram Loss: Matches color distribution to ground truth
    
    The weights balance these objectives - perceptual and style losses are
    particularly important for producing visually pleasing results.
    """
    
    # L1 (pixel-wise) reconstruction loss weight
    l1_weight: float = 10.0
    
    # Perceptual loss weight (VGG feature matching)
    # Higher values produce sharper, more detailed results but may introduce artifacts
    perceptual_weight: float = 0.0
    
    # Style loss weight (Gram matrix matching)
    # Helps transfer color statistics from reference
    # Note: Reduced from 50.0 to 10.0 for numerical stability
    style_weight: float = 0.0
    
    # Color histogram loss weight
    # Encourages the output to have similar color distribution to ground truth
    histogram_weight: float = 0.0
    
    # VGG layers to use for perceptual loss
    # Earlier layers capture low-level features; later layers capture semantics
    vgg_layers: List[str] = field(default_factory=lambda: [
        'relu1_2', 'relu2_2', 'relu3_4', 'relu4_4', 'relu5_4'
    ])
    
    # VGG layers to use for style loss (typically use more layers)
    style_layers: List[str] = field(default_factory=lambda: [
        'relu1_2', 'relu2_2', 'relu3_4', 'relu4_4'
    ])


@dataclass
class TrainingConfig:
    """
    Configuration for the training process.
    
    Training uses AdamW optimizer with cosine annealing learning rate schedule.
    Gradient clipping helps stabilize training, especially early on.
    """
    
    # Batch size - adjust based on GPU memory
    # RTX 3090 (24GB): batch_size=16 should work
    # RTX 4090 (24GB): batch_size=16-20
    # A100 (40GB): batch_size=32
    batch_size: int = 32 # 10-12 seems to work well with 12 GB VRAM
    
    # Number of training epochs
    num_epochs: int = 100
    
    # Learning rate
    # 1e-4 is a good starting point for Adam with pretrained features
    learning_rate: float = 2e-4
    
    # Weight decay for regularization
    weight_decay: float = 1e-5
    
    # Learning rate scheduler: 'cosine', 'step', 'plateau', 'none'
    lr_scheduler: str = "cosine"
    
    # For step scheduler: decay factor and step milestones
    lr_decay_factor: float = 0.5
    lr_milestones: List[int] = field(default_factory=lambda: [30, 60, 80])
    
    # For cosine scheduler: minimum learning rate
    lr_min: float = 2e-6
    
    # Gradient clipping (max norm) - helps stabilize training
    # Reduced from 1.0 to 0.5 for better stability during early training
    gradient_clip_norm: float = 0.5
    
    # How often to save checkpoints (in epochs)
    save_every: int = 5
    
    # How often to log training metrics (in iterations)
    log_every: int = 50
    
    # How often to save sample visualizations (in epochs)
    visualize_every: int = 1
    
    # Number of samples to visualize
    num_visualize_samples: int = 7
    
    # Resume from checkpoint path (None to start fresh)
    resume_checkpoint: Optional[str] = None
    
    # Output directory for checkpoints and logs
    output_dir: str = "./outputs"
    
    # Random seed for reproducibility
    seed: int = 42
    
    # Mixed precision training (faster on modern GPUs)
    # Note: Disabled by default as it can cause NaN issues with VGG perceptual loss
    # Enable once training is stable: use_amp: bool = True
    use_amp: bool = True
    
    # Device: 'cuda', 'cpu', or specific GPU like 'cuda:0'
    device: str = "cuda"


@dataclass
class Config:
    """
    Master configuration combining all sub-configurations.
    
    Usage:
        config = Config()
        # Access settings like:
        # config.training.batch_size
        # config.model.encoder_backbone
        # etc.
    """
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    
    def __post_init__(self):
        """Make sure required output directories exist."""
        self.ensure_output_dirs()

    def ensure_output_dirs(self) -> None:
        """Create the directories that training expects to write to."""
        base = self.training.output_dir
        os.makedirs(base, exist_ok=True)
        os.makedirs(os.path.join(base, "checkpoints"), exist_ok=True)
        os.makedirs(os.path.join(base, "visualizations"), exist_ok=True)
        os.makedirs(os.path.join(base, "logs"), exist_ok=True)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the configuration making all nested dataclasses plain dicts."""
        return asdict(self)

    def save_yaml(self, path: Union[str, Path]) -> None:
        """Persist the configuration as YAML to the provided path."""
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("w", encoding="utf-8") as stream:
            yaml.safe_dump(self.to_dict(), stream, sort_keys=False)

    @staticmethod
    def _build_section(section_cls: Any, values: Optional[Mapping[str, Any]]) -> Any:
        """Instantiate a dataclass section, defaulting when nothing is provided."""
        if values is None:
            return section_cls()
        return section_cls(**values)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Config":
        """Create a Config object from a dictionary (e.g., parsed YAML)."""
        return cls(
            data=cls._build_section(DataConfig, data.get("data")),
            model=cls._build_section(ModelConfig, data.get("model")),
            loss=cls._build_section(LossConfig, data.get("loss")),
            training=cls._build_section(TrainingConfig, data.get("training"))
        )

    @classmethod
    def from_yaml(cls, path: Union[str, Path]) -> "Config":
        """Load configuration values from a YAML file."""
        source = Path(path)
        with source.open("r", encoding="utf-8") as stream:
            raw = yaml.safe_load(stream)
        if raw is None:
            raw = {}
        return cls.from_dict(raw)


def get_config() -> Config:
    """
    Factory function to get the default configuration.
    
    TODO: can be extended to support loading from YAML/JSON files.
    
    Returns:
        Config: The complete configuration object
    """
    return Config()


# Quick sanity check when module is run directly
if __name__ == "__main__":
    # Reflectively serialize the full default configuration and pretty-print it.
    config = get_config()
    print(yaml.safe_dump(config.to_dict(), sort_keys=False, allow_unicode=True))
