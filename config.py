"""
┌─────────────────────────────────────────────────────────────────────────────┐
│ IR-to-Color Image Translation Network — Configuration                       │
├─────────────────────────────────────────────────────────────────────────────┤
│ Purpose:                                                                    │
│   Centralized configuration for the reference-based IR→Color colorization   │
│   pipeline. This module defines dataclasses that encapsulate dataset,       │
│   model, loss, and training hyperparameters so experiments are reproducible │
│   and easy to modify without touching training logic.                       │
│                                                                             │
│ Key Features:                                                               │
│   - Typed dataclasses for Data, Model, Loss, and Training settings.         │
│   - Curriculum support: an ordered list of Config stages (Curriculum)       │
│     with helpers to serialize/deserialize curriculum YAML and enforce a     │
│     shared output directory across stages.                                  │
│   - Convenience helpers: to_dict, from_dict, save_yaml, from_yaml, and      │
│     load_experiment_config (supports either a Config or a Curriculum)       │
│   - Default 3-stage curriculum factory (get_curriculum) for common runs.    │
│   - Ensures output directories exist for checkpoints, logs, and visuals.    │
│   - Reasonable defaults for quick experiments and clear knobs for tuning.   │
│                                                                             │
│ Typical Usage:                                                              │
│   from config import get_config, Config, Curriculum                         │
│   cfg = get_config()                                                        │
│   # or load from YAML (single config or curriculum YAML supported)          │
│   cfg_or_curr = load_experiment_config('my_run/config_or_curriculum.yaml')  │
│                                                                             │
│ Notes & Caveats:                                                            │
│   - This file is purely declarative: training logic uses these values but   │
│     does not hard-code them.                                                │
│   - When saving YAML, nested dataclasses are flattened via asdict().        │
│   - Curriculum enforces a single shared training.output_dir across stages;  │
│     ensure_output_dirs() is called on initialization to avoid surprises     │
│     when training attempts to write files.                                  │
└─────────────────────────────────────────────────────────────────────────────┘
"""

import os
from dataclasses import asdict, dataclass, field, replace
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
    crop_ratio_range: Tuple[float, float] = (0.5, 1.0)
    
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
    max_rotation_angle: float = 90.0  # degrees
    random_perspective: bool = True
    perspective_distortion: float = 0.2  # 0.0 to 0.5, higher = more distortion
    
    # IR simulation parameters
    # Enable advanced IR simulation (channel subtraction + noise)
    # If False, uses simple red channel extraction (legacy behavior)
    use_ir_augmentation: bool = True
    
    # Maximum weight for channel subtraction when simulating IR
    # IR = R - random(0, max_channel_subtract) * (G + B)
    max_channel_subtract: float = 0.2
    
    # Maximum pixel noise standard deviation for simulated IR (0-255 scale)
    # Actual noise std is randomly selected from [0, ir_noise_std] per image
    ir_noise_std: float = 10.0
    
    # Black-hot IR simulation
    # If True, inverts the grayscale IR image (255 - value) to simulate
    # black-hot IR sensors where hot objects appear dark
    # If False, uses white-hot convention (hot objects appear bright)
    black_hot: bool = True
    
    # Maximum number of training samples to use (None = use all)
    # Useful for debugging or quick experiments with smaller subsets
    max_train_samples: Optional[int] = 10000
    
    # Maximum number of validation samples to use (None = use all)
    # Useful for faster validation during training
    max_validation_samples: Optional[int] = 500

    def copy(self, **overrides: Any) -> "DataConfig":
        """Return a copy of this config, applying any field overrides."""
        return replace(self, **overrides)


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
    use_skip_connections: bool = True
    use_instance_norm: bool = True
    output_channels: int = 3

    def copy(self, **overrides: Any) -> "ModelConfig":
        """Return a copy of this config, applying any field overrides."""
        return replace(self, **overrides)


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
    perceptual_weight: float = 10.0
    
    # Style loss weight (Gram matrix matching)
    # Helps transfer color statistics from reference
    # Note: Reduced from 50.0 to 10.0 for numerical stability
    style_weight: float = 10.0
    
    # Color histogram loss weight
    # Encourages the output to have similar color distribution to ground truth
    histogram_weight: float = 1.0
    
    # VGG layers to use for perceptual loss
    # Earlier layers capture low-level features; later layers capture semantics
    vgg_layers: List[str] = field(default_factory=lambda: [
        'relu1_2', 'relu2_2', 'relu3_4', 'relu4_4', 'relu5_4'
    ])
    
    # VGG layers to use for style loss (typically use more layers)
    style_layers: List[str] = field(default_factory=lambda: [
        'relu1_2', 'relu2_2', 'relu3_4', 'relu4_4'
    ])

    def copy(self, **overrides: Any) -> "LossConfig":
        """Return a copy of this config, applying any field overrides."""
        return replace(self, **overrides)


@dataclass
class TrainingConfig:
    """
    Configuration for the training process.
    
    Training uses AdamW optimizer with cosine annealing learning rate schedule.
    Gradient clipping helps stabilize training, especially early on.
    """
    
    # Batch size - adjust based on GPU memory
    batch_size: int = 8  # 8 seems to work well with 12 GB VRAM
    
    # Number of training epochs
    num_epochs: int = 100
    
    # Learning rate
    # 1e-4 is a good starting point for Adam with pretrained features
    learning_rate: float = 1e-4
    
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
    log_every: int = 100
    
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

    def copy(self, **overrides: Any) -> "TrainingConfig":
        """Return a copy of this config, applying any field overrides."""
        return replace(self, **overrides)


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

    def copy(
        self,
        *,
        data: Optional[Union[DataConfig, Mapping[str, Any]]] = None,
        model: Optional[Union[ModelConfig, Mapping[str, Any]]] = None,
        loss: Optional[Union[LossConfig, Mapping[str, Any]]] = None,
        training: Optional[Union[TrainingConfig, Mapping[str, Any]]] = None,
    ) -> "Config":
        """Copy this Config, reusing unchanged sub-objects by default.

        Passing a section as a dict applies overrides via that section's copy().
        Passing a section as an instance replaces the section.
        Passing None reuses the existing section object.
        """

        def _maybe_copy_section(current: Any, patch: Optional[Union[Any, Mapping[str, Any]]]) -> Any:
            if patch is None:
                return current
            if isinstance(patch, Mapping):
                # All sub-configs implement .copy(**overrides)
                return current.copy(**dict(patch))
            return patch

        return Config(
            data=_maybe_copy_section(self.data, data),
            model=_maybe_copy_section(self.model, model),
            loss=_maybe_copy_section(self.loss, loss),
            training=_maybe_copy_section(self.training, training),
        )

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


@dataclass
class Curriculum:
    """An ordered list of Config stages for curriculum learning.

    Each stage's training.num_epochs is interpreted as the number of epochs to
    run for that stage. The total epochs for the curriculum is the sum of all
    stage num_epochs.
    """

    stages: List[Config]

    def __post_init__(self) -> None:
        if not self.stages:
            raise ValueError("Curriculum must contain at least one Config stage")

        # Enforce single output directory per run (user preference).
        out_dir = self.stages[0].training.output_dir
        for stage in self.stages[1:]:
            if stage.training.output_dir != out_dir:
                raise ValueError(
                    "All curriculum stages must share the same training.output_dir"
                )

    @property
    def total_epochs(self) -> int:
        return int(sum(stage.training.num_epochs for stage in self.stages))

    @property
    def output_dir(self) -> str:
        return self.stages[0].training.output_dir

    def ensure_output_dirs(self) -> None:
        # Ensure the shared output directory exists.
        self.stages[0].ensure_output_dirs()

    def to_dict(self) -> Dict[str, Any]:
        return {"curriculum": [stage.to_dict() for stage in self.stages]}

    def save_yaml(self, path: Union[str, Path]) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("w", encoding="utf-8") as stream:
            yaml.safe_dump(self.to_dict(), stream, sort_keys=False)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Curriculum":
        raw_stages = data.get("curriculum")
        if raw_stages is None:
            raw_stages = data.get("stages")
        if not isinstance(raw_stages, list):
            raise ValueError("Curriculum YAML must contain a 'curriculum' list")
        stages = [Config.from_dict(stage_dict or {}) for stage_dict in raw_stages]
        return cls(stages=stages)

    @classmethod
    def from_yaml(cls, path: Union[str, Path]) -> "Curriculum":
        source = Path(path)
        with source.open("r", encoding="utf-8") as stream:
            raw = yaml.safe_load(stream)
        if raw is None:
            raw = {}
        return cls.from_dict(raw)

    def get_config_for_epoch(self, global_epoch: int) -> Tuple[Config, int, int]:
        """Get the config stage for a given global epoch.
        
        Args:
            global_epoch: The global epoch number (0-indexed)
            
        Returns:
            Tuple of (config, stage_idx, stage_epoch) where:
            - config: The Config for the stage containing this global epoch
            - stage_idx: The index of the stage (0-indexed)
            - stage_epoch: The epoch within that stage (0-indexed)
        """
        if global_epoch < 0:
            return self.stages[0], 0, 0
            
        cumulative = 0
        for stage_idx, stage in enumerate(self.stages):
            stage_epochs = int(stage.training.num_epochs)
            if global_epoch < cumulative + stage_epochs:
                stage_epoch = global_epoch - cumulative
                return stage, stage_idx, stage_epoch
            cumulative += stage_epochs
        
        # Beyond curriculum end - return last stage
        last_idx = len(self.stages) - 1
        last_stage = self.stages[last_idx]
        last_epoch = int(last_stage.training.num_epochs) - 1
        return last_stage, last_idx, last_epoch


def load_experiment_config(path: Union[str, Path]) -> Union[Config, Curriculum]:
    """Load either a single Config or a Curriculum from YAML."""
    source = Path(path)
    with source.open("r", encoding="utf-8") as stream:
        raw = yaml.safe_load(stream)
    if raw is None:
        raw = {}

    if isinstance(raw, Mapping) and "curriculum" in raw:
        return Curriculum.from_dict(raw)
    return Config.from_dict(raw if isinstance(raw, Mapping) else {})


def get_curriculum() -> Curriculum:
    """Create the default 3-stage curriculum:
      1) 50 epochs, perceptual/style/histogram only, COCO, 1/3 geometric augmentation
      2) 75 epochs, full loss including L1, COCO, 2/3 geometric augmentation
      3) 150 epochs, full loss, Cityscapes, full geometric augmentation
    """

    base = get_config()

    # Shared training section across all stages
    training_50 = base.training.copy(num_epochs=50)
    training_75 = base.training.copy(num_epochs=75)
    training_150 = base.training.copy(num_epochs=150)

    # Shared model across all stages
    shared_model = base.model

    # Dataset sections with progressive geometric augmentation
    data_coco_stage1 = base.data.copy(
        dataset_name="coco",
        max_rotation_angle=20.0,
        perspective_distortion=0.067
    )
    data_coco_stage2 = base.data.copy(
        dataset_name="coco",
        max_rotation_angle=60.0,
        perspective_distortion=0.133
    )
    data_cityscapes = base.data.copy(dataset_name="cityscapes")

    # Loss sections
    loss_no_l1 = base.loss.copy(l1_weight=0.0)
    loss_full = base.loss

    stage1 = Config(data=data_coco_stage1, model=shared_model, loss=loss_no_l1, training=training_50)
    stage2 = Config(data=data_coco_stage2, model=shared_model, loss=loss_full, training=training_75)
    stage3 = Config(data=data_cityscapes, model=shared_model, loss=loss_full, training=training_150)

    return Curriculum(stages=[stage1, stage2, stage3])


def get_config() -> Config:
    """
    Factory function to get the default configuration.
    
    Returns:
        Config: The complete configuration object
    """
    return Config()


# Quick sanity check when module is run directly
if __name__ == "__main__":
    # Reflectively serialize the full default configuration and pretty-print it.
    curriculum = get_curriculum()
    print(yaml.safe_dump(curriculum.to_dict(), sort_keys=False, allow_unicode=True))
