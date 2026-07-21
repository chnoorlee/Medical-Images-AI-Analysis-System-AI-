"""
Training configuration with sensible defaults for medical imaging.
All values can be overridden via environment variables or command line.
"""

import os
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from pathlib import Path


@dataclass
class ModelConfig:
    """Model architecture configuration."""
    name: str = "resnet50"
    num_classes: int = 2
    pretrained: bool = True
    in_channels: int = 3
    input_size: int = 224
    dropout: float = 0.3
    # For segmentation
    decoder_channels: List[int] = field(default_factory=lambda: [256, 128, 64, 32, 16])
    # For ViT
    vit_patch_size: int = 16
    vit_hidden_size: int = 768
    vit_num_layers: int = 12
    vit_num_heads: int = 12


@dataclass
class DataConfig:
    """Dataset and data loading configuration."""
    dataset_name: str = "chest_xray"
    data_dir: str = "./data"
    batch_size: int = 32
    val_batch_size: int = 64
    num_workers: int = 8
    pin_memory: bool = True
    prefetch_factor: int = 2
    # Augmentation
    use_augmentation: bool = True
    augmentation_intensity: str = "medium"  # low, medium, high
    # Normalization (ImageNet stats by default)
    mean: List[float] = field(default_factory=lambda: [0.485, 0.456, 0.406])
    std: List[float] = field(default_factory=lambda: [0.229, 0.224, 0.225])
    # Validation split
    val_split: float = 0.15
    test_split: float = 0.05
    # Class balancing
    use_weighted_sampler: bool = True


@dataclass
class OptimizerConfig:
    """Optimizer and learning rate configuration."""
    name: str = "adamw"
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    betas: tuple = (0.9, 0.999)
    eps: float = 1e-8
    # Scheduler
    scheduler: str = "cosine_warmup"  # cosine_warmup, cosine, plateau, onecycle
    warmup_epochs: int = 5
    min_lr: float = 1e-7
    plateau_patience: int = 5
    plateau_factor: float = 0.5


@dataclass
class TrainingConfig:
    """Full training configuration."""
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)

    # Training loop
    epochs: int = 100
    mixed_precision: bool = True
    gradient_accumulation_steps: int = 1
    max_grad_norm: float = 1.0
    gradient_clip_val: float = 1.0

    # Distributed training
    distributed: bool = False
    local_rank: int = -1
    world_size: int = 1

    # Checkpointing
    checkpoint_dir: str = "./checkpoints"
    save_top_k: int = 3
    save_every_n_epochs: int = 5
    resume_from: Optional[str] = None

    # Early stopping
    early_stopping_patience: int = 15
    early_stopping_min_delta: float = 1e-4
    early_stopping_metric: str = "val_auc"

    # Logging
    use_wandb: bool = False
    wandb_project: str = "medical-ai"
    wandb_entity: Optional[str] = None
    use_tensorboard: bool = True
    log_every_n_steps: int = 50

    # Reproducibility
    seed: int = 42
    deterministic: bool = False

    # Output
    output_dir: str = "./trained_models"
    experiment_name: str = "medical_ai_v2"

    def __post_init__(self):
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.data.data_dir, exist_ok=True)


# Pre-defined configurations for different medical imaging tasks
CHEST_XRAY_CONFIG = TrainingConfig(
    model=ModelConfig(name="densenet121", num_classes=14, input_size=320),
    data=DataConfig(dataset_name="chest_xray", batch_size=64, augmentation_intensity="high"),
    optimizer=OptimizerConfig(learning_rate=3e-4, warmup_epochs=3),
    epochs=50,
    experiment_name="chest_xray_classifier"
)

BRAIN_MRI_CONFIG = TrainingConfig(
    model=ModelConfig(name="resnet50", num_classes=2, input_size=256),
    data=DataConfig(dataset_name="brain_mri", batch_size=32, augmentation_intensity="medium"),
    optimizer=OptimizerConfig(learning_rate=2e-4),
    epochs=60,
    experiment_name="brain_mri_classifier"
)

SKIN_LESION_CONFIG = TrainingConfig(
    model=ModelConfig(name="efficientnet_b0", num_classes=7, input_size=224),
    data=DataConfig(dataset_name="skin_lesion", batch_size=48, augmentation_intensity="high"),
    optimizer=OptimizerConfig(learning_rate=5e-4),
    epochs=70,
    experiment_name="skin_lesion_classifier"
)

RETINA_OCT_CONFIG = TrainingConfig(
    model=ModelConfig(name="resnet50", num_classes=4, input_size=256),
    data=DataConfig(dataset_name="retina_oct", batch_size=32),
    optimizer=OptimizerConfig(learning_rate=2e-4),
    epochs=50,
    experiment_name="retina_oct_classifier"
)

PATHOLOGY_CONFIG = TrainingConfig(
    model=ModelConfig(name="densenet121", num_classes=9, input_size=256),
    data=DataConfig(dataset_name="pathology", batch_size=32, augmentation_intensity="high"),
    optimizer=OptimizerConfig(learning_rate=2e-4),
    epochs=60,
    experiment_name="pathology_classifier"
)

# Task registry
TASK_CONFIGS: Dict[str, TrainingConfig] = {
    "chest_xray": CHEST_XRAY_CONFIG,
    "brain_mri": BRAIN_MRI_CONFIG,
    "skin_lesion": SKIN_LESION_CONFIG,
    "retina_oct": RETINA_OCT_CONFIG,
    "pathology": PATHOLOGY_CONFIG,
}
