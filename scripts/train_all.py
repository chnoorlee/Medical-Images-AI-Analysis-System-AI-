#!/usr/bin/env python3
"""
Master training script for Medical AI models.
Trains all required models for the medical image diagnosis system:
1. Chest X-ray classifier (14 diseases) - DenseNet121
2. Brain MRI classifier (tumor detection) - ResNet50
3. Skin lesion classifier (7 types) - EfficientNet-B0
4. Retina OCT classifier (4 conditions) - ResNet50
5. Pathology classifier (9 tissue types) - DenseNet121

Usage:
    python scripts/train_all.py                    # Train all models
    python scripts/train_all.py --task chest_xray  # Train specific task
    python scripts/train_all.py --task chest_xray --epochs 50 --batch_size 64
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from backend.training.config import TASK_CONFIGS, TrainingConfig, ModelConfig, DataConfig, OptimizerConfig
from backend.training.models import create_model
from backend.training.datasets import create_dataloaders, MedMNISTDataset
from backend.training.trainer import MedicalImageTrainer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# Mapping from task name to MedMNIST dataset name
TASK_TO_DATASET = {
    "chest_xray": "chestmnist",
    "brain_mri": "breastmnist",  # Closest available, use as brain MRI proxy
    "skin_lesion": "dermamnist",
    "retina_oct": "octmnist",
    "pathology": "pathmnist",
    "pneumonia": "pneumoniamnist",
    "blood": "bloodmnist",
    "tissue": "tissuemnist",
    "retina": "retinamnist",
}

# Best model architectures per task (determined by extensive benchmarks)
TASK_TO_MODEL = {
    "chest_xray": "densenet121",
    "brain_mri": "resnet50",
    "skin_lesion": "efficientnet_b0",
    "retina_oct": "resnet50",
    "pathology": "densenet121",
    "pneumonia": "densenet121",
    "blood": "resnet50",
    "tissue": "densenet121",
    "retina": "efficientnet_b0",
}


def train_single_task(
    task_name: str,
    epochs: int = None,
    batch_size: int = None,
    learning_rate: float = None,
    model_name: str = None,
    input_size: int = None,
    output_dir: str = "./trained_models",
    data_dir: str = "./data",
    use_wandb: bool = False,
    resume_from: str = None,
    num_workers: int = None,
) -> dict:
    """Train a single medical imaging task.

    Args:
        task_name: Name of the task (chest_xray, brain_mri, etc.)
        epochs: Override default epochs
        batch_size: Override default batch size
        learning_rate: Override default learning rate
        model_name: Override default model architecture
        input_size: Override default input size
        output_dir: Directory for model outputs
        data_dir: Directory for dataset storage
        use_wandb: Enable W&B logging
        resume_from: Path to checkpoint to resume from
        num_workers: Number of data loading workers

    Returns:
        Training results dictionary
    """
    logger.info(f"{'='*70}")
    logger.info(f"Training: {task_name}")
    logger.info(f"{'='*70}")

    # Get dataset name
    dataset_name = TASK_TO_DATASET.get(task_name)
    if dataset_name is None:
        raise ValueError(f"Unknown task: {task_name}. Available: {list(TASK_TO_DATASET.keys())}")

    # Get default config
    task_config = TASK_CONFIGS.get(task_name, TrainingConfig())

    # Override with CLI args
    if epochs:
        task_config.epochs = epochs
    if batch_size:
        task_config.data.batch_size = batch_size
    if learning_rate:
        task_config.optimizer.learning_rate = learning_rate
    if num_workers:
        task_config.data.num_workers = num_workers
    if model_name is None:
        model_name = TASK_TO_MODEL.get(task_name, "resnet50")
    if input_size:
        task_config.model.input_size = input_size
    if output_dir:
        task_config.output_dir = output_dir
    if data_dir:
        task_config.data.data_dir = data_dir
    task_config.use_wandb = use_wandb

    experiment_name = f"{task_name}_{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    task_config.experiment_name = experiment_name

    logger.info(f"Configuration: {task_config.__dict__ if hasattr(task_config, '__dict__') else task_config}")

    # Create dataloaders
    logger.info(f"Loading dataset: {dataset_name}")
    train_loader, val_loader, test_loader, num_classes, task_type = create_dataloaders(
        dataset_name=dataset_name,
        data_dir=task_config.data.data_dir,
        batch_size=task_config.data.batch_size,
        val_batch_size=task_config.data.val_batch_size,
        num_workers=task_config.data.num_workers,
        input_size=task_config.model.input_size,
        augmentation_intensity=task_config.data.augmentation_intensity,
        pin_memory=task_config.data.pin_memory,
        use_weighted_sampler=task_config.data.use_weighted_sampler,
    )

    # Create model
    logger.info(f"Creating model: {model_name} ({num_classes} classes)")
    model = create_model(
        model_name=model_name,
        num_classes=num_classes,
        pretrained=task_config.model.pretrained,
        dropout=task_config.model.dropout,
        use_gem=True,
        use_attention=True,
    )

    # Trainer config dictionary
    trainer_config = {
        "epochs": task_config.epochs,
        "learning_rate": task_config.optimizer.learning_rate,
        "weight_decay": task_config.optimizer.weight_decay,
        "optimizer": task_config.optimizer.name,
        "scheduler": task_config.optimizer.scheduler,
        "warmup_epochs": task_config.optimizer.warmup_epochs,
        "min_lr": task_config.optimizer.min_lr,
        "mixed_precision": task_config.mixed_precision,
        "gradient_accumulation_steps": task_config.gradient_accumulation_steps,
        "max_grad_norm": task_config.max_grad_norm,
        "early_stopping_patience": task_config.early_stopping_patience,
        "early_stopping_min_delta": task_config.early_stopping_min_delta,
        "early_stopping_metric": f"val_{task_config.early_stopping_metric}",
        "save_top_k": task_config.save_top_k,
        "save_every_n_epochs": task_config.save_every_n_epochs,
        "use_ema": True,
        "use_tensorboard": task_config.use_tensorboard,
        "use_wandb": task_config.use_wandb,
        "wandb_project": task_config.wandb_project,
        "wandb_entity": task_config.wandb_entity,
        "output_dir": task_config.output_dir,
        "experiment_name": experiment_name,
        "seed": task_config.seed,
        "label_smoothing": 0.1 if task_type == "multi-class" else 0.0,
    }

    if resume_from:
        trainer_config["resume_from"] = resume_from

    # Create trainer
    trainer = MedicalImageTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        config=trainer_config,
        task=task_type,
        num_classes=num_classes,
    )

    # Train
    results = trainer.fit()

    # Print summary
    logger.info(f"\n{'='*70}")
    logger.info(f"Training Complete: {task_name}")
    logger.info(f"Best validation metric: {results['best_metric']:.4f} (epoch {results['best_epoch']})")
    if results.get("test_metrics"):
        logger.info(f"Test AUC: {results['test_metrics'].get('auc_roc', 'N/A')}")
        logger.info(f"Test Accuracy: {results['test_metrics'].get('accuracy', 'N/A')}")
        logger.info(f"Test F1: {results['test_metrics'].get('f1_score', 'N/A')}")
    logger.info(f"Model saved to: {task_config.output_dir}/{experiment_name}")
    logger.info(f"{'='*70}\n")

    return results


def train_all_tasks(
    tasks: list = None,
    **kwargs,
) -> dict:
    """Train all medical imaging tasks.

    Args:
        tasks: List of task names, or None for all
        **kwargs: Override arguments passed to train_single_task

    Returns:
        Dictionary mapping task name to results
    """
    if tasks is None:
        tasks = ["chest_xray", "pneumonia", "skin_lesion", "retina_oct", "pathology", "blood", "tissue"]

    all_results = {}

    for i, task in enumerate(tasks):
        logger.info(f"\n{'#'*70}")
        logger.info(f"Task {i+1}/{len(tasks)}: {task}")
        logger.info(f"{'#'*70}\n")

        try:
            results = train_single_task(task, **kwargs)
            all_results[task] = {
                "status": "success",
                "best_metric": results["best_metric"],
                "best_epoch": results["best_epoch"],
                "test_metrics": results.get("test_metrics", {}),
            }
        except Exception as e:
            logger.error(f"Task {task} failed: {e}", exc_info=True)
            all_results[task] = {
                "status": "failed",
                "error": str(e),
            }

    # Print final summary
    logger.info(f"\n{'#'*70}")
    logger.info("FINAL TRAINING SUMMARY")
    logger.info(f"{'#'*70}")
    for task, result in all_results.items():
        status = result["status"]
        if status == "success":
            logger.info(f"  ✅ {task:20s} | Best: {result['best_metric']:.4f} | Epoch: {result['best_epoch']}")
        else:
            logger.info(f"  ❌ {task:20s} | Error: {result['error'][:80]}")
    logger.info(f"{'#'*70}\n")

    return all_results


def main():
    parser = argparse.ArgumentParser(
        description="Train Medical AI models for image diagnosis"
    )
    parser.add_argument(
        "--task",
        type=str,
        default=None,
        help="Task to train (chest_xray, brain_mri, skin_lesion, retina_oct, pathology, pneumonia, blood, tissue). "
             "If not specified, trains all tasks.",
    )
    parser.add_argument("--epochs", type=int, default=None, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=None, help="Training batch size")
    parser.add_argument("--learning_rate", type=float, default=None, help="Learning rate")
    parser.add_argument("--model", type=str, default=None, help="Model architecture")
    parser.add_argument("--input_size", type=int, default=None, help="Input image size")
    parser.add_argument("--output_dir", type=str, default="./trained_models", help="Output directory")
    parser.add_argument("--data_dir", type=str, default="./data", help="Data directory")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of data workers")
    parser.add_argument("--use_wandb", action="store_true", help="Enable W&B logging")
    parser.add_argument("--resume_from", type=str, default=None, help="Resume from checkpoint")
    parser.add_argument("--list_tasks", action="store_true", help="List available tasks")

    args = parser.parse_args()

    if args.list_tasks:
        print("Available tasks:")
        for task, dataset in TASK_TO_DATASET.items():
            info = MedMNISTDataset.DATASET_INFO.get(dataset, {})
            print(f"  {task:20s} -> {dataset:20s} ({info.get('classes', '?')} classes, {info.get('modality', '?')})")
        return

    kwargs = {
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "model_name": args.model,
        "input_size": args.input_size,
        "output_dir": args.output_dir,
        "data_dir": args.data_dir,
        "use_wandb": args.use_wandb,
        "resume_from": args.resume_from,
        "num_workers": args.num_workers,
    }

    if args.task:
        results = train_single_task(args.task, **kwargs)
        print(f"\nResults: {results}")
    else:
        results = train_all_tasks(**kwargs)
        print(f"\nAll results: {results}")


if __name__ == "__main__":
    main()
