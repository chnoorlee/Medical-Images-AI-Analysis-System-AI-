#!/usr/bin/env python3
"""
Quick training test to validate the pipeline works end-to-end.
Trains a pneumonia classifier on MedMNIST data.
"""

import sys
import os
import logging

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

from backend.training.datasets import create_dataloaders, MedMNISTDataset
from backend.training.models import create_model
from backend.training.trainer import MedicalImageTrainer

def main():
    print("=" * 60)
    print("MEDICAL AI - Training Pipeline Validation")
    print("=" * 60)

    # Use a small, fast dataset for validation
    dataset_name = "pneumoniamnist"  # Binary: pneumonia vs normal
    task_type = "binary"
    num_classes = 2

    print(f"\n[1/4] Loading dataset: {dataset_name}")
    train_loader, val_loader, test_loader, num_classes, task_type = create_dataloaders(
        dataset_name=dataset_name,
        data_dir="./data",
        batch_size=32,
        val_batch_size=64,
        num_workers=2,
        input_size=224,
        augmentation_intensity="medium",
        pin_memory=False,
        use_weighted_sampler=False,
    )

    print(f"[2/4] Creating model: densenet121 ({num_classes} classes)")
    model = create_model(
        model_name="densenet121",
        num_classes=num_classes,
        pretrained=True,
        dropout=0.3,
    )

    print(f"[3/4] Starting training...")
    trainer = MedicalImageTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        config={
            "epochs": 30,
            "learning_rate": 1e-4,
            "weight_decay": 1e-4,
            "optimizer": "adamw",
            "scheduler": "cosine_warmup",
            "warmup_epochs": 3,
            "min_lr": 1e-7,
            "mixed_precision": False,  # CPU training
            "gradient_accumulation_steps": 1,
            "max_grad_norm": 1.0,
            "early_stopping_patience": 10,
            "early_stopping_min_delta": 1e-4,
            "early_stopping_metric": "val_auc",
            "use_ema": True,
            "use_tensorboard": False,
            "use_wandb": False,
            "output_dir": "./trained_models",
            "experiment_name": "pneumonia_densenet121_validation",
            "seed": 42,
        },
        task=task_type,
        num_classes=num_classes,
    )

    results = trainer.fit()

    print(f"\n[4/4] Training complete!")
    print(f"  Best metric: {results['best_metric']:.4f}")
    print(f"  Best epoch: {results['best_epoch']}")
    if results.get('test_metrics'):
        tm = results['test_metrics']
        print(f"  Test Accuracy: {tm.get('accuracy', 'N/A'):.4f}")
        print(f"  Test AUC: {tm.get('auc_roc', 'N/A'):.4f}")
        print(f"  Test F1: {tm.get('f1_score', 'N/A'):.4f}")
    print(f"  Total time: {results.get('total_time_minutes', 0):.1f} min")

    return results


if __name__ == "__main__":
    main()
