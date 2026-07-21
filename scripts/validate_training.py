#!/usr/bin/env python3
"""Quick 3-epoch validation to confirm the training pipeline works end-to-end."""

import sys, os, logging, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    stream=sys.stdout,
    force=True,
)

logger = logging.getLogger(__name__)

from backend.training.datasets import create_dataloaders
from backend.training.models import create_model
from backend.training.trainer import MedicalImageTrainer

def main():
    logger.info("Loading data...")
    train_loader, val_loader, test_loader, n_cls, task_type = create_dataloaders(
        dataset_name="pneumoniamnist", data_dir="./data",
        batch_size=32, val_batch_size=64, num_workers=0,
        input_size=224, pin_memory=False,
    )

    logger.info(f"Data: train={len(train_loader)} val={len(val_loader)} test={len(test_loader)} batches")

    logger.info(f"Creating model (n_cls={n_cls})...")
    model = create_model("densenet121", num_classes=n_cls, pretrained=True)

    logger.info("Starting 3-epoch test...")
    t0 = time.time()

    trainer = MedicalImageTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        config={
            "epochs": 3,
            "learning_rate": 1e-4,
            "mixed_precision": False,
            "use_ema": False,
            "use_tensorboard": False,
            "use_wandb": False,
            "output_dir": "./trained_models",
            "experiment_name": "validate_3ep",
            "log_every_n_steps": 30,
            "seed": 42,
        },
        task=task_type,
        num_classes=n_cls,
    )

    results = trainer.fit()
    elapsed = time.time() - t0

    logger.info(f"Done in {elapsed/60:.1f} min!")
    logger.info(f"Best val metric: {results['best_metric']:.4f} at epoch {results['best_epoch']}")
    if results.get("test_metrics"):
        tm = results["test_metrics"]
        logger.info(f"Test: Acc={tm.get('accuracy',0):.3f} AUC={tm.get('auc_roc',0):.3f} F1={tm.get('f1_score',0):.3f}")

    return results

if __name__ == "__main__":
    main()
