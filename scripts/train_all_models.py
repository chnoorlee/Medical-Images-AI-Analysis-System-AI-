#!/usr/bin/env python3
"""
Train all Medical AI models on the RTX 5090.
Once PyTorch CUDA nightly is ready, this handles the full training queue.

Usage:
    python scripts/train_all_models.py
    python scripts/train_all_models.py --tasks pneumonia blood --epochs 20
"""

import sys, os, argparse, logging, time, json
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

import torch

# Task definitions - ordered from fastest to slowest
TASKS = [
    {"task": "pneumonia",    "dataset": "pneumoniamnist", "model": "densenet121", "classes": 2,  "type": "binary",       "epochs": 30, "batch": 64,  "size": 224},
    {"task": "breast",       "dataset": "breastmnist",    "model": "resnet50",    "classes": 2,  "type": "binary",       "epochs": 30, "batch": 64,  "size": 224},
    {"task": "blood",        "dataset": "bloodmnist",     "model": "resnet50",    "classes": 8,  "type": "multi-class",  "epochs": 40, "batch": 48,  "size": 224},
    {"task": "retina_oct",   "dataset": "octmnist",       "model": "resnet50",    "classes": 4,  "type": "multi-class",  "epochs": 40, "batch": 48,  "size": 224},
    {"task": "retina",       "dataset": "retinamnist",    "model": "resnet50",    "classes": 5,  "type": "multi-class",  "epochs": 40, "batch": 48,  "size": 224},
    {"task": "tissue",       "dataset": "tissuemnist",    "model": "densenet121", "classes": 8,  "type": "multi-class",  "epochs": 50, "batch": 32,  "size": 224},
    {"task": "skin_lesion",  "dataset": "dermamnist",     "model": "efficientnet_b0", "classes": 7,  "type": "multi-class",  "epochs": 50, "batch": 48,  "size": 224},
    {"task": "pathology",    "dataset": "pathmnist",      "model": "densenet121", "classes": 9,  "type": "multi-class",  "epochs": 50, "batch": 32,  "size": 224},
    {"task": "chest_xray",   "dataset": "chestmnist",     "model": "densenet121", "classes": 14, "type": "multi-label",  "epochs": 60, "batch": 32,  "size": 320},
]

RESULTS = {}


def train_task(cfg: dict, output_dir: str = "./trained_models") -> dict:
    from backend.training.datasets import create_dataloaders
    from backend.training.models import create_model
    from backend.training.trainer import MedicalImageTrainer

    task_name = cfg["task"]
    logger.info(f"\n{'='*60}\n  TRAINING: {task_name} ({cfg['model']})\n{'='*60}")

    # Device
    use_gpu = torch.cuda.is_available()
    gpu_name = torch.cuda.get_device_name(0) if use_gpu else "CPU"
    logger.info(f"Device: {gpu_name}")

    # Data
    logger.info(f"Loading {cfg['dataset']}...")
    train_loader, val_loader, test_loader, n_cls, task_type = create_dataloaders(
        dataset_name=cfg["dataset"],
        data_dir="./data",
        batch_size=cfg["batch"],
        val_batch_size=cfg["batch"] * 2,
        num_workers=4 if use_gpu else 0,
        input_size=cfg["size"],
        augmentation_intensity="medium",
        pin_memory=use_gpu,
    )
    n_cls = cfg.get("classes", n_cls)
    task_type = cfg.get("type", task_type)

    # Model
    model = create_model(cfg["model"], num_classes=n_cls, pretrained=True)

    # Trainer
    trainer = MedicalImageTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        config={
            "epochs": cfg["epochs"],
            "learning_rate": 3e-4 if not use_gpu else 1e-4,
            "weight_decay": 1e-4,
            "optimizer": "adamw",
            "scheduler": "cosine_warmup",
            "warmup_epochs": min(5, cfg["epochs"] // 6),
            "min_lr": 1e-7,
            "mixed_precision": use_gpu,
            "gradient_accumulation_steps": 1,
            "max_grad_norm": 1.0,
            "early_stopping_patience": min(12, cfg["epochs"] // 3),
            "early_stopping_metric": "val_auc",
            "use_ema": True,
            "use_tensorboard": False,
            "use_wandb": False,
            "output_dir": output_dir,
            "experiment_name": f"{task_name}_{cfg['model']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "seed": 42,
            "label_smoothing": 0.1 if task_type == "multi-class" else 0.0,
        },
        task=task_type,
        num_classes=n_cls,
    )

    results = trainer.fit()

    # Save results summary
    summary = {
        "task": task_name,
        "model": cfg["model"],
        "device": gpu_name,
        "best_metric": results["best_metric"],
        "best_epoch": results["best_epoch"],
        "test_metrics": results.get("test_metrics", {}),
        "time_min": results.get("total_time_minutes", 0),
        "timestamp": datetime.now().isoformat(),
    }

    summary_path = Path(output_dir) / f"{task_name}_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    RESULTS[task_name] = summary
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", nargs="+", default=None, help="Tasks to train")
    parser.add_argument("--output_dir", default="./trained_models")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--skip_until", type=str, default=None, help="Skip until this task name")
    args = parser.parse_args()

    tasks = TASKS
    if args.tasks:
        tasks = [t for t in TASKS if t["task"] in args.tasks]
    if args.skip_until:
        skip = True
        filtered = []
        for t in tasks:
            if skip and t["task"] == args.skip_until:
                skip = False
            if not skip:
                filtered.append(t)
        tasks = filtered

    if args.epochs:
        for t in tasks:
            t["epochs"] = args.epochs

    logger.info(f"Training {len(tasks)} models on {'GPU' if torch.cuda.is_available() else 'CPU'}")
    for i, t in enumerate(tasks):
        logger.info(f"  {i+1}. {t['task']}: {t['model']} ({t['classes']}cls, {t['epochs']}ep)")

    start_time = time.time()

    for i, cfg in enumerate(tasks):
        logger.info(f"\n{'#'*60}")
        logger.info(f"  Task {i+1}/{len(tasks)}: {cfg['task']}")
        logger.info(f"{'#'*60}")
        try:
            train_task(cfg, args.output_dir)
        except Exception as e:
            logger.error(f"Task {cfg['task']} FAILED: {e}", exc_info=True)
            RESULTS[cfg["task"]] = {"status": "failed", "error": str(e)}

    # Final summary
    total_time = (time.time() - start_time) / 60
    logger.info(f"\n{'='*70}")
    logger.info(f"  ALL TRAINING COMPLETE ({total_time:.0f} min)")
    logger.info(f"{'='*70}")
    for t, r in RESULTS.items():
        if "error" in r:
            logger.info(f"  ❌ {t:20s}: {r['error'][:60]}")
        else:
            logger.info(f"  ✅ {t:20s}: AUC={r['test_metrics'].get('auc_roc', 0):.3f} Acc={r['test_metrics'].get('accuracy', 0):.3f} ({r['time_min']:.0f}min)")

    # Save master results
    master_path = Path(args.output_dir) / f"all_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(master_path, "w") as f:
        json.dump(RESULTS, f, indent=2)
    logger.info(f"\nResults saved to: {master_path}")
    logger.info(f"Models ready for HuggingFace upload!")


if __name__ == "__main__":
    main()
