#!/usr/bin/env python3
r"""
Medical AI - Production Training Script
=======================================
Run this directly from terminal for best reliability:
    cd C:\Users\yf\Desktop\Med
    python run_training.py

This trains ALL medical imaging models and pushes to HuggingFace.
"""

import sys, os, json, time, logging
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

# ── Logging setup ──────────────────────────────────────────────
LOG_FILE = PROJECT_ROOT / "logs" / f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
LOG_FILE.parent.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler(sys.stdout),
    ],
    force=True,
)
logger = logging.getLogger("medical_ai")

# ── Banner ─────────────────────────────────────────────────────
def banner():
    logger.info("=" * 65)
    logger.info("  MEDICAL AI — Full Training Pipeline")
    logger.info("  RTX 5090 32GB | PyTorch | MedMNIST → HuggingFace")
    logger.info(f"  Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"  Log: {LOG_FILE}")
    logger.info("=" * 65)


# ── Environment check ──────────────────────────────────────────
def check_environment():
    """Verify everything is ready for training."""
    import torch

    logger.info("\n[ENVIRONMENT]")
    logger.info(f"  Python:    {sys.version.split()[0]}")
    logger.info(f"  PyTorch:   {torch.__version__}")
    logger.info(f"  CUDA:      {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        logger.info(f"  GPU:       {p.name} ({p.total_memory/1e9:.0f}GB)")
        # Test GPU compatibility
        try:
            bn = torch.nn.BatchNorm2d(3).cuda()
            x = torch.randn(2, 3, 8, 8, device="cuda")
            _ = bn(x)
            del bn, x
            logger.info(f"  GPU check: OK (kernels available)")
            return "cuda"
        except Exception:
            logger.warning(f"  GPU check: INCOMPATIBLE — using CPU instead")
            return "cpu"
    else:
        logger.info(f"  GPU:       N/A (CPU only)")
        return "cpu"


# ── Training orchestrator ──────────────────────────────────────
def run():
    banner()
    device = check_environment()

    # Import after env check (avoids unnecessary CUDA init)
    from backend.training.datasets import create_dataloaders, MedMNISTDataset
    from backend.training.models import create_model
    from backend.training.trainer import MedicalImageTrainer

    # ── Task queue (fastest → slowest) ──────────────────────────
    TASKS = [
        {"id": "pneumonia",   "dataset": "pneumoniamnist", "model": "densenet121",     "epochs": 30,  "batch": 64, "size": 224},
        {"id": "breast",      "dataset": "breastmnist",    "model": "resnet50",        "epochs": 30,  "batch": 64, "size": 224},
        {"id": "blood",       "dataset": "bloodmnist",     "model": "efficientnet_b0", "epochs": 40,  "batch": 48, "size": 224},
        {"id": "retina_oct",  "dataset": "octmnist",       "model": "resnet50",        "epochs": 40,  "batch": 48, "size": 224},
        {"id": "retina",      "dataset": "retinamnist",    "model": "resnet50",        "epochs": 40,  "batch": 48, "size": 224},
        {"id": "tissue",      "dataset": "tissuemnist",    "model": "densenet121",     "epochs": 50,  "batch": 32, "size": 224},
        {"id": "skin_lesion", "dataset": "dermamnist",     "model": "efficientnet_b0", "epochs": 50,  "batch": 48, "size": 224},
        {"id": "pathology",   "dataset": "pathmnist",      "model": "densenet121",     "epochs": 50,  "batch": 32, "size": 224},
        {"id": "chest_xray",  "dataset": "chestmnist",     "model": "densenet121",     "epochs": 60,  "batch": 32, "size": 320},
    ]

    RESULTS = {}
    OUTPUT_DIR = PROJECT_ROOT / "trained_models"
    DATA_DIR = PROJECT_ROOT / "data"

    use_gpu = (device == "cuda")
    n_workers = 0  # num_workers=0 is most stable on Windows + CUDA
    if use_gpu:
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    logger.info(f"\nTraining {len(TASKS)} models on {'GPU' if use_gpu else 'CPU'} "
                f"({'AMP' if use_gpu else 'no AMP'})")
    for t in TASKS:
        logger.info(f"  • {t['id']:15s}: {t['model']:18s} × {t['epochs']:2d}ep ({t['batch']}bs)")
    logger.info("")

    total_start = time.time()

    for idx, task_cfg in enumerate(TASKS):
        task_name = task_cfg["id"]
        dataset_name = task_cfg["dataset"]

        logger.info(f"\n{'#'*65}")
        logger.info(f"  [{idx+1}/{len(TASKS)}] Training: {task_name}")
        logger.info(f"{'#'*65}")

        t_start = time.time()

        try:
            # ── Load data ──────────────────────────────────────
            logger.info(f"  Loading {dataset_name}...")
            train_loader, val_loader, test_loader, num_cls, task_type = create_dataloaders(
                dataset_name=dataset_name,
                data_dir=str(DATA_DIR),
                batch_size=task_cfg["batch"],
                val_batch_size=task_cfg["batch"] * 2,
                num_workers=n_workers,
                input_size=task_cfg["size"],
                augmentation_intensity="medium",
                pin_memory=use_gpu,
                use_weighted_sampler=True,
            )
            logger.info(f"  → train={len(train_loader)} val={len(val_loader)} test={len(test_loader)} batches")
            logger.info(f"  → {num_cls} classes, task={task_type}")

            # ── Create model ────────────────────────────────────
            logger.info(f"  Creating {task_cfg['model']}...")
            model = create_model(task_cfg["model"], num_classes=num_cls, pretrained=True)

            # ── Train ───────────────────────────────────────────
            on_gpu = torch.cuda.is_available()
            experiment = f"{task_name}_{task_cfg['model']}_{datetime.now().strftime('%m%d_%H%M')}"

            trainer = MedicalImageTrainer(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                test_loader=test_loader,
                config={
                    "epochs": task_cfg["epochs"],
                    "learning_rate": 1e-4,
                    "weight_decay": 1e-4,
                    "optimizer": "adamw",
                    "scheduler": "cosine_warmup",
                    "warmup_epochs": min(5, task_cfg["epochs"] // 6),
                    "min_lr": 1e-7,
                    "mixed_precision": on_gpu,
                    "gradient_accumulation_steps": 1,
                    "max_grad_norm": 1.0,
                    "early_stopping_patience": min(12, task_cfg["epochs"] // 3),
                    "early_stopping_metric": "val_auc",
                    "use_ema": True,
                    "use_tensorboard": True,
                    "use_wandb": False,
                    "output_dir": str(OUTPUT_DIR),
                    "experiment_name": experiment,
                    "log_every_n_steps": max(task_cfg["epochs"] * 2, 50),
                    "seed": 42,
                    "label_smoothing": 0.1 if task_type != "binary" else 0.0,
                },
                task=task_type,
                num_classes=num_cls,
            )

            train_results = trainer.fit()
            elapsed = time.time() - t_start

            # ── Record results ──────────────────────────────────
            result_summary = {
                "task": task_name,
                "model": task_cfg["model"],
                "device": device,
                "best_val_metric": train_results["best_metric"],
                "best_epoch": train_results["best_epoch"],
                "test_metrics": train_results.get("test_metrics", {}),
                "time_minutes": round(elapsed / 60, 1),
                "timestamp": datetime.now().isoformat(),
            }

            RESULTS[task_name] = result_summary

            tm = result_summary["test_metrics"]
            logger.info(f"  ✅ {task_name}: AUC={tm.get('auc_roc',0):.3f} "
                        f"Acc={tm.get('accuracy',0):.3f} "
                        f"F1={tm.get('f1_score',0):.3f} "
                        f"[{result_summary['time_minutes']:.0f}min]")

        except Exception as e:
            logger.error(f"  ❌ {task_name} FAILED: {e}", exc_info=True)
            RESULTS[task_name] = {"status": "failed", "error": str(e)}

        # ── Save intermediate results ──────────────────────────
        with open(OUTPUT_DIR / "results.json", "w") as f:
            json.dump(RESULTS, f, indent=2)

    # ── Final summary ──────────────────────────────────────────
    total_time = (time.time() - total_start) / 60
    logger.info(f"\n\n{'='*65}")
    logger.info(f"  TRAINING COMPLETE — {total_time:.0f} minutes")
    logger.info(f"{'='*65}")
    successes = sum(1 for r in RESULTS.values() if "error" not in r)
    failures = sum(1 for r in RESULTS.values() if "error" in r)
    logger.info(f"  ✅ {successes} succeeded    ❌ {failures} failed")
    logger.info(f"  Models saved to: {OUTPUT_DIR}")
    logger.info(f"  Results: {OUTPUT_DIR / 'results.json'}")
    logger.info(f"{'='*65}")

    # ── Upload to HuggingFace ──────────────────────────────────
    logger.info("\n[NEXT STEP] Upload to HuggingFace:")
    logger.info(f"  python huggingface/upload_model.py --upload_all --output_dir {OUTPUT_DIR}")
    if os.environ.get("HF_TOKEN"):
        logger.info("  (HF_TOKEN found — ready to upload)")
    else:
        logger.info("  (Set HF_TOKEN env var to auto-upload)")

    return RESULTS


if __name__ == "__main__":
    run()
