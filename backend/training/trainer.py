"""
Modern medical image model trainer with:
- Automatic Mixed Precision (AMP) for 2-3x faster training
- Gradient accumulation for larger effective batch sizes
- Cosine annealing with linear warmup
- Exponential Moving Average (EMA) of model weights
- Early stopping with multiple monitoring metrics
- Comprehensive logging (TensorBoard + W&B + console)
- Automatic checkpointing and best model saving
- Multi-GPU Distributed Data Parallel (DDP) support
"""

import os
import sys
import time
import json
import logging
import warnings
from pathlib import Path
from typing import Optional, Dict, Any, Tuple, List
from datetime import datetime
from collections import defaultdict
from copy import deepcopy

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from .metrics import MetricsTracker

logger = logging.getLogger(__name__)


class EMAModel:
    """Exponential Moving Average of model weights for more stable inference."""

    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        self._register()

    def _register(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()

    def apply_shadow(self):
        """Apply EMA weights to model."""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data.clone()
                param.data = self.shadow[name]

    def restore(self):
        """Restore original model weights."""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data = self.backup[name]
        self.backup = {}

    @property
    def model_with_ema(self):
        return self.apply_shadow()


class MedicalImageTrainer:
    """Complete training pipeline for medical image models."""

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        test_loader: Optional[DataLoader] = None,
        config: Optional[Dict[str, Any]] = None,
        task: str = "multi-class",
        num_classes: int = 2,
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.task = task
        self.num_classes = num_classes

        # Configuration
        self.config = self._merge_config(config or {})

        # Device - detect and validate GPU
        force_cpu = self.config.get("force_cpu", False)
        if force_cpu or not torch.cuda.is_available():
            self.device = torch.device("cpu")
        else:
            # Test if GPU is actually usable (handles RTX 5090 sm_120 incompatibility)
            try:
                props = torch.cuda.get_device_properties(0)
                if props.major >= 10:
                    # Blackwell or newer — check kernel compatibility
                    test_bn = torch.nn.BatchNorm2d(3).cuda()
                    test_x = torch.zeros(2, 3, 4, 4, device="cuda")
                    _ = test_bn(test_x)
                    del test_bn, test_x
                self.device = torch.device("cuda")
            except Exception:
                logger.warning(
                    f"GPU incompatible with current PyTorch build. "
                    f"Using CPU. Install PyTorch cu130+ for RTX 5090."
                )
                self.device = torch.device("cpu")
        self.is_distributed = self.config.get("distributed", False)
        self.local_rank = self.config.get("local_rank", 0)
        self.is_main_process = (not self.is_distributed) or (
            self.is_distributed and self.local_rank == 0
        )

        # Move model to device
        self.model = self.model.to(self.device)

        # DDP wrapping
        if self.is_distributed:
            self.model = DDP(
                self.model,
                device_ids=[self.local_rank],
                output_device=self.local_rank,
                find_unused_parameters=False,
            )

        # EMA
        self.use_ema = self.config.get("use_ema", True)
        if self.use_ema and self.is_main_process:
            self.ema = EMAModel(self.model.module if self.is_distributed else self.model)
        else:
            self.ema = None

        # Loss function
        self.criterion = self._get_loss_function()

        # Optimizer
        self.optimizer = self._create_optimizer()

        # Scheduler
        self.scheduler = self._create_scheduler()

        # Mixed precision
        self.use_amp = self.config.get("mixed_precision", True) and self.device.type == "cuda"
        if self.use_amp:
            self.scaler = torch.amp.GradScaler('cuda')
        else:
            self.scaler = None

        # Metrics trackers
        self.train_metrics = MetricsTracker(num_classes, task)
        self.val_metrics = MetricsTracker(num_classes, task)

        # Logging
        self.output_dir = Path(self.config.get("output_dir", "./trained_models"))
        self.experiment_name = self.config.get(
            "experiment_name",
            f"medical_ai_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # TensorBoard
        if self.is_main_process and self.config.get("use_tensorboard", True):
            self.writer = SummaryWriter(
                log_dir=str(self.output_dir / "tensorboard" / self.experiment_name)
            )
        else:
            self.writer = None

        # W&B
        self.use_wandb = self.config.get("use_wandb", False) and self.is_main_process
        if self.use_wandb:
            try:
                import wandb
                wandb.init(
                    project=self.config.get("wandb_project", "medical-ai"),
                    name=self.experiment_name,
                    config=self.config,
                    entity=self.config.get("wandb_entity"),
                )
                self.wandb = wandb
            except ImportError:
                logger.warning("W&B not installed. Skipping W&B logging.")
                self.use_wandb = False
                self.wandb = None
        else:
            self.wandb = None

        # Training state
        self.current_epoch = 0
        self.best_metric = -float("inf")
        self.best_epoch = 0
        self.patience_counter = 0
        self.global_step = 0
        self.training_history = defaultdict(list)

        # Log config
        if self.is_main_process:
            logger.info(f"Device: {self.device}")
            logger.info(f"Mixed precision: {self.use_amp}")
            logger.info(f"Distributed: {self.is_distributed}")
            logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
            logger.info(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    def _merge_config(self, config: Dict) -> Dict:
        """Merge user config with defaults."""
        defaults = {
            "epochs": 100,
            "learning_rate": 1e-4,
            "weight_decay": 1e-4,
            "optimizer": "adamw",
            "scheduler": "cosine_warmup",
            "warmup_epochs": 5,
            "min_lr": 1e-7,
            "mixed_precision": True,
            "gradient_accumulation_steps": 1,
            "max_grad_norm": 1.0,
            "early_stopping_patience": 15,
            "early_stopping_min_delta": 1e-4,
            "early_stopping_metric": "val_auc",
            "save_top_k": 3,
            "save_every_n_epochs": 5,
            "use_ema": True,
            "ema_decay": 0.999,
            "use_tensorboard": True,
            "use_wandb": False,
            "seed": 42,
            "class_weights": None,
            "label_smoothing": 0.0,
        }
        defaults.update(config)
        return defaults

    def _get_loss_function(self) -> nn.Module:
        """Get appropriate loss function for the task."""
        class_weights = self.config.get("class_weights")
        if class_weights is not None and isinstance(class_weights, (list, np.ndarray)):
            class_weights = torch.tensor(class_weights).float().to(self.device)
        elif class_weights is not None:
            class_weights = class_weights.to(self.device)

        label_smoothing = self.config.get("label_smoothing", 0.0)

        if self.task == "multi-label":
            return nn.BCEWithLogitsLoss(pos_weight=class_weights)
        elif self.task == "multi-class":
            return nn.CrossEntropyLoss(
                weight=class_weights,
                label_smoothing=label_smoothing,
            )
        else:  # binary
            if self.num_classes == 1:
                return nn.BCEWithLogitsLoss(pos_weight=class_weights)
            else:
                return nn.CrossEntropyLoss(
                    weight=class_weights,
                    label_smoothing=label_smoothing,
                )

    def _create_optimizer(self) -> torch.optim.Optimizer:
        """Create optimizer with layer-wise learning rates for transfer learning."""
        lr = self.config["learning_rate"]
        wd = self.config["weight_decay"]
        opt_name = self.config["optimizer"]

        # Separate backbone (lower LR) and head (higher LR) parameters
        backbone_params = []
        head_params = []
        other_params = []

        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if "backbone" in name or "features" in name:
                backbone_params.append(param)
            elif "head" in name or "classifier" in name or "fc" in name:
                head_params.append(param)
            else:
                other_params.append(param)

        param_groups = []

        if backbone_params:
            param_groups.append({
                "params": backbone_params,
                "lr": lr * 0.1,  # Lower LR for backbone (transfer learning)
                "weight_decay": wd,
            })
        if head_params:
            param_groups.append({
                "params": head_params,
                "lr": lr,
                "weight_decay": wd,
            })
        if other_params:
            param_groups.append({
                "params": other_params,
                "lr": lr,
                "weight_decay": wd,
            })

        if not param_groups:
            param_groups = [{"params": self.model.parameters(), "lr": lr, "weight_decay": wd}]

        if opt_name == "adamw":
            return torch.optim.AdamW(param_groups, eps=self.config.get("eps", 1e-8))
        elif opt_name == "adam":
            return torch.optim.Adam(param_groups, eps=self.config.get("eps", 1e-8))
        elif opt_name == "sgd":
            return torch.optim.SGD(
                param_groups,
                momentum=0.9,
                nesterov=True,
            )
        elif opt_name == "lamb":
            try:
                from torch_optimizer import Lamb
                return Lamb(param_groups)
            except ImportError:
                logger.warning("torch_optimizer not installed. Falling back to AdamW.")
                return torch.optim.AdamW(param_groups)
        else:
            return torch.optim.AdamW(param_groups)

    def _create_scheduler(self):
        """Create learning rate scheduler."""
        scheduler_name = self.config.get("scheduler", "cosine_warmup")
        total_epochs = self.config["epochs"]
        warmup_epochs = self.config.get("warmup_epochs", 5)
        steps_per_epoch = len(self.train_loader) // self.config.get("gradient_accumulation_steps", 1)

        if scheduler_name == "cosine_warmup":
            # Linear warmup + cosine annealing
            warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
                self.optimizer,
                start_factor=0.1,
                end_factor=1.0,
                total_iters=warmup_epochs * steps_per_epoch,
            )
            cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=(total_epochs - warmup_epochs) * steps_per_epoch,
                eta_min=self.config.get("min_lr", 1e-7),
            )
            return torch.optim.lr_scheduler.SequentialLR(
                self.optimizer,
                schedulers=[warmup_scheduler, cosine_scheduler],
                milestones=[warmup_epochs * steps_per_epoch],
            )
        elif scheduler_name == "cosine":
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=total_epochs * steps_per_epoch,
                eta_min=self.config.get("min_lr", 1e-7),
            )
        elif scheduler_name == "plateau":
            return torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode="max",
                factor=0.5,
                patience=5,
                min_lr=self.config.get("min_lr", 1e-7),
            )
        elif scheduler_name == "onecycle":
            return torch.optim.lr_scheduler.OneCycleLR(
                self.optimizer,
                max_lr=self.config["learning_rate"],
                epochs=total_epochs,
                steps_per_epoch=steps_per_epoch,
                pct_start=warmup_epochs / total_epochs,
            )
        else:
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=total_epochs * steps_per_epoch,
            )

    def set_seed(self):
        """Set random seeds for reproducibility."""
        seed = self.config.get("seed", 42)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        if self.config.get("deterministic", False):
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        self.train_metrics.reset()

        accumulation_steps = self.config.get("gradient_accumulation_steps", 1)
        epoch_loss = 0.0
        batch_count = 0

        for batch_idx, (data, target) in enumerate(self.train_loader):
            data = data.to(self.device, non_blocking=True)
            target = target.to(self.device, non_blocking=True)

            # Handle target shape for different tasks
            if self.task == "multi-label":
                target = target.float()
            elif target.ndim > 1:
                # Squeeze (B,1) to (B,) for CrossEntropyLoss compatibility
                target = target.squeeze(1).long()

            # Forward pass
            with torch.amp.autocast('cuda', enabled=self.use_amp):
                outputs = self.model(data)

                if isinstance(outputs, dict):
                    logits = outputs["logits"]
                else:
                    logits = outputs

                loss = self.criterion(logits, target)
                loss = loss / accumulation_steps

            # Backward pass
            if self.use_amp and self.scaler is not None:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()

            # Gradient accumulation
            if (batch_idx + 1) % accumulation_steps == 0:
                if self.use_amp and self.scaler is not None:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config.get("max_grad_norm", 1.0),
                    )
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config.get("max_grad_norm", 1.0),
                    )
                    self.optimizer.step()
                self.optimizer.zero_grad(set_to_none=True)

                # Update EMA
                if self.ema is not None:
                    self.ema.update()

                # Step scheduler (per-batch schedulers)
                if self.config.get("scheduler") in ("cosine_warmup", "cosine", "onecycle"):
                    self.scheduler.step()

                self.global_step += 1

            # Track metrics
            epoch_loss += loss.item() * accumulation_steps
            batch_count += 1

            with torch.no_grad():
                self.train_metrics.update(logits.detach(), target, loss.item() * accumulation_steps)

            # Log every N steps
            if self.is_main_process and batch_idx % self.config.get("log_every_n_steps", 50) == 0:
                current_lr = self.optimizer.param_groups[-1]["lr"]
                logger.info(
                    f"Epoch {self.current_epoch:3d} | "
                    f"Batch {batch_idx:4d}/{len(self.train_loader):4d} | "
                    f"LR: {current_lr:.2e} | "
                    f"Loss: {loss.item() * accumulation_steps:.4f}"
                )

        # End of epoch metrics
        train_metrics = self.train_metrics.compute()

        if self.is_main_process:
            logger.info(
                f"Epoch {self.current_epoch:3d} | "
                f"Train Loss: {train_metrics.get('loss', 0):.4f} | "
                f"Train Acc: {train_metrics.get('accuracy', 0):.4f} | "
                f"Train AUC: {train_metrics.get('auc_roc', 0):.4f} | "
                f"LR: {self.optimizer.param_groups[-1]['lr']:.2e}"
            )

        return train_metrics

    @torch.no_grad()
    def validate_epoch(self) -> Dict[str, float]:
        """Validate for one epoch."""
        self.model.eval()

        # Use EMA weights for validation if available
        if self.ema is not None:
            self.ema.apply_shadow()

        self.val_metrics.reset()
        epoch_loss = 0.0
        batch_count = 0

        for data, target in self.val_loader:
            data = data.to(self.device, non_blocking=True)
            target = target.to(self.device, non_blocking=True)

            if self.task == "multi-label":
                target = target.float()
            elif target.ndim > 1:
                target = target.squeeze(1).long()

            # No AMP needed for validation, but we use it for consistency
            with torch.amp.autocast('cuda', enabled=self.use_amp):
                outputs = self.model(data)

                if isinstance(outputs, dict):
                    logits = outputs["logits"]
                else:
                    logits = outputs

                loss = self.criterion(logits, target)

            epoch_loss += loss.item()
            batch_count += 1

            self.val_metrics.update(logits, target, loss.item())

        # Restore original weights
        if self.ema is not None:
            self.ema.restore()

        metrics = self.val_metrics.compute()

        if self.is_main_process:
            logger.info(
                f"Epoch {self.current_epoch:3d} | "
                f"Val Loss: {metrics.get('loss', 0):.4f} | "
                f"Val Acc: {metrics.get('accuracy', 0):.4f} | "
                f"Val AUC: {metrics.get('auc_roc', 0):.4f} | "
                f"Val F1: {metrics.get('f1_score', 0):.4f}"
            )

        return metrics

    def _save_checkpoint(self, metrics: Dict[str, float], is_best: bool = False):
        """Save model checkpoint."""
        save_dir = Path(self.config.get("output_dir", "./trained_models")) / self.experiment_name
        save_dir.mkdir(parents=True, exist_ok=True)

        model_to_save = self.model.module if self.is_distributed else self.model

        checkpoint = {
            "epoch": self.current_epoch,
            "model_state_dict": model_to_save.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "scaler_state_dict": self.scaler.state_dict() if self.scaler is not None else {},
            "metrics": metrics,
            "config": self.config,
            "best_metric": self.best_metric,
            "task": self.task,
            "num_classes": self.num_classes,
        }

        if self.ema is not None:
            checkpoint["ema_shadow"] = self.ema.shadow

        # Save regular checkpoint
        if self.current_epoch % self.config.get("save_every_n_epochs", 5) == 0:
            path = save_dir / f"checkpoint_epoch_{self.current_epoch}.pt"
            torch.save(checkpoint, path)
            logger.info(f"Checkpoint saved: {path}")

        # Save best model
        if is_best:
            best_path = save_dir / "best_model.pt"
            torch.save(checkpoint, best_path)

            # Also save just the model weights for deployment
            model_path = save_dir / "best_model_weights.pth"
            torch.save(model_to_save.state_dict(), model_path)

            # Save config alongside
            config_path = save_dir / "model_config.json"
            with open(config_path, "w") as f:
                json.dump({
                    "task": self.task,
                    "num_classes": self.num_classes,
                    "metrics": {k: float(v) if isinstance(v, (np.floating,)) else v for k, v in metrics.items()},
                    "config": {k: str(v) if not isinstance(v, (int, float, bool, list, dict, type(None))) else v
                               for k, v in self.config.items()},
                }, f, indent=2)

            logger.info(f"Best model saved: {best_path} (metric: {self.best_metric:.4f})")

    def _log_metrics(
        self,
        train_metrics: Dict[str, float],
        val_metrics: Dict[str, float],
    ):
        """Log metrics to TensorBoard and W&B."""
        epoch = self.current_epoch

        # TensorBoard
        if self.writer is not None:
            for key, value in train_metrics.items():
                self.writer.add_scalar(f"train/{key}", value, epoch)
            for key, value in val_metrics.items():
                self.writer.add_scalar(f"val/{key}", value, epoch)
            self.writer.add_scalar(
                "lr",
                self.optimizer.param_groups[-1]["lr"],
                epoch,
            )

        # W&B
        if self.use_wandb and self.wandb is not None:
            log_dict = {}
            for key, value in train_metrics.items():
                log_dict[f"train/{key}"] = value
            for key, value in val_metrics.items():
                log_dict[f"val/{key}"] = value
            log_dict["lr"] = self.optimizer.param_groups[-1]["lr"]
            log_dict["epoch"] = epoch
            self.wandb.log(log_dict, step=epoch)

    def _get_monitor_metric(self, metrics: Dict[str, float]) -> float:
        """Extract the metric used for model selection."""
        metric_name = self.config.get("early_stopping_metric", "val_auc")
        # Try exact match first
        if metric_name in metrics:
            return metrics[metric_name]
        # Try without val_ prefix
        key = metric_name.replace("val_", "")
        if key in metrics:
            return metrics[key]
        # Default to auc_roc
        return metrics.get("auc_roc", metrics.get("accuracy", 0.0))

    def fit(self) -> Dict[str, Any]:
        """Main training loop."""
        self.set_seed()

        epochs = self.config["epochs"]
        start_time = time.time()

        logger.info(f"Starting training: {epochs} epochs, {self.experiment_name}")
        logger.info(f"Output directory: {self.output_dir / self.experiment_name}")

        for epoch in range(1, epochs + 1):
            self.current_epoch = epoch
            epoch_start = time.time()

            # Training phase
            train_metrics = self.train_epoch()

            # Validation phase
            val_metrics = self.validate_epoch()

            # Step epoch-based scheduler
            if self.config.get("scheduler") == "plateau":
                monitor = self._get_monitor_metric(val_metrics)
                self.scheduler.step(monitor)

            # Logging
            if self.is_main_process:
                self._log_metrics(train_metrics, val_metrics)

                # Track history
                self.training_history["train_loss"].append(train_metrics.get("loss", 0))
                self.training_history["val_loss"].append(val_metrics.get("loss", 0))
                self.training_history["val_auc"].append(val_metrics.get("auc_roc", 0))
                self.training_history["val_accuracy"].append(val_metrics.get("accuracy", 0))
                self.training_history["lr"].append(self.optimizer.param_groups[-1]["lr"])

            # Model selection
            current_metric = self._get_monitor_metric(val_metrics)
            is_best = current_metric > self.best_metric + self.config.get("early_stopping_min_delta", 1e-4)

            if is_best:
                self.best_metric = current_metric
                self.best_epoch = epoch
                self.patience_counter = 0
                if self.is_main_process:
                    logger.info(f"[BEST] New best model! {self.config['early_stopping_metric']}: {current_metric:.4f}")
                    self._save_checkpoint(val_metrics, is_best=True)
            else:
                self.patience_counter += 1

            # Save periodic checkpoint
            if self.is_main_process and epoch % self.config.get("save_every_n_epochs", 5) == 0:
                self._save_checkpoint(val_metrics)

            # Early stopping
            patience = self.config.get("early_stopping_patience", 15)
            if self.patience_counter >= patience:
                logger.info(f"Early stopping triggered after {epoch} epochs "
                           f"(no improvement for {patience} epochs)")
                break

            # Progress bar
            epoch_time = time.time() - epoch_start
            if self.is_main_process and epoch % 5 == 0:
                remaining = (epoch_time) * (epochs - epoch) / 60
                logger.info(f"[TIME] Epoch {epoch} took {epoch_time:.1f}s | ~{remaining:.1f} min remaining")

        # Training complete
        total_time = time.time() - start_time
        logger.info(f"Training complete! Total time: {total_time/60:.1f} min")
        logger.info(f"Best {self.config['early_stopping_metric']}: {self.best_metric:.4f} at epoch {self.best_epoch}")

        # Final evaluation on test set
        test_metrics = {}
        if self.test_loader is not None and self.is_main_process:
            logger.info("Running final evaluation on test set...")
            test_metrics = self.evaluate()

        if self.is_main_process:
            # Save training history
            history_path = self.output_dir / self.experiment_name / "training_history.json"
            history_path.parent.mkdir(parents=True, exist_ok=True)
            with open(history_path, "w") as f:
                json.dump(self.training_history, f, indent=2)

            # Close loggers
            if self.writer is not None:
                self.writer.close()
            if self.use_wandb and self.wandb is not None:
                self.wandb.finish()

        return {
            "best_metric": self.best_metric,
            "best_epoch": self.best_epoch,
            "train_metrics": train_metrics,
            "val_metrics": val_metrics,
            "test_metrics": test_metrics,
            "total_time_minutes": total_time / 60,
        }

    @torch.no_grad()
    def evaluate(self) -> Dict[str, float]:
        """Evaluate on test set."""
        self.model.eval()

        if self.ema is not None:
            self.ema.apply_shadow()

        test_metrics = MetricsTracker(self.num_classes, self.task)
        total_loss = 0.0

        for data, target in self.test_loader:
            data = data.to(self.device, non_blocking=True)
            target = target.to(self.device, non_blocking=True)

            if self.task == "multi-label":
                target = target.float()
            elif target.ndim > 1:
                target = target.squeeze(1).long()

            with torch.amp.autocast('cuda', enabled=self.use_amp):
                outputs = self.model(data)
                logits = outputs["logits"] if isinstance(outputs, dict) else outputs
                loss = self.criterion(logits, target)

            total_loss += loss.item()
            test_metrics.update(logits, target, loss.item())

        if self.ema is not None:
            self.ema.restore()

        metrics = test_metrics.compute()

        logger.info("=" * 60)
        logger.info("Test Set Results:")
        logger.info(f"  Loss:      {metrics.get('loss', 0):.4f}")
        logger.info(f"  Accuracy:  {metrics.get('accuracy', 0):.4f}")
        logger.info(f"  AUC-ROC:   {metrics.get('auc_roc', 0):.4f}")
        logger.info(f"  F1 Score:  {metrics.get('f1_score', 0):.4f}")
        logger.info(f"  Precision: {metrics.get('precision', 0):.4f}")
        logger.info(f"  Recall:    {metrics.get('recall', 0):.4f}")
        logger.info(f"  Kappa:     {metrics.get('cohen_kappa', 0):.4f}")
        if "sensitivity" in metrics:
            logger.info(f"  Sensitivity: {metrics['sensitivity']:.4f}")
            logger.info(f"  Specificity: {metrics['specificity']:.4f}")
        logger.info("=" * 60)

        return metrics
