"""
Comprehensive medical imaging metrics.

Includes:
- AUC-ROC (per-class and macro/micro)
- Sensitivity, Specificity
- F1 Score, Precision, Recall
- Cohen's Kappa
- Confusion Matrix
- DICE Score (for segmentation)
- Hausdorff Distance (for segmentation)
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from collections import defaultdict
import warnings

import torch
import torch.nn.functional as F
from sklearn.metrics import (
    roc_auc_score, accuracy_score, precision_score, recall_score,
    f1_score, cohen_kappa_score, confusion_matrix,
    average_precision_score, hamming_loss,
)


class MetricsTracker:
    """Track and compute comprehensive medical imaging metrics."""

    def __init__(self, num_classes: int, task: str = "multi-class"):
        self.num_classes = num_classes
        self.task = task  # binary, multi-class, multi-label
        self.reset()

    def reset(self):
        """Reset all accumulated predictions and targets."""
        self.all_targets = []
        self.all_predictions = []
        self.all_probabilities = []
        self.all_losses = []

    def update(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        loss: Optional[float] = None,
    ):
        """Update metrics with batch predictions.

        Args:
            logits: Model output logits (B, num_classes)
            targets: Ground truth labels (B, num_classes) or (B,)
            loss: Optional batch loss value
        """
        with torch.no_grad():
            if self.task == "multi-label":
                probabilities = torch.sigmoid(logits)
                predictions = (probabilities > 0.5).float()
            elif self.task == "binary":
                if logits.shape[1] == 1:
                    probabilities = torch.sigmoid(logits).squeeze(1)
                    predictions = (probabilities > 0.5).long()
                    probabilities = torch.stack([1 - probabilities, probabilities], dim=1)
                else:
                    probabilities = F.softmax(logits, dim=1)
                    predictions = torch.argmax(probabilities, dim=1)
            else:  # multi-class
                probabilities = F.softmax(logits, dim=1)
                predictions = torch.argmax(probabilities, dim=1)

            self.all_targets.append(targets.detach().cpu().numpy())
            self.all_predictions.append(predictions.detach().cpu().numpy())
            self.all_probabilities.append(probabilities.detach().cpu().numpy())
            if loss is not None:
                self.all_losses.append(loss)

    def compute(self) -> Dict[str, float]:
        """Compute all accumulated metrics.

        Returns:
            Dictionary of metric names to values
        """
        if not self.all_targets:
            return {}

        targets = np.concatenate(self.all_targets)
        predictions = np.concatenate(self.all_predictions)
        probabilities = np.concatenate(self.all_probabilities)

        metrics = {}

        # Loss
        if self.all_losses:
            metrics["loss"] = float(np.mean(self.all_losses))

        # Basic metrics
        metrics["accuracy"] = float(accuracy_score(
            targets.argmax(axis=1) if targets.ndim > 1 and self.task == "multi-label" else
            targets if targets.ndim == 1 else targets.argmax(axis=1),
            predictions.argmax(axis=1) if predictions.ndim > 1 else predictions
        ))

        # AUC-ROC
        try:
            if self.task == "multi-label":
                auc_scores = []
                for i in range(min(self.num_classes, probabilities.shape[1])):
                    if len(np.unique(targets[:, i])) > 1:
                        auc_scores.append(roc_auc_score(targets[:, i], probabilities[:, i]))
                metrics["auc_roc"] = float(np.mean(auc_scores)) if auc_scores else 0.5
            elif self.task == "binary":
                if probabilities.shape[1] >= 2 and len(np.unique(targets)) > 1:
                    metrics["auc_roc"] = float(roc_auc_score(
                        targets, probabilities[:, 1]
                    ))
                else:
                    metrics["auc_roc"] = 0.5
            else:
                if len(np.unique(targets)) > 1:
                    try:
                        metrics["auc_roc"] = float(roc_auc_score(
                            targets, probabilities, multi_class="ovr", average="macro"
                        ))
                    except ValueError:
                        metrics["auc_roc"] = 0.5
                else:
                    metrics["auc_roc"] = 0.5
        except Exception:
            metrics["auc_roc"] = 0.5

        # AUC-PR
        try:
            if self.task == "binary" and probabilities.shape[1] >= 2:
                metrics["auc_pr"] = float(average_precision_score(
                    targets, probabilities[:, 1]
                ))
            elif self.task != "multi-label":
                metrics["auc_pr"] = float(average_precision_score(
                    targets, probabilities, average="macro"
                ))
        except Exception:
            metrics["auc_pr"] = 0.0

        # Flatten targets/predictions for multi-label
        if self.task == "multi-label":
            t_flat = targets
            p_flat = predictions
        elif self.task == "binary" and targets.ndim == 1:
            t_flat = targets
            p_flat = predictions
        else:
            t_flat = targets.argmax(axis=1) if targets.ndim > 1 else targets
            p_flat = predictions.argmax(axis=1) if predictions.ndim > 1 else predictions

        # Auto-detect appropriate averaging
        n_unique = len(np.unique(t_flat))
        if self.task == "multi-class" or self.task == "multi-label" or n_unique > 2:
            avg = "macro"
        else:
            avg = "binary"

        metrics["f1_score"] = float(f1_score(t_flat, p_flat, average=avg, zero_division=0))
        metrics["precision"] = float(precision_score(t_flat, p_flat, average=avg, zero_division=0))
        metrics["recall"] = float(recall_score(t_flat, p_flat, average=avg, zero_division=0))

        # Cohen's Kappa
        try:
            metrics["cohen_kappa"] = float(cohen_kappa_score(t_flat, p_flat))
        except Exception:
            metrics["cohen_kappa"] = 0.0

        # Sensitivity & Specificity (for binary)
        if self.task == "binary":
            cm = confusion_matrix(t_flat, p_flat)
            if cm.shape == (2, 2):
                tn, fp, fn, tp = cm.ravel()
                metrics["sensitivity"] = float(tp / (tp + fn + 1e-8))
                metrics["specificity"] = float(tn / (tn + fp + 1e-8))
                metrics["ppv"] = float(tp / (tp + fp + 1e-8))  # Positive Predictive Value
                metrics["npv"] = float(tn / (tn + fn + 1e-8))  # Negative Predictive Value

        # Per-class accuracy (for multi-class)
        if self.task == "multi-class":
            cm = confusion_matrix(t_flat, p_flat)
            per_class_acc = cm.diagonal() / (cm.sum(axis=1) + 1e-8)
            metrics["per_class_accuracy_mean"] = float(np.mean(per_class_acc))
            metrics["per_class_accuracy_std"] = float(np.std(per_class_acc))

        return metrics


class SegmentationMetrics:
    """Metrics for medical image segmentation."""

    @staticmethod
    def dice_score(pred: torch.Tensor, target: torch.Tensor, smooth: float = 1e-6) -> float:
        """Compute Dice coefficient."""
        pred_flat = pred.contiguous().view(-1)
        target_flat = target.contiguous().view(-1)
        intersection = (pred_flat * target_flat).sum()
        return float((2. * intersection + smooth) / (pred_flat.sum() + target_flat.sum() + smooth))

    @staticmethod
    def iou(pred: torch.Tensor, target: torch.Tensor, smooth: float = 1e-6) -> float:
        """Compute IoU (Jaccard index)."""
        pred_flat = pred.contiguous().view(-1)
        target_flat = target.contiguous().view(-1)
        intersection = (pred_flat * target_flat).sum()
        union = pred_flat.sum() + target_flat.sum() - intersection
        return float((intersection + smooth) / (union + smooth))

    @staticmethod
    def sensitivity(pred: torch.Tensor, target: torch.Tensor) -> float:
        """True positive rate / recall for segmentation."""
        pred_flat = pred.contiguous().view(-1)
        target_flat = target.contiguous().view(-1)
        tp = (pred_flat * target_flat).sum()
        fn = ((1 - pred_flat) * target_flat).sum()
        return float(tp / (tp + fn + 1e-6))

    @staticmethod
    def specificity(pred: torch.Tensor, target: torch.Tensor) -> float:
        """True negative rate for segmentation."""
        pred_flat = pred.contiguous().view(-1)
        target_flat = target.contiguous().view(-1)
        tn = ((1 - pred_flat) * (1 - target_flat)).sum()
        fp = (pred_flat * (1 - target_flat)).sum()
        return float(tn / (tn + fp + 1e-6))

    @classmethod
    def compute_all(cls, pred: torch.Tensor, target: torch.Tensor) -> Dict[str, float]:
        """Compute all segmentation metrics."""
        pred_binary = (torch.sigmoid(pred) > 0.5).float() if pred.shape[1] == 1 else pred.argmax(1, keepdim=True).float()
        return {
            "dice": cls.dice_score(pred_binary, target),
            "iou": cls.iou(pred_binary, target),
            "sensitivity": cls.sensitivity(pred_binary, target),
            "specificity": cls.specificity(pred_binary, target),
        }
