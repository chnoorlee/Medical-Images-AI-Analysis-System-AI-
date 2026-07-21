"""
Medical image dataset loaders with support for multiple data sources:
- MedMNIST (lightweight benchmark)
- Custom DICOM/PNG datasets
- NIH Chest X-ray
"""

import os
import time
import glob
import numpy as np
from pathlib import Path
from typing import Optional, Callable, Tuple, List, Dict, Any
import logging

import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms
import cv2
from PIL import Image

logger = logging.getLogger(__name__)


class MedMNISTDataset(Dataset):
    """Dataset wrapper for MedMNIST datasets with robust download handling.

    MedMNIST provides standardized medical image datasets in 28x28, 64x64, 128x128, and 224x224 sizes.
    Uses smaller size (28) for fast initial download, with images resized in our transform pipeline.
    Available datasets include:
    - pathmnist (colon pathology, 9 classes)
    - chestmnist (chest x-ray, 14 classes)
    - dermamnist (skin lesions, 7 classes)
    - octmnist (retina OCT, 4 classes)
    - pneumoniamnist (chest x-ray pneumonia, 2 classes)
    - retinamnist (retina, 5 classes)
    - breastmnist (breast ultrasound, 2 classes)
    - bloodmnist (blood cells, 8 classes)
    - tissuemnist (kidney tissue, 8 classes)
    - organamnist (abdominal CT organs, 11 classes)
    - organcmnist (abdominal CT organs, 11 classes)
    - organsmnist (abdominal CT organs, 11 classes)
    """

    DATASET_INFO = {
        "pathmnist": {"classes": 9, "task": "multi-class", "modality": "pathology"},
        "chestmnist": {"classes": 14, "task": "multi-label", "modality": "chest_xray"},
        "dermamnist": {"classes": 7, "task": "multi-class", "modality": "dermatology"},
        "octmnist": {"classes": 4, "task": "multi-class", "modality": "retina_oct"},
        "pneumoniamnist": {"classes": 2, "task": "binary", "modality": "chest_xray"},
        "retinamnist": {"classes": 5, "task": "multi-class", "modality": "retina"},
        "breastmnist": {"classes": 2, "task": "binary", "modality": "ultrasound"},
        "bloodmnist": {"classes": 8, "task": "multi-class", "modality": "microscopy"},
        "tissuemnist": {"classes": 8, "task": "multi-class", "modality": "pathology"},
        "organamnist": {"classes": 11, "task": "multi-class", "modality": "ct"},
        "organcmnist": {"classes": 11, "task": "multi-class", "modality": "ct"},
        "organsmnist": {"classes": 11, "task": "multi-class", "modality": "ct"},
    }

    def __init__(
        self,
        dataset_name: str,
        split: str = "train",
        size: int = 224,
        transform: Optional[Callable] = None,
        data_dir: str = "./data",
        download: bool = True,
        as_rgb: bool = True,
    ):
        super().__init__()
        self.dataset_name = dataset_name
        self.split = split
        self.size = size
        self.as_rgb = as_rgb
        self.data_dir = Path(data_dir)

        try:
            import medmnist
            from medmnist import INFO

            dataset_cls = getattr(medmnist, INFO[dataset_name]['python_class'])
            self.dataset_info = INFO[dataset_name]

            # Use size=28 for download (small ~3MB files, fast), then resize in transforms
            max_retries = 5
            last_error = None

            for attempt in range(max_retries):
                try:
                    self._dataset = dataset_cls(
                        split=split,
                        transform=transform,
                        download=download,
                        size=28,  # Always download 28x28 for speed, resize in pipeline
                        root=data_dir,
                        as_rgb=as_rgb,
                    )
                    break
                except (RuntimeError, Exception) as e:
                    last_error = e
                    if attempt < max_retries - 1:
                        wait = (attempt + 1) * 10
                        logger.warning(
                            f"Download attempt {attempt + 1}/{max_retries} failed for {dataset_name}. "
                            f"Retrying in {wait}s... (Error: {str(e)[:100]})"
                        )
                        # Clean corrupt file
                        for pattern in [f"{dataset_name}*.npz", f"{dataset_name}*.npz.*"]:
                            for f in glob.glob(str(self.data_dir / pattern)):
                                try:
                                    os.remove(f)
                                except Exception:
                                    pass
                        time.sleep(wait)
                    else:
                        raise RuntimeError(
                            f"Failed to download {dataset_name} after {max_retries} attempts. "
                            f"Last error: {last_error}"
                        ) from last_error

            self.num_classes = len(INFO[dataset_name]['label'])
            self.task = self.DATASET_INFO.get(dataset_name, {}).get("task", "multi-class")
            self.modality = self.DATASET_INFO.get(dataset_name, {}).get("modality", "unknown")

            logger.info(
                f"Loaded {dataset_name} ({split}): {len(self._dataset)} samples, "
                f"{self.num_classes} classes, task={self.task}, modality={self.modality}"
            )

        except ImportError:
            raise ImportError(
                "medmnist package is required. Install with: pip install medmnist"
            )

    def __len__(self) -> int:
        return len(self._dataset)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        image, label = self._dataset[idx]
        return image, label

    def get_class_weights(self) -> torch.Tensor:
        """Compute class weights for balanced loss."""
        labels = []
        for i in range(len(self)):
            _, label = self[i]
            if self.task == "multi-label":
                labels.append(label.numpy())
            else:
                labels.append(label.item() if isinstance(label, torch.Tensor) else label)
        labels = [l.item() if isinstance(l, (torch.Tensor, np.ndarray)) else l for l in labels]

        labels = np.array(labels)

        if self.task == "multi-label":
            # Per-class positive counts
            pos_counts = labels.sum(axis=0)
            neg_counts = len(labels) - pos_counts
            weights = neg_counts / (pos_counts + 1e-6)
            return torch.from_numpy(weights).float()
        else:
            # Per-class counts
            class_counts = np.bincount(labels.flatten(), minlength=self.num_classes)
            total = len(labels)
            weights = total / (self.num_classes * (class_counts + 1e-6))
            return torch.from_numpy(weights).float()


class ImageFolderDataset(Dataset):
    """Dataset for image folders organized by class subdirectories."""

    VALID_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.dcm', '.dicom'}

    def __init__(
        self,
        root_dir: str,
        transform: Optional[Callable] = None,
        is_dicom: bool = False,
    ):
        super().__init__()
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.is_dicom = is_dicom

        self.classes = sorted([
            d.name for d in self.root_dir.iterdir()
            if d.is_dir() and not d.name.startswith('.')
        ])
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}

        self.samples = []
        for cls_name in self.classes:
            cls_dir = self.root_dir / cls_name
            for ext in self.VALID_EXTENSIONS:
                for file_path in cls_dir.glob(f"*{ext}"):
                    self.samples.append((str(file_path), self.class_to_idx[cls_name]))
                for file_path in cls_dir.glob(f"*{ext.upper()}"):
                    self.samples.append((str(file_path), self.class_to_idx[cls_name]))

        logger.info(f"Loaded {len(self.samples)} images from {root_dir}, {len(self.classes)} classes")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        path, label = self.samples[idx]

        if path.lower().endswith(('.dcm', '.dicom')):
            import pydicom
            dcm = pydicom.dcmread(path)
            image = dcm.pixel_array.astype(np.float32)
            image = (image - image.min()) / (image.max() - image.min() + 1e-6)
            image = (image * 255).astype(np.uint8)
            image = Image.fromarray(image)
        else:
            image = Image.open(path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image, label


def get_medical_transforms(
    input_size: int = 224,
    is_train: bool = True,
    mean: List[float] = None,
    std: List[float] = None,
    augmentation_intensity: str = "medium",
    modality: str = "xray",
) -> transforms.Compose:
    """Get standardized transforms for medical images.

    Args:
        input_size: Target image size
        is_train: Whether this is for training (enables augmentations)
        mean: Normalization mean
        std: Normalization std
        augmentation_intensity: low/medium/high
        modality: Image modality (xray, ct, mri, pathology, etc.)
    """
    if mean is None:
        mean = [0.485, 0.456, 0.406]
    if std is None:
        std = [0.229, 0.224, 0.225]

    if is_train:
        # Define augmentation strength
        intensity_configs = {
            "low": {
                "rotation": 5,
                "translate": (0.02, 0.02),
                "brightness": 0.05,
                "contrast": 0.05,
            },
            "medium": {
                "rotation": 15,
                "translate": (0.05, 0.05),
                "brightness": 0.15,
                "contrast": 0.15,
            },
            "high": {
                "rotation": 30,
                "translate": (0.1, 0.1),
                "brightness": 0.3,
                "contrast": 0.3,
            },
        }
        cfg = intensity_configs.get(augmentation_intensity, intensity_configs["medium"])

        train_transforms = [
            transforms.Resize((input_size, input_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.1),
            transforms.RandomRotation(degrees=cfg["rotation"]),
            transforms.RandomAffine(
                degrees=0,
                translate=cfg["translate"],
                scale=(0.95, 1.05),
            ),
            transforms.ColorJitter(
                brightness=cfg["brightness"],
                contrast=cfg["contrast"],
            ),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]

        # Modality-specific augmentations
        if modality in ("xray", "ct", "mri"):
            train_transforms.insert(-2, transforms.RandomEqualize(p=0.2))

        return transforms.Compose(train_transforms)
    else:
        return transforms.Compose([
            transforms.Resize((input_size, input_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])


def create_dataloaders(
    dataset_name: str,
    data_dir: str = "./data",
    batch_size: int = 32,
    val_batch_size: int = 64,
    num_workers: int = 8,
    input_size: int = 224,
    augmentation_intensity: str = "medium",
    pin_memory: bool = True,
    use_weighted_sampler: bool = False,
) -> Tuple[DataLoader, DataLoader, DataLoader, int, str]:
    """Create train, validation, and test dataloaders.

    Args:
        dataset_name: Name of the MedMNIST dataset or 'imagefolder'
        data_dir: Directory for data storage
        batch_size: Training batch size
        val_batch_size: Validation batch size
        num_workers: Number of data loading workers
        input_size: Target image size
        augmentation_intensity: Augmentation intensity
        pin_memory: Pin memory for faster GPU transfer
        use_weighted_sampler: Use weighted sampling for class balance

    Returns:
        train_loader, val_loader, test_loader, num_classes, task_type
    """
    # Determine dataset info and transforms
    dataset_info = MedMNISTDataset.DATASET_INFO
    if dataset_name in dataset_info:
        info = dataset_info[dataset_name]
        num_classes = info["classes"]
        task = info["task"]
        modality = info["modality"]
    else:
        num_classes = 2
        task = "binary"
        modality = "xray"

    train_transform = get_medical_transforms(
        input_size=input_size,
        is_train=True,
        augmentation_intensity=augmentation_intensity,
        modality=modality,
    )
    eval_transform = get_medical_transforms(
        input_size=input_size,
        is_train=False,
        modality=modality,
    )

    # Create datasets
    train_dataset = MedMNISTDataset(
        dataset_name=dataset_name,
        split="train",
        size=input_size,
        transform=train_transform,
        data_dir=data_dir,
        download=True,
    )
    val_dataset = MedMNISTDataset(
        dataset_name=dataset_name,
        split="val",
        size=input_size,
        transform=eval_transform,
        data_dir=data_dir,
        download=True,
    )
    test_dataset = MedMNISTDataset(
        dataset_name=dataset_name,
        split="test",
        size=input_size,
        transform=eval_transform,
        data_dir=data_dir,
        download=True,
    )

    # DataLoader kwargs
    train_kwargs = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "persistent_workers": num_workers > 0,
        "drop_last": True,
    }
    if num_workers > 0:
        train_kwargs["prefetch_factor"] = 2

    eval_kwargs = {
        "batch_size": val_batch_size,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "persistent_workers": num_workers > 0,
    }

    # Weighted sampler for class imbalance
    if use_weighted_sampler and task != "multi-label":
        class_weights = train_dataset.get_class_weights()
        labels = []
        for i in range(len(train_dataset)):
            _, label = train_dataset[i]
            labels.append(label.item() if isinstance(label, torch.Tensor) else label)
        labels = [l.item() if isinstance(l, (torch.Tensor, np.ndarray)) else l for l in labels]
        sample_weights = class_weights[torch.tensor(labels)]
        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True,
        )
        train_loader = DataLoader(train_dataset, sampler=sampler, **train_kwargs)
    else:
        train_loader = DataLoader(train_dataset, shuffle=True, **train_kwargs)

    val_loader = DataLoader(val_dataset, shuffle=False, **eval_kwargs)
    test_loader = DataLoader(test_dataset, shuffle=False, **eval_kwargs)

    logger.info(
        f"Created dataloaders: train={len(train_loader)} batches, "
        f"val={len(val_loader)} batches, test={len(test_loader)} batches"
    )

    return train_loader, val_loader, test_loader, num_classes, task
