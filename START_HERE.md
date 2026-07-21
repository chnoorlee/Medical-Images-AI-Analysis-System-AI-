# Medical AI — Training Instructions

## 第一步：安装依赖 / Step 1: Install Dependencies

```bash
cd C:\Users\yf\Desktop\Med

# Install PyTorch (pick ONE):
# For your RTX 5090 — needs PyTorch Nightly for Blackwell (sm_120):
pip install --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/cu130

# If that fails, use stable CPU-only:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Install remaining dependencies:
pip install medmnist scikit-learn tensorboard huggingface_hub tqdm
```

## 第二步：验证环境 / Step 2: Verify

```bash
python -c "import torch; print('GPU:', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'); from backend.training.models import create_model; m=create_model('resnet50', 2); print('Model OK')"
```

## 第三步：开始训练 / Step 3: Train

```bash
# Full training (all 9 models, takes 1-4 hours on GPU, 8-24h on CPU):
python run_training.py

# Or train individual models:
python -c "
from scripts.train_all_models import train_task, TASKS
train_task(TASKS[0])  # Train pneumonia first
"
```

## 第四步：上传到 HuggingFace / Step 4: Upload

```bash
set HF_TOKEN=your_token_here
python huggingface/upload_model.py --upload_all
```

## 项目结构 / Project Structure

```
Med/
├── run_training.py              ← 🚀 MAIN: Run this to train everything
├── scripts/
│   ├── train_all_models.py      ← Individual model training
│   └── validate_training.py     ← Quick 3-epoch validation
├── backend/training/            ← Core training pipeline
│   ├── models/__init__.py       ← 9 architectures + UNet++
│   ├── datasets.py              ← MedMNIST + DICOM loaders
│   ├── trainer.py               ← AMP/EMA/Cosine/ES trainer
│   ├── metrics.py               ← AUC/Sensitivity/Kappa
│   └── config.py                ← Pre-configured tasks
├── huggingface/
│   └── upload_model.py          ← HF Hub upload with model cards
├── trained_models/              ← Output directory
└── data/                        ← MedMNIST datasets
```

## 训练的模型 / Models Being Trained

| # | Model | Dataset | Type | Classes |
|---|-------|---------|------|---------|
| 1 | DenseNet121 | PneumoniaMNIST | Binary | 2 |
| 2 | ResNet50 | BreastMNIST | Binary | 2 |
| 3 | EfficientNet-B0 | BloodMNIST | Multi-class | 8 |
| 4 | ResNet50 | OCTMNIST | Multi-class | 4 |
| 5 | ResNet50 | RetinaMNIST | Multi-class | 5 |
| 6 | DenseNet121 | TissueMNIST | Multi-class | 8 |
| 7 | EfficientNet-B0 | DermaMNIST | Multi-class | 7 |
| 8 | DenseNet121 | PathMNIST | Multi-class | 9 |
| 9 | DenseNet121 | ChestMNIST | Multi-label | 14 |

## 技术特性 / Technical Features

- **Mixed Precision (AMP)** — 2-3x faster on GPU
- **Cosine Annealing + Linear Warmup** — Optimal learning rate schedule
- **EMA (Exponential Moving Average)** — More stable inference
- **Gradient Accumulation** — Larger effective batch sizes
- **Early Stopping** — Prevents overfitting
- **Layer-wise Learning Rates** — Transfer learning (backbone LR ×0.1)
- **GeM Pooling + CBAM Attention** — Better features for medical images
- **MedMNIST with auto-retry** — Robust dataset download
