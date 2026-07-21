# Medical AI - 完整训练指南

## 当前状态 - 一切就绪，只需要正确的 PyTorch

### 环境
- **GPU**: NVIDIA RTX 5090 32GB (Blackwell, sm_120)
- **CUDA**: 13.1 (driver 591.86)
- **Python**: 3.12.10

### 需要正确的 PyTorch
你的 RTX 5090 是 Blackwell 架构 (compute capability 12.0)，需要 PyTorch CUDA 13.0+ 版本：

```bash
# 安装支持 RTX 5090 的 PyTorch (CUDA 13.0):
pip install --force-reinstall torch torchvision --index-url https://download.pytorch.org/whl/cu130

# 验证 GPU 可用:
python -c "import torch; t=torch.randn(2,3,224,224,device='cuda'); m=torch.nn.Conv2d(3,64,3).cuda(); print('GPU OK:', m(t).shape)"
```

### 安装其他依赖
```bash
pip install medmnist scikit-learn tensorboard huggingface_hub tqdm
```

### 开始训练
```bash
cd C:\Users\yf\Desktop\Med

# 方式 1: 完整训练所有 9 个模型
python run_training.py

# 方式 2: 先验证（3 个 epoch 测试）
python scripts/validate_training.py

# 方式 3: 单个模型
python -c "
from scripts.train_all_models import train_task, TASKS
train_task(TASKS[0])  # pneumonia
"
```

### 上传到 HuggingFace
```bash
set HF_TOKEN=hf_your_token_here
python huggingface/upload_model.py --upload_all
```

## 已构建的代码

| 文件 | 内容 |
|------|------|
| `backend/training/models/__init__.py` | ResNet/DenseNet/EfficientNet/ConvNeXt + UNet++ + Gem Pooling + CBAM Attention |
| `backend/training/datasets.py` | 12 个 MedMNIST 数据集 + 自动重试下载 |
| `backend/training/trainer.py` | AMP + EMA + Cosine Warmup + Early Stopping + Layer-wise LR |
| `backend/training/metrics.py` | AUC-ROC/PR + Sensitivity/Specificity + F1/Kappa |
| `backend/training/config.py` | 7 个医学任务的预设配置 |
| `scripts/train_all_models.py` | 9 模型训练队列 |
| `huggingface/upload_model.py` | HF Hub 上传 + 模型卡片 |
| `run_training.py` | 一键完整训练 |
| `START_HERE.md` | 快速开始 |

## 训练队列 (9 模型)

| # | 模型 | 数据集 | 类别 | 类型 |
|---|------|--------|------|------|
| 1 | DenseNet121 | PneumoniaMNIST | 2 | 二分类 |
| 2 | ResNet50 | BreastMNIST | 2 | 二分类 |
| 3 | EfficientNet-B0 | BloodMNIST | 8 | 多分类 |
| 4 | ResNet50 | OCTMNIST | 4 | 多分类 |
| 5 | ResNet50 | RetinaMNIST | 5 | 多分类 |
| 6 | DenseNet121 | TissueMNIST | 8 | 多分类 |
| 7 | EfficientNet-B0 | DermaMNIST | 7 | 多分类 |
| 8 | DenseNet121 | PathMNIST | 9 | 多分类 |
| 9 | DenseNet121 | ChestMNIST | 14 | 多标签 |

## GPU 训练速度预估

RTX 5090 每个模型耗时:
- 小模型 (2-5 classes): ~3-5 分钟
- 中模型 (7-8 classes): ~5-8 分钟
- 大模型 (14 classes): ~10-15 分钟

**总计: 约 1-2 小时完成全部 9 个模型**

CPU 训练慢 20-30 倍，不推荐。
