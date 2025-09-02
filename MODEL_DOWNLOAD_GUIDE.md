# 医疗AI模型下载指南

## 模型存放位置

根据系统配置，模型文件应存放在以下位置：

### 1. 主要模型存储路径
- **配置路径**: `./models` (在项目根目录)
- **上传路径**: `./uploads/models` (用于用户上传的模型)
- **临时下载路径**: `./temp_models` (系统自动创建，用于临时存储)

### 2. 模型目录结构
```
models/
├── model_name/
│   ├── model.pth          # PyTorch模型权重文件
│   ├── config.json        # 模型配置文件
│   └── metadata.json      # 模型元数据
└── ...

uploads/models/
├── model_name/
│   ├── model.pth
│   ├── config.json
│   └── ...
└── ...
```

## 推荐的医疗AI模型

### 1. 胸部X光分析模型

#### 推荐模型类型：
- **ResNet50** - 适合一般分类任务
- **DenseNet121** - 在医疗图像上表现优异
- **EfficientNet-B0/B4** - 效率和准确性平衡
- **Vision Transformer (ViT)** - 最新的Transformer架构

#### 具体推荐：
1. **CheXNet (DenseNet121)**
   - 用途：胸部X光14种疾病分类
   - 准确率：在某些疾病上超过放射科医生
   - 下载：Stanford ML Group开源

2. **COVID-Net**
   - 用途：COVID-19检测
   - 特点：专门针对COVID-19优化
   - 下载：GitHub开源项目

### 2. CT扫描分析模型

#### 推荐模型类型：
- **3D ResNet** - 处理3D医疗图像
- **U-Net** - 医疗图像分割
- **V-Net** - 3D图像分割

### 3. MRI分析模型

#### 推荐模型类型：
- **3D U-Net** - MRI图像分割
- **ResNet3D** - 3D图像分类
- **TransUNet** - 结合Transformer和U-Net

## 模型下载来源

### 1. 官方预训练模型

#### PyTorch Hub
```python
# 下载预训练的医疗模型
import torch
model = torch.hub.load('pytorch/vision', 'densenet121', pretrained=True)
```

#### Hugging Face Model Hub
- 网址：https://huggingface.co/models
- 搜索关键词：medical, chest-xray, covid, radiology
- 推荐模型：
  - `microsoft/DialoGPT-medium-medical`
  - `emilyalsentzer/Bio_ClinicalBERT`

### 2. 学术研究模型

#### Papers with Code
- 网址：https://paperswithcode.com/
- 分类：Medical Image Analysis
- 查找最新的SOTA模型

#### GitHub开源项目
1. **CheXNet**
   - 仓库：https://github.com/arnoweng/CheXNet
   - 模型：DenseNet121预训练权重

2. **COVID-Net**
   - 仓库：https://github.com/lindawangg/COVID-Net
   - 模型：COVID-19检测模型

3. **MedicalNet**
   - 仓库：https://github.com/Tencent/MedicalNet
   - 模型：3D医疗图像预训练模型

### 3. 商业模型平台

#### NVIDIA NGC
- 网址：https://ngc.nvidia.com/
- 医疗AI模型集合
- 需要注册账号

#### AWS SageMaker
- 预训练医疗模型
- 按使用付费

## 模型下载和部署步骤

### 1. 手动下载步骤

```bash
# 1. 创建模型目录
mkdir -p ./uploads/models/your_model_name

# 2. 下载模型文件
# 方法1：使用wget/curl
wget https://example.com/model.pth -O ./uploads/models/your_model_name/model.pth

# 方法2：使用Python脚本
python download_model.py --model-name your_model_name --output-dir ./uploads/models/

# 3. 验证模型文件
ls -la ./uploads/models/your_model_name/
```

### 2. 使用系统API下载

```python
# 通过系统API下载公共模型
import requests

response = requests.post(
    'http://localhost:8000/api/marketplace/models/model_id/download',
    headers={'Authorization': 'Bearer your_token'}
)
```

### 3. 模型注册

下载完成后，需要在系统中注册模型：

```python
# 注册模型到系统
model_data = {
    "name": "chest_xray_classifier",
    "version": "1.0.0",
    "description": "胸部X光分类模型",
    "model_type": "classification",
    "architecture": "DenseNet121",
    "input_shape": [1, 224, 224],
    "output_shape": [14],
    "file_path": "./uploads/models/chest_xray_classifier/model.pth"
}

response = requests.post(
    'http://localhost:8000/api/models/register',
    json=model_data,
    headers={'Authorization': 'Bearer your_token'}
)
```

## 模型文件格式要求

### 1. PyTorch模型 (.pth)
- 保存模型状态字典：`torch.save(model.state_dict(), 'model.pth')`
- 文件大小：建议小于500MB
- 包含所有必要的权重参数

### 2. 配置文件 (config.json)
```json
{
  "model_name": "chest_xray_classifier",
  "version": "1.0.0",
  "architecture": "DenseNet121",
  "input_shape": [1, 224, 224],
  "output_shape": [14],
  "num_classes": 14,
  "class_names": ["Atelectasis", "Cardiomegaly", ...],
  "preprocessing": {
    "resize": [224, 224],
    "normalize": {
      "mean": [0.485, 0.456, 0.406],
      "std": [0.229, 0.224, 0.225]
    }
  }
}
```

## 性能优化建议

### 1. 模型选择原则
- **准确性优先**：选择在医疗数据集上验证过的模型
- **效率考虑**：根据硬件资源选择合适大小的模型
- **可解释性**：医疗应用需要模型决策的可解释性

### 2. 硬件要求
- **CPU推理**：轻量级模型（MobileNet, EfficientNet-B0）
- **GPU推理**：大型模型（ResNet152, ViT-Large）
- **内存要求**：至少8GB RAM，推荐16GB+

### 3. 部署优化
- 使用模型量化减少文件大小
- 启用混合精度推理
- 考虑模型蒸馏技术

## 常见问题解决

### 1. 模型加载失败
- 检查文件路径是否正确
- 验证模型文件完整性
- 确认PyTorch版本兼容性

### 2. 内存不足
- 减少批处理大小
- 使用模型量化
- 考虑使用更小的模型

### 3. 推理速度慢
- 启用GPU加速
- 使用TensorRT优化
- 考虑模型并行处理

## 示例：完整的模型下载和部署流程

```python
#!/usr/bin/env python3
# 完整的模型下载和部署示例

import os
import torch
import requests
from pathlib import Path

def download_and_deploy_model():
    # 1. 设置路径
    model_dir = Path("./uploads/models/chest_xray_classifier")
    model_dir.mkdir(parents=True, exist_ok=True)
    
    # 2. 下载模型（示例URL）
    model_url = "https://example.com/chest_xray_model.pth"
    model_path = model_dir / "model.pth"
    
    if not model_path.exists():
        print("正在下载模型...")
        response = requests.get(model_url)
        with open(model_path, 'wb') as f:
            f.write(response.content)
        print(f"模型已下载到: {model_path}")
    
    # 3. 验证模型
    try:
        model_state = torch.load(model_path, map_location='cpu')
        print(f"模型验证成功，包含 {len(model_state)} 个参数组")
    except Exception as e:
        print(f"模型验证失败: {e}")
        return False
    
    # 4. 注册到系统
    # 这里调用系统API注册模型
    print("模型部署完成！")
    return True

if __name__ == "__main__":
    download_and_deploy_model()
```

---

**注意事项：**
1. 下载模型前请确认许可证和使用条款
2. 医疗AI模型需要在真实临床环境中验证
3. 定期更新模型以获得最佳性能
4. 遵循医疗器械相关法规要求