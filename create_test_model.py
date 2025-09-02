#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
创建测试用的胸部X光分类模型
"""

import torch
import torch.nn as nn
import os

class ChestXrayClassifier(nn.Module):
    """简单的胸部X光分类模型"""
    
    def __init__(self, num_classes=3):
        super(ChestXrayClassifier, self).__init__()
        # 简单的CNN架构
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((7, 7))
        )
        
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(128 * 7 * 7, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

def create_test_model():
    """创建并保存测试模型"""
    # 创建模型
    model = ChestXrayClassifier(num_classes=3)  # 正常、肺炎、COVID-19
    
    # 创建一些随机权重（模拟训练后的模型）
    model.eval()
    
    # 保存模型
    model_dir = "./uploads/models/test_chest_xray_classifier"
    os.makedirs(model_dir, exist_ok=True)
    
    model_path = os.path.join(model_dir, "model.pth")
    
    # 保存模型状态字典
    torch.save(model.state_dict(), model_path)
    
    print(f"测试模型已保存到: {model_path}")
    print(f"模型参数数量: {sum(p.numel() for p in model.parameters())}")
    print(f"模型大小: {os.path.getsize(model_path) / 1024 / 1024:.2f} MB")
    
    # 创建模型配置文件
    config = {
        "model_name": "test_chest_xray_classifier",
        "version": "1.0.0",
        "architecture": "CNN",
        "input_shape": [1, 224, 224],
        "output_shape": [3],
        "num_classes": 3,
        "class_names": ["正常", "肺炎", "COVID-19"],
        "preprocessing": {
            "resize": [224, 224],
            "normalize": {
                "mean": [0.485],
                "std": [0.229]
            }
        }
    }
    
    import json
    config_path = os.path.join(model_dir, "config.json")
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(config, f, ensure_ascii=False, indent=2)
    
    print(f"模型配置已保存到: {config_path}")
    
    return model_path, config_path

if __name__ == "__main__":
    create_test_model()