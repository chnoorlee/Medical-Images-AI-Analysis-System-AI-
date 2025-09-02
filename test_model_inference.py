#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试AI模型推理功能
"""

import requests
import json
import os
from pathlib import Path

# 配置
BASE_URL = "http://localhost:8000"
LOGIN_URL = f"{BASE_URL}/api/auth/login"
MODELS_URL = f"{BASE_URL}/api/models/list"
REGISTER_MODEL_URL = f"{BASE_URL}/api/models/register"
LOAD_MODEL_URL = f"{BASE_URL}/api/models/{{model_id}}/load"
PREDICT_URL = f"{BASE_URL}/api/models/{{model_id}}/predict"

# 登录凭据
USERNAME = "admin"
PASSWORD = "admin123456"

def login():
    """登录获取token"""
    login_data = {
        "username": USERNAME,
        "password": PASSWORD
    }
    
    response = requests.post(LOGIN_URL, json=login_data)
    if response.status_code == 200:
        result = response.json()
        token = result.get('access_token')
        print(f"登录成功，获取token: {token[:50]}...")
        return token
    else:
        print(f"登录失败: {response.status_code} - {response.text}")
        return None

def get_models(token):
    """获取可用模型列表"""
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json"
    }
    
    response = requests.get(MODELS_URL, headers=headers)
    if response.status_code == 200:
        models = response.json()
        print(f"模型列表响应: {models}")
        # 如果返回的是字典格式，可能包含models字段
        if isinstance(models, dict) and 'models' in models:
            models = models['models']
        print(f"获取到 {len(models) if isinstance(models, list) else 0} 个模型")
        return models
    else:
        print(f"获取模型列表失败: {response.status_code} - {response.text}")
        return []

def register_test_model(token):
    """注册一个测试模型"""
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json"
    }
    
    model_data = {
        "name": "test_chest_xray_classifier",
        "description": "测试用胸部X光分类模型",
        "model_type": "classification",
        "architecture": "resnet50",
        "version": "1.0.0",
        "config": {
            "input_shape": [224, 224, 3],
            "num_classes": 2,
            "preprocessing": "normalize"
        }
    }
    
    response = requests.post(REGISTER_MODEL_URL, headers=headers, json=model_data)
    if response.status_code == 200:
        result = response.json()
        model_id = result.get('model_id')
        print(f"模型注册成功，模型ID: {model_id}")
        return model_id
    else:
        print(f"模型注册失败: {response.status_code} - {response.text}")
        return None

def load_model(token, model_id):
    """加载模型"""
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json"
    }
    
    url = LOAD_MODEL_URL.format(model_id=model_id)
    response = requests.post(url, headers=headers)
    
    if response.status_code == 200:
        result = response.json()
        print(f"模型加载成功: {result.get('message')}")
        return True
    else:
        print(f"模型加载失败: {response.status_code} - {response.text}")
        return False

def test_inference(token, model_id):
    """测试模型推理"""
    headers = {
        "Authorization": f"Bearer {token}"
    }
    
    # 使用之前上传的测试图像
    image_path = "test_image.jpg"
    if not os.path.exists(image_path):
        print(f"测试图像文件不存在: {image_path}")
        return False
    
    url = PREDICT_URL.format(model_id=model_id)
    
    with open(image_path, 'rb') as f:
        files = {
            'file': ('test_image.jpg', f, 'image/jpeg')
        }
        data = {
            'config': json.dumps({
                'confidence_threshold': 0.5,
                'enable_heatmap': True
            })
        }
        
        response = requests.post(url, headers=headers, files=files, data=data)
    
    print(f"推理响应状态码: {response.status_code}")
    print(f"推理响应内容: {response.text}")
    
    if response.status_code == 200:
        result = response.json()
        print(f"推理成功! 结果: {json.dumps(result, indent=2, ensure_ascii=False)}")
        return True
    else:
        print(f"推理失败: {response.status_code} - {response.text}")
        return False

def main():
    """主函数"""
    print("=== 开始测试AI模型推理功能 ===")
    
    # 1. 登录
    token = login()
    if not token:
        return
    
    # 2. 获取现有模型
    models = get_models(token)
    
    model_id = None
    if models and len(models) > 0:
        # 使用第一个可用模型
        model_id = models[0].get('id') or models[0].get('model_id')
        print(f"使用现有模型: {model_id}")
    else:
        # 注册新的测试模型
        print("没有可用模型，注册新的测试模型...")
        model_id = register_test_model(token)
    
    if not model_id:
        print("无法获取或创建模型")
        return
    
    # 3. 加载模型
    print(f"加载模型: {model_id}")
    if not load_model(token, model_id):
        print("模型加载失败，跳过推理测试")
        return
    
    # 4. 测试推理
    print("开始推理测试...")
    test_inference(token, model_id)
    
    print("=== AI模型推理功能测试完成 ===")

if __name__ == "__main__":
    main()