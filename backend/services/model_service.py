import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
from torchvision.models import resnet50, densenet121, efficientnet_b0
import cv2
from typing import Dict, List, Optional, Any, Tuple, Union
import logging
from datetime import datetime, timezone
import json
from pathlib import Path
import pickle
import joblib
import uuid
import asyncio
from concurrent.futures import ThreadPoolExecutor
import threading
from dataclasses import dataclass
from enum import Enum

from backend.models.ai_model import AIModel, ModelVersion, Inference, ModelTrainingJob
from backend.core.database import get_db_context
from backend.core.config import settings
from backend.services.storage_service import StorageService

logger = logging.getLogger(__name__)

class ModelType(Enum):
    """模型类型枚举"""
    CLASSIFICATION = "classification"
    SEGMENTATION = "segmentation"
    DETECTION = "detection"
    REGRESSION = "regression"

class ModelStatus(Enum):
    """模型状态枚举"""
    TRAINING = "training"
    READY = "ready"
    FAILED = "failed"
    DEPRECATED = "deprecated"

@dataclass
class InferenceResult:
    """推理结果数据类"""
    predictions: Dict[str, Any]
    confidence_scores: Dict[str, float]
    processing_time: float
    model_version: str
    metadata: Dict[str, Any]

class MedicalImageDataset(Dataset):
    """医学图像数据集"""
    
    def __init__(self, images: List[np.ndarray], labels: Optional[List] = None, 
                 transform: Optional[transforms.Compose] = None):
        self.images = images
        self.labels = labels
        self.transform = transform
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image = self.images[idx]
        
        # 确保图像是3通道
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.shape[2] == 1:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        
        if self.transform:
            image = self.transform(image)
        
        if self.labels is not None:
            return image, self.labels[idx]
        return image

class ModelService:
    """AI模型服务
    
    提供AI模型管理、推理和训练功能，包括：
    - 模型加载和管理
    - 模型推理
    - 模型训练和微调
    - 模型版本控制
    - 模型性能监控
    """
    
    def __init__(self):
        self.storage_service = StorageService()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.loaded_models = {}  # 缓存已加载的模型
        self.model_lock = threading.Lock()
        self.executor = ThreadPoolExecutor(max_workers=4)
        
        # 预定义的模型架构
        self.model_architectures = {
            'resnet50': self._create_resnet50,
            'densenet121': self._create_densenet121,
            'efficientnet_b0': self._create_efficientnet_b0,
            'custom_cnn': self._create_custom_cnn
        }
        
        logger.info(f"模型服务初始化完成，使用设备: {self.device}")
    
    # 模型管理
    async def register_model(self, name: str, description: str, model_type: str,
                           architecture: str, version: str = "1.0.0",
                           created_by: Optional[str] = None,
                           config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """注册新模型
        
        Args:
            name: 模型名称
            description: 模型描述
            model_type: 模型类型
            architecture: 模型架构
            version: 版本号
            created_by: 创建者ID
            
        Returns:
            注册结果
        """
        try:
            with get_db_context() as db:
                # 检查模型是否已存在
                existing_model = db.query(AIModel).filter(
                    AIModel.model_name == name
                ).first()
                
                if existing_model:
                    return {
                        'success': False,
                        'error': f'模型 {name} 已存在'
                    }
                
                # 创建模型记录
                model = AIModel(
                    model_id=uuid.uuid4(),
                    model_name=name,
                    description=description,
                    model_type=model_type,
                    architecture=architecture,
                    created_by=uuid.UUID(created_by) if created_by else None,
                    created_at=datetime.now(timezone.utc)
                )
                
                db.add(model)
                db.commit()
                db.refresh(model)
                
                # 创建初始版本
                model_version = ModelVersion(
                    version_id=uuid.uuid4(),
                    model_id=model.model_id,
                    version=version,
                    status='development',
                    config=config,
                    created_at=datetime.now(timezone.utc)
                )
                
                db.add(model_version)
                db.commit()
                
                logger.info(f"模型注册成功: {name} v{version}")
                return {
                    'success': True,
                    'model_id': str(model.model_id),
                    'version_id': str(model_version.version_id),
                    'message': '模型注册成功'
                }
                
        except Exception as e:
            logger.error(f"模型注册失败: {e}")
            return {
                'success': False,
                'error': '模型注册失败'
            }
    
    async def load_model(self, model_id: str, version: Optional[str] = None) -> Dict[str, Any]:
        """加载模型
        
        Args:
            model_id: 模型ID
            version: 版本号，如果为None则加载最新版本
            
        Returns:
            加载结果
        """
        try:
            with get_db_context() as db:
                # 查找模型
                model = db.query(AIModel).filter(
                    AIModel.model_id == uuid.UUID(model_id)
                ).first()
                
                if not model:
                    return {
                        'success': False,
                        'error': '模型不存在'
                    }
                
                # 查找版本
                if version:
                    model_version = db.query(ModelVersion).filter(
                        ModelVersion.model_id == model.model_id,
                        ModelVersion.version_number == version
                    ).first()
                else:
                    model_version = db.query(ModelVersion).filter(
                        ModelVersion.model_id == model.model_id,
                        ModelVersion.deployment_status == 'deployed'
                    ).order_by(ModelVersion.created_at.desc()).first()
                
                if not model_version:
                    return {
                        'success': False,
                        'error': '模型版本不存在或未就绪'
                    }
                
                # 检查是否已加载
                cache_key = f"{model_id}_{model_version.version_number}"
                if cache_key in self.loaded_models:
                    return {
                        'success': True,
                        'message': '模型已加载',
                        'cache_key': cache_key
                    }
                
                # 加载模型文件
                if model_version.model_file_path:
                    # 创建临时目录和本地路径
                    os.makedirs("./temp_models", exist_ok=True)
                    local_path = f"./temp_models/{cache_key}.pth"
                    download_result = await self.storage_service.download_file(model_version.model_file_path, local_path)
                    
                    # 根据架构创建模型
                    pytorch_model = self._create_model(model.architecture, model_version.deployment_config or {})
                    
                    # 加载权重
                    if download_result.get('success') and os.path.exists(local_path):
                        state_dict = torch.load(local_path, map_location=self.device)
                        pytorch_model.load_state_dict(state_dict)
                        # 清理临时文件
                        os.remove(local_path)
                    
                    pytorch_model.to(self.device)
                    pytorch_model.eval()
                    
                    # 缓存模型
                    with self.model_lock:
                        self.loaded_models[cache_key] = {
                            'model': pytorch_model,
                            'metadata': {
                                'model_id': model_id,
                                'version': model_version.version_number,
                                'version_id': str(model_version.version_id),
                                'architecture': model.architecture,
                                'model_type': model.model_type,
                                'config': model_version.deployment_config
                            },
                            'loaded_at': datetime.now(timezone.utc)
                        }
                    
                    logger.info(f"模型加载成功: {model.model_name} v{model_version.version_number}")
                    return {
                        'success': True,
                        'message': '模型加载成功',
                        'cache_key': cache_key
                    }
                else:
                    return {
                        'success': False,
                        'error': '模型文件不存在'
                    }
                    
        except Exception as e:
            logger.error(f"模型加载失败: {e}")
            return {
                'success': False,
                'error': '模型加载失败'
            }
    
    async def unload_model(self, model_id: str, version: Optional[str] = None) -> Dict[str, Any]:
        """卸载模型
        
        Args:
            model_id: 模型ID
            version: 版本号
            
        Returns:
            卸载结果
        """
        try:
            cache_key = f"{model_id}_{version}" if version else None
            
            with self.model_lock:
                if cache_key and cache_key in self.loaded_models:
                    del self.loaded_models[cache_key]
                    logger.info(f"模型卸载成功: {cache_key}")
                    return {
                        'success': True,
                        'message': '模型卸载成功'
                    }
                elif not cache_key:
                    # 卸载所有版本
                    keys_to_remove = [k for k in self.loaded_models.keys() if k.startswith(model_id)]
                    for key in keys_to_remove:
                        del self.loaded_models[key]
                    
                    logger.info(f"模型所有版本卸载成功: {model_id}")
                    return {
                        'success': True,
                        'message': f'模型所有版本卸载成功，共{len(keys_to_remove)}个版本'
                    }
                else:
                    return {
                        'success': False,
                        'error': '模型未加载'
                    }
                    
        except Exception as e:
            logger.error(f"模型卸载失败: {e}")
            return {
                'success': False,
                'error': '模型卸载失败'
            }
    
    # 模型推理
    async def predict(self, model_id: str, image: np.ndarray, 
                     version: Optional[str] = None,
                     preprocessing_config: Optional[Dict[str, Any]] = None) -> InferenceResult:
        """模型推理
        
        Args:
            model_id: 模型ID
            image: 输入图像
            version: 模型版本
            preprocessing_config: 预处理配置
            
        Returns:
            推理结果
        """
        start_time = datetime.now()
        
        try:
            # 确保模型已加载
            cache_key = f"{model_id}_{version}" if version else None
            if not cache_key or cache_key not in self.loaded_models:
                load_result = await self.load_model(model_id, version)
                if not load_result['success']:
                    raise Exception(f"模型加载失败: {load_result['error']}")
                cache_key = load_result['cache_key']
            
            model_info = self.loaded_models[cache_key]
            pytorch_model = model_info['model']
            metadata = model_info['metadata']
            
            # 预处理图像
            processed_image = await self._preprocess_image(image, preprocessing_config)
            
            # 执行推理
            with torch.no_grad():
                if len(processed_image.shape) == 3:
                    processed_image = processed_image.unsqueeze(0)  # 添加batch维度
                
                processed_image = processed_image.to(self.device)
                outputs = pytorch_model(processed_image)
                
                # 后处理结果
                predictions, confidence_scores = await self._postprocess_outputs(
                    outputs, metadata['model_type'], metadata.get('config', {})
                )
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            # 创建推理结果
            result = InferenceResult(
                predictions=predictions,
                confidence_scores=confidence_scores,
                processing_time=processing_time,
                model_version=metadata['version'],
                metadata={
                    'model_id': model_id,
                    'architecture': metadata['architecture'],
                    'model_type': metadata['model_type'],
                    'device': str(self.device),
                    'input_shape': list(processed_image.shape)
                }
            )
            
            # 记录推理日志
            await self._log_inference(model_id, metadata['version_id'], result)
            
            logger.debug(f"推理完成，耗时: {processing_time:.3f}秒")
            return result
            
        except Exception as e:
            logger.error(f"模型推理失败: {e}")
            processing_time = (datetime.now() - start_time).total_seconds()
            return InferenceResult(
                predictions={},
                confidence_scores={},
                processing_time=processing_time,
                model_version=version or 'unknown',
                metadata={'error': str(e)}
            )
    
    async def batch_predict(self, model_id: str, images: List[np.ndarray],
                          version: Optional[str] = None,
                          batch_size: int = 8) -> List[InferenceResult]:
        """批量推理
        
        Args:
            model_id: 模型ID
            images: 图像列表
            version: 模型版本
            batch_size: 批次大小
            
        Returns:
            推理结果列表
        """
        try:
            results = []
            
            for i in range(0, len(images), batch_size):
                batch_images = images[i:i + batch_size]
                batch_results = await asyncio.gather(*[
                    self.predict(model_id, image, version)
                    for image in batch_images
                ])
                results.extend(batch_results)
            
            logger.info(f"批量推理完成，共处理{len(images)}张图像")
            return results
            
        except Exception as e:
            logger.error(f"批量推理失败: {e}")
            return []
    
    # 模型训练
    async def train_model(self, model_id: str, training_data: Dict[str, Any],
                        training_config: Dict[str, Any],
                        created_by: Optional[str] = None) -> Dict[str, Any]:
        """训练模型
        
        Args:
            model_id: 模型ID
            training_data: 训练数据配置
            training_config: 训练配置
            created_by: 创建者ID
            
        Returns:
            训练结果
        """
        try:
            with get_db_context() as db:
                # 创建训练任务记录
                training_job = ModelTrainingJob(
                    job_id=uuid.uuid4(),
                    model_id=uuid.UUID(model_id),
                    status='running',
                    training_config=json.dumps(training_config),
                    created_by=uuid.UUID(created_by) if created_by else None,
                    started_at=datetime.now(timezone.utc)
                )
                
                db.add(training_job)
                db.commit()
                db.refresh(training_job)
                
                # 异步执行训练
                asyncio.create_task(self._execute_training(
                    str(training_job.job_id), model_id, training_data, training_config
                ))
                
                return {
                    'success': True,
                    'job_id': str(training_job.job_id),
                    'message': '训练任务已启动'
                }
                
        except Exception as e:
            logger.error(f"启动训练任务失败: {e}")
            return {
                'success': False,
                'error': '启动训练任务失败'
            }
    
    async def _execute_training(self, job_id: str, model_id: str,
                              training_data: Dict[str, Any],
                              training_config: Dict[str, Any]):
        """执行模型训练"""
        try:
            with get_db_context() as db:
                training_job = db.query(ModelTrainingJob).filter(
                    ModelTrainingJob.job_id == uuid.UUID(job_id)
                ).first()
                
                if not training_job:
                    return
                
                # 获取模型信息
                model = db.query(AIModel).filter(
                    AIModel.model_id == uuid.UUID(model_id)
                ).first()
                
                if not model:
                    training_job.status = 'failed'
                    training_job.error_message = '模型不存在'
                    db.commit()
                    return
                
                # 准备训练数据
                train_loader, val_loader = await self._prepare_training_data(
                    training_data, training_config
                )
                
                # 创建模型
                pytorch_model = self._create_model(model.architecture, training_config)
                pytorch_model.to(self.device)
                
                # 训练模型
                training_metrics = await self._train_pytorch_model(
                    pytorch_model, train_loader, val_loader, training_config
                )
                
                # 保存模型
                model_path = await self._save_trained_model(
                    pytorch_model, model_id, training_config
                )
                
                # 创建新版本
                new_version = await self._create_model_version(
                    model_id, model_path, training_config, training_metrics
                )
                
                # 更新训练任务状态
                training_job.status = 'completed'
                training_job.completed_at = datetime.now(timezone.utc)
                training_job.metrics = json.dumps(training_metrics)
                training_job.model_version_id = new_version['version_id']
                
                db.commit()
                
                logger.info(f"模型训练完成: {model.model_name}")
                
        except Exception as e:
            logger.error(f"模型训练失败: {e}")
            
            with get_db_context() as db:
                training_job = db.query(ModelTrainingJob).filter(
                    ModelTrainingJob.job_id == uuid.UUID(job_id)
                ).first()
                
                if training_job:
                    training_job.status = 'failed'
                    training_job.error_message = str(e)
                    training_job.completed_at = datetime.now(timezone.utc)
                    db.commit()
    
    # 模型架构创建
    def _create_model(self, architecture: str, config: Optional[Dict[str, Any]] = None) -> nn.Module:
        """创建模型"""
        if architecture in self.model_architectures:
            return self.model_architectures[architecture](config or {})
        else:
            raise ValueError(f"不支持的模型架构: {architecture}")
    
    def _create_resnet50(self, config: Dict[str, Any]) -> nn.Module:
        """创建ResNet50模型"""
        num_classes = config.get('num_classes', 2)
        pretrained = config.get('pretrained', True)
        
        model = resnet50(pretrained=pretrained)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        
        return model
    
    def _create_densenet121(self, config: Dict[str, Any]) -> nn.Module:
        """创建DenseNet121模型"""
        num_classes = config.get('num_classes', 2)
        pretrained = config.get('pretrained', True)
        
        model = densenet121(pretrained=pretrained)
        model.classifier = nn.Linear(model.classifier.in_features, num_classes)
        
        return model
    
    def _create_efficientnet_b0(self, config: Dict[str, Any]) -> nn.Module:
        """创建EfficientNet-B0模型"""
        num_classes = config.get('num_classes', 2)
        pretrained = config.get('pretrained', True)
        
        model = efficientnet_b0(pretrained=pretrained)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
        
        return model
    
    def _create_custom_cnn(self, config: Dict[str, Any]) -> nn.Module:
        """创建自定义CNN模型"""
        class CustomCNN(nn.Module):
            def __init__(self, num_classes=2):
                super(CustomCNN, self).__init__()
                self.features = nn.Sequential(
                    nn.Conv2d(3, 64, kernel_size=3, padding=1),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(kernel_size=2, stride=2),
                    
                    nn.Conv2d(64, 128, kernel_size=3, padding=1),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(kernel_size=2, stride=2),
                    
                    nn.Conv2d(128, 256, kernel_size=3, padding=1),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(kernel_size=2, stride=2),
                    
                    nn.Conv2d(256, 512, kernel_size=3, padding=1),
                    nn.ReLU(inplace=True),
                    nn.AdaptiveAvgPool2d((1, 1))
                )
                
                self.classifier = nn.Sequential(
                    nn.Dropout(0.5),
                    nn.Linear(512, 256),
                    nn.ReLU(inplace=True),
                    nn.Dropout(0.5),
                    nn.Linear(256, num_classes)
                )
            
            def forward(self, x):
                x = self.features(x)
                x = x.view(x.size(0), -1)
                x = self.classifier(x)
                return x
        
        num_classes = config.get('num_classes', 2)
        return CustomCNN(num_classes)
    
    # 辅助方法
    async def _preprocess_image(self, image: np.ndarray, 
                              config: Optional[Dict[str, Any]] = None) -> torch.Tensor:
        """预处理图像"""
        # 默认预处理配置
        default_config = {
            'resize': (224, 224),
            'normalize': True,
            'mean': [0.485, 0.456, 0.406],
            'std': [0.229, 0.224, 0.225]
        }
        
        if config:
            default_config.update(config)
        
        # 确保图像是RGB格式
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.shape[2] == 1:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        
        # 调整大小
        if default_config['resize']:
            image = cv2.resize(image, default_config['resize'])
        
        # 转换为tensor
        image = torch.from_numpy(image).float()
        
        # 调整维度顺序 (H, W, C) -> (C, H, W)
        image = image.permute(2, 0, 1)
        
        # 归一化到[0, 1]
        image = image / 255.0
        
        # 标准化
        if default_config['normalize']:
            mean = torch.tensor(default_config['mean']).view(3, 1, 1)
            std = torch.tensor(default_config['std']).view(3, 1, 1)
            image = (image - mean) / std
        
        return image
    
    async def _postprocess_outputs(self, outputs: torch.Tensor, model_type: str,
                                 config: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, float]]:
        """后处理模型输出"""
        predictions = {}
        confidence_scores = {}
        
        if model_type == ModelType.CLASSIFICATION.value:
            # 分类任务
            probabilities = torch.softmax(outputs, dim=1)
            predicted_class = torch.argmax(probabilities, dim=1)
            confidence = torch.max(probabilities, dim=1)[0]
            
            predictions['class'] = predicted_class.cpu().numpy().tolist()
            predictions['probabilities'] = probabilities.cpu().numpy().tolist()
            confidence_scores['classification'] = float(confidence.cpu().numpy()[0])
            
        elif model_type == ModelType.SEGMENTATION.value:
            # 分割任务
            if len(outputs.shape) == 4:  # (B, C, H, W)
                segmentation_mask = torch.argmax(outputs, dim=1)
                predictions['segmentation_mask'] = segmentation_mask.cpu().numpy().tolist()
                
                # 计算每个类别的置信度
                probabilities = torch.softmax(outputs, dim=1)
                max_probs = torch.max(probabilities, dim=1)[0]
                confidence_scores['segmentation'] = torch.mean(max_probs).item()
            
        elif model_type == ModelType.DETECTION.value:
            # 检测任务（简化处理）
            predictions['detections'] = outputs.cpu().numpy().tolist()
            confidence_scores['detection'] = 0.8  # 简化处理
            
        elif model_type == ModelType.REGRESSION.value:
            # 回归任务
            predictions['values'] = outputs.cpu().numpy().tolist()
            confidence_scores['regression'] = 1.0  # 回归任务的置信度处理
        
        return predictions, confidence_scores
    
    async def _log_inference(self, model_id: str, model_version_id: str, result: InferenceResult):
        """记录推理日志"""
        try:
            with get_db_context() as db:
                inference = Inference(
                    inference_id=uuid.uuid4(),
                    model_id=uuid.UUID(model_id),
                    model_version_id=uuid.UUID(model_version_id),
                    input_data=json.dumps(result.metadata.get('input_shape', [])),
                    processed_output=json.dumps(result.predictions),
                    confidence_scores=json.dumps(result.confidence_scores),
                    processing_time_ms=int(result.processing_time * 1000),
                    status='completed',
                    created_at=datetime.now(timezone.utc)
                )
                
                db.add(inference)
                db.commit()
                
        except Exception as e:
            logger.error(f"记录推理日志失败: {e}")
    
    async def _prepare_training_data(self, training_data: Dict[str, Any],
                                   training_config: Dict[str, Any]) -> Tuple[DataLoader, DataLoader]:
        """准备训练数据"""
        # 这里是简化的数据准备逻辑
        # 实际应用中需要根据具体的数据格式和存储方式来实现
        
        # 数据变换
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # 创建数据集（这里使用模拟数据）
        train_images = [np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8) for _ in range(100)]
        train_labels = [np.random.randint(0, 2) for _ in range(100)]
        
        val_images = [np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8) for _ in range(20)]
        val_labels = [np.random.randint(0, 2) for _ in range(20)]
        
        train_dataset = MedicalImageDataset(train_images, train_labels, transform)
        val_dataset = MedicalImageDataset(val_images, val_labels, transform)
        
        batch_size = training_config.get('batch_size', 16)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        return train_loader, val_loader
    
    async def _train_pytorch_model(self, model: nn.Module, train_loader: DataLoader,
                                 val_loader: DataLoader, config: Dict[str, Any]) -> Dict[str, Any]:
        """训练PyTorch模型"""
        # 训练配置
        epochs = config.get('epochs', 10)
        learning_rate = config.get('learning_rate', 0.001)
        
        # 优化器和损失函数
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        criterion = nn.CrossEntropyLoss()
        
        # 训练指标
        train_losses = []
        val_losses = []
        val_accuracies = []
        
        for epoch in range(epochs):
            # 训练阶段
            model.train()
            train_loss = 0.0
            
            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = data.to(self.device), target.to(self.device)
                
                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
            
            train_loss /= len(train_loader)
            train_losses.append(train_loss)
            
            # 验证阶段
            model.eval()
            val_loss = 0.0
            correct = 0
            total = 0
            
            with torch.no_grad():
                for data, target in val_loader:
                    data, target = data.to(self.device), target.to(self.device)
                    output = model(data)
                    val_loss += criterion(output, target).item()
                    
                    _, predicted = torch.max(output.data, 1)
                    total += target.size(0)
                    correct += (predicted == target).sum().item()
            
            val_loss /= len(val_loader)
            val_accuracy = 100 * correct / total
            
            val_losses.append(val_loss)
            val_accuracies.append(val_accuracy)
            
            logger.info(f"Epoch {epoch+1}/{epochs}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.2f}%")
        
        return {
            'train_losses': train_losses,
            'val_losses': val_losses,
            'val_accuracies': val_accuracies,
            'final_val_accuracy': val_accuracies[-1] if val_accuracies else 0.0
        }
    
    async def _save_trained_model(self, model: nn.Module, model_id: str,
                                config: Dict[str, Any]) -> str:
        """保存训练好的模型"""
        # 生成模型文件路径
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        model_filename = f"model_{model_id}_{timestamp}.pth"
        model_path = f"models/{model_id}/{model_filename}"
        
        # 保存模型状态字典
        model_state = {
            'state_dict': model.state_dict(),
            'config': config,
            'timestamp': timestamp
        }
        
        # 临时保存到本地
        temp_path = Path(f"/tmp/{model_filename}")
        torch.save(model_state, temp_path)
        
        # 上传到存储服务
        with open(temp_path, 'rb') as f:
            await self.storage_service.upload_file(f, model_path)
        
        # 清理临时文件
        temp_path.unlink()
        
        return model_path
    
    async def _create_model_version(self, model_id: str, model_path: str,
                                  config: Dict[str, Any], metrics: Dict[str, Any]) -> Dict[str, Any]:
        """创建模型版本"""
        try:
            with get_db_context() as db:
                # 获取下一个版本号
                latest_version = db.query(ModelVersion).filter(
                    ModelVersion.model_id == uuid.UUID(model_id)
                ).order_by(ModelVersion.created_at.desc()).first()
                
                if latest_version:
                    version_parts = latest_version.version.split('.')
                    major, minor, patch = int(version_parts[0]), int(version_parts[1]), int(version_parts[2])
                    new_version = f"{major}.{minor}.{patch + 1}"
                else:
                    new_version = "1.0.0"
                
                # 创建新版本
                model_version = ModelVersion(
                    version_id=uuid.uuid4(),
                    model_id=uuid.UUID(model_id),
                    version=new_version,
                    model_path=model_path,
                    config=json.dumps(config),
                    metrics=json.dumps(metrics),
                    status='ready',
                    created_at=datetime.now(timezone.utc)
                )
                
                db.add(model_version)
                db.commit()
                
                return {
                    'version_id': str(model_version.version_id),
                    'version': new_version
                }
                
        except Exception as e:
            logger.error(f"创建模型版本失败: {e}")
            raise
    
    # 模型管理API
    async def get_model_list(self) -> List[Dict[str, Any]]:
        """获取模型列表"""
        try:
            with get_db_context() as db:
                models = db.query(AIModel).all()
                
                model_list = []
                for model in models:
                    latest_version = db.query(ModelVersion).filter(
                        ModelVersion.model_id == model.model_id
                    ).order_by(ModelVersion.created_at.desc()).first()
                    
                    model_list.append({
                        'model_id': str(model.model_id),
                        'name': model.model_name,
                        'description': model.description,
                        'model_type': model.model_type,
                        'architecture': model.architecture,
                        'is_active': model.is_active,
                        'latest_version': latest_version.version_number if latest_version else None,
                        'created_at': model.created_at.isoformat()
                    })
                
                return model_list
                
        except Exception as e:
            logger.error(f"获取模型列表失败: {e}")
            return []
    
    async def get_model_info(self, model_id: str) -> Dict[str, Any]:
        """获取模型详细信息"""
        try:
            with get_db_context() as db:
                model = db.query(AIModel).filter(
                    AIModel.model_id == uuid.UUID(model_id)
                ).first()
                
                if not model:
                    return {'error': '模型不存在'}
                
                versions = db.query(ModelVersion).filter(
                    ModelVersion.model_id == model.model_id
                ).order_by(ModelVersion.created_at.desc()).all()
                
                return {
                    'model_id': str(model.model_id),
                    'name': model.model_name,
                    'description': model.description,
                    'model_type': model.model_type,
                    'architecture': model.architecture,
                    'status': model.status,
                    'created_at': model.created_at.isoformat(),
                    'versions': [
                        {
                            'version_id': str(version.version_id),
                            'version': version.version,
                            'status': version.status,
                            'metrics': json.loads(version.metrics) if version.metrics else {},
                            'created_at': version.created_at.isoformat()
                        }
                        for version in versions
                    ]
                }
                
        except Exception as e:
            logger.error(f"获取模型信息失败: {e}")
            return {'error': '获取模型信息失败'}
    
    async def get_loaded_models(self) -> Dict[str, Any]:
        """获取已加载的模型信息"""
        with self.model_lock:
            return {
                cache_key: {
                    'metadata': info['metadata'],
                    'loaded_at': info['loaded_at'].isoformat()
                }
                for cache_key, info in self.loaded_models.items()
            }