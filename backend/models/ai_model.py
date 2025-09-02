from sqlalchemy import Column, String, Integer, DateTime, Text, Boolean, DECIMAL, ForeignKey, BigInteger
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy import JSON
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
import uuid
from .base import Base

class AIModel(Base):
    """AI模型模型"""
    __tablename__ = 'ai_models'
    
    model_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    model_name = Column(String(100), nullable=False, index=True)
    model_type = Column(String(50), nullable=False)          # classification, segmentation, detection, diagnosis
    category = Column(String(50))                            # 模型分类：chest_xray, ct_scan, mri等
    description = Column(Text)
    target_disease = Column(String(100))                     # 目标疾病
    modality = Column(String(20))                            # 影像模态：CT, MRI, X-Ray等
    body_part = Column(String(50))                           # 身体部位
    framework = Column(String(50))                           # 框架：pytorch, tensorflow等
    architecture = Column(String(100))                       # 模型架构：ResNet, UNet等
    input_shape = Column(JSON)                              # 输入形状
    output_shape = Column(JSON)                             # 输出形状
    preprocessing_config = Column(JSON)                     # 预处理配置
    postprocessing_config = Column(JSON)                    # 后处理配置
    training_dataset_info = Column(JSON)                    # 训练数据集信息
    validation_metrics = Column(JSON)                       # 验证指标
    clinical_validation = Column(JSON)                      # 临床验证信息
    regulatory_approval = Column(JSON)                      # 监管批准信息
    is_active = Column(Boolean, default=True)
    is_production_ready = Column(Boolean, default=False)
    created_by = Column(UUID(as_uuid=True), ForeignKey('users.user_id'))
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    # 关联关系
    creator = relationship("User")
    versions = relationship("ModelVersion", back_populates="model", cascade="all, delete-orphan")
    inferences = relationship("Inference", back_populates="model")
    
    def __repr__(self):
        return f"<AIModel(model_name='{self.model_name}', type='{self.model_type}', category='{self.category}')>"
    
    def get_latest_version(self):
        """获取最新版本"""
        return max(self.versions, key=lambda v: v.version_number) if self.versions else None
    
    def get_production_version(self):
        """获取生产版本"""
        return next((v for v in self.versions if v.is_production), None)

class ModelVersion(Base):
    """模型版本模型"""
    __tablename__ = 'model_versions'
    
    version_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    model_id = Column(UUID(as_uuid=True), ForeignKey('ai_models.model_id'), nullable=False)
    version_number = Column(String(20), nullable=False)      # 版本号：1.0.0, 1.1.0等
    version_name = Column(String(100))                       # 版本名称
    description = Column(Text)
    model_file_path = Column(String(500), nullable=False)    # 模型文件路径
    model_file_size = Column(BigInteger)                     # 模型文件大小
    model_checksum = Column(String(64))                      # 模型文件校验和
    config_file_path = Column(String(500))                   # 配置文件路径
    weights_file_path = Column(String(500))                  # 权重文件路径
    training_config = Column(JSON)                          # 训练配置
    training_metrics = Column(JSON)                         # 训练指标
    validation_metrics = Column(JSON)                       # 验证指标
    test_metrics = Column(JSON)                             # 测试指标
    performance_benchmarks = Column(JSON)                   # 性能基准
    resource_requirements = Column(JSON)                    # 资源需求
    deployment_config = Column(JSON)                        # 部署配置
    is_production = Column(Boolean, default=False)          # 是否为生产版本
    is_deprecated = Column(Boolean, default=False)          # 是否已弃用
    deployment_status = Column(String(20), default='pending') # pending, deployed, failed, retired
    created_by = Column(UUID(as_uuid=True), ForeignKey('users.user_id'))
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    deployed_at = Column(DateTime(timezone=True))
    retired_at = Column(DateTime(timezone=True))
    
    # 关联关系
    model = relationship("AIModel", back_populates="versions")
    creator = relationship("User")
    inferences = relationship("Inference", back_populates="model_version")
    
    def __repr__(self):
        return f"<ModelVersion(version_id='{self.version_id}', version_number='{self.version_number}', is_production={self.is_production})>"

class Inference(Base):
    """推理记录模型"""
    __tablename__ = 'inferences'
    
    inference_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    image_id = Column(UUID(as_uuid=True), ForeignKey('images.image_id'), nullable=False)
    model_id = Column(UUID(as_uuid=True), ForeignKey('ai_models.model_id'), nullable=False)
    model_version_id = Column(UUID(as_uuid=True), ForeignKey('model_versions.version_id'), nullable=False)
    user_id = Column(UUID(as_uuid=True), ForeignKey('users.user_id'))  # 触发推理的用户
    inference_type = Column(String(50), nullable=False)      # batch, real_time, scheduled
    input_data = Column(JSON)                               # 输入数据信息
    preprocessing_params = Column(JSON)                     # 预处理参数
    inference_params = Column(JSON)                         # 推理参数
    raw_output = Column(JSON)                               # 原始输出
    processed_output = Column(JSON)                         # 处理后输出
    confidence_scores = Column(JSON)                        # 置信度分数
    prediction_summary = Column(JSON)                       # 预测摘要
    processing_time_ms = Column(Integer)                     # 处理时间（毫秒）
    gpu_memory_used_mb = Column(Integer)                     # GPU内存使用（MB）
    cpu_usage_percent = Column(DECIMAL(5, 2))               # CPU使用率
    status = Column(String(20), default='pending')          # pending, running, completed, failed
    error_message = Column(Text)                             # 错误信息
    quality_score = Column(DECIMAL(3, 2))                   # 推理质量评分
    is_reviewed = Column(Boolean, default=False)            # 是否已审核
    reviewed_by = Column(UUID(as_uuid=True), ForeignKey('users.user_id'))
    review_notes = Column(Text)                              # 审核备注
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    completed_at = Column(DateTime(timezone=True))
    reviewed_at = Column(DateTime(timezone=True))
    
    # 关联关系
    image = relationship("Image")
    model = relationship("AIModel", back_populates="inferences")
    model_version = relationship("ModelVersion", back_populates="inferences")
    user = relationship("User", foreign_keys=[user_id])
    reviewer = relationship("User", foreign_keys=[reviewed_by])
    
    def __repr__(self):
        return f"<Inference(inference_id='{self.inference_id}', status='{self.status}', processing_time={self.processing_time_ms}ms)>"

class ModelTrainingJob(Base):
    """模型训练任务模型"""
    __tablename__ = 'model_training_jobs'
    
    job_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    model_id = Column(UUID(as_uuid=True), ForeignKey('ai_models.model_id'), nullable=False)
    job_name = Column(String(200), nullable=False)
    description = Column(Text)
    training_config = Column(JSON, nullable=False)          # 训练配置
    dataset_config = Column(JSON, nullable=False)           # 数据集配置
    hyperparameters = Column(JSON)                          # 超参数
    resource_allocation = Column(JSON)                      # 资源分配
    status = Column(String(20), default='created')          # created, queued, running, completed, failed, cancelled
    progress_percent = Column(DECIMAL(5, 2), default=0.0)   # 进度百分比
    current_epoch = Column(Integer, default=0)
    total_epochs = Column(Integer)
    training_metrics = Column(JSON)                         # 训练指标历史
    validation_metrics = Column(JSON)                       # 验证指标历史
    resource_usage = Column(JSON)                           # 资源使用情况
    logs_path = Column(String(500))                          # 日志文件路径
    checkpoint_path = Column(String(500))                    # 检查点路径
    output_model_path = Column(String(500))                  # 输出模型路径
    error_message = Column(Text)                             # 错误信息
    created_by = Column(UUID(as_uuid=True), ForeignKey('users.user_id'))
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    started_at = Column(DateTime(timezone=True))
    completed_at = Column(DateTime(timezone=True))
    
    # 关联关系
    model = relationship("AIModel")
    creator = relationship("User")
    
    def __repr__(self):
        return f"<ModelTrainingJob(job_id='{self.job_id}', job_name='{self.job_name}', status='{self.status}')>"