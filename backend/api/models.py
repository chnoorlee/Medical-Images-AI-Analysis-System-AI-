from fastapi import APIRouter, Depends, HTTPException, status, UploadFile, File, Form, Query, BackgroundTasks
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List, Union
from datetime import datetime
import logging
import uuid
import json
import numpy as np
from PIL import Image
import io

from backend.services.model_service import ModelService, InferenceResult
from backend.api.auth import get_current_active_user, require_permission
from backend.models.ai_model import AIModel, ModelVersion, Inference, ModelTrainingJob
from backend.core.database import get_db_context

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/models", tags=["AI模型管理"])
model_service = ModelService()

# Pydantic模型
class ModelRegisterRequest(BaseModel):
    """模型注册请求模型"""
    name: str = Field(..., min_length=1, max_length=100, description="模型名称")
    description: str = Field(..., min_length=1, max_length=500, description="模型描述")
    model_type: str = Field(..., description="模型类型 (classification/segmentation/detection/regression)")
    architecture: str = Field(..., description="模型架构")
    version: str = Field("1.0.0", description="版本号")
    config: Optional[Dict[str, Any]] = Field(None, description="模型配置")

class ModelTrainingRequest(BaseModel):
    """模型训练请求模型"""
    training_data_config: Dict[str, Any] = Field(..., description="训练数据配置")
    training_config: Dict[str, Any] = Field(..., description="训练配置")
    validation_split: float = Field(0.2, ge=0.1, le=0.5, description="验证集比例")
    epochs: int = Field(10, ge=1, le=1000, description="训练轮数")
    batch_size: int = Field(16, ge=1, le=128, description="批次大小")
    learning_rate: float = Field(0.001, gt=0, le=1, description="学习率")
    early_stopping: bool = Field(True, description="是否启用早停")
    save_best_only: bool = Field(True, description="是否只保存最佳模型")

class InferenceRequest(BaseModel):
    """推理请求模型"""
    model_version: Optional[str] = Field(None, description="模型版本")
    preprocessing_config: Optional[Dict[str, Any]] = Field(None, description="预处理配置")
    return_probabilities: bool = Field(True, description="是否返回概率")
    return_features: bool = Field(False, description="是否返回特征")

class BatchInferenceRequest(BaseModel):
    """批量推理请求模型"""
    image_ids: List[str] = Field(..., description="图像ID列表")
    model_version: Optional[str] = Field(None, description="模型版本")
    batch_size: int = Field(8, ge=1, le=32, description="批次大小")
    preprocessing_config: Optional[Dict[str, Any]] = Field(None, description="预处理配置")

class ModelResponse(BaseModel):
    """模型信息响应模型"""
    model_id: str
    name: str
    description: str
    model_type: str
    architecture: str
    status: str
    latest_version: Optional[str]
    created_at: datetime
    updated_at: Optional[datetime]
    versions: List[Dict[str, Any]]

class InferenceResponse(BaseModel):
    """推理响应模型"""
    success: bool
    predictions: Dict[str, Any]
    confidence_scores: Dict[str, float]
    processing_time: float
    model_version: str
    metadata: Dict[str, Any]

class TrainingJobResponse(BaseModel):
    """训练任务响应模型"""
    job_id: str
    model_id: str
    status: str
    progress: Optional[float]
    current_epoch: Optional[int]
    total_epochs: Optional[int]
    metrics: Optional[Dict[str, Any]]
    started_at: datetime
    estimated_completion: Optional[datetime]
    error_message: Optional[str]

# API端点
@router.post("/register", summary="注册新模型")
async def register_model(
    model_data: ModelRegisterRequest,
    current_user: Dict[str, Any] = Depends(require_permission("model_management"))
):
    """注册新的AI模型
    
    创建新的模型记录，包括模型基本信息和初始版本。
    """
    try:
        result = await model_service.register_model(
            name=model_data.name,
            description=model_data.description,
            model_type=model_data.model_type,
            architecture=model_data.architecture,
            version=model_data.version,
            created_by=current_user['user_id'],
            config=model_data.config
        )
        
        if result['success']:
            return {
                "success": True,
                "message": "模型注册成功",
                "model_id": result['model_id'],
                "version_id": result['version_id']
            }
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=result['error']
            )
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"模型注册失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="模型注册失败"
        )

@router.get("/list", summary="获取模型列表")
async def get_models(
    skip: int = Query(0, ge=0, description="跳过数量"),
    limit: int = Query(20, ge=1, le=100, description="返回数量"),
    model_type: Optional[str] = Query(None, description="模型类型筛选"),
    status: Optional[str] = Query(None, description="状态筛选"),
    current_user: Dict[str, Any] = Depends(require_permission("model_view"))
):
    """获取模型列表
    
    返回系统中所有AI模型的列表，支持分页和筛选。
    """
    try:
        models = await model_service.get_model_list()
        
        # 应用筛选
        if model_type:
            models = [m for m in models if m['model_type'] == model_type]
        if status:
            models = [m for m in models if m['status'] == status]
        
        # 应用分页
        total = len(models)
        models = models[skip:skip + limit]
        
        return {
            "success": True,
            "models": models,
            "total": total,
            "skip": skip,
            "limit": limit
        }
        
    except Exception as e:
        logger.error(f"获取模型列表失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="获取模型列表失败"
        )

@router.get("/{model_id}", response_model=ModelResponse, summary="获取模型详细信息")
async def get_model_info(
    model_id: str,
    current_user: Dict[str, Any] = Depends(require_permission("model_view"))
):
    """获取模型详细信息
    
    返回指定模型的详细信息，包括所有版本。
    """
    try:
        model_info = await model_service.get_model_info(model_id)
        
        if 'error' in model_info:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=model_info['error']
            )
        
        return ModelResponse(**model_info)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"获取模型信息失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="获取模型信息失败"
        )

@router.post("/{model_id}/load", summary="加载模型")
async def load_model(
    model_id: str,
    version: Optional[str] = Query(None, description="模型版本"),
    current_user: Dict[str, Any] = Depends(require_permission("model_inference"))
):
    """加载模型到内存
    
    将指定版本的模型加载到内存中，准备进行推理。
    """
    try:
        result = await model_service.load_model(model_id, version)
        
        if result['success']:
            return {
                "success": True,
                "message": "模型加载成功",
                "cache_key": result.get('cache_key')
            }
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=result['error']
            )
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"模型加载失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="模型加载失败"
        )

@router.post("/{model_id}/unload", summary="卸载模型")
async def unload_model(
    model_id: str,
    version: Optional[str] = Query(None, description="模型版本"),
    current_user: Dict[str, Any] = Depends(require_permission("model_management"))
):
    """卸载模型
    
    从内存中卸载指定的模型。
    """
    try:
        result = await model_service.unload_model(model_id, version)
        
        if result['success']:
            return {
                "success": True,
                "message": result['message']
            }
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=result['error']
            )
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"模型卸载失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="模型卸载失败"
        )

@router.post("/{model_id}/predict", response_model=InferenceResponse, summary="单图像推理")
async def predict_image(
    model_id: str,
    file: UploadFile = File(..., description="图像文件"),
    config: Optional[str] = Form(None, description="推理配置(JSON格式)"),
    current_user: Dict[str, Any] = Depends(require_permission("model_inference"))
):
    """单图像推理
    
    对上传的图像进行AI模型推理。
    """
    try:
        # 验证文件类型
        allowed_types = ['image/jpeg', 'image/png', 'image/tiff']
        if file.content_type not in allowed_types:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"不支持的文件类型: {file.content_type}"
            )
        
        # 解析配置
        inference_config = {}
        if config:
            try:
                inference_config = json.loads(config)
            except json.JSONDecodeError:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="配置格式错误"
                )
        
        # 读取图像
        file_content = await file.read()
        image = Image.open(io.BytesIO(file_content))
        image_array = np.array(image)
        
        # 执行推理
        result = await model_service.predict(
            model_id=model_id,
            image=image_array,
            version=inference_config.get('model_version'),
            preprocessing_config=inference_config.get('preprocessing_config')
        )
        
        return InferenceResponse(
            success=True,
            predictions=result.predictions,
            confidence_scores=result.confidence_scores,
            processing_time=result.processing_time,
            model_version=result.model_version,
            metadata=result.metadata
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"图像推理失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="图像推理失败"
        )

@router.post("/{model_id}/predict/batch", summary="批量推理")
async def batch_predict(
    model_id: str,
    request: BatchInferenceRequest,
    background_tasks: BackgroundTasks,
    current_user: Dict[str, Any] = Depends(require_permission("model_inference"))
):
    """批量推理
    
    对多个图像进行批量推理。
    """
    try:
        if len(request.image_ids) > 100:  # 限制批量推理数量
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="批量推理图像数量不能超过100个"
            )
        
        # 启动后台任务进行批量推理
        task_id = str(uuid.uuid4())
        background_tasks.add_task(
            _execute_batch_inference,
            task_id,
            model_id,
            request.image_ids,
            request.model_version,
            request.batch_size,
            request.preprocessing_config,
            current_user['user_id']
        )
        
        return {
            "success": True,
            "message": "批量推理任务已启动",
            "task_id": task_id,
            "image_count": len(request.image_ids)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"批量推理失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="批量推理失败"
        )

@router.post("/{model_id}/train", summary="训练模型")
async def train_model(
    model_id: str,
    training_request: ModelTrainingRequest,
    background_tasks: BackgroundTasks,
    current_user: Dict[str, Any] = Depends(require_permission("model_training"))
):
    """训练模型
    
    启动模型训练任务。
    """
    try:
        # 准备训练配置
        training_config = {
            'epochs': training_request.epochs,
            'batch_size': training_request.batch_size,
            'learning_rate': training_request.learning_rate,
            'validation_split': training_request.validation_split,
            'early_stopping': training_request.early_stopping,
            'save_best_only': training_request.save_best_only,
            **training_request.training_config
        }
        
        result = await model_service.train_model(
            model_id=model_id,
            training_data=training_request.training_data_config,
            training_config=training_config,
            created_by=current_user['user_id']
        )
        
        if result['success']:
            return {
                "success": True,
                "message": "训练任务已启动",
                "job_id": result['job_id']
            }
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=result['error']
            )
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"启动训练失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="启动训练失败"
        )

@router.get("/training/jobs", summary="获取训练任务列表")
async def get_training_jobs(
    skip: int = Query(0, ge=0, description="跳过数量"),
    limit: int = Query(20, ge=1, le=100, description="返回数量"),
    status: Optional[str] = Query(None, description="状态筛选"),
    current_user: Dict[str, Any] = Depends(require_permission("model_view"))
):
    """获取训练任务列表
    
    返回系统中的模型训练任务列表。
    """
    try:
        with get_db_context() as db:
            query = db.query(ModelTrainingJob)
            
            if status:
                query = query.filter(ModelTrainingJob.status == status)
            
            total = query.count()
            jobs = query.offset(skip).limit(limit).all()
            
            job_list = []
            for job in jobs:
                job_list.append({
                    "job_id": str(job.job_id),
                    "model_id": str(job.model_id),
                    "status": job.status,
                    "training_config": json.loads(job.training_config) if job.training_config else {},
                    "metrics": json.loads(job.metrics) if job.metrics else {},
                    "started_at": job.started_at.isoformat() if job.started_at else None,
                    "completed_at": job.completed_at.isoformat() if job.completed_at else None,
                    "error_message": job.error_message
                })
            
            return {
                "success": True,
                "jobs": job_list,
                "total": total,
                "skip": skip,
                "limit": limit
            }
            
    except Exception as e:
        logger.error(f"获取训练任务列表失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="获取训练任务列表失败"
        )

@router.get("/training/jobs/{job_id}", response_model=TrainingJobResponse, summary="获取训练任务详情")
async def get_training_job(
    job_id: str,
    current_user: Dict[str, Any] = Depends(require_permission("model_view"))
):
    """获取训练任务详情
    
    返回指定训练任务的详细信息。
    """
    try:
        with get_db_context() as db:
            job = db.query(ModelTrainingJob).filter(
                ModelTrainingJob.job_id == uuid.UUID(job_id)
            ).first()
            
            if not job:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="训练任务不存在"
                )
            
            metrics = json.loads(job.metrics) if job.metrics else {}
            
            return TrainingJobResponse(
                job_id=str(job.job_id),
                model_id=str(job.model_id),
                status=job.status,
                progress=metrics.get('progress'),
                current_epoch=metrics.get('current_epoch'),
                total_epochs=metrics.get('total_epochs'),
                metrics=metrics,
                started_at=job.started_at,
                estimated_completion=None,  # 可以根据进度计算
                error_message=job.error_message
            )
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"获取训练任务详情失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="获取训练任务详情失败"
        )

@router.delete("/training/jobs/{job_id}", summary="取消训练任务")
async def cancel_training_job(
    job_id: str,
    current_user: Dict[str, Any] = Depends(require_permission("model_training"))
):
    """取消训练任务
    
    取消正在运行的训练任务。
    """
    try:
        with get_db_context() as db:
            job = db.query(ModelTrainingJob).filter(
                ModelTrainingJob.job_id == uuid.UUID(job_id)
            ).first()
            
            if not job:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="训练任务不存在"
                )
            
            if job.status not in ['running', 'pending']:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="只能取消运行中或待运行的任务"
                )
            
            job.status = 'cancelled'
            job.completed_at = datetime.now()
            db.commit()
            
            return {
                "success": True,
                "message": "训练任务已取消"
            }
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"取消训练任务失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="取消训练任务失败"
        )

@router.get("/loaded", summary="获取已加载的模型")
async def get_loaded_models(
    current_user: Dict[str, Any] = Depends(require_permission("model_view"))
):
    """获取已加载的模型
    
    返回当前在内存中加载的所有模型。
    """
    try:
        loaded_models = await model_service.get_loaded_models()
        
        return {
            "success": True,
            "loaded_models": loaded_models
        }
        
    except Exception as e:
        logger.error(f"获取已加载模型失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="获取已加载模型失败"
        )

@router.get("/inference/history", summary="获取推理历史")
async def get_inference_history(
    skip: int = Query(0, ge=0, description="跳过数量"),
    limit: int = Query(20, ge=1, le=100, description="返回数量"),
    model_id: Optional[str] = Query(None, description="模型ID筛选"),
    date_from: Optional[datetime] = Query(None, description="开始日期"),
    date_to: Optional[datetime] = Query(None, description="结束日期"),
    current_user: Dict[str, Any] = Depends(require_permission("model_view"))
):
    """获取推理历史
    
    返回系统中的推理记录历史。
    """
    try:
        with get_db_context() as db:
            query = db.query(Inference)
            
            if model_id:
                query = query.filter(Inference.model_id == uuid.UUID(model_id))
            if date_from:
                query = query.filter(Inference.created_at >= date_from)
            if date_to:
                query = query.filter(Inference.created_at <= date_to)
            
            total = query.count()
            inferences = query.order_by(Inference.created_at.desc()).offset(skip).limit(limit).all()
            
            inference_list = []
            for inference in inferences:
                inference_list.append({
                    "inference_id": str(inference.inference_id),
                    "model_id": str(inference.model_id),
                    "version": inference.version,
                    "input_data": json.loads(inference.input_data) if inference.input_data else {},
                    "output_data": json.loads(inference.output_data) if inference.output_data else {},
                    "confidence_score": inference.confidence_score,
                    "processing_time": inference.processing_time,
                    "created_at": inference.created_at.isoformat()
                })
            
            return {
                "success": True,
                "inferences": inference_list,
                "total": total,
                "skip": skip,
                "limit": limit
            }
            
    except Exception as e:
        logger.error(f"获取推理历史失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="获取推理历史失败"
        )

@router.get("/statistics/overview", summary="获取模型统计概览")
async def get_model_statistics(
    current_user: Dict[str, Any] = Depends(require_permission("model_view"))
):
    """获取模型统计概览
    
    返回系统中模型的统计信息。
    """
    try:
        with get_db_context() as db:
            # 模型总数
            total_models = db.query(AIModel).count()
            
            # 按类型分组
            model_types = db.query(AIModel.model_type, db.func.count(AIModel.model_id)).group_by(AIModel.model_type).all()
            
            # 按状态分组
            model_statuses = db.query(AIModel.status, db.func.count(AIModel.model_id)).group_by(AIModel.status).all()
            
            # 推理统计
            total_inferences = db.query(Inference).count()
            avg_processing_time = db.query(db.func.avg(Inference.processing_time)).scalar() or 0
            
            # 训练任务统计
            total_training_jobs = db.query(ModelTrainingJob).count()
            running_jobs = db.query(ModelTrainingJob).filter(ModelTrainingJob.status == 'running').count()
            
            return {
                "success": True,
                "statistics": {
                    "total_models": total_models,
                    "model_types": dict(model_types),
                    "model_statuses": dict(model_statuses),
                    "total_inferences": total_inferences,
                    "avg_processing_time": round(avg_processing_time, 3),
                    "total_training_jobs": total_training_jobs,
                    "running_training_jobs": running_jobs
                }
            }
            
    except Exception as e:
        logger.error(f"获取模型统计失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="获取模型统计失败"
        )

# 辅助函数
async def _execute_batch_inference(task_id: str, model_id: str, image_ids: List[str],
                                  model_version: Optional[str], batch_size: int,
                                  preprocessing_config: Optional[Dict[str, Any]],
                                  user_id: str):
    """执行批量推理的后台任务"""
    try:
        # 这里应该实现实际的批量推理逻辑
        # 包括从数据库获取图像、执行推理、保存结果等
        logger.info(f"开始执行批量推理任务: {task_id}")
        
        # 模拟批量推理过程
        # 实际实现中需要:
        # 1. 从数据库获取图像数据
        # 2. 分批执行推理
        # 3. 保存推理结果
        # 4. 更新任务状态
        
        logger.info(f"批量推理任务完成: {task_id}")
        
    except Exception as e:
        logger.error(f"批量推理任务失败: {task_id}, 错误: {e}")