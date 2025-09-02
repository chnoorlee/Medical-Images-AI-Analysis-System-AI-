from fastapi import APIRouter, Depends, HTTPException, status, UploadFile, File, Form, Query
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List, Union
from datetime import datetime
import logging
import uuid
import json
import io
from PIL import Image
import numpy as np

from backend.services.image_service import ImageService
from backend.services.preprocessing_service import PreprocessingService
from backend.services.feature_extraction_service import FeatureExtractionService
from backend.services.quality_service import QualityService
from backend.api.auth import get_current_active_user, require_permission
from backend.models.patient import Image as ImageModel
from backend.core.database import get_db_context

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/images", tags=["图像管理"])
image_service = ImageService()
preprocessing_service = PreprocessingService()
feature_service = FeatureExtractionService()
quality_service = QualityService()

# Pydantic模型
class ImageUploadResponse(BaseModel):
    """图像上传响应模型"""
    success: bool
    image_id: Optional[str] = None
    message: str
    file_path: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

class ImageMetadata(BaseModel):
    """图像元数据模型"""
    patient_id: Optional[str] = Field(None, description="患者ID")
    study_id: Optional[str] = Field(None, description="检查ID")
    series_id: Optional[str] = Field(None, description="序列ID")
    modality: Optional[str] = Field(None, description="影像模态")
    body_part: Optional[str] = Field(None, description="检查部位")
    acquisition_date: Optional[datetime] = Field(None, description="采集日期")
    institution: Optional[str] = Field(None, description="医疗机构")
    description: Optional[str] = Field(None, description="图像描述")
    tags: Optional[List[str]] = Field(None, description="标签")

class PreprocessingConfig(BaseModel):
    """预处理配置模型"""
    resize: Optional[tuple] = Field(None, description="调整大小")
    normalize: bool = Field(True, description="是否归一化")
    denoise: bool = Field(False, description="是否去噪")
    enhance_contrast: bool = Field(False, description="是否增强对比度")
    window_level: Optional[Dict[str, float]] = Field(None, description="窗宽窗位")
    custom_params: Optional[Dict[str, Any]] = Field(None, description="自定义参数")

class FeatureExtractionConfig(BaseModel):
    """特征提取配置模型"""
    extract_texture: bool = Field(True, description="提取纹理特征")
    extract_shape: bool = Field(True, description="提取形状特征")
    extract_statistical: bool = Field(True, description="提取统计特征")
    extract_frequency: bool = Field(False, description="提取频域特征")
    extract_deep: bool = Field(False, description="提取深度特征")
    deep_model: Optional[str] = Field("resnet50", description="深度学习模型")

class ImageSearchParams(BaseModel):
    """图像搜索参数模型"""
    patient_id: Optional[str] = None
    study_id: Optional[str] = None
    modality: Optional[str] = None
    body_part: Optional[str] = None
    date_from: Optional[datetime] = None
    date_to: Optional[datetime] = None
    tags: Optional[List[str]] = None
    quality_min: Optional[float] = None
    has_annotations: Optional[bool] = None

class ImageResponse(BaseModel):
    """图像信息响应模型"""
    image_id: str
    filename: str
    file_path: str
    file_size: int
    image_format: str
    width: int
    height: int
    channels: int
    metadata: Dict[str, Any]
    quality_score: Optional[float]
    processing_status: str
    created_at: datetime
    updated_at: Optional[datetime]

class BatchProcessingRequest(BaseModel):
    """批量处理请求模型"""
    image_ids: List[str]
    preprocessing_config: Optional[PreprocessingConfig] = None
    feature_extraction_config: Optional[FeatureExtractionConfig] = None
    quality_assessment: bool = Field(True, description="是否进行质量评估")

# API端点
@router.post("/upload", response_model=ImageUploadResponse, summary="上传图像")
async def upload_image(
    file: UploadFile = File(..., description="图像文件"),
    metadata: Optional[str] = Form(None, description="图像元数据(JSON格式)"),
    auto_process: bool = Form(True, description="是否自动处理"),
    current_user: Dict[str, Any] = Depends(require_permission("image.create"))
):
    """上传医学图像
    
    支持DICOM、JPEG、PNG等格式的医学图像上传。
    可以同时上传图像元数据，并选择是否自动进行预处理和质量评估。
    """
    try:
        # 验证文件类型
        allowed_types = ['image/jpeg', 'image/png', 'image/tiff', 'application/dicom']
        if file.content_type not in allowed_types:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"不支持的文件类型: {file.content_type}"
            )
        
        # 解析元数据
        parsed_metadata = {}
        if metadata:
            try:
                parsed_metadata = json.loads(metadata)
            except json.JSONDecodeError:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="元数据格式错误"
                )
        
        # 读取文件内容
        file_content = await file.read()
        
        # 上传图像
        result = await image_service.upload_image_from_content(
            file_content=file_content,
            filename=file.filename,
            content_type=file.content_type,
            metadata=parsed_metadata,
            uploaded_by=current_user['user_id'],
            auto_process=auto_process
        )
        
        if result['success']:
            return ImageUploadResponse(
                success=True,
                image_id=result['image_id'],
                message="图像上传成功",
                file_path=result.get('file_path'),
                metadata=result.get('metadata')
            )
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=result['error']
            )
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"图像上传失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="图像上传失败"
        )

@router.post("/upload/batch", summary="批量上传图像")
async def upload_images_batch(
    files: List[UploadFile] = File(..., description="图像文件列表"),
    auto_process: bool = Form(True, description="是否自动处理"),
    current_user: Dict[str, Any] = Depends(require_permission("image.create"))
):
    """批量上传医学图像
    
    一次性上传多个图像文件。
    """
    try:
        if len(files) > 50:  # 限制批量上传数量
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="批量上传文件数量不能超过50个"
            )
        
        results = []
        for file in files:
            try:
                file_content = await file.read()
                result = await image_service.upload_image(
                    file_content=file_content,
                    filename=file.filename,
                    content_type=file.content_type,
                    metadata={},
                    uploaded_by=current_user['user_id'],
                    auto_process=auto_process
                )
                results.append({
                    'filename': file.filename,
                    'success': result['success'],
                    'image_id': result.get('image_id'),
                    'error': result.get('error')
                })
            except Exception as e:
                results.append({
                    'filename': file.filename,
                    'success': False,
                    'error': str(e)
                })
        
        successful_uploads = sum(1 for r in results if r['success'])
        
        return {
            "success": True,
            "message": f"批量上传完成，成功: {successful_uploads}/{len(files)}",
            "results": results
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"批量上传失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="批量上传失败"
        )

@router.get("/list", summary="获取图像列表")
async def get_images(
    skip: int = Query(0, ge=0, description="跳过数量"),
    limit: int = Query(20, ge=1, le=100, description="返回数量"),
    patient_id: Optional[str] = Query(None, description="患者ID"),
    modality: Optional[str] = Query(None, description="影像模态"),
    body_part: Optional[str] = Query(None, description="检查部位"),
    date_from: Optional[datetime] = Query(None, description="开始日期"),
    date_to: Optional[datetime] = Query(None, description="结束日期"),
    current_user: Dict[str, Any] = Depends(require_permission("image_view"))
):
    """获取图像列表
    
    支持多种筛选条件的图像列表查询。
    """
    try:
        search_params = ImageSearchParams(
            patient_id=patient_id,
            modality=modality,
            body_part=body_part,
            date_from=date_from,
            date_to=date_to
        )
        
        result = await image_service.search_images(
            search_params=search_params.dict(exclude_unset=True),
            skip=skip,
            limit=limit
        )
        
        return {
            "success": True,
            "images": result['images'],
            "total": result['total'],
            "skip": skip,
            "limit": limit
        }
        
    except Exception as e:
        logger.error(f"获取图像列表失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="获取图像列表失败"
        )

@router.get("/{image_id}", response_model=ImageResponse, summary="获取图像信息")
async def get_image_info(
    image_id: str,
    current_user: Dict[str, Any] = Depends(require_permission("image_view"))
):
    """获取图像详细信息
    
    返回指定图像的详细信息，包括元数据、质量评分等。
    """
    try:
        with get_db_context() as db:
            image = db.query(ImageModel).filter(
                ImageModel.image_id == uuid.UUID(image_id)
            ).first()
            
            if not image:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="图像不存在"
                )
            
            return ImageResponse(
                image_id=str(image.image_id),
                filename=image.filename,
                file_path=image.file_path,
                file_size=image.file_size,
                image_format=image.image_format,
                width=image.width,
                height=image.height,
                channels=image.channels,
                metadata=json.loads(image.metadata) if image.metadata else {},
                quality_score=image.quality_score,
                processing_status=image.processing_status,
                created_at=image.created_at,
                updated_at=image.updated_at
            )
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"获取图像信息失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="获取图像信息失败"
        )

@router.get("/{image_id}/download", summary="下载图像")
async def download_image(
    image_id: str,
    format: Optional[str] = Query("original", description="下载格式"),
    current_user: Dict[str, Any] = Depends(require_permission("image_download"))
):
    """下载图像文件
    
    支持原始格式或转换后的格式下载。
    """
    try:
        result = await image_service.get_image_file(image_id, format)
        
        if not result['success']:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=result['error']
            )
        
        file_content = result['file_content']
        filename = result['filename']
        content_type = result['content_type']
        
        return StreamingResponse(
            io.BytesIO(file_content),
            media_type=content_type,
            headers={"Content-Disposition": f"attachment; filename={filename}"}
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"下载图像失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="下载图像失败"
        )

@router.get("/{image_id}/thumbnail", summary="获取图像缩略图")
async def get_image_thumbnail(
    image_id: str,
    size: int = Query(256, ge=64, le=512, description="缩略图大小"),
    current_user: Dict[str, Any] = Depends(require_permission("image_view"))
):
    """获取图像缩略图
    
    返回指定大小的图像缩略图。
    """
    try:
        result = await image_service.get_image_thumbnail(image_id, size)
        
        if not result['success']:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=result['error']
            )
        
        thumbnail_data = result['thumbnail_data']
        
        return StreamingResponse(
            io.BytesIO(thumbnail_data),
            media_type="image/jpeg",
            headers={"Content-Disposition": f"inline; filename=thumbnail_{image_id}.jpg"}
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"获取缩略图失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="获取缩略图失败"
        )

@router.post("/{image_id}/preprocess", summary="预处理图像")
async def preprocess_image(
    image_id: str,
    config: PreprocessingConfig,
    current_user: Dict[str, Any] = Depends(require_permission("image_process"))
):
    """预处理图像
    
    对指定图像进行预处理操作，如调整大小、去噪、增强对比度等。
    """
    try:
        result = await image_service.preprocess_image(
            image_id=image_id,
            config=config.dict(exclude_unset=True)
        )
        
        if result['success']:
            return {
                "success": True,
                "message": "图像预处理完成",
                "processed_image_id": result.get('processed_image_id'),
                "processing_info": result.get('processing_info')
            }
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=result['error']
            )
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"图像预处理失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="图像预处理失败"
        )

@router.post("/{image_id}/extract-features", summary="提取图像特征")
async def extract_image_features(
    image_id: str,
    config: FeatureExtractionConfig,
    current_user: Dict[str, Any] = Depends(require_permission("image_process"))
):
    """提取图像特征
    
    从指定图像中提取各种特征，包括纹理、形状、统计、频域和深度特征。
    """
    try:
        result = await image_service.extract_features(
            image_id=image_id,
            config=config.dict(exclude_unset=True)
        )
        
        if result['success']:
            return {
                "success": True,
                "message": "特征提取完成",
                "features": result.get('features'),
                "feature_count": result.get('feature_count')
            }
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=result['error']
            )
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"特征提取失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="特征提取失败"
        )

@router.post("/{image_id}/assess-quality", summary="评估图像质量")
async def assess_image_quality(
    image_id: str,
    current_user: Dict[str, Any] = Depends(require_permission("quality_assessment"))
):
    """评估图像质量
    
    对指定图像进行质量评估，包括技术质量和临床质量。
    """
    try:
        result = await image_service.assess_quality(image_id)
        
        if result['success']:
            return {
                "success": True,
                "message": "质量评估完成",
                "quality_assessment": result.get('quality_assessment')
            }
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=result['error']
            )
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"质量评估失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="质量评估失败"
        )

@router.post("/batch/process", summary="批量处理图像")
async def batch_process_images(
    request: BatchProcessingRequest,
    current_user: Dict[str, Any] = Depends(require_permission("image_process"))
):
    """批量处理图像
    
    对多个图像进行批量预处理、特征提取和质量评估。
    """
    try:
        if len(request.image_ids) > 100:  # 限制批量处理数量
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="批量处理图像数量不能超过100个"
            )
        
        result = await image_service.batch_process_images(
            image_ids=request.image_ids,
            preprocessing_config=request.preprocessing_config.dict(exclude_unset=True) if request.preprocessing_config else None,
            feature_extraction_config=request.feature_extraction_config.dict(exclude_unset=True) if request.feature_extraction_config else None,
            quality_assessment=request.quality_assessment
        )
        
        return {
            "success": True,
            "message": "批量处理已启动",
            "task_id": result.get('task_id'),
            "processed_count": result.get('processed_count'),
            "failed_count": result.get('failed_count')
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"批量处理失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="批量处理失败"
        )

@router.put("/{image_id}/metadata", summary="更新图像元数据")
async def update_image_metadata(
    image_id: str,
    metadata: ImageMetadata,
    current_user: Dict[str, Any] = Depends(require_permission("image_edit"))
):
    """更新图像元数据
    
    更新指定图像的元数据信息。
    """
    try:
        result = await image_service.update_image_metadata(
            image_id=image_id,
            metadata=metadata.dict(exclude_unset=True)
        )
        
        if result['success']:
            return {
                "success": True,
                "message": "图像元数据更新成功"
            }
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=result['error']
            )
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"更新图像元数据失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="更新图像元数据失败"
        )

@router.delete("/{image_id}", summary="删除图像")
async def delete_image(
    image_id: str,
    current_user: Dict[str, Any] = Depends(require_permission("image_delete"))
):
    """删除图像
    
    删除指定的图像及其相关数据。
    """
    try:
        result = await image_service.delete_image(image_id)
        
        if result['success']:
            return {
                "success": True,
                "message": "图像删除成功"
            }
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=result['error']
            )
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"删除图像失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="删除图像失败"
        )

@router.get("/statistics/overview", summary="获取图像统计概览")
async def get_image_statistics(
    current_user: Dict[str, Any] = Depends(require_permission("image_view"))
):
    """获取图像统计概览
    
    返回系统中图像的统计信息，包括总数、格式分布、质量分布等。
    """
    try:
        stats = await image_service.get_image_statistics()
        
        return {
            "success": True,
            "statistics": stats
        }
        
    except Exception as e:
        logger.error(f"获取图像统计失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="获取图像统计失败"
        )

@router.post("/search/similar", summary="相似图像搜索")
async def search_similar_images(
    image_id: str,
    limit: int = Query(10, ge=1, le=50, description="返回数量"),
    threshold: float = Query(0.8, ge=0.0, le=1.0, description="相似度阈值"),
    current_user: Dict[str, Any] = Depends(require_permission("image_view"))
):
    """相似图像搜索
    
    基于图像特征搜索相似的图像。
    """
    try:
        result = await image_service.search_similar_images(
            image_id=image_id,
            limit=limit,
            threshold=threshold
        )
        
        if result['success']:
            return {
                "success": True,
                "similar_images": result['similar_images'],
                "search_time": result.get('search_time')
            }
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=result['error']
            )
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"相似图像搜索失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="相似图像搜索失败"
        )

@router.get("/{image_id}/processing-history", summary="获取图像处理历史")
async def get_image_processing_history(
    image_id: str,
    current_user: Dict[str, Any] = Depends(require_permission("image_view"))
):
    """获取图像处理历史
    
    返回指定图像的所有处理记录。
    """
    try:
        result = await image_service.get_processing_history(image_id)
        
        if result['success']:
            return {
                "success": True,
                "processing_history": result['history']
            }
        else:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=result['error']
            )
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"获取处理历史失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="获取处理历史失败"
        )