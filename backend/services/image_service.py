import numpy as np
import cv2
from typing import Dict, List, Tuple, Optional, Any, Union
import logging
from pathlib import Path
import json
import uuid
from datetime import datetime, timezone
import asyncio
from concurrent.futures import ThreadPoolExecutor
import hashlib
import os

from .preprocessing_service import PreprocessingService
from .feature_extraction_service import FeatureExtractionService
from .storage_service import StorageService
from .quality_service import QualityService
from backend.models.patient import Image, ImageMetadata
from backend.core.database import get_db_context
from backend.core.config import settings

logger = logging.getLogger(__name__)

class ImageService:
    """图像服务
    
    提供完整的医学图像处理服务，包括：
    - 图像上传和存储
    - 图像预处理
    - 特征提取
    - 质量评估
    - 图像检索和管理
    """
    
    def __init__(self):
        self.preprocessing_service = PreprocessingService()
        self.feature_extraction_service = FeatureExtractionService()
        self.storage_service = StorageService()
        self.quality_service = QualityService()
        self.executor = ThreadPoolExecutor(max_workers=4)
    
    async def upload_image_from_content(self, file_content: bytes, filename: str, 
                                      content_type: str, metadata: Optional[Dict[str, Any]] = None,
                                      uploaded_by: str = None, auto_process: bool = True) -> Dict[str, Any]:
        """从文件内容上传图像
        
        Args:
            file_content: 文件内容字节
            filename: 文件名
            content_type: 文件MIME类型
            metadata: 额外的元数据
            uploaded_by: 上传者ID
            auto_process: 是否自动处理
            
        Returns:
            上传结果字典
        """
        import tempfile
        import os
        
        try:
            # 生成图像ID
            image_id = str(uuid.uuid4())
            
            # 创建临时文件
            suffix = Path(filename).suffix if filename else '.tmp'
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
                temp_file.write(file_content)
                temp_file_path = temp_file.name
            
            try:
                # 验证文件格式
                if suffix.lower() not in self.preprocessing_service.supported_formats:
                    raise ValueError(f"不支持的文件格式: {suffix}")
                
                # 计算文件哈希
                file_hash = await self._calculate_file_hash(temp_file_path)
                
                # 检查是否已存在相同的图像
                existing_image = await self._check_duplicate_image(file_hash)
                if existing_image:
                    logger.warning(f"发现重复图像: {file_hash}")
                    return {
                        'success': False,
                        'error': '图像已存在',
                        'existing_image_id': existing_image['image_id']
                    }
                
                # 处理图像（如果启用自动处理）
                processing_result = {}
                if auto_process:
                    processing_result = await self._process_image_async(temp_file_path, image_id)
                
                # 存储到云存储
                storage_result = await self.storage_service.upload_file(
                    temp_file_path, f"images/{image_id}/{filename}"
                )
                
                # 保存到数据库（使用默认series_id）
                default_series_id = str(uuid.uuid4())
                image_record = await self._save_image_to_database(
                    image_id=image_id,
                    series_id=default_series_id,
                    file_path=storage_result['url'],
                    file_hash=file_hash,
                    file_size=len(file_content),
                    processing_result=processing_result,
                    metadata=metadata
                )
                
                logger.info(f"图像上传成功: {image_id}")
                return {
                    'success': True,
                    'image_id': image_id,
                    'file_path': storage_result['url'],
                    'processing_info': processing_result,
                    'metadata': metadata
                }
                
            finally:
                # 清理临时文件
                if os.path.exists(temp_file_path):
                    os.unlink(temp_file_path)
                    
        except Exception as e:
            logger.error(f"图像上传失败: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    async def upload_image(self, file_path: str, series_id: str, 
                          metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """上传图像文件
        
        Args:
            file_path: 图像文件路径
            series_id: 所属系列ID
            metadata: 额外的元数据
            
        Returns:
            上传结果字典
        """
        try:
            # 验证文件
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"文件不存在: {file_path}")
            
            file_path_obj = Path(file_path)
            if file_path_obj.suffix.lower() not in self.preprocessing_service.supported_formats:
                raise ValueError(f"不支持的文件格式: {file_path_obj.suffix}")
            
            # 生成图像ID
            image_id = str(uuid.uuid4())
            
            # 计算文件哈希
            file_hash = await self._calculate_file_hash(file_path)
            
            # 检查是否已存在相同的图像
            existing_image = await self._check_duplicate_image(file_hash)
            if existing_image:
                logger.warning(f"发现重复图像: {file_hash}")
                return {
                    'status': 'duplicate',
                    'image_id': existing_image['image_id'],
                    'message': '图像已存在'
                }
            
            # 异步处理图像
            processing_result = await self._process_image_async(file_path, image_id)
            
            # 存储到云存储
            storage_result = await self.storage_service.upload_file(
                file_path, f"images/{image_id}/{file_path_obj.name}"
            )
            
            # 保存到数据库
            image_record = await self._save_image_to_database(
                image_id=image_id,
                series_id=series_id,
                file_path=storage_result['url'],
                file_hash=file_hash,
                file_size=os.path.getsize(file_path),
                processing_result=processing_result,
                metadata=metadata
            )
            
            logger.info(f"图像上传成功: {image_id}")
            return {
                'status': 'success',
                'image_id': image_id,
                'storage_url': storage_result['url'],
                'processing_info': processing_result,
                'message': '图像上传并处理完成'
            }
            
        except Exception as e:
            logger.error(f"图像上传失败: {e}")
            raise
    
    async def _process_image_async(self, file_path: str, image_id: str) -> Dict[str, Any]:
        """异步处理图像"""
        loop = asyncio.get_event_loop()
        
        # 在线程池中执行CPU密集型任务
        processing_result = await loop.run_in_executor(
            self.executor, self._process_image_sync, file_path, image_id
        )
        
        return processing_result
    
    def _process_image_sync(self, file_path: str, image_id: str) -> Dict[str, Any]:
        """同步处理图像（在线程池中执行）"""
        try:
            result = {
                'image_id': image_id,
                'preprocessing': {},
                'features': {},
                'quality': {},
                'thumbnails': []
            }
            
            # 1. 预处理
            preprocessing_config = self.preprocessing_service.get_default_config()
            processed_image, preprocessing_info = self.preprocessing_service.preprocess_pipeline(
                file_path, preprocessing_config
            )
            result['preprocessing'] = preprocessing_info
            
            # 2. 特征提取
            features = self.feature_extraction_service.extract_all_features(
                processed_image, include_cnn=True
            )
            result['features'] = features
            
            # 3. 质量评估
            quality_metrics = self.quality_service.assess_image_quality(
                processed_image, preprocessing_info.get('metadata', {})
            )
            result['quality'] = quality_metrics
            
            # 4. 生成缩略图
            thumbnails = self._generate_thumbnails(processed_image, image_id)
            result['thumbnails'] = thumbnails
            
            # 5. 保存处理结果
            self._save_processing_results(image_id, result)
            
            return result
            
        except Exception as e:
            logger.error(f"图像处理失败: {e}")
            raise
    
    def _generate_thumbnails(self, image: np.ndarray, image_id: str) -> List[Dict[str, Any]]:
        """生成缩略图"""
        thumbnails = []
        thumbnail_sizes = [(128, 128), (256, 256), (512, 512)]
        
        try:
            for size in thumbnail_sizes:
                # 调整尺寸
                thumbnail = cv2.resize(image, size, interpolation=cv2.INTER_AREA)
                
                # 转换为uint8
                if thumbnail.dtype != np.uint8:
                    thumbnail = ((thumbnail - thumbnail.min()) / 
                               (thumbnail.max() - thumbnail.min()) * 255).astype(np.uint8)
                
                # 保存缩略图
                thumbnail_filename = f"thumbnail_{size[0]}x{size[1]}.jpg"
                thumbnail_path = os.path.join(settings.upload_dir, "thumbnails", 
                                            image_id, thumbnail_filename)
                
                os.makedirs(os.path.dirname(thumbnail_path), exist_ok=True)
                cv2.imwrite(thumbnail_path, thumbnail)
                
                thumbnails.append({
                    'size': size,
                    'path': thumbnail_path,
                    'filename': thumbnail_filename
                })
            
            return thumbnails
            
        except Exception as e:
            logger.error(f"生成缩略图失败: {e}")
            return []
    
    def _save_processing_results(self, image_id: str, results: Dict[str, Any]) -> None:
        """保存处理结果到文件"""
        try:
            results_dir = os.path.join(settings.upload_dir, "processing_results", image_id)
            os.makedirs(results_dir, exist_ok=True)
            
            # 保存特征
            features_path = os.path.join(results_dir, "features.json")
            self.feature_extraction_service.save_features_to_json(
                results['features'], features_path
            )
            
            # 保存完整结果
            results_path = os.path.join(results_dir, "processing_results.json")
            with open(results_path, 'w', encoding='utf-8') as f:
                # 确保结果可序列化
                serializable_results = self._make_serializable(results)
                json.dump(serializable_results, f, indent=2, ensure_ascii=False)
            
            logger.debug(f"处理结果已保存: {results_dir}")
            
        except Exception as e:
            logger.error(f"保存处理结果失败: {e}")
    
    def _make_serializable(self, obj: Any) -> Any:
        """使对象可序列化"""
        if isinstance(obj, dict):
            return {k: self._make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_serializable(item) for item in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        elif isinstance(obj, datetime):
            return obj.isoformat()
        else:
            return obj
    
    async def _calculate_file_hash(self, file_path: str) -> str:
        """计算文件哈希值"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self.executor, self._calculate_hash_sync, file_path)
    
    def _calculate_hash_sync(self, file_path: str) -> str:
        """同步计算文件哈希"""
        hash_md5 = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()
    
    async def _check_duplicate_image(self, file_hash: str) -> Optional[Dict[str, Any]]:
        """检查重复图像"""
        try:
            with get_db_context() as db:
                existing_image = db.query(Image).filter(
                    Image.file_hash == file_hash
                ).first()
                
                if existing_image:
                    return {
                        'image_id': str(existing_image.image_id),
                        'file_path': existing_image.file_path,
                        'created_at': existing_image.created_at
                    }
                
                return None
                
        except Exception as e:
            logger.error(f"检查重复图像失败: {e}")
            return None
    
    async def _save_image_to_database(self, image_id: str, series_id: str, 
                                    file_path: str, file_hash: str, file_size: int,
                                    processing_result: Dict[str, Any],
                                    metadata: Optional[Dict[str, Any]] = None) -> Image:
        """保存图像记录到数据库"""
        try:
            with get_db_context() as db:
                # 创建图像记录
                image_record = Image(
                    image_id=uuid.UUID(image_id),
                    series_id=uuid.UUID(series_id),
                    file_path=file_path,
                    image_hash=file_hash,
                    file_size=file_size,
                    processing_status='completed',
                    quality_score=processing_result.get('quality', {}).get('overall_score', 0.0)
                )
                
                db.add(image_record)
                db.flush()
                
                # 创建元数据记录
                if metadata or processing_result.get('preprocessing', {}).get('metadata'):
                    combined_metadata = {}
                    if metadata:
                        combined_metadata.update(metadata)
                    if processing_result.get('preprocessing', {}).get('metadata'):
                        combined_metadata.update(processing_result['preprocessing']['metadata'])
                    
                    metadata_record = ImageMetadata(
                        metadata_id=uuid.uuid4(),
                        image_id=uuid.UUID(image_id),
                        metadata_json=json.dumps(combined_metadata),
                        extraction_method='automatic'
                    )
                    
                    db.add(metadata_record)
                
                db.commit()
                
                logger.info(f"图像记录已保存到数据库: {image_id}")
                return image_record
                
        except Exception as e:
            logger.error(f"保存图像记录失败: {e}")
            raise
    
    async def get_image_info(self, image_id: str) -> Optional[Dict[str, Any]]:
        """获取图像信息
        
        Args:
            image_id: 图像ID
            
        Returns:
            图像信息字典
        """
        try:
            with get_db_context() as db:
                image_record = db.query(Image).filter(
                    Image.image_id == uuid.UUID(image_id)
                ).first()
                
                if not image_record:
                    return None
                
                # 获取元数据
                metadata_record = db.query(ImageMetadata).filter(
                    ImageMetadata.image_id == uuid.UUID(image_id)
                ).first()
                
                metadata = {}
                if metadata_record:
                    metadata = json.loads(metadata_record.metadata_json)
                
                # 加载处理结果
                processing_results = await self._load_processing_results(image_id)
                
                return {
                    'image_id': image_id,
                    'series_id': str(image_record.series_id),
                    'file_path': image_record.file_path,
                    'file_size': image_record.file_size,
                    'image_type': image_record.image_type,
                    'processing_status': image_record.processing_status,
                    'quality_score': float(image_record.quality_score),
                    'created_at': image_record.created_at.isoformat(),
                    'metadata': metadata,
                    'processing_results': processing_results
                }
                
        except Exception as e:
            logger.error(f"获取图像信息失败: {e}")
            return None
    
    async def _load_processing_results(self, image_id: str) -> Dict[str, Any]:
        """加载处理结果"""
        try:
            results_path = os.path.join(settings.upload_dir, "processing_results", 
                                      image_id, "processing_results.json")
            
            if os.path.exists(results_path):
                with open(results_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            
            return {}
            
        except Exception as e:
            logger.error(f"加载处理结果失败: {e}")
            return {}
    
    async def search_images(self, query: Dict[str, Any], 
                          limit: int = 50, offset: int = 0) -> Dict[str, Any]:
        """搜索图像
        
        Args:
            query: 搜索条件
            limit: 返回数量限制
            offset: 偏移量
            
        Returns:
            搜索结果
        """
        try:
            with get_db_context() as db:
                # 构建查询
                db_query = db.query(Image)
                
                # 应用过滤条件
                if 'series_id' in query:
                    db_query = db_query.filter(Image.series_id == uuid.UUID(query['series_id']))
                
                if 'image_type' in query:
                    db_query = db_query.filter(Image.image_type == query['image_type'])
                
                if 'processing_status' in query:
                    db_query = db_query.filter(Image.processing_status == query['processing_status'])
                
                if 'min_quality_score' in query:
                    db_query = db_query.filter(Image.quality_score >= query['min_quality_score'])
                
                if 'date_from' in query:
                    db_query = db_query.filter(Image.created_at >= query['date_from'])
                
                if 'date_to' in query:
                    db_query = db_query.filter(Image.created_at <= query['date_to'])
                
                # 获取总数
                total_count = db_query.count()
                
                # 应用分页
                images = db_query.offset(offset).limit(limit).all()
                
                # 构建结果
                results = []
                for image in images:
                    results.append({
                        'image_id': str(image.image_id),
                        'series_id': str(image.series_id),
                        'file_path': image.file_path,
                        'file_size': image.file_size,
                        'image_type': image.image_type,
                        'processing_status': image.processing_status,
                        'quality_score': float(image.quality_score),
                        'created_at': image.created_at.isoformat()
                    })
                
                return {
                    'images': results,
                    'total_count': total_count,
                    'limit': limit,
                    'offset': offset,
                    'has_more': offset + limit < total_count
                }
                
        except Exception as e:
            logger.error(f"搜索图像失败: {e}")
            raise
    
    async def delete_image(self, image_id: str) -> bool:
        """删除图像
        
        Args:
            image_id: 图像ID
            
        Returns:
            是否删除成功
        """
        try:
            with get_db_context() as db:
                # 获取图像记录
                image_record = db.query(Image).filter(
                    Image.image_id == uuid.UUID(image_id)
                ).first()
                
                if not image_record:
                    logger.warning(f"图像不存在: {image_id}")
                    return False
                
                # 删除云存储文件
                try:
                    await self.storage_service.delete_file(image_record.file_path)
                except Exception as e:
                    logger.warning(f"删除云存储文件失败: {e}")
                
                # 删除本地处理结果
                self._cleanup_local_files(image_id)
                
                # 删除数据库记录
                db.delete(image_record)
                db.commit()
                
                logger.info(f"图像删除成功: {image_id}")
                return True
                
        except Exception as e:
            logger.error(f"删除图像失败: {e}")
            return False
    
    def _cleanup_local_files(self, image_id: str) -> None:
        """清理本地文件"""
        try:
            import shutil
            
            # 删除处理结果目录
            results_dir = os.path.join(settings.upload_dir, "processing_results", image_id)
            if os.path.exists(results_dir):
                shutil.rmtree(results_dir)
            
            # 删除缩略图目录
            thumbnails_dir = os.path.join(settings.upload_dir, "thumbnails", image_id)
            if os.path.exists(thumbnails_dir):
                shutil.rmtree(thumbnails_dir)
            
            logger.debug(f"本地文件清理完成: {image_id}")
            
        except Exception as e:
            logger.error(f"清理本地文件失败: {e}")
    
    async def get_image_statistics(self) -> Dict[str, Any]:
        """获取图像统计信息
        
        Returns:
            统计信息字典
        """
        try:
            with get_db_context() as db:
                # 总图像数
                total_images = db.query(Image).count()
                
                # 按类型统计
                type_stats = db.query(
                    Image.image_type, 
                    db.func.count(Image.image_id)
                ).group_by(Image.image_type).all()
                
                # 按处理状态统计
                status_stats = db.query(
                    Image.processing_status,
                    db.func.count(Image.image_id)
                ).group_by(Image.processing_status).all()
                
                # 质量分数统计
                quality_stats = db.query(
                    db.func.avg(Image.quality_score),
                    db.func.min(Image.quality_score),
                    db.func.max(Image.quality_score)
                ).first()
                
                # 存储大小统计
                storage_stats = db.query(
                    db.func.sum(Image.file_size),
                    db.func.avg(Image.file_size)
                ).first()
                
                return {
                    'total_images': total_images,
                    'by_type': {image_type: count for image_type, count in type_stats},
                    'by_status': {status: count for status, count in status_stats},
                    'quality_scores': {
                        'average': float(quality_stats[0] or 0),
                        'minimum': float(quality_stats[1] or 0),
                        'maximum': float(quality_stats[2] or 0)
                    },
                    'storage': {
                        'total_size_bytes': int(storage_stats[0] or 0),
                        'average_size_bytes': int(storage_stats[1] or 0)
                    }
                }
                
        except Exception as e:
            logger.error(f"获取图像统计失败: {e}")
            raise