import os
import shutil
from typing import Dict, List, Optional, Any, BinaryIO
import logging
from pathlib import Path
import uuid
from datetime import datetime, timezone
import asyncio
from concurrent.futures import ThreadPoolExecutor
import hashlib
import mimetypes
import aiofiles
import boto3
from botocore.exceptions import ClientError, NoCredentialsError
from minio import Minio
from minio.error import S3Error

from backend.core.config import settings

logger = logging.getLogger(__name__)

class StorageService:
    """存储服务
    
    提供文件存储和管理功能，支持：
    - 本地文件系统存储
    - MinIO对象存储
    - AWS S3存储
    - 文件上传、下载、删除
    - 文件完整性验证
    """
    
    def __init__(self):
        self.storage_type = getattr(settings, 'storage_type', 'local')
        self.local_storage_path = getattr(settings, 'upload_dir', './uploads')
        self.executor = ThreadPoolExecutor(max_workers=4)
        
        # 确保本地存储目录存在
        os.makedirs(self.local_storage_path, exist_ok=True)
        
        # 初始化云存储客户端
        self.minio_client = None
        self.s3_client = None
        
        if self.storage_type == 'minio':
            self._init_minio_client()
        elif self.storage_type == 's3':
            self._init_s3_client()
    
    def _init_minio_client(self):
        """初始化MinIO客户端"""
        try:
            self.minio_client = Minio(
                endpoint=getattr(settings, 'minio_endpoint', 'localhost:9000'),
                access_key=getattr(settings, 'minio_access_key', ''),
                secret_key=getattr(settings, 'minio_secret_key', ''),
                secure=getattr(settings, 'minio_secure', False)
            )
            
            # 检查连接
            self.minio_client.list_buckets()
            logger.info("MinIO客户端初始化成功")
            
        except Exception as e:
            logger.error(f"MinIO客户端初始化失败: {e}")
            self.minio_client = None
    
    def _init_s3_client(self):
        """初始化S3客户端"""
        try:
            self.s3_client = boto3.client(
                's3',
                aws_access_key_id=getattr(settings, 'aws_access_key_id', ''),
                aws_secret_access_key=getattr(settings, 'aws_secret_access_key', ''),
                region_name=getattr(settings, 'aws_region', 'us-east-1')
            )
            
            # 检查连接
            self.s3_client.list_buckets()
            logger.info("S3客户端初始化成功")
            
        except Exception as e:
            logger.error(f"S3客户端初始化失败: {e}")
            self.s3_client = None
    
    async def upload_file(self, file_path: str, object_name: Optional[str] = None,
                         metadata: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
        """上传文件
        
        Args:
            file_path: 本地文件路径
            object_name: 存储对象名称，如果为None则使用文件名
            metadata: 文件元数据
            
        Returns:
            上传结果字典
        """
        try:
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"文件不存在: {file_path}")
            
            if object_name is None:
                object_name = os.path.basename(file_path)
            
            # 计算文件信息
            file_info = await self._get_file_info(file_path)
            
            # 根据存储类型选择上传方法
            if self.storage_type == 'local':
                result = await self._upload_to_local(file_path, object_name, metadata)
            elif self.storage_type == 'minio':
                result = await self._upload_to_minio(file_path, object_name, metadata)
            elif self.storage_type == 's3':
                result = await self._upload_to_s3(file_path, object_name, metadata)
            else:
                raise ValueError(f"不支持的存储类型: {self.storage_type}")
            
            # 添加文件信息
            result.update(file_info)
            
            logger.info(f"文件上传成功: {object_name}")
            return result
            
        except Exception as e:
            logger.error(f"文件上传失败: {e}")
            raise
    
    async def _upload_to_local(self, file_path: str, object_name: str,
                              metadata: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
        """上传到本地存储"""
        try:
            # 构建目标路径
            target_path = os.path.join(self.local_storage_path, object_name)
            target_dir = os.path.dirname(target_path)
            
            # 确保目标目录存在
            os.makedirs(target_dir, exist_ok=True)
            
            # 异步复制文件
            await self._copy_file_async(file_path, target_path)
            
            # 保存元数据
            if metadata:
                await self._save_metadata(target_path, metadata)
            
            return {
                'storage_type': 'local',
                'url': target_path,
                'object_name': object_name,
                'status': 'success'
            }
            
        except Exception as e:
            logger.error(f"本地存储上传失败: {e}")
            raise
    
    async def _upload_to_minio(self, file_path: str, object_name: str,
                              metadata: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
        """上传到MinIO"""
        try:
            if not self.minio_client:
                raise RuntimeError("MinIO客户端未初始化")
            
            bucket_name = getattr(settings, 'minio_bucket', 'medical-images')
            
            # 确保存储桶存在
            await self._ensure_bucket_exists(bucket_name)
            
            # 获取文件MIME类型
            content_type = mimetypes.guess_type(file_path)[0] or 'application/octet-stream'
            
            # 异步上传
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                self.executor,
                self._upload_to_minio_sync,
                file_path, bucket_name, object_name, content_type, metadata
            )
            
            # 构建URL
            url = f"http://{getattr(settings, 'minio_endpoint', 'localhost:9000')}/{bucket_name}/{object_name}"
            
            return {
                'storage_type': 'minio',
                'url': url,
                'bucket': bucket_name,
                'object_name': object_name,
                'status': 'success'
            }
            
        except Exception as e:
            logger.error(f"MinIO上传失败: {e}")
            raise
    
    def _upload_to_minio_sync(self, file_path: str, bucket_name: str, object_name: str,
                             content_type: str, metadata: Optional[Dict[str, str]] = None):
        """同步上传到MinIO"""
        try:
            self.minio_client.fput_object(
                bucket_name=bucket_name,
                object_name=object_name,
                file_path=file_path,
                content_type=content_type,
                metadata=metadata or {}
            )
        except S3Error as e:
            logger.error(f"MinIO上传错误: {e}")
            raise
    
    async def _upload_to_s3(self, file_path: str, object_name: str,
                           metadata: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
        """上传到S3"""
        try:
            if not self.s3_client:
                raise RuntimeError("S3客户端未初始化")
            
            bucket_name = getattr(settings, 'aws_s3_bucket', 'medical-images')
            
            # 准备上传参数
            extra_args = {}
            if metadata:
                extra_args['Metadata'] = metadata
            
            # 设置内容类型
            content_type = mimetypes.guess_type(file_path)[0]
            if content_type:
                extra_args['ContentType'] = content_type
            
            # 异步上传
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                self.executor,
                self.s3_client.upload_file,
                file_path, bucket_name, object_name, extra_args
            )
            
            # 构建URL
            region = getattr(settings, 'aws_region', 'us-east-1')
            url = f"https://{bucket_name}.s3.{region}.amazonaws.com/{object_name}"
            
            return {
                'storage_type': 's3',
                'url': url,
                'bucket': bucket_name,
                'object_name': object_name,
                'status': 'success'
            }
            
        except Exception as e:
            logger.error(f"S3上传失败: {e}")
            raise
    
    async def download_file(self, object_name: str, local_path: str) -> Dict[str, Any]:
        """下载文件
        
        Args:
            object_name: 存储对象名称
            local_path: 本地保存路径
            
        Returns:
            下载结果字典
        """
        try:
            # 确保本地目录存在
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            
            # 根据存储类型选择下载方法
            if self.storage_type == 'local':
                result = await self._download_from_local(object_name, local_path)
            elif self.storage_type == 'minio':
                result = await self._download_from_minio(object_name, local_path)
            elif self.storage_type == 's3':
                result = await self._download_from_s3(object_name, local_path)
            else:
                raise ValueError(f"不支持的存储类型: {self.storage_type}")
            
            logger.info(f"文件下载成功: {object_name}")
            return result
            
        except Exception as e:
            logger.error(f"文件下载失败: {e}")
            raise
    
    async def _download_from_local(self, object_name: str, local_path: str) -> Dict[str, Any]:
        """从本地存储下载"""
        try:
            source_path = os.path.join(self.local_storage_path, object_name)
            
            if not os.path.exists(source_path):
                raise FileNotFoundError(f"文件不存在: {source_path}")
            
            # 异步复制文件
            await self._copy_file_async(source_path, local_path)
            
            return {
                'storage_type': 'local',
                'source_path': source_path,
                'local_path': local_path,
                'status': 'success'
            }
            
        except Exception as e:
            logger.error(f"本地存储下载失败: {e}")
            raise
    
    async def _download_from_minio(self, object_name: str, local_path: str) -> Dict[str, Any]:
        """从MinIO下载"""
        try:
            if not self.minio_client:
                raise RuntimeError("MinIO客户端未初始化")
            
            bucket_name = getattr(settings, 'minio_bucket', 'medical-images')
            
            # 异步下载
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                self.executor,
                self.minio_client.fget_object,
                bucket_name, object_name, local_path
            )
            
            return {
                'storage_type': 'minio',
                'bucket': bucket_name,
                'object_name': object_name,
                'local_path': local_path,
                'status': 'success'
            }
            
        except Exception as e:
            logger.error(f"MinIO下载失败: {e}")
            raise
    
    async def _download_from_s3(self, object_name: str, local_path: str) -> Dict[str, Any]:
        """从S3下载"""
        try:
            if not self.s3_client:
                raise RuntimeError("S3客户端未初始化")
            
            bucket_name = getattr(settings, 'aws_s3_bucket', 'medical-images')
            
            # 异步下载
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                self.executor,
                self.s3_client.download_file,
                bucket_name, object_name, local_path
            )
            
            return {
                'storage_type': 's3',
                'bucket': bucket_name,
                'object_name': object_name,
                'local_path': local_path,
                'status': 'success'
            }
            
        except Exception as e:
            logger.error(f"S3下载失败: {e}")
            raise
    
    async def delete_file(self, object_name: str) -> bool:
        """删除文件
        
        Args:
            object_name: 存储对象名称
            
        Returns:
            是否删除成功
        """
        try:
            # 根据存储类型选择删除方法
            if self.storage_type == 'local':
                success = await self._delete_from_local(object_name)
            elif self.storage_type == 'minio':
                success = await self._delete_from_minio(object_name)
            elif self.storage_type == 's3':
                success = await self._delete_from_s3(object_name)
            else:
                raise ValueError(f"不支持的存储类型: {self.storage_type}")
            
            if success:
                logger.info(f"文件删除成功: {object_name}")
            else:
                logger.warning(f"文件删除失败: {object_name}")
            
            return success
            
        except Exception as e:
            logger.error(f"文件删除失败: {e}")
            return False
    
    async def _delete_from_local(self, object_name: str) -> bool:
        """从本地存储删除"""
        try:
            file_path = os.path.join(self.local_storage_path, object_name)
            
            if os.path.exists(file_path):
                os.remove(file_path)
                
                # 删除元数据文件
                metadata_path = file_path + '.metadata'
                if os.path.exists(metadata_path):
                    os.remove(metadata_path)
                
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"本地存储删除失败: {e}")
            return False
    
    async def _delete_from_minio(self, object_name: str) -> bool:
        """从MinIO删除"""
        try:
            if not self.minio_client:
                raise RuntimeError("MinIO客户端未初始化")
            
            bucket_name = getattr(settings, 'minio_bucket', 'medical-images')
            
            # 异步删除
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                self.executor,
                self.minio_client.remove_object,
                bucket_name, object_name
            )
            
            return True
            
        except Exception as e:
            logger.error(f"MinIO删除失败: {e}")
            return False
    
    async def _delete_from_s3(self, object_name: str) -> bool:
        """从S3删除"""
        try:
            if not self.s3_client:
                raise RuntimeError("S3客户端未初始化")
            
            bucket_name = getattr(settings, 'aws_s3_bucket', 'medical-images')
            
            # 异步删除
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                self.executor,
                self.s3_client.delete_object,
                Bucket=bucket_name, Key=object_name
            )
            
            return True
            
        except Exception as e:
            logger.error(f"S3删除失败: {e}")
            return False
    
    async def file_exists(self, object_name: str) -> bool:
        """检查文件是否存在
        
        Args:
            object_name: 存储对象名称
            
        Returns:
            文件是否存在
        """
        try:
            # 根据存储类型选择检查方法
            if self.storage_type == 'local':
                return await self._file_exists_local(object_name)
            elif self.storage_type == 'minio':
                return await self._file_exists_minio(object_name)
            elif self.storage_type == 's3':
                return await self._file_exists_s3(object_name)
            else:
                return False
                
        except Exception as e:
            logger.error(f"检查文件存在性失败: {e}")
            return False
    
    async def _file_exists_local(self, object_name: str) -> bool:
        """检查本地文件是否存在"""
        file_path = os.path.join(self.local_storage_path, object_name)
        return os.path.exists(file_path)
    
    async def _file_exists_minio(self, object_name: str) -> bool:
        """检查MinIO文件是否存在"""
        try:
            if not self.minio_client:
                return False
            
            bucket_name = getattr(settings, 'minio_bucket', 'medical-images')
            
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                self.executor,
                self.minio_client.stat_object,
                bucket_name, object_name
            )
            
            return True
            
        except S3Error:
            return False
    
    async def _file_exists_s3(self, object_name: str) -> bool:
        """检查S3文件是否存在"""
        try:
            if not self.s3_client:
                return False
            
            bucket_name = getattr(settings, 'aws_s3_bucket', 'medical-images')
            
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                self.executor,
                self.s3_client.head_object,
                Bucket=bucket_name, Key=object_name
            )
            
            return True
            
        except ClientError:
            return False
    
    async def get_file_info(self, object_name: str) -> Optional[Dict[str, Any]]:
        """获取文件信息
        
        Args:
            object_name: 存储对象名称
            
        Returns:
            文件信息字典
        """
        try:
            # 根据存储类型选择获取方法
            if self.storage_type == 'local':
                return await self._get_file_info_local(object_name)
            elif self.storage_type == 'minio':
                return await self._get_file_info_minio(object_name)
            elif self.storage_type == 's3':
                return await self._get_file_info_s3(object_name)
            else:
                return None
                
        except Exception as e:
            logger.error(f"获取文件信息失败: {e}")
            return None
    
    async def _get_file_info_local(self, object_name: str) -> Optional[Dict[str, Any]]:
        """获取本地文件信息"""
        try:
            file_path = os.path.join(self.local_storage_path, object_name)
            
            if not os.path.exists(file_path):
                return None
            
            stat = os.stat(file_path)
            
            return {
                'object_name': object_name,
                'size': stat.st_size,
                'modified_time': datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc),
                'content_type': mimetypes.guess_type(file_path)[0],
                'storage_type': 'local'
            }
            
        except Exception as e:
            logger.error(f"获取本地文件信息失败: {e}")
            return None
    
    async def _get_file_info_minio(self, object_name: str) -> Optional[Dict[str, Any]]:
        """获取MinIO文件信息"""
        try:
            if not self.minio_client:
                return None
            
            bucket_name = getattr(settings, 'minio_bucket', 'medical-images')
            
            loop = asyncio.get_event_loop()
            stat = await loop.run_in_executor(
                self.executor,
                self.minio_client.stat_object,
                bucket_name, object_name
            )
            
            return {
                'object_name': object_name,
                'size': stat.size,
                'modified_time': stat.last_modified,
                'content_type': stat.content_type,
                'etag': stat.etag,
                'storage_type': 'minio'
            }
            
        except Exception as e:
            logger.error(f"获取MinIO文件信息失败: {e}")
            return None
    
    async def _get_file_info_s3(self, object_name: str) -> Optional[Dict[str, Any]]:
        """获取S3文件信息"""
        try:
            if not self.s3_client:
                return None
            
            bucket_name = getattr(settings, 'aws_s3_bucket', 'medical-images')
            
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                self.executor,
                self.s3_client.head_object,
                Bucket=bucket_name, Key=object_name
            )
            
            return {
                'object_name': object_name,
                'size': response['ContentLength'],
                'modified_time': response['LastModified'],
                'content_type': response.get('ContentType'),
                'etag': response['ETag'].strip('"'),
                'storage_type': 's3'
            }
            
        except Exception as e:
            logger.error(f"获取S3文件信息失败: {e}")
            return None
    
    async def _get_file_info(self, file_path: str) -> Dict[str, Any]:
        """获取本地文件信息"""
        try:
            stat = os.stat(file_path)
            
            # 计算文件哈希
            file_hash = await self._calculate_file_hash(file_path)
            
            return {
                'size': stat.st_size,
                'modified_time': datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc),
                'content_type': mimetypes.guess_type(file_path)[0],
                'hash': file_hash
            }
            
        except Exception as e:
            logger.error(f"获取文件信息失败: {e}")
            return {}
    
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
    
    async def _copy_file_async(self, source_path: str, target_path: str):
        """异步复制文件"""
        async with aiofiles.open(source_path, 'rb') as src:
            async with aiofiles.open(target_path, 'wb') as dst:
                while True:
                    chunk = await src.read(8192)
                    if not chunk:
                        break
                    await dst.write(chunk)
    
    async def _save_metadata(self, file_path: str, metadata: Dict[str, str]):
        """保存文件元数据"""
        try:
            metadata_path = file_path + '.metadata'
            
            async with aiofiles.open(metadata_path, 'w', encoding='utf-8') as f:
                import json
                await f.write(json.dumps(metadata, indent=2, ensure_ascii=False))
                
        except Exception as e:
            logger.error(f"保存元数据失败: {e}")
    
    async def _ensure_bucket_exists(self, bucket_name: str):
        """确保存储桶存在"""
        try:
            if self.storage_type == 'minio' and self.minio_client:
                loop = asyncio.get_event_loop()
                exists = await loop.run_in_executor(
                    self.executor,
                    self.minio_client.bucket_exists,
                    bucket_name
                )
                
                if not exists:
                    await loop.run_in_executor(
                        self.executor,
                        self.minio_client.make_bucket,
                        bucket_name
                    )
                    logger.info(f"创建MinIO存储桶: {bucket_name}")
                    
        except Exception as e:
            logger.error(f"确保存储桶存在失败: {e}")
    
    async def list_files(self, prefix: str = '', limit: int = 100) -> List[Dict[str, Any]]:
        """列出文件
        
        Args:
            prefix: 文件名前缀
            limit: 返回数量限制
            
        Returns:
            文件列表
        """
        try:
            # 根据存储类型选择列出方法
            if self.storage_type == 'local':
                return await self._list_files_local(prefix, limit)
            elif self.storage_type == 'minio':
                return await self._list_files_minio(prefix, limit)
            elif self.storage_type == 's3':
                return await self._list_files_s3(prefix, limit)
            else:
                return []
                
        except Exception as e:
            logger.error(f"列出文件失败: {e}")
            return []
    
    async def _list_files_local(self, prefix: str, limit: int) -> List[Dict[str, Any]]:
        """列出本地文件"""
        try:
            files = []
            search_path = os.path.join(self.local_storage_path, prefix)
            
            if os.path.isfile(search_path):
                # 单个文件
                info = await self._get_file_info_local(prefix)
                if info:
                    files.append(info)
            else:
                # 目录搜索
                for root, dirs, filenames in os.walk(self.local_storage_path):
                    for filename in filenames:
                        if filename.startswith(prefix) and not filename.endswith('.metadata'):
                            rel_path = os.path.relpath(os.path.join(root, filename), self.local_storage_path)
                            info = await self._get_file_info_local(rel_path)
                            if info:
                                files.append(info)
                            
                            if len(files) >= limit:
                                break
                    
                    if len(files) >= limit:
                        break
            
            return files[:limit]
            
        except Exception as e:
            logger.error(f"列出本地文件失败: {e}")
            return []
    
    async def _list_files_minio(self, prefix: str, limit: int) -> List[Dict[str, Any]]:
        """列出MinIO文件"""
        try:
            if not self.minio_client:
                return []
            
            bucket_name = getattr(settings, 'minio_bucket', 'medical-images')
            
            loop = asyncio.get_event_loop()
            objects = await loop.run_in_executor(
                self.executor,
                lambda: list(self.minio_client.list_objects(bucket_name, prefix=prefix))
            )
            
            files = []
            for obj in objects[:limit]:
                files.append({
                    'object_name': obj.object_name,
                    'size': obj.size,
                    'modified_time': obj.last_modified,
                    'etag': obj.etag,
                    'storage_type': 'minio'
                })
            
            return files
            
        except Exception as e:
            logger.error(f"列出MinIO文件失败: {e}")
            return []
    
    async def _list_files_s3(self, prefix: str, limit: int) -> List[Dict[str, Any]]:
        """列出S3文件"""
        try:
            if not self.s3_client:
                return []
            
            bucket_name = getattr(settings, 'aws_s3_bucket', 'medical-images')
            
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                self.executor,
                self.s3_client.list_objects_v2,
                Bucket=bucket_name, Prefix=prefix, MaxKeys=limit
            )
            
            files = []
            for obj in response.get('Contents', []):
                files.append({
                    'object_name': obj['Key'],
                    'size': obj['Size'],
                    'modified_time': obj['LastModified'],
                    'etag': obj['ETag'].strip('"'),
                    'storage_type': 's3'
                })
            
            return files
            
        except Exception as e:
            logger.error(f"列出S3文件失败: {e}")
            return []
    
    async def get_storage_stats(self) -> Dict[str, Any]:
        """获取存储统计信息
        
        Returns:
            存储统计信息
        """
        try:
            stats = {
                'storage_type': self.storage_type,
                'total_files': 0,
                'total_size': 0,
                'available_space': 0
            }
            
            if self.storage_type == 'local':
                # 统计本地存储
                total_size = 0
                total_files = 0
                
                for root, dirs, files in os.walk(self.local_storage_path):
                    for file in files:
                        if not file.endswith('.metadata'):
                            file_path = os.path.join(root, file)
                            total_size += os.path.getsize(file_path)
                            total_files += 1
                
                stats['total_files'] = total_files
                stats['total_size'] = total_size
                
                # 获取可用空间
                import shutil
                _, _, free_space = shutil.disk_usage(self.local_storage_path)
                stats['available_space'] = free_space
            
            return stats
            
        except Exception as e:
            logger.error(f"获取存储统计失败: {e}")
            return {'storage_type': self.storage_type, 'error': str(e)}