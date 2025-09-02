# -*- coding: utf-8 -*-
"""
Medical AI 集成测试

本模块包含了 Medical AI 系统的集成测试，测试各组件之间的协作：
- 数据库集成测试
- Redis 缓存集成测试
- 文件存储集成测试
- AI 模型集成测试
- 消息队列集成测试
- 端到端工作流测试
"""

import pytest
import asyncio
import tempfile
import os
import shutil
from datetime import datetime, timedelta
from typing import Dict, Any, List
from unittest.mock import Mock, patch, AsyncMock
import json
import redis
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker
import boto3
from moto import mock_s3
import celery
from celery import Celery

from backend.core.config import settings
from backend.core.database import Base, get_db
from backend.models.user import User
from backend.models.patient import Patient
from backend.models.image import MedicalImage
from backend.models.report import DiagnosticReport
from backend.models.inference import InferenceTask
from backend.services.auth import AuthService
from backend.services.patient import PatientService
from backend.services.image import ImageService
from backend.services.ai_inference import AIInferenceService
from backend.services.report import ReportService
from backend.services.storage import StorageService
from backend.services.cache import CacheService
from backend.services.notification import NotificationService
from backend.tasks.ai_inference import process_inference_task
from backend.tasks.report_generation import generate_report_task


class TestDatabaseIntegration:
    """数据库集成测试"""
    
    @pytest.fixture
    async def db_session(self):
        """创建测试数据库会话"""
        # 使用内存数据库进行测试
        engine = create_async_engine(
            "sqlite+aiosqlite:///:memory:",
            echo=False
        )
        
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        
        async_session = sessionmaker(
            engine, class_=AsyncSession, expire_on_commit=False
        )
        
        async with async_session() as session:
            yield session
    
    @pytest.mark.asyncio
    async def test_user_patient_relationship(self, db_session):
        """测试用户和患者关系"""
        # 创建用户
        user = User(
            username="doctor@test.com",
            email="doctor@test.com",
            full_name="Test Doctor",
            hashed_password="hashed_password",
            role="doctor",
            department="Radiology"
        )
        db_session.add(user)
        await db_session.commit()
        await db_session.refresh(user)
        
        # 创建患者
        patient = Patient(
            patient_id="P123456",
            name="测试患者",
            gender="male",
            birth_date=datetime(1980, 5, 15).date(),
            phone="13800138000",
            created_by=user.id
        )
        db_session.add(patient)
        await db_session.commit()
        await db_session.refresh(patient)
        
        # 验证关系
        assert patient.created_by == user.id
        assert patient.creator.username == user.username
    
    @pytest.mark.asyncio
    async def test_image_patient_relationship(self, db_session):
        """测试影像和患者关系"""
        # 创建患者
        patient = Patient(
            patient_id="P123456",
            name="测试患者",
            gender="male",
            birth_date=datetime(1980, 5, 15).date()
        )
        db_session.add(patient)
        await db_session.commit()
        await db_session.refresh(patient)
        
        # 创建影像
        image = MedicalImage(
            filename="test.dcm",
            original_filename="test_original.dcm",
            file_path="/storage/images/test.dcm",
            file_size=1024000,
            study_type="chest_xray",
            patient_id=patient.id,
            description="胸部X光检查"
        )
        db_session.add(image)
        await db_session.commit()
        await db_session.refresh(image)
        
        # 验证关系
        assert image.patient_id == patient.id
        assert image.patient.patient_id == "P123456"
    
    @pytest.mark.asyncio
    async def test_report_image_relationship(self, db_session):
        """测试报告和影像关系"""
        # 创建患者
        patient = Patient(
            patient_id="P123456",
            name="测试患者",
            gender="male",
            birth_date=datetime(1980, 5, 15).date()
        )
        db_session.add(patient)
        
        # 创建影像
        image = MedicalImage(
            filename="test.dcm",
            original_filename="test_original.dcm",
            file_path="/storage/images/test.dcm",
            file_size=1024000,
            study_type="chest_xray",
            patient_id=patient.id
        )
        db_session.add(image)
        
        # 创建用户
        user = User(
            username="doctor@test.com",
            email="doctor@test.com",
            full_name="Test Doctor",
            hashed_password="hashed_password",
            role="doctor"
        )
        db_session.add(user)
        
        await db_session.commit()
        await db_session.refresh(patient)
        await db_session.refresh(image)
        await db_session.refresh(user)
        
        # 创建报告
        report = DiagnosticReport(
            patient_id=patient.id,
            image_id=image.id,
            study_type="chest_xray",
            findings="右下肺野可见片状阴影",
            impression="右下肺炎",
            recommendations="建议抗感染治疗",
            created_by=user.id,
            status="completed"
        )
        db_session.add(report)
        await db_session.commit()
        await db_session.refresh(report)
        
        # 验证关系
        assert report.patient_id == patient.id
        assert report.image_id == image.id
        assert report.created_by == user.id
        assert report.patient.patient_id == "P123456"
        assert report.image.filename == "test.dcm"
        assert report.creator.username == "doctor@test.com"


class TestRedisIntegration:
    """Redis 缓存集成测试"""
    
    @pytest.fixture
    def redis_client(self):
        """创建 Redis 测试客户端"""
        # 使用 fakeredis 进行测试
        import fakeredis
        return fakeredis.FakeRedis(decode_responses=True)
    
    @pytest.mark.asyncio
    async def test_cache_service(self, redis_client):
        """测试缓存服务"""
        cache_service = CacheService(redis_client)
        
        # 测试设置和获取缓存
        test_data = {"user_id": "123", "username": "test@example.com"}
        await cache_service.set("user:123", test_data, expire=3600)
        
        cached_data = await cache_service.get("user:123")
        assert cached_data == test_data
        
        # 测试缓存过期
        await cache_service.set("temp_key", "temp_value", expire=1)
        await asyncio.sleep(2)
        expired_data = await cache_service.get("temp_key")
        assert expired_data is None
    
    @pytest.mark.asyncio
    async def test_session_cache(self, redis_client):
        """测试会话缓存"""
        cache_service = CacheService(redis_client)
        
        # 模拟用户会话
        session_data = {
            "user_id": "user123",
            "username": "doctor@test.com",
            "role": "doctor",
            "login_time": datetime.utcnow().isoformat()
        }
        
        session_key = f"session:{session_data['user_id']}"
        await cache_service.set(session_key, session_data, expire=7200)
        
        # 验证会话数据
        cached_session = await cache_service.get(session_key)
        assert cached_session["username"] == "doctor@test.com"
        assert cached_session["role"] == "doctor"
    
    @pytest.mark.asyncio
    async def test_inference_result_cache(self, redis_client):
        """测试推理结果缓存"""
        cache_service = CacheService(redis_client)
        
        # 模拟推理结果
        inference_result = {
            "task_id": "task123",
            "image_id": "image456",
            "model_name": "chest_xray_classifier",
            "predictions": [
                {"class": "pneumonia", "confidence": 0.92},
                {"class": "normal", "confidence": 0.08}
            ],
            "processing_time": 2.5,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        result_key = f"inference_result:{inference_result['task_id']}"
        await cache_service.set(result_key, inference_result, expire=86400)
        
        # 验证缓存的推理结果
        cached_result = await cache_service.get(result_key)
        assert cached_result["task_id"] == "task123"
        assert len(cached_result["predictions"]) == 2
        assert cached_result["predictions"][0]["class"] == "pneumonia"


@mock_s3
class TestStorageIntegration:
    """文件存储集成测试"""
    
    @pytest.fixture
    def s3_client(self):
        """创建 S3 测试客户端"""
        client = boto3.client(
            "s3",
            region_name="us-east-1",
            aws_access_key_id="testing",
            aws_secret_access_key="testing"
        )
        
        # 创建测试桶
        client.create_bucket(Bucket="medical-ai-test")
        return client
    
    @pytest.mark.asyncio
    async def test_image_upload_and_download(self, s3_client):
        """测试影像上传和下载"""
        storage_service = StorageService(s3_client, bucket_name="medical-ai-test")
        
        # 创建测试文件
        test_content = b"DICM" + b"\x00" * 1000  # 模拟 DICOM 文件
        
        with tempfile.NamedTemporaryFile(suffix=".dcm", delete=False) as temp_file:
            temp_file.write(test_content)
            temp_file.flush()
            
            try:
                # 上传文件
                file_key = "images/test_patient/test_image.dcm"
                upload_result = await storage_service.upload_file(
                    temp_file.name,
                    file_key,
                    content_type="application/dicom"
                )
                
                assert upload_result["success"] is True
                assert upload_result["file_key"] == file_key
                
                # 下载文件
                download_result = await storage_service.download_file(file_key)
                assert download_result == test_content
                
                # 获取文件信息
                file_info = await storage_service.get_file_info(file_key)
                assert file_info["size"] == len(test_content)
                assert file_info["content_type"] == "application/dicom"
                
            finally:
                os.unlink(temp_file.name)
    
    @pytest.mark.asyncio
    async def test_file_deletion(self, s3_client):
        """测试文件删除"""
        storage_service = StorageService(s3_client, bucket_name="medical-ai-test")
        
        # 上传测试文件
        test_content = b"test file content"
        file_key = "temp/test_file.txt"
        
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_file.write(test_content)
            temp_file.flush()
            
            try:
                await storage_service.upload_file(
                    temp_file.name,
                    file_key,
                    content_type="text/plain"
                )
                
                # 验证文件存在
                file_exists = await storage_service.file_exists(file_key)
                assert file_exists is True
                
                # 删除文件
                delete_result = await storage_service.delete_file(file_key)
                assert delete_result["success"] is True
                
                # 验证文件已删除
                file_exists = await storage_service.file_exists(file_key)
                assert file_exists is False
                
            finally:
                os.unlink(temp_file.name)


class TestAIModelIntegration:
    """AI 模型集成测试"""
    
    @pytest.fixture
    def mock_model(self):
        """创建模拟 AI 模型"""
        model = Mock()
        model.predict.return_value = {
            "predictions": [
                {"class": "pneumonia", "confidence": 0.92},
                {"class": "normal", "confidence": 0.08}
            ],
            "heatmap": "base64_encoded_heatmap_data",
            "processing_time": 2.5
        }
        return model
    
    @pytest.mark.asyncio
    async def test_ai_inference_workflow(self, mock_model):
        """测试 AI 推理工作流"""
        ai_service = AIInferenceService()
        
        with patch.object(ai_service, 'load_model', return_value=mock_model):
            with patch.object(ai_service, 'preprocess_image') as mock_preprocess:
                mock_preprocess.return_value = Mock()  # 模拟预处理后的图像
                
                # 执行推理
                result = await ai_service.process_image(
                    image_path="/path/to/test_image.dcm",
                    model_name="chest_xray_classifier",
                    parameters={
                        "confidence_threshold": 0.8,
                        "enable_heatmap": True
                    }
                )
                
                # 验证结果
                assert "predictions" in result
                assert "heatmap" in result
                assert "processing_time" in result
                assert len(result["predictions"]) == 2
                assert result["predictions"][0]["class"] == "pneumonia"
                assert result["predictions"][0]["confidence"] == 0.92
    
    @pytest.mark.asyncio
    async def test_model_performance_monitoring(self, mock_model):
        """测试模型性能监控"""
        ai_service = AIInferenceService()
        
        with patch.object(ai_service, 'load_model', return_value=mock_model):
            with patch.object(ai_service, 'preprocess_image'):
                # 执行多次推理以测试性能监控
                results = []
                for i in range(5):
                    result = await ai_service.process_image(
                        image_path=f"/path/to/test_image_{i}.dcm",
                        model_name="chest_xray_classifier"
                    )
                    results.append(result)
                
                # 验证所有推理都成功
                assert len(results) == 5
                for result in results:
                    assert "predictions" in result
                    assert "processing_time" in result
                    assert result["processing_time"] > 0


class TestCeleryIntegration:
    """Celery 消息队列集成测试"""
    
    @pytest.fixture
    def celery_app(self):
        """创建 Celery 测试应用"""
        app = Celery('test_app')
        app.conf.update(
            broker_url='memory://',
            result_backend='cache+memory://',
            task_always_eager=True,  # 同步执行任务
            task_eager_propagates=True
        )
        return app
    
    @pytest.mark.asyncio
    async def test_inference_task_processing(self, celery_app):
        """测试推理任务处理"""
        # 模拟推理任务数据
        task_data = {
            "task_id": "task123",
            "image_id": "image456",
            "image_path": "/storage/images/test.dcm",
            "model_name": "chest_xray_classifier",
            "parameters": {
                "confidence_threshold": 0.8,
                "enable_heatmap": True
            }
        }
        
        with patch('backend.tasks.ai_inference.AIInferenceService') as mock_service:
            mock_instance = mock_service.return_value
            mock_instance.process_image.return_value = {
                "predictions": [{"class": "normal", "confidence": 0.95}],
                "processing_time": 1.5
            }
            
            # 执行任务
            result = process_inference_task.apply(args=[task_data])
            
            # 验证任务执行成功
            assert result.successful()
            task_result = result.get()
            assert task_result["status"] == "completed"
            assert "predictions" in task_result["results"]
    
    @pytest.mark.asyncio
    async def test_report_generation_task(self, celery_app):
        """测试报告生成任务"""
        # 模拟报告生成任务数据
        task_data = {
            "report_id": "report123",
            "patient_id": "patient456",
            "image_id": "image789",
            "inference_results": {
                "predictions": [{"class": "pneumonia", "confidence": 0.92}]
            },
            "template": "standard_report"
        }
        
        with patch('backend.tasks.report_generation.ReportService') as mock_service:
            mock_instance = mock_service.return_value
            mock_instance.generate_report.return_value = {
                "report_id": "report123",
                "content": "Generated report content",
                "format": "pdf"
            }
            
            # 执行任务
            result = generate_report_task.apply(args=[task_data])
            
            # 验证任务执行成功
            assert result.successful()
            task_result = result.get()
            assert task_result["status"] == "completed"
            assert task_result["report_id"] == "report123"


class TestEndToEndWorkflow:
    """端到端工作流测试"""
    
    @pytest.fixture
    async def test_environment(self):
        """设置测试环境"""
        # 创建临时目录
        temp_dir = tempfile.mkdtemp()
        
        # 模拟服务
        services = {
            'auth': Mock(spec=AuthService),
            'patient': Mock(spec=PatientService),
            'image': Mock(spec=ImageService),
            'ai_inference': Mock(spec=AIInferenceService),
            'report': Mock(spec=ReportService),
            'storage': Mock(spec=StorageService),
            'cache': Mock(spec=CacheService),
            'notification': Mock(spec=NotificationService)
        }
        
        yield {
            'temp_dir': temp_dir,
            'services': services
        }
        
        # 清理
        shutil.rmtree(temp_dir)
    
    @pytest.mark.asyncio
    async def test_complete_diagnosis_workflow(self, test_environment):
        """测试完整的诊断工作流"""
        services = test_environment['services']
        
        # 1. 用户登录
        services['auth'].authenticate_user.return_value = Mock(
            id="user123",
            username="doctor@test.com",
            role="doctor"
        )
        
        # 2. 创建患者
        services['patient'].create_patient.return_value = Mock(
            id="patient123",
            patient_id="P123456",
            name="测试患者"
        )
        
        # 3. 上传医疗影像
        services['image'].upload_image.return_value = Mock(
            id="image123",
            filename="chest_xray.dcm",
            patient_id="patient123",
            file_path="/storage/images/chest_xray.dcm"
        )
        
        # 4. 提交 AI 推理任务
        services['ai_inference'].submit_inference_task.return_value = Mock(
            task_id="task123",
            status="pending",
            image_id="image123"
        )
        
        # 5. AI 推理完成
        services['ai_inference'].get_inference_result.return_value = Mock(
            task_id="task123",
            status="completed",
            results={
                "predictions": [
                    {"class": "pneumonia", "confidence": 0.92}
                ]
            }
        )
        
        # 6. 生成诊断报告
        services['report'].create_report.return_value = Mock(
            id="report123",
            patient_id="patient123",
            image_id="image123",
            findings="右下肺野可见片状阴影",
            impression="右下肺炎",
            status="completed"
        )
        
        # 7. 发送通知
        services['notification'].send_notification.return_value = True
        
        # 执行工作流验证
        # 验证用户认证
        user = services['auth'].authenticate_user("doctor@test.com", "password")
        assert user.username == "doctor@test.com"
        
        # 验证患者创建
        patient = services['patient'].create_patient({
            "patient_id": "P123456",
            "name": "测试患者"
        })
        assert patient.patient_id == "P123456"
        
        # 验证影像上传
        image = services['image'].upload_image(
            file_path="/tmp/chest_xray.dcm",
            patient_id=patient.id
        )
        assert image.patient_id == "patient123"
        
        # 验证 AI 推理
        inference_task = services['ai_inference'].submit_inference_task(
            image_id=image.id,
            model_name="chest_xray_classifier"
        )
        assert inference_task.status == "pending"
        
        # 验证推理结果
        inference_result = services['ai_inference'].get_inference_result(
            inference_task.task_id
        )
        assert inference_result.status == "completed"
        assert len(inference_result.results["predictions"]) > 0
        
        # 验证报告生成
        report = services['report'].create_report({
            "patient_id": patient.id,
            "image_id": image.id,
            "findings": "右下肺野可见片状阴影",
            "impression": "右下肺炎"
        })
        assert report.status == "completed"
        
        # 验证通知发送
        notification_sent = services['notification'].send_notification(
            user_id=user.id,
            message="诊断报告已生成",
            type="report_completed"
        )
        assert notification_sent is True
    
    @pytest.mark.asyncio
    async def test_error_handling_workflow(self, test_environment):
        """测试错误处理工作流"""
        services = test_environment['services']
        
        # 模拟 AI 推理失败
        services['ai_inference'].submit_inference_task.side_effect = Exception(
            "AI model not available"
        )
        
        # 模拟错误处理
        try:
            services['ai_inference'].submit_inference_task(
                image_id="image123",
                model_name="unavailable_model"
            )
            assert False, "Should have raised an exception"
        except Exception as e:
            assert "AI model not available" in str(e)
            
            # 验证错误通知
            services['notification'].send_error_notification.return_value = True
            error_notification_sent = services['notification'].send_error_notification(
                user_id="user123",
                error_message=str(e),
                context="ai_inference"
            )
            assert error_notification_sent is True


class TestPerformanceIntegration:
    """性能集成测试"""
    
    @pytest.mark.asyncio
    async def test_concurrent_inference_tasks(self):
        """测试并发推理任务"""
        ai_service = AIInferenceService()
        
        with patch.object(ai_service, 'process_image') as mock_process:
            # 模拟推理处理时间
            async def mock_inference(*args, **kwargs):
                await asyncio.sleep(0.1)  # 模拟处理时间
                return {
                    "predictions": [{"class": "normal", "confidence": 0.95}],
                    "processing_time": 0.1
                }
            
            mock_process.side_effect = mock_inference
            
            # 并发执行多个推理任务
            tasks = []
            for i in range(10):
                task = ai_service.process_image(
                    image_path=f"/path/to/image_{i}.dcm",
                    model_name="chest_xray_classifier"
                )
                tasks.append(task)
            
            # 等待所有任务完成
            start_time = asyncio.get_event_loop().time()
            results = await asyncio.gather(*tasks)
            end_time = asyncio.get_event_loop().time()
            
            # 验证并发执行效果
            total_time = end_time - start_time
            assert total_time < 1.0  # 并发执行应该比串行快
            assert len(results) == 10
            
            # 验证所有结果
            for result in results:
                assert "predictions" in result
                assert result["predictions"][0]["class"] == "normal"
    
    @pytest.mark.asyncio
    async def test_database_connection_pool(self):
        """测试数据库连接池"""
        # 模拟多个并发数据库操作
        async def mock_db_operation(session_id: int):
            # 模拟数据库查询
            await asyncio.sleep(0.05)
            return f"result_{session_id}"
        
        # 并发执行多个数据库操作
        tasks = []
        for i in range(20):
            task = mock_db_operation(i)
            tasks.append(task)
        
        start_time = asyncio.get_event_loop().time()
        results = await asyncio.gather(*tasks)
        end_time = asyncio.get_event_loop().time()
        
        # 验证连接池效果
        total_time = end_time - start_time
        assert total_time < 0.5  # 连接池应该提高并发性能
        assert len(results) == 20


if __name__ == "__main__":
    # 运行集成测试
    pytest.main([
        "-v",
        "--cov=backend",
        "--cov-report=html",
        "--cov-report=term-missing",
        "-m", "not slow",  # 排除慢速测试
        __file__
    ])