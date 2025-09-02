# -*- coding: utf-8 -*-
"""
Pytest 配置文件

本文件包含了测试的全局配置、fixtures 和工具函数：
- 测试数据库配置
- 测试客户端配置
- 模拟数据生成
- 测试环境设置
- 通用测试工具
"""

import pytest
import asyncio
import tempfile
import os
import shutil
from datetime import datetime, timedelta
from typing import Dict, Any, List, Generator
from unittest.mock import Mock, patch, AsyncMock
import json
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker
from httpx import AsyncClient
from fastapi.testclient import TestClient
import redis
import fakeredis
from moto import mock_s3
import boto3
from celery import Celery

from backend.main import app
from backend.core.config import settings
from backend.core.database import Base, get_db
from backend.models.user import User
from backend.models.patient import Patient
from backend.models.image import MedicalImage
from backend.models.report import DiagnosticReport
from backend.models.inference import InferenceTask
from backend.services.auth import create_access_token


class TestConfig:
    """测试配置类"""
    
    # 数据库配置
    TEST_DATABASE_URL = "sqlite+aiosqlite:///:memory:"
    
    # Redis 配置
    TEST_REDIS_URL = "redis://localhost:6379/15"  # 使用测试数据库
    
    # 文件存储配置
    TEST_STORAGE_BUCKET = "medical-ai-test"
    TEST_STORAGE_PATH = "/tmp/medical_ai_test"
    
    # AI 模型配置
    TEST_MODEL_PATH = "/tmp/test_models"
    
    # 测试用户数据
    TEST_USERS = {
        "doctor": {
            "id": "doctor_uuid",
            "username": "doctor@test.com",
            "email": "doctor@test.com",
            "full_name": "Test Doctor",
            "password": "test_password123",
            "role": "doctor",
            "department": "Radiology",
            "license_number": "DOC123456",
            "is_active": True
        },
        "admin": {
            "id": "admin_uuid",
            "username": "admin@test.com",
            "email": "admin@test.com",
            "full_name": "Test Admin",
            "password": "admin_password123",
            "role": "admin",
            "department": "IT",
            "is_active": True
        },
        "radiologist": {
            "id": "radiologist_uuid",
            "username": "radiologist@test.com",
            "email": "radiologist@test.com",
            "full_name": "Test Radiologist",
            "password": "radio_password123",
            "role": "radiologist",
            "department": "Radiology",
            "license_number": "RAD123456",
            "is_active": True
        }
    }
    
    # 测试患者数据
    TEST_PATIENTS = [
        {
            "id": "patient1_uuid",
            "patient_id": "P001",
            "name": "张三",
            "gender": "male",
            "birth_date": "1980-05-15",
            "phone": "13800138001",
            "email": "zhangsan@test.com",
            "address": "北京市朝阳区",
            "emergency_contact": "李四",
            "emergency_phone": "13900139001"
        },
        {
            "id": "patient2_uuid",
            "patient_id": "P002",
            "name": "李四",
            "gender": "female",
            "birth_date": "1985-08-20",
            "phone": "13800138002",
            "email": "lisi@test.com",
            "address": "上海市浦东新区",
            "emergency_contact": "王五",
            "emergency_phone": "13900139002"
        },
        {
            "id": "patient3_uuid",
            "patient_id": "P003",
            "name": "王五",
            "gender": "male",
            "birth_date": "1975-12-10",
            "phone": "13800138003",
            "email": "wangwu@test.com",
            "address": "广州市天河区",
            "emergency_contact": "赵六",
            "emergency_phone": "13900139003"
        }
    ]
    
    # 测试影像数据
    TEST_IMAGES = [
        {
            "id": "image1_uuid",
            "filename": "chest_xray_001.dcm",
            "original_filename": "chest_xray_001_original.dcm",
            "file_path": "/storage/images/chest_xray_001.dcm",
            "file_size": 1024000,
            "study_type": "chest_xray",
            "patient_id": "patient1_uuid",
            "description": "胸部X光检查",
            "acquisition_date": "2024-01-15T10:30:00",
            "modality": "CR",
            "body_part": "CHEST"
        },
        {
            "id": "image2_uuid",
            "filename": "brain_mri_001.dcm",
            "original_filename": "brain_mri_001_original.dcm",
            "file_path": "/storage/images/brain_mri_001.dcm",
            "file_size": 5120000,
            "study_type": "brain_mri",
            "patient_id": "patient2_uuid",
            "description": "脑部MRI检查",
            "acquisition_date": "2024-01-16T14:20:00",
            "modality": "MR",
            "body_part": "BRAIN"
        }
    ]
    
    # 测试推理结果
    TEST_INFERENCE_RESULTS = {
        "chest_xray": {
            "predictions": [
                {"class": "pneumonia", "confidence": 0.92, "bbox": [100, 150, 200, 250]},
                {"class": "normal", "confidence": 0.08, "bbox": None}
            ],
            "heatmap": "base64_encoded_heatmap_data",
            "processing_time": 2.5,
            "model_version": "1.2.0"
        },
        "brain_mri": {
            "predictions": [
                {"class": "tumor", "confidence": 0.85, "bbox": [50, 75, 150, 175]},
                {"class": "normal", "confidence": 0.15, "bbox": None}
            ],
            "heatmap": "base64_encoded_heatmap_data",
            "processing_time": 5.2,
            "model_version": "2.1.0"
        }
    }


# Pytest 配置
def pytest_configure(config):
    """Pytest 配置"""
    # 设置测试标记
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "e2e: marks tests as end-to-end tests"
    )
    config.addinivalue_line(
        "markers", "unit: marks tests as unit tests"
    )


# 异步测试支持
@pytest.fixture(scope="session")
def event_loop():
    """创建事件循环"""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


# 数据库相关 fixtures
@pytest.fixture(scope="session")
async def test_engine():
    """创建测试数据库引擎"""
    engine = create_async_engine(
        TestConfig.TEST_DATABASE_URL,
        echo=False,
        future=True
    )
    
    # 创建所有表
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    
    yield engine
    
    # 清理
    await engine.dispose()


@pytest.fixture
async def db_session(test_engine):
    """创建数据库会话"""
    async_session = sessionmaker(
        test_engine, class_=AsyncSession, expire_on_commit=False
    )
    
    async with async_session() as session:
        yield session
        await session.rollback()


@pytest.fixture
def override_get_db(db_session):
    """覆盖数据库依赖"""
    async def _override_get_db():
        yield db_session
    
    app.dependency_overrides[get_db] = _override_get_db
    yield
    app.dependency_overrides.clear()


# Redis 相关 fixtures
@pytest.fixture
def redis_client():
    """创建 Redis 测试客户端"""
    client = fakeredis.FakeRedis(decode_responses=True)
    yield client
    client.flushall()


# 文件存储相关 fixtures
@pytest.fixture
def temp_storage_dir():
    """创建临时存储目录"""
    temp_dir = tempfile.mkdtemp(prefix="medical_ai_test_")
    yield temp_dir
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
@mock_s3
def s3_client():
    """创建 S3 测试客户端"""
    client = boto3.client(
        "s3",
        region_name="us-east-1",
        aws_access_key_id="testing",
        aws_secret_access_key="testing"
    )
    
    # 创建测试桶
    client.create_bucket(Bucket=TestConfig.TEST_STORAGE_BUCKET)
    
    yield client


# HTTP 客户端相关 fixtures
@pytest.fixture
def client():
    """创建测试客户端"""
    return TestClient(app)


@pytest.fixture
async def async_client():
    """创建异步测试客户端"""
    async with AsyncClient(app=app, base_url="http://test") as ac:
        yield ac


# 认证相关 fixtures
@pytest.fixture
def test_user_doctor():
    """创建测试医生用户"""
    return TestConfig.TEST_USERS["doctor"]


@pytest.fixture
def test_user_admin():
    """创建测试管理员用户"""
    return TestConfig.TEST_USERS["admin"]


@pytest.fixture
def test_user_radiologist():
    """创建测试放射科医生用户"""
    return TestConfig.TEST_USERS["radiologist"]


@pytest.fixture
def doctor_token(test_user_doctor):
    """创建医生用户令牌"""
    token_data = {
        "sub": test_user_doctor["username"],
        "role": test_user_doctor["role"],
        "user_id": test_user_doctor["id"],
        "exp": datetime.utcnow() + timedelta(hours=1)
    }
    return create_access_token(data=token_data)


@pytest.fixture
def admin_token(test_user_admin):
    """创建管理员用户令牌"""
    token_data = {
        "sub": test_user_admin["username"],
        "role": test_user_admin["role"],
        "user_id": test_user_admin["id"],
        "exp": datetime.utcnow() + timedelta(hours=1)
    }
    return create_access_token(data=token_data)


@pytest.fixture
def radiologist_token(test_user_radiologist):
    """创建放射科医生用户令牌"""
    token_data = {
        "sub": test_user_radiologist["username"],
        "role": test_user_radiologist["role"],
        "user_id": test_user_radiologist["id"],
        "exp": datetime.utcnow() + timedelta(hours=1)
    }
    return create_access_token(data=token_data)


@pytest.fixture
def auth_headers_doctor(doctor_token):
    """创建医生认证头"""
    return {"Authorization": f"Bearer {doctor_token}"}


@pytest.fixture
def auth_headers_admin(admin_token):
    """创建管理员认证头"""
    return {"Authorization": f"Bearer {admin_token}"}


@pytest.fixture
def auth_headers_radiologist(radiologist_token):
    """创建放射科医生认证头"""
    return {"Authorization": f"Bearer {radiologist_token}"}


# 测试数据相关 fixtures
@pytest.fixture
def test_patients():
    """获取测试患者数据"""
    return TestConfig.TEST_PATIENTS


@pytest.fixture
def test_images():
    """获取测试影像数据"""
    return TestConfig.TEST_IMAGES


@pytest.fixture
def test_inference_results():
    """获取测试推理结果"""
    return TestConfig.TEST_INFERENCE_RESULTS


# 文件相关 fixtures
@pytest.fixture
def sample_dicom_file():
    """创建示例 DICOM 文件"""
    with tempfile.NamedTemporaryFile(suffix=".dcm", delete=False) as f:
        # 创建简化的 DICOM 文件头
        dicom_header = b"DICM"
        dicom_data = b"\x00" * 1000  # 1KB 的测试数据
        f.write(dicom_header + dicom_data)
        f.flush()
        yield f.name
    os.unlink(f.name)


@pytest.fixture
def sample_image_files():
    """创建多个示例影像文件"""
    files = []
    
    for i in range(3):
        with tempfile.NamedTemporaryFile(suffix=f"_test_{i}.dcm", delete=False) as f:
            dicom_header = b"DICM"
            dicom_data = b"\x00" * (1000 + i * 500)  # 不同大小的文件
            f.write(dicom_header + dicom_data)
            f.flush()
            files.append(f.name)
    
    yield files
    
    # 清理文件
    for file_path in files:
        try:
            os.unlink(file_path)
        except FileNotFoundError:
            pass


# Celery 相关 fixtures
@pytest.fixture
def celery_app():
    """创建 Celery 测试应用"""
    app = Celery('test_app')
    app.conf.update(
        broker_url='memory://',
        result_backend='cache+memory://',
        task_always_eager=True,  # 同步执行任务
        task_eager_propagates=True,
        task_store_eager_result=True
    )
    return app


# 模拟服务 fixtures
@pytest.fixture
def mock_ai_service():
    """模拟 AI 推理服务"""
    service = Mock()
    
    async def mock_process_image(*args, **kwargs):
        study_type = kwargs.get('model_name', 'chest_xray_classifier')
        if 'chest' in study_type:
            return TestConfig.TEST_INFERENCE_RESULTS['chest_xray']
        elif 'brain' in study_type:
            return TestConfig.TEST_INFERENCE_RESULTS['brain_mri']
        else:
            return {
                "predictions": [{"class": "normal", "confidence": 0.95}],
                "processing_time": 1.0
            }
    
    service.process_image = AsyncMock(side_effect=mock_process_image)
    service.get_available_models = Mock(return_value=[
        {
            "name": "chest_xray_classifier",
            "version": "1.2.0",
            "description": "胸部X光分类模型",
            "accuracy": 0.94
        },
        {
            "name": "brain_mri_segmentation",
            "version": "2.1.0",
            "description": "脑部MRI分割模型",
            "accuracy": 0.89
        }
    ])
    
    return service


@pytest.fixture
def mock_storage_service():
    """模拟存储服务"""
    service = Mock()
    
    async def mock_upload_file(file_path, key, **kwargs):
        return {
            "success": True,
            "file_key": key,
            "file_size": os.path.getsize(file_path) if os.path.exists(file_path) else 1024,
            "upload_time": datetime.utcnow().isoformat()
        }
    
    async def mock_download_file(key):
        return b"mock_file_content"
    
    async def mock_delete_file(key):
        return {"success": True}
    
    async def mock_file_exists(key):
        return True
    
    service.upload_file = AsyncMock(side_effect=mock_upload_file)
    service.download_file = AsyncMock(side_effect=mock_download_file)
    service.delete_file = AsyncMock(side_effect=mock_delete_file)
    service.file_exists = AsyncMock(side_effect=mock_file_exists)
    
    return service


@pytest.fixture
def mock_notification_service():
    """模拟通知服务"""
    service = Mock()
    
    async def mock_send_notification(*args, **kwargs):
        return True
    
    async def mock_send_email(*args, **kwargs):
        return {"success": True, "message_id": "test_message_id"}
    
    service.send_notification = AsyncMock(side_effect=mock_send_notification)
    service.send_email = AsyncMock(side_effect=mock_send_email)
    
    return service


# 数据库数据填充 fixtures
@pytest.fixture
async def populated_db(db_session):
    """填充测试数据到数据库"""
    # 创建测试用户
    users = []
    for user_data in TestConfig.TEST_USERS.values():
        user = User(
            id=user_data["id"],
            username=user_data["username"],
            email=user_data["email"],
            full_name=user_data["full_name"],
            hashed_password="hashed_" + user_data["password"],
            role=user_data["role"],
            department=user_data["department"],
            license_number=user_data.get("license_number"),
            is_active=user_data["is_active"]
        )
        users.append(user)
        db_session.add(user)
    
    # 创建测试患者
    patients = []
    for patient_data in TestConfig.TEST_PATIENTS:
        patient = Patient(
            id=patient_data["id"],
            patient_id=patient_data["patient_id"],
            name=patient_data["name"],
            gender=patient_data["gender"],
            birth_date=datetime.strptime(patient_data["birth_date"], "%Y-%m-%d").date(),
            phone=patient_data["phone"],
            email=patient_data["email"],
            address=patient_data["address"],
            emergency_contact=patient_data["emergency_contact"],
            emergency_phone=patient_data["emergency_phone"],
            created_by=users[0].id  # 由第一个用户创建
        )
        patients.append(patient)
        db_session.add(patient)
    
    # 创建测试影像
    images = []
    for i, image_data in enumerate(TestConfig.TEST_IMAGES):
        image = MedicalImage(
            id=image_data["id"],
            filename=image_data["filename"],
            original_filename=image_data["original_filename"],
            file_path=image_data["file_path"],
            file_size=image_data["file_size"],
            study_type=image_data["study_type"],
            patient_id=patients[i].id,  # 关联到对应患者
            description=image_data["description"],
            acquisition_date=datetime.fromisoformat(image_data["acquisition_date"]),
            modality=image_data["modality"],
            body_part=image_data["body_part"],
            uploaded_by=users[0].id
        )
        images.append(image)
        db_session.add(image)
    
    await db_session.commit()
    
    return {
        "users": users,
        "patients": patients,
        "images": images
    }


# 工具函数
def create_test_user_data(role="doctor", **overrides):
    """创建测试用户数据"""
    base_data = TestConfig.TEST_USERS.get(role, TestConfig.TEST_USERS["doctor"]).copy()
    base_data.update(overrides)
    return base_data


def create_test_patient_data(**overrides):
    """创建测试患者数据"""
    base_data = TestConfig.TEST_PATIENTS[0].copy()
    base_data.update(overrides)
    return base_data


def create_test_image_data(**overrides):
    """创建测试影像数据"""
    base_data = TestConfig.TEST_IMAGES[0].copy()
    base_data.update(overrides)
    return base_data


def assert_response_success(response, expected_status=200):
    """断言响应成功"""
    assert response.status_code == expected_status
    if response.headers.get("content-type", "").startswith("application/json"):
        data = response.json()
        assert "error" not in data or data["error"] is None


def assert_response_error(response, expected_status=400):
    """断言响应错误"""
    assert response.status_code == expected_status
    if response.headers.get("content-type", "").startswith("application/json"):
        data = response.json()
        assert "detail" in data or "error" in data


def generate_mock_dicom_data(size_kb=1):
    """生成模拟 DICOM 数据"""
    header = b"DICM"
    data = b"\x00" * (size_kb * 1024 - len(header))
    return header + data


def wait_for_async_task(task_func, timeout=10, interval=0.1):
    """等待异步任务完成"""
    import time
    start_time = time.time()
    
    while time.time() - start_time < timeout:
        try:
            result = task_func()
            if result:
                return result
        except Exception:
            pass
        time.sleep(interval)
    
    raise TimeoutError(f"Task did not complete within {timeout} seconds")


# 性能测试工具
class PerformanceTimer:
    """性能计时器"""
    
    def __init__(self):
        self.start_time = None
        self.end_time = None
    
    def __enter__(self):
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = time.time()
    
    @property
    def elapsed_time(self):
        if self.start_time and self.end_time:
            return self.end_time - self.start_time
        return None


@pytest.fixture
def performance_timer():
    """性能计时器 fixture"""
    return PerformanceTimer