# -*- coding: utf-8 -*-
"""
Medical AI API 测试用例

本模块包含了 Medical AI 系统的 API 测试用例，包括：
- 用户认证和授权测试
- 患者管理测试
- 医疗影像管理测试
- AI 推理测试
- 诊断报告测试
"""

import pytest
import asyncio
from httpx import AsyncClient
from fastapi.testclient import TestClient
from sqlalchemy.ext.asyncio import AsyncSession
from unittest.mock import Mock, patch, AsyncMock
import tempfile
import os
from datetime import datetime, timedelta
import jwt
from typing import Dict, Any

from backend.main import app
from backend.core.config import settings
from backend.core.database import get_db
from backend.models.user import User
from backend.models.patient import Patient
from backend.models.image import MedicalImage
from backend.models.report import DiagnosticReport
from backend.services.auth import create_access_token
from backend.services.ai_inference import AIInferenceService


class TestConfig:
    """测试配置"""
    
    # 测试数据库 URL
    TEST_DATABASE_URL = "sqlite+aiosqlite:///./test.db"
    
    # 测试用户数据
    TEST_USER_DATA = {
        "username": "test_doctor@example.com",
        "email": "test_doctor@example.com",
        "full_name": "Test Doctor",
        "password": "test_password123",
        "role": "doctor",
        "department": "Radiology",
        "license_number": "TEST123456"
    }
    
    # 测试患者数据
    TEST_PATIENT_DATA = {
        "patient_id": "P123456",
        "name": "测试患者",
        "gender": "male",
        "birth_date": "1980-05-15",
        "phone": "13800138000",
        "email": "patient@example.com",
        "address": "北京市朝阳区"
    }


@pytest.fixture
def client():
    """创建测试客户端"""
    return TestClient(app)


@pytest.fixture
async def async_client():
    """创建异步测试客户端"""
    async with AsyncClient(app=app, base_url="http://test") as ac:
        yield ac


@pytest.fixture
def test_user_token():
    """创建测试用户令牌"""
    user_data = {
        "sub": TestConfig.TEST_USER_DATA["username"],
        "role": TestConfig.TEST_USER_DATA["role"],
        "exp": datetime.utcnow() + timedelta(hours=1)
    }
    return create_access_token(data=user_data)


@pytest.fixture
def auth_headers(test_user_token):
    """创建认证头"""
    return {"Authorization": f"Bearer {test_user_token}"}


@pytest.fixture
def sample_dicom_file():
    """创建示例 DICOM 文件"""
    with tempfile.NamedTemporaryFile(suffix=".dcm", delete=False) as f:
        # 创建一个简单的测试文件
        f.write(b"DICM" + b"\x00" * 1000)  # 简化的 DICOM 文件头
        f.flush()
        yield f.name
    os.unlink(f.name)


class TestAuthentication:
    """认证相关测试"""
    
    def test_login_success(self, client):
        """测试登录成功"""
        with patch('backend.services.auth.authenticate_user') as mock_auth:
            mock_user = Mock()
            mock_user.username = TestConfig.TEST_USER_DATA["username"]
            mock_user.role = TestConfig.TEST_USER_DATA["role"]
            mock_auth.return_value = mock_user
            
            response = client.post(
                "/api/v1/auth/login",
                data={
                    "username": TestConfig.TEST_USER_DATA["username"],
                    "password": TestConfig.TEST_USER_DATA["password"]
                }
            )
            
            assert response.status_code == 200
            data = response.json()
            assert "access_token" in data
            assert "token_type" in data
            assert data["token_type"] == "bearer"
    
    def test_login_invalid_credentials(self, client):
        """测试登录失败 - 无效凭据"""
        with patch('backend.services.auth.authenticate_user') as mock_auth:
            mock_auth.return_value = None
            
            response = client.post(
                "/api/v1/auth/login",
                data={
                    "username": "invalid@example.com",
                    "password": "wrong_password"
                }
            )
            
            assert response.status_code == 401
            assert "Incorrect username or password" in response.json()["detail"]
    
    def test_protected_endpoint_without_token(self, client):
        """测试访问受保护端点时未提供令牌"""
        response = client.get("/api/v1/users/me")
        assert response.status_code == 401
    
    def test_protected_endpoint_with_invalid_token(self, client):
        """测试使用无效令牌访问受保护端点"""
        headers = {"Authorization": "Bearer invalid_token"}
        response = client.get("/api/v1/users/me", headers=headers)
        assert response.status_code == 401
    
    def test_token_refresh(self, client, test_user_token):
        """测试令牌刷新"""
        with patch('backend.services.auth.verify_refresh_token') as mock_verify:
            mock_verify.return_value = TestConfig.TEST_USER_DATA["username"]
            
            response = client.post(
                "/api/v1/auth/refresh",
                json={"refresh_token": "valid_refresh_token"}
            )
            
            assert response.status_code == 200
            data = response.json()
            assert "access_token" in data


class TestUserManagement:
    """用户管理测试"""
    
    def test_get_current_user(self, client, auth_headers):
        """测试获取当前用户信息"""
        with patch('backend.services.user.get_current_user') as mock_get_user:
            mock_user = Mock()
            mock_user.username = TestConfig.TEST_USER_DATA["username"]
            mock_user.email = TestConfig.TEST_USER_DATA["email"]
            mock_user.full_name = TestConfig.TEST_USER_DATA["full_name"]
            mock_user.role = TestConfig.TEST_USER_DATA["role"]
            mock_get_user.return_value = mock_user
            
            response = client.get("/api/v1/users/me", headers=auth_headers)
            
            assert response.status_code == 200
            data = response.json()
            assert data["username"] == TestConfig.TEST_USER_DATA["username"]
            assert data["role"] == TestConfig.TEST_USER_DATA["role"]
    
    def test_update_user_profile(self, client, auth_headers):
        """测试更新用户资料"""
        update_data = {
            "full_name": "Updated Doctor Name",
            "department": "Emergency Medicine"
        }
        
        with patch('backend.services.user.update_user_profile') as mock_update:
            mock_user = Mock()
            mock_user.full_name = update_data["full_name"]
            mock_user.department = update_data["department"]
            mock_update.return_value = mock_user
            
            response = client.put(
                "/api/v1/users/me",
                json=update_data,
                headers=auth_headers
            )
            
            assert response.status_code == 200
            data = response.json()
            assert data["full_name"] == update_data["full_name"]
    
    def test_change_password(self, client, auth_headers):
        """测试修改密码"""
        password_data = {
            "current_password": "old_password",
            "new_password": "new_password123"
        }
        
        with patch('backend.services.user.change_password') as mock_change:
            mock_change.return_value = True
            
            response = client.post(
                "/api/v1/users/change-password",
                json=password_data,
                headers=auth_headers
            )
            
            assert response.status_code == 200
            assert response.json()["message"] == "Password changed successfully"


class TestPatientManagement:
    """患者管理测试"""
    
    def test_create_patient(self, client, auth_headers):
        """测试创建患者"""
        with patch('backend.services.patient.create_patient') as mock_create:
            mock_patient = Mock()
            mock_patient.id = "patient_uuid"
            mock_patient.patient_id = TestConfig.TEST_PATIENT_DATA["patient_id"]
            mock_patient.name = TestConfig.TEST_PATIENT_DATA["name"]
            mock_create.return_value = mock_patient
            
            response = client.post(
                "/api/v1/patients",
                json=TestConfig.TEST_PATIENT_DATA,
                headers=auth_headers
            )
            
            assert response.status_code == 201
            data = response.json()
            assert data["patient_id"] == TestConfig.TEST_PATIENT_DATA["patient_id"]
    
    def test_get_patient_list(self, client, auth_headers):
        """测试获取患者列表"""
        with patch('backend.services.patient.get_patient_list') as mock_get_list:
            mock_patients = [
                Mock(id="1", patient_id="P001", name="患者1"),
                Mock(id="2", patient_id="P002", name="患者2")
            ]
            mock_get_list.return_value = (mock_patients, 2)
            
            response = client.get(
                "/api/v1/patients?page=1&size=10",
                headers=auth_headers
            )
            
            assert response.status_code == 200
            data = response.json()
            assert len(data["items"]) == 2
            assert data["total"] == 2
    
    def test_get_patient_detail(self, client, auth_headers):
        """测试获取患者详情"""
        patient_id = "patient_uuid"
        
        with patch('backend.services.patient.get_patient_by_id') as mock_get:
            mock_patient = Mock()
            mock_patient.id = patient_id
            mock_patient.patient_id = TestConfig.TEST_PATIENT_DATA["patient_id"]
            mock_patient.name = TestConfig.TEST_PATIENT_DATA["name"]
            mock_get.return_value = mock_patient
            
            response = client.get(
                f"/api/v1/patients/{patient_id}",
                headers=auth_headers
            )
            
            assert response.status_code == 200
            data = response.json()
            assert data["id"] == patient_id
    
    def test_patient_not_found(self, client, auth_headers):
        """测试患者不存在"""
        with patch('backend.services.patient.get_patient_by_id') as mock_get:
            mock_get.return_value = None
            
            response = client.get(
                "/api/v1/patients/nonexistent",
                headers=auth_headers
            )
            
            assert response.status_code == 404


class TestImageManagement:
    """医疗影像管理测试"""
    
    def test_upload_image(self, client, auth_headers, sample_dicom_file):
        """测试上传医疗影像"""
        with patch('backend.services.image.save_uploaded_image') as mock_save:
            mock_image = Mock()
            mock_image.id = "image_uuid"
            mock_image.filename = "test.dcm"
            mock_image.patient_id = TestConfig.TEST_PATIENT_DATA["patient_id"]
            mock_save.return_value = mock_image
            
            with open(sample_dicom_file, "rb") as f:
                response = client.post(
                    "/api/v1/images/upload",
                    files={"file": ("test.dcm", f, "application/dicom")},
                    data={
                        "patient_id": TestConfig.TEST_PATIENT_DATA["patient_id"],
                        "study_type": "chest_xray",
                        "description": "胸部X光检查"
                    },
                    headers=auth_headers
                )
            
            assert response.status_code == 201
            data = response.json()
            assert data["filename"] == "test.dcm"
    
    def test_get_image_list(self, client, auth_headers):
        """测试获取影像列表"""
        with patch('backend.services.image.get_image_list') as mock_get_list:
            mock_images = [
                Mock(id="1", filename="image1.dcm", study_type="chest_xray"),
                Mock(id="2", filename="image2.dcm", study_type="brain_mri")
            ]
            mock_get_list.return_value = (mock_images, 2)
            
            response = client.get(
                "/api/v1/images?patient_id=P123456",
                headers=auth_headers
            )
            
            assert response.status_code == 200
            data = response.json()
            assert len(data["items"]) == 2
    
    def test_get_image_detail(self, client, auth_headers):
        """测试获取影像详情"""
        image_id = "image_uuid"
        
        with patch('backend.services.image.get_image_by_id') as mock_get:
            mock_image = Mock()
            mock_image.id = image_id
            mock_image.filename = "test.dcm"
            mock_image.study_type = "chest_xray"
            mock_get.return_value = mock_image
            
            response = client.get(
                f"/api/v1/images/{image_id}",
                headers=auth_headers
            )
            
            assert response.status_code == 200
            data = response.json()
            assert data["id"] == image_id
    
    def test_download_image(self, client, auth_headers):
        """测试下载影像文件"""
        image_id = "image_uuid"
        
        with patch('backend.services.image.get_image_file_path') as mock_get_path:
            mock_get_path.return_value = "/path/to/image.dcm"
            
            with patch('builtins.open', create=True) as mock_open:
                mock_open.return_value.__enter__.return_value.read.return_value = b"dicom_data"
                
                response = client.get(
                    f"/api/v1/images/{image_id}/download",
                    headers=auth_headers
                )
                
                assert response.status_code == 200
                assert response.headers["content-type"] == "application/dicom"


class TestAIInference:
    """AI 推理测试"""
    
    def test_submit_inference_task(self, client, auth_headers):
        """测试提交推理任务"""
        inference_data = {
            "image_id": "image_uuid",
            "model_name": "chest_xray_classifier",
            "parameters": {
                "confidence_threshold": 0.8,
                "enable_heatmap": True
            }
        }
        
        with patch('backend.services.ai_inference.submit_inference_task') as mock_submit:
            mock_task = Mock()
            mock_task.task_id = "task_uuid"
            mock_task.status = "pending"
            mock_submit.return_value = mock_task
            
            response = client.post(
                "/api/v1/ai/inference",
                json=inference_data,
                headers=auth_headers
            )
            
            assert response.status_code == 202
            data = response.json()
            assert data["task_id"] == "task_uuid"
            assert data["status"] == "pending"
    
    def test_get_inference_result(self, client, auth_headers):
        """测试获取推理结果"""
        task_id = "task_uuid"
        
        with patch('backend.services.ai_inference.get_inference_result') as mock_get:
            mock_result = Mock()
            mock_result.task_id = task_id
            mock_result.status = "completed"
            mock_result.results = {
                "predictions": [
                    {"class": "pneumonia", "confidence": 0.92}
                ]
            }
            mock_get.return_value = mock_result
            
            response = client.get(
                f"/api/v1/ai/inference/{task_id}",
                headers=auth_headers
            )
            
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "completed"
            assert "predictions" in data["results"]
    
    def test_get_available_models(self, client, auth_headers):
        """测试获取可用模型列表"""
        with patch('backend.services.ai_inference.get_available_models') as mock_get:
            mock_models = [
                {
                    "name": "chest_xray_classifier",
                    "version": "1.2.0",
                    "description": "胸部X光分类模型",
                    "accuracy": 0.94
                }
            ]
            mock_get.return_value = mock_models
            
            response = client.get(
                "/api/v1/ai/models",
                headers=auth_headers
            )
            
            assert response.status_code == 200
            data = response.json()
            assert len(data["models"]) == 1
            assert data["models"][0]["name"] == "chest_xray_classifier"
    
    @pytest.mark.asyncio
    async def test_ai_inference_service(self):
        """测试 AI 推理服务"""
        service = AIInferenceService()
        
        with patch.object(service, 'load_model') as mock_load:
            with patch.object(service, 'preprocess_image') as mock_preprocess:
                with patch.object(service, 'run_inference') as mock_inference:
                    mock_load.return_value = Mock()
                    mock_preprocess.return_value = Mock()
                    mock_inference.return_value = {
                        "predictions": [{"class": "normal", "confidence": 0.95}]
                    }
                    
                    result = await service.process_image(
                        image_path="/path/to/image.dcm",
                        model_name="chest_xray_classifier"
                    )
                    
                    assert "predictions" in result
                    assert result["predictions"][0]["class"] == "normal"


class TestReportManagement:
    """诊断报告测试"""
    
    def test_create_report(self, client, auth_headers):
        """测试创建诊断报告"""
        report_data = {
            "patient_id": TestConfig.TEST_PATIENT_DATA["patient_id"],
            "image_id": "image_uuid",
            "study_type": "chest_xray",
            "findings": "右下肺野可见片状阴影",
            "impression": "右下肺炎",
            "recommendations": "建议抗感染治疗"
        }
        
        with patch('backend.services.report.create_report') as mock_create:
            mock_report = Mock()
            mock_report.id = "report_uuid"
            mock_report.patient_id = report_data["patient_id"]
            mock_report.findings = report_data["findings"]
            mock_create.return_value = mock_report
            
            response = client.post(
                "/api/v1/reports",
                json=report_data,
                headers=auth_headers
            )
            
            assert response.status_code == 201
            data = response.json()
            assert data["findings"] == report_data["findings"]
    
    def test_get_report_list(self, client, auth_headers):
        """测试获取报告列表"""
        with patch('backend.services.report.get_report_list') as mock_get_list:
            mock_reports = [
                Mock(id="1", patient_id="P001", status="completed"),
                Mock(id="2", patient_id="P002", status="draft")
            ]
            mock_get_list.return_value = (mock_reports, 2)
            
            response = client.get(
                "/api/v1/reports?status=completed",
                headers=auth_headers
            )
            
            assert response.status_code == 200
            data = response.json()
            assert len(data["items"]) == 2
    
    def test_update_report(self, client, auth_headers):
        """测试更新报告"""
        report_id = "report_uuid"
        update_data = {
            "findings": "更新后的发现",
            "impression": "更新后的印象",
            "status": "completed"
        }
        
        with patch('backend.services.report.update_report') as mock_update:
            mock_report = Mock()
            mock_report.id = report_id
            mock_report.findings = update_data["findings"]
            mock_report.status = update_data["status"]
            mock_update.return_value = mock_report
            
            response = client.put(
                f"/api/v1/reports/{report_id}",
                json=update_data,
                headers=auth_headers
            )
            
            assert response.status_code == 200
            data = response.json()
            assert data["findings"] == update_data["findings"]
    
    def test_export_report(self, client, auth_headers):
        """测试导出报告"""
        report_id = "report_uuid"
        
        with patch('backend.services.report.export_report_pdf') as mock_export:
            mock_export.return_value = b"PDF content"
            
            response = client.get(
                f"/api/v1/reports/{report_id}/export?format=pdf",
                headers=auth_headers
            )
            
            assert response.status_code == 200
            assert response.headers["content-type"] == "application/pdf"


class TestSystemHealth:
    """系统健康检查测试"""
    
    def test_health_check(self, client):
        """测试健康检查端点"""
        with patch('backend.services.health.check_database') as mock_db:
            with patch('backend.services.health.check_redis') as mock_redis:
                with patch('backend.services.health.check_ai_service') as mock_ai:
                    mock_db.return_value = {"status": "healthy", "response_time": 10}
                    mock_redis.return_value = {"status": "healthy", "response_time": 5}
                    mock_ai.return_value = {"status": "healthy", "response_time": 100}
                    
                    response = client.get("/api/v1/system/health")
                    
                    assert response.status_code == 200
                    data = response.json()
                    assert data["status"] == "healthy"
                    assert "services" in data
    
    def test_health_check_with_service_down(self, client):
        """测试服务故障时的健康检查"""
        with patch('backend.services.health.check_database') as mock_db:
            with patch('backend.services.health.check_redis') as mock_redis:
                mock_db.return_value = {"status": "unhealthy", "error": "Connection failed"}
                mock_redis.return_value = {"status": "healthy", "response_time": 5}
                
                response = client.get("/api/v1/system/health")
                
                assert response.status_code == 503
                data = response.json()
                assert data["status"] == "unhealthy"


class TestErrorHandling:
    """错误处理测试"""
    
    def test_validation_error(self, client, auth_headers):
        """测试验证错误"""
        invalid_patient_data = {
            "name": "",  # 空名称
            "gender": "invalid",  # 无效性别
            "birth_date": "invalid-date"  # 无效日期
        }
        
        response = client.post(
            "/api/v1/patients",
            json=invalid_patient_data,
            headers=auth_headers
        )
        
        assert response.status_code == 422
        data = response.json()
        assert "detail" in data
    
    def test_rate_limiting(self, client, auth_headers):
        """测试请求限流"""
        # 模拟大量请求
        with patch('backend.middleware.rate_limiter.is_rate_limited') as mock_limit:
            mock_limit.return_value = True
            
            response = client.get("/api/v1/users/me", headers=auth_headers)
            
            assert response.status_code == 429
            assert "Rate limit exceeded" in response.json()["detail"]
    
    def test_internal_server_error(self, client, auth_headers):
        """测试服务器内部错误"""
        with patch('backend.services.user.get_current_user') as mock_get_user:
            mock_get_user.side_effect = Exception("Database connection failed")
            
            response = client.get("/api/v1/users/me", headers=auth_headers)
            
            assert response.status_code == 500
            data = response.json()
            assert "Internal server error" in data["detail"]


class TestWebSocketConnection:
    """WebSocket 连接测试"""
    
    @pytest.mark.asyncio
    async def test_websocket_connection(self):
        """测试 WebSocket 连接"""
        with patch('backend.websocket.verify_token') as mock_verify:
            mock_verify.return_value = TestConfig.TEST_USER_DATA["username"]
            
            async with AsyncClient(app=app, base_url="http://test") as client:
                with client.websocket_connect("/ws?token=valid_token") as websocket:
                    # 发送测试消息
                    await websocket.send_json({
                        "type": "ping",
                        "data": {"message": "test"}
                    })
                    
                    # 接收响应
                    response = await websocket.receive_json()
                    assert response["type"] == "pong"
    
    @pytest.mark.asyncio
    async def test_websocket_authentication_failure(self):
        """测试 WebSocket 认证失败"""
        with patch('backend.websocket.verify_token') as mock_verify:
            mock_verify.return_value = None
            
            async with AsyncClient(app=app, base_url="http://test") as client:
                with pytest.raises(Exception):  # 连接应该被拒绝
                    async with client.websocket_connect("/ws?token=invalid_token"):
                        pass


if __name__ == "__main__":
    # 运行测试
    pytest.main([
        "-v",
        "--cov=backend",
        "--cov-report=html",
        "--cov-report=term-missing",
        __file__
    ])