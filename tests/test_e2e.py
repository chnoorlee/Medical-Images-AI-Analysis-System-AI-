# -*- coding: utf-8 -*-
"""
Medical AI 端到端测试

本模块包含了 Medical AI 系统的端到端测试，模拟真实用户场景：
- 医生工作流测试
- 患者管理流程测试
- 影像诊断完整流程测试
- 报告生成和审核流程测试
- 系统管理员操作测试
- 性能和负载测试
"""

import pytest
import asyncio
import tempfile
import os
import json
from datetime import datetime, timedelta
from typing import Dict, Any, List
from unittest.mock import Mock, patch, AsyncMock
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options
from selenium.common.exceptions import TimeoutException
import requests
from httpx import AsyncClient
import websockets
import time

from backend.main import app
from backend.core.config import settings
from tests.conftest import TestConfig


class E2ETestConfig:
    """端到端测试配置"""
    
    # 测试环境配置
    BASE_URL = "http://localhost:8000"
    FRONTEND_URL = "http://localhost:3000"
    WEBSOCKET_URL = "ws://localhost:8000/ws"
    
    # 测试用户配置
    DOCTOR_USER = {
        "username": "doctor@test.com",
        "password": "test_password123",
        "role": "doctor",
        "department": "Radiology"
    }
    
    ADMIN_USER = {
        "username": "admin@test.com",
        "password": "admin_password123",
        "role": "admin",
        "department": "IT"
    }
    
    # 测试数据
    TEST_PATIENT = {
        "patient_id": "E2E001",
        "name": "端到端测试患者",
        "gender": "male",
        "birth_date": "1985-03-20",
        "phone": "13900139000",
        "email": "patient.e2e@test.com",
        "address": "北京市海淀区测试街道123号"
    }


class BaseE2ETest:
    """端到端测试基类"""
    
    @pytest.fixture(scope="class")
    def browser(self):
        """创建浏览器实例"""
        chrome_options = Options()
        chrome_options.add_argument("--headless")  # 无头模式
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        chrome_options.add_argument("--window-size=1920,1080")
        
        driver = webdriver.Chrome(options=chrome_options)
        driver.implicitly_wait(10)
        
        yield driver
        
        driver.quit()
    
    @pytest.fixture
    async def api_client(self):
        """创建 API 客户端"""
        async with AsyncClient(app=app, base_url=E2ETestConfig.BASE_URL) as client:
            yield client
    
    @pytest.fixture
    def auth_token(self, api_client):
        """获取认证令牌"""
        # 模拟登录获取令牌
        login_data = {
            "username": E2ETestConfig.DOCTOR_USER["username"],
            "password": E2ETestConfig.DOCTOR_USER["password"]
        }
        
        with patch('backend.services.auth.authenticate_user') as mock_auth:
            mock_user = Mock()
            mock_user.username = login_data["username"]
            mock_user.role = E2ETestConfig.DOCTOR_USER["role"]
            mock_auth.return_value = mock_user
            
            response = requests.post(
                f"{E2ETestConfig.BASE_URL}/api/v1/auth/login",
                data=login_data
            )
            
            if response.status_code == 200:
                return response.json()["access_token"]
            return None
    
    def wait_for_element(self, driver, locator, timeout=10):
        """等待元素出现"""
        return WebDriverWait(driver, timeout).until(
            EC.presence_of_element_located(locator)
        )
    
    def wait_for_clickable(self, driver, locator, timeout=10):
        """等待元素可点击"""
        return WebDriverWait(driver, timeout).until(
            EC.element_to_be_clickable(locator)
        )


class TestDoctorWorkflow(BaseE2ETest):
    """医生工作流测试"""
    
    def test_doctor_login_workflow(self, browser):
        """测试医生登录工作流"""
        # 访问登录页面
        browser.get(f"{E2ETestConfig.FRONTEND_URL}/login")
        
        # 输入用户名和密码
        username_input = self.wait_for_element(
            browser, (By.ID, "username")
        )
        password_input = browser.find_element(By.ID, "password")
        login_button = browser.find_element(By.ID, "login-button")
        
        username_input.send_keys(E2ETestConfig.DOCTOR_USER["username"])
        password_input.send_keys(E2ETestConfig.DOCTOR_USER["password"])
        
        # 点击登录
        login_button.click()
        
        # 验证登录成功，跳转到仪表板
        self.wait_for_element(browser, (By.ID, "dashboard"))
        assert "dashboard" in browser.current_url
        
        # 验证用户信息显示
        user_info = browser.find_element(By.ID, "user-info")
        assert E2ETestConfig.DOCTOR_USER["username"] in user_info.text
    
    def test_patient_management_workflow(self, browser, auth_token):
        """测试患者管理工作流"""
        # 模拟已登录状态
        browser.get(f"{E2ETestConfig.FRONTEND_URL}/dashboard")
        browser.execute_script(
            f"localStorage.setItem('auth_token', '{auth_token}')"
        )
        
        # 导航到患者管理页面
        browser.get(f"{E2ETestConfig.FRONTEND_URL}/patients")
        
        # 点击添加患者按钮
        add_patient_button = self.wait_for_clickable(
            browser, (By.ID, "add-patient-button")
        )
        add_patient_button.click()
        
        # 填写患者信息
        patient_form = self.wait_for_element(
            browser, (By.ID, "patient-form")
        )
        
        # 填写表单字段
        form_fields = {
            "patient-id": E2ETestConfig.TEST_PATIENT["patient_id"],
            "patient-name": E2ETestConfig.TEST_PATIENT["name"],
            "patient-gender": E2ETestConfig.TEST_PATIENT["gender"],
            "patient-birth-date": E2ETestConfig.TEST_PATIENT["birth_date"],
            "patient-phone": E2ETestConfig.TEST_PATIENT["phone"],
            "patient-email": E2ETestConfig.TEST_PATIENT["email"],
            "patient-address": E2ETestConfig.TEST_PATIENT["address"]
        }
        
        for field_id, value in form_fields.items():
            field = browser.find_element(By.ID, field_id)
            field.clear()
            field.send_keys(value)
        
        # 提交表单
        submit_button = browser.find_element(By.ID, "submit-patient")
        submit_button.click()
        
        # 验证患者创建成功
        success_message = self.wait_for_element(
            browser, (By.CLASS_NAME, "success-message")
        )
        assert "患者创建成功" in success_message.text
        
        # 验证患者出现在列表中
        patient_list = browser.find_element(By.ID, "patient-list")
        assert E2ETestConfig.TEST_PATIENT["name"] in patient_list.text
    
    def test_image_upload_workflow(self, browser, auth_token):
        """测试影像上传工作流"""
        # 创建测试 DICOM 文件
        with tempfile.NamedTemporaryFile(suffix=".dcm", delete=False) as temp_file:
            temp_file.write(b"DICM" + b"\x00" * 1000)
            temp_file.flush()
            test_file_path = temp_file.name
        
        try:
            # 模拟已登录状态
            browser.get(f"{E2ETestConfig.FRONTEND_URL}/dashboard")
            browser.execute_script(
                f"localStorage.setItem('auth_token', '{auth_token}')"
            )
            
            # 导航到影像上传页面
            browser.get(f"{E2ETestConfig.FRONTEND_URL}/images/upload")
            
            # 选择患者
            patient_select = self.wait_for_element(
                browser, (By.ID, "patient-select")
            )
            patient_select.send_keys(E2ETestConfig.TEST_PATIENT["name"])
            
            # 选择检查类型
            study_type_select = browser.find_element(By.ID, "study-type-select")
            study_type_select.send_keys("胸部X光")
            
            # 上传文件
            file_input = browser.find_element(By.ID, "file-input")
            file_input.send_keys(test_file_path)
            
            # 添加描述
            description_input = browser.find_element(By.ID, "description-input")
            description_input.send_keys("端到端测试影像")
            
            # 提交上传
            upload_button = browser.find_element(By.ID, "upload-button")
            upload_button.click()
            
            # 验证上传成功
            success_message = self.wait_for_element(
                browser, (By.CLASS_NAME, "upload-success")
            )
            assert "影像上传成功" in success_message.text
            
            # 验证影像出现在列表中
            image_list = browser.find_element(By.ID, "image-list")
            assert "端到端测试影像" in image_list.text
            
        finally:
            os.unlink(test_file_path)
    
    def test_ai_inference_workflow(self, browser, auth_token):
        """测试 AI 推理工作流"""
        # 模拟已登录状态
        browser.get(f"{E2ETestConfig.FRONTEND_URL}/dashboard")
        browser.execute_script(
            f"localStorage.setItem('auth_token', '{auth_token}')"
        )
        
        # 导航到影像列表页面
        browser.get(f"{E2ETestConfig.FRONTEND_URL}/images")
        
        # 选择一个影像进行分析
        image_item = self.wait_for_element(
            browser, (By.CLASS_NAME, "image-item")
        )
        analyze_button = image_item.find_element(By.CLASS_NAME, "analyze-button")
        analyze_button.click()
        
        # 选择 AI 模型
        model_select = self.wait_for_element(
            browser, (By.ID, "model-select")
        )
        model_select.send_keys("胸部X光分类模型")
        
        # 设置参数
        confidence_threshold = browser.find_element(By.ID, "confidence-threshold")
        confidence_threshold.clear()
        confidence_threshold.send_keys("0.8")
        
        enable_heatmap = browser.find_element(By.ID, "enable-heatmap")
        enable_heatmap.click()
        
        # 提交分析请求
        analyze_submit = browser.find_element(By.ID, "analyze-submit")
        analyze_submit.click()
        
        # 等待分析完成
        analysis_result = self.wait_for_element(
            browser, (By.ID, "analysis-result"), timeout=30
        )
        
        # 验证分析结果
        assert "分析完成" in analysis_result.text
        
        # 验证预测结果显示
        predictions = browser.find_element(By.ID, "predictions")
        assert len(predictions.find_elements(By.CLASS_NAME, "prediction-item")) > 0
        
        # 验证热力图显示（如果启用）
        heatmap = browser.find_element(By.ID, "heatmap")
        assert heatmap.is_displayed()
    
    def test_report_generation_workflow(self, browser, auth_token):
        """测试报告生成工作流"""
        # 模拟已登录状态
        browser.get(f"{E2ETestConfig.FRONTEND_URL}/dashboard")
        browser.execute_script(
            f"localStorage.setItem('auth_token', '{auth_token}')"
        )
        
        # 导航到报告页面
        browser.get(f"{E2ETestConfig.FRONTEND_URL}/reports/new")
        
        # 选择患者
        patient_select = self.wait_for_element(
            browser, (By.ID, "report-patient-select")
        )
        patient_select.send_keys(E2ETestConfig.TEST_PATIENT["name"])
        
        # 选择影像
        image_select = browser.find_element(By.ID, "report-image-select")
        image_select.send_keys("端到端测试影像")
        
        # 填写报告内容
        findings_textarea = browser.find_element(By.ID, "findings")
        findings_textarea.send_keys("右下肺野可见片状阴影，边界模糊")
        
        impression_textarea = browser.find_element(By.ID, "impression")
        impression_textarea.send_keys("考虑右下肺炎可能")
        
        recommendations_textarea = browser.find_element(By.ID, "recommendations")
        recommendations_textarea.send_keys("建议抗感染治疗，复查胸片")
        
        # 保存报告
        save_button = browser.find_element(By.ID, "save-report")
        save_button.click()
        
        # 验证报告保存成功
        success_message = self.wait_for_element(
            browser, (By.CLASS_NAME, "report-saved")
        )
        assert "报告保存成功" in success_message.text
        
        # 提交审核
        submit_review_button = browser.find_element(By.ID, "submit-review")
        submit_review_button.click()
        
        # 验证报告状态更新
        report_status = browser.find_element(By.ID, "report-status")
        assert "待审核" in report_status.text


class TestSystemAdminWorkflow(BaseE2ETest):
    """系统管理员工作流测试"""
    
    def test_user_management_workflow(self, browser):
        """测试用户管理工作流"""
        # 管理员登录
        browser.get(f"{E2ETestConfig.FRONTEND_URL}/login")
        
        username_input = self.wait_for_element(browser, (By.ID, "username"))
        password_input = browser.find_element(By.ID, "password")
        
        username_input.send_keys(E2ETestConfig.ADMIN_USER["username"])
        password_input.send_keys(E2ETestConfig.ADMIN_USER["password"])
        
        login_button = browser.find_element(By.ID, "login-button")
        login_button.click()
        
        # 导航到用户管理页面
        browser.get(f"{E2ETestConfig.FRONTEND_URL}/admin/users")
        
        # 添加新用户
        add_user_button = self.wait_for_clickable(
            browser, (By.ID, "add-user-button")
        )
        add_user_button.click()
        
        # 填写用户信息
        user_form_fields = {
            "new-username": "newdoctor@test.com",
            "new-email": "newdoctor@test.com",
            "new-full-name": "新医生",
            "new-role": "doctor",
            "new-department": "内科",
            "new-password": "newpassword123"
        }
        
        for field_id, value in user_form_fields.items():
            field = browser.find_element(By.ID, field_id)
            field.clear()
            field.send_keys(value)
        
        # 提交用户创建
        create_user_button = browser.find_element(By.ID, "create-user")
        create_user_button.click()
        
        # 验证用户创建成功
        success_message = self.wait_for_element(
            browser, (By.CLASS_NAME, "user-created")
        )
        assert "用户创建成功" in success_message.text
    
    def test_system_monitoring_workflow(self, browser):
        """测试系统监控工作流"""
        # 导航到监控页面
        browser.get(f"{E2ETestConfig.FRONTEND_URL}/admin/monitoring")
        
        # 验证系统状态显示
        system_status = self.wait_for_element(
            browser, (By.ID, "system-status")
        )
        assert system_status.is_displayed()
        
        # 验证服务状态
        service_status_items = browser.find_elements(
            By.CLASS_NAME, "service-status-item"
        )
        assert len(service_status_items) > 0
        
        # 验证性能指标
        performance_metrics = browser.find_element(By.ID, "performance-metrics")
        assert performance_metrics.is_displayed()
        
        # 验证日志查看
        log_viewer = browser.find_element(By.ID, "log-viewer")
        assert log_viewer.is_displayed()
    
    def test_ai_model_management_workflow(self, browser):
        """测试 AI 模型管理工作流"""
        # 导航到模型管理页面
        browser.get(f"{E2ETestConfig.FRONTEND_URL}/admin/models")
        
        # 验证模型列表显示
        model_list = self.wait_for_element(browser, (By.ID, "model-list"))
        assert model_list.is_displayed()
        
        # 查看模型详情
        model_item = browser.find_element(By.CLASS_NAME, "model-item")
        view_details_button = model_item.find_element(
            By.CLASS_NAME, "view-details-button"
        )
        view_details_button.click()
        
        # 验证模型详情显示
        model_details = self.wait_for_element(
            browser, (By.ID, "model-details")
        )
        assert model_details.is_displayed()
        
        # 验证模型性能指标
        performance_metrics = browser.find_element(
            By.ID, "model-performance-metrics"
        )
        assert performance_metrics.is_displayed()


class TestWebSocketCommunication(BaseE2ETest):
    """WebSocket 通信测试"""
    
    @pytest.mark.asyncio
    async def test_real_time_notifications(self, auth_token):
        """测试实时通知"""
        # 建立 WebSocket 连接
        uri = f"{E2ETestConfig.WEBSOCKET_URL}?token={auth_token}"
        
        async with websockets.connect(uri) as websocket:
            # 发送心跳消息
            await websocket.send(json.dumps({
                "type": "ping",
                "data": {"timestamp": datetime.utcnow().isoformat()}
            }))
            
            # 接收响应
            response = await websocket.recv()
            response_data = json.loads(response)
            
            assert response_data["type"] == "pong"
            
            # 模拟接收推理完成通知
            notification_message = {
                "type": "inference_completed",
                "data": {
                    "task_id": "task123",
                    "image_id": "image456",
                    "results": {
                        "predictions": [
                            {"class": "pneumonia", "confidence": 0.92}
                        ]
                    }
                }
            }
            
            # 验证能够接收通知
            await websocket.send(json.dumps(notification_message))
            
            # 等待确认消息
            confirmation = await websocket.recv()
            confirmation_data = json.loads(confirmation)
            
            assert confirmation_data["type"] == "notification_received"
    
    @pytest.mark.asyncio
    async def test_real_time_progress_updates(self, auth_token):
        """测试实时进度更新"""
        uri = f"{E2ETestConfig.WEBSOCKET_URL}?token={auth_token}"
        
        async with websockets.connect(uri) as websocket:
            # 模拟推理进度更新
            progress_updates = [
                {"type": "inference_progress", "data": {"task_id": "task123", "progress": 25}},
                {"type": "inference_progress", "data": {"task_id": "task123", "progress": 50}},
                {"type": "inference_progress", "data": {"task_id": "task123", "progress": 75}},
                {"type": "inference_progress", "data": {"task_id": "task123", "progress": 100}}
            ]
            
            for update in progress_updates:
                await websocket.send(json.dumps(update))
                
                # 接收确认
                response = await websocket.recv()
                response_data = json.loads(response)
                
                assert response_data["type"] == "progress_received"
                assert response_data["data"]["task_id"] == "task123"


class TestPerformanceAndLoad(BaseE2ETest):
    """性能和负载测试"""
    
    @pytest.mark.asyncio
    async def test_concurrent_api_requests(self, api_client, auth_token):
        """测试并发 API 请求"""
        headers = {"Authorization": f"Bearer {auth_token}"}
        
        # 并发发送多个请求
        async def make_request(client, endpoint):
            response = await client.get(endpoint, headers=headers)
            return response.status_code
        
        # 创建并发任务
        tasks = []
        endpoints = [
            "/api/v1/users/me",
            "/api/v1/patients",
            "/api/v1/images",
            "/api/v1/reports",
            "/api/v1/ai/models"
        ]
        
        for _ in range(10):  # 每个端点10个并发请求
            for endpoint in endpoints:
                task = make_request(api_client, endpoint)
                tasks.append(task)
        
        # 执行并发请求
        start_time = time.time()
        results = await asyncio.gather(*tasks, return_exceptions=True)
        end_time = time.time()
        
        # 验证性能
        total_time = end_time - start_time
        successful_requests = sum(1 for result in results if result == 200)
        
        assert total_time < 10.0  # 总时间应该在10秒内
        assert successful_requests >= len(tasks) * 0.95  # 95%的请求应该成功
    
    def test_page_load_performance(self, browser):
        """测试页面加载性能"""
        pages_to_test = [
            f"{E2ETestConfig.FRONTEND_URL}/login",
            f"{E2ETestConfig.FRONTEND_URL}/dashboard",
            f"{E2ETestConfig.FRONTEND_URL}/patients",
            f"{E2ETestConfig.FRONTEND_URL}/images",
            f"{E2ETestConfig.FRONTEND_URL}/reports"
        ]
        
        load_times = []
        
        for page_url in pages_to_test:
            start_time = time.time()
            browser.get(page_url)
            
            # 等待页面加载完成
            WebDriverWait(browser, 10).until(
                lambda driver: driver.execute_script("return document.readyState") == "complete"
            )
            
            end_time = time.time()
            load_time = end_time - start_time
            load_times.append(load_time)
        
        # 验证页面加载性能
        average_load_time = sum(load_times) / len(load_times)
        max_load_time = max(load_times)
        
        assert average_load_time < 3.0  # 平均加载时间应该在3秒内
        assert max_load_time < 5.0  # 最大加载时间应该在5秒内
    
    @pytest.mark.asyncio
    async def test_memory_usage_monitoring(self):
        """测试内存使用监控"""
        import psutil
        import gc
        
        # 获取初始内存使用
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # 模拟大量数据处理
        large_data_sets = []
        for i in range(100):
            # 创建模拟的大数据集
            data_set = {
                "id": i,
                "data": [j for j in range(1000)],
                "metadata": {f"key_{k}": f"value_{k}" for k in range(100)}
            }
            large_data_sets.append(data_set)
        
        # 获取处理后内存使用
        current_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = current_memory - initial_memory
        
        # 清理数据
        del large_data_sets
        gc.collect()
        
        # 获取清理后内存使用
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # 验证内存使用合理
        assert memory_increase < 500  # 内存增长应该在500MB以内
        assert final_memory < initial_memory + 100  # 清理后内存应该接近初始值


class TestErrorRecovery(BaseE2ETest):
    """错误恢复测试"""
    
    def test_network_error_recovery(self, browser, auth_token):
        """测试网络错误恢复"""
        # 模拟已登录状态
        browser.get(f"{E2ETestConfig.FRONTEND_URL}/dashboard")
        browser.execute_script(
            f"localStorage.setItem('auth_token', '{auth_token}')"
        )
        
        # 模拟网络中断
        browser.execute_script(
            "window.fetch = () => Promise.reject(new Error('Network error'))"
        )
        
        # 尝试执行需要网络的操作
        browser.get(f"{E2ETestConfig.FRONTEND_URL}/patients")
        
        # 验证错误提示显示
        error_message = self.wait_for_element(
            browser, (By.CLASS_NAME, "network-error")
        )
        assert "网络连接失败" in error_message.text
        
        # 验证重试按钮存在
        retry_button = browser.find_element(By.ID, "retry-button")
        assert retry_button.is_displayed()
        
        # 恢复网络连接
        browser.execute_script("delete window.fetch")
        
        # 点击重试
        retry_button.click()
        
        # 验证页面正常加载
        patient_list = self.wait_for_element(
            browser, (By.ID, "patient-list")
        )
        assert patient_list.is_displayed()
    
    def test_session_timeout_recovery(self, browser):
        """测试会话超时恢复"""
        # 模拟过期的令牌
        expired_token = "expired.jwt.token"
        browser.get(f"{E2ETestConfig.FRONTEND_URL}/dashboard")
        browser.execute_script(
            f"localStorage.setItem('auth_token', '{expired_token}')"
        )
        
        # 尝试访问需要认证的页面
        browser.get(f"{E2ETestConfig.FRONTEND_URL}/patients")
        
        # 验证自动跳转到登录页面
        self.wait_for_element(browser, (By.ID, "login-form"))
        assert "login" in browser.current_url
        
        # 验证会话过期提示
        session_expired_message = browser.find_element(
            By.CLASS_NAME, "session-expired"
        )
        assert "会话已过期" in session_expired_message.text


if __name__ == "__main__":
    # 运行端到端测试
    pytest.main([
        "-v",
        "-s",  # 显示输出
        "--tb=short",  # 简短的错误信息
        "-m", "not slow",  # 排除慢速测试
        __file__
    ])