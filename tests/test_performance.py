# -*- coding: utf-8 -*-
"""
性能测试

本文件包含了医疗AI系统的性能测试：
- API响应时间测试
- 并发处理能力测试
- 数据库性能测试
- AI推理性能测试
- 内存使用测试
- 文件上传/下载性能测试
- 缓存性能测试
- 负载测试
"""

import pytest
import asyncio
import time
import psutil
import statistics
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Any, Tuple
import json
import tempfile
import os
from datetime import datetime, timedelta
import aiohttp
import aiofiles
from httpx import AsyncClient
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from conftest import (
    TestConfig, PerformanceTimer, create_test_user_data,
    create_test_patient_data, create_test_image_data
)
from factories import (
    BatchFactory, UserFactory, PatientFactory, MedicalImageFactory,
    InferenceTaskFactory
)


class PerformanceMetrics:
    """性能指标收集器"""
    
    def __init__(self):
        self.metrics = {
            'response_times': [],
            'throughput': [],
            'error_rates': [],
            'memory_usage': [],
            'cpu_usage': [],
            'concurrent_users': [],
            'database_query_times': [],
            'ai_inference_times': [],
            'file_upload_times': [],
            'cache_hit_rates': []
        }
        self.start_time = None
        self.end_time = None
    
    def start_monitoring(self):
        """开始监控"""
        self.start_time = time.time()
        self.initial_memory = psutil.virtual_memory().used
        self.initial_cpu = psutil.cpu_percent()
    
    def stop_monitoring(self):
        """停止监控"""
        self.end_time = time.time()
        self.final_memory = psutil.virtual_memory().used
        self.final_cpu = psutil.cpu_percent()
    
    def add_response_time(self, response_time: float):
        """添加响应时间"""
        self.metrics['response_times'].append(response_time)
    
    def add_error(self):
        """添加错误"""
        self.metrics['error_rates'].append(1)
    
    def add_success(self):
        """添加成功"""
        self.metrics['error_rates'].append(0)
    
    def record_memory_usage(self):
        """记录内存使用"""
        memory_mb = psutil.virtual_memory().used / 1024 / 1024
        self.metrics['memory_usage'].append(memory_mb)
    
    def record_cpu_usage(self):
        """记录CPU使用"""
        cpu_percent = psutil.cpu_percent()
        self.metrics['cpu_usage'].append(cpu_percent)
    
    def get_summary(self) -> Dict[str, Any]:
        """获取性能摘要"""
        response_times = self.metrics['response_times']
        error_rates = self.metrics['error_rates']
        
        summary = {
            'duration': self.end_time - self.start_time if self.end_time and self.start_time else 0,
            'total_requests': len(response_times),
            'successful_requests': len([r for r in error_rates if r == 0]),
            'failed_requests': len([r for r in error_rates if r == 1]),
            'error_rate': sum(error_rates) / len(error_rates) if error_rates else 0,
        }
        
        if response_times:
            summary.update({
                'avg_response_time': statistics.mean(response_times),
                'min_response_time': min(response_times),
                'max_response_time': max(response_times),
                'p50_response_time': statistics.median(response_times),
                'p95_response_time': np.percentile(response_times, 95),
                'p99_response_time': np.percentile(response_times, 99),
                'throughput': len(response_times) / summary['duration'] if summary['duration'] > 0 else 0
            })
        
        if self.metrics['memory_usage']:
            summary.update({
                'avg_memory_mb': statistics.mean(self.metrics['memory_usage']),
                'max_memory_mb': max(self.metrics['memory_usage']),
                'memory_increase_mb': (self.final_memory - self.initial_memory) / 1024 / 1024
            })
        
        if self.metrics['cpu_usage']:
            summary.update({
                'avg_cpu_percent': statistics.mean(self.metrics['cpu_usage']),
                'max_cpu_percent': max(self.metrics['cpu_usage'])
            })
        
        return summary
    
    def save_report(self, filepath: str):
        """保存性能报告"""
        report = {
            'timestamp': datetime.now().isoformat(),
            'summary': self.get_summary(),
            'raw_metrics': self.metrics
        }
        
        with open(filepath, 'w') as f:
            json.dump(report, f, indent=2)
    
    def plot_metrics(self, output_dir: str):
        """绘制性能图表"""
        os.makedirs(output_dir, exist_ok=True)
        
        # 响应时间分布
        if self.metrics['response_times']:
            plt.figure(figsize=(10, 6))
            plt.hist(self.metrics['response_times'], bins=50, alpha=0.7)
            plt.xlabel('Response Time (seconds)')
            plt.ylabel('Frequency')
            plt.title('Response Time Distribution')
            plt.savefig(os.path.join(output_dir, 'response_time_distribution.png'))
            plt.close()
        
        # 内存使用趋势
        if self.metrics['memory_usage']:
            plt.figure(figsize=(12, 6))
            plt.plot(self.metrics['memory_usage'])
            plt.xlabel('Time')
            plt.ylabel('Memory Usage (MB)')
            plt.title('Memory Usage Over Time')
            plt.savefig(os.path.join(output_dir, 'memory_usage.png'))
            plt.close()
        
        # CPU使用趋势
        if self.metrics['cpu_usage']:
            plt.figure(figsize=(12, 6))
            plt.plot(self.metrics['cpu_usage'])
            plt.xlabel('Time')
            plt.ylabel('CPU Usage (%)')
            plt.title('CPU Usage Over Time')
            plt.savefig(os.path.join(output_dir, 'cpu_usage.png'))
            plt.close()


@pytest.mark.slow
@pytest.mark.performance
class TestAPIPerformance:
    """API性能测试"""
    
    @pytest.mark.asyncio
    async def test_api_response_times(self, async_client: AsyncClient, auth_headers_doctor):
        """测试API响应时间"""
        metrics = PerformanceMetrics()
        metrics.start_monitoring()
        
        endpoints = [
            ('/api/v1/health', 'GET'),
            ('/api/v1/users/me', 'GET'),
            ('/api/v1/patients', 'GET'),
            ('/api/v1/images', 'GET'),
            ('/api/v1/reports', 'GET')
        ]
        
        # 测试每个端点的响应时间
        for endpoint, method in endpoints:
            for _ in range(10):  # 每个端点测试10次
                start_time = time.time()
                
                try:
                    if method == 'GET':
                        response = await async_client.get(endpoint, headers=auth_headers_doctor)
                    else:
                        response = await async_client.request(method, endpoint, headers=auth_headers_doctor)
                    
                    response_time = time.time() - start_time
                    metrics.add_response_time(response_time)
                    
                    if response.status_code < 400:
                        metrics.add_success()
                    else:
                        metrics.add_error()
                        
                except Exception as e:
                    metrics.add_error()
                    print(f"Error testing {endpoint}: {e}")
                
                metrics.record_memory_usage()
                metrics.record_cpu_usage()
        
        metrics.stop_monitoring()
        summary = metrics.get_summary()
        
        # 性能断言
        assert summary['avg_response_time'] < 2.0, f"Average response time too high: {summary['avg_response_time']}s"
        assert summary['p95_response_time'] < 5.0, f"95th percentile response time too high: {summary['p95_response_time']}s"
        assert summary['error_rate'] < 0.05, f"Error rate too high: {summary['error_rate']}"
        
        # 保存报告
        metrics.save_report('tests/reports/api_performance.json')
        metrics.plot_metrics('tests/reports/charts')
    
    @pytest.mark.asyncio
    async def test_concurrent_api_requests(self, async_client: AsyncClient, auth_headers_doctor):
        """测试并发API请求"""
        metrics = PerformanceMetrics()
        metrics.start_monitoring()
        
        async def make_request(session, endpoint):
            """发送单个请求"""
            start_time = time.time()
            try:
                async with session.get(endpoint, headers=auth_headers_doctor) as response:
                    response_time = time.time() - start_time
                    return response_time, response.status
            except Exception as e:
                return time.time() - start_time, 500
        
        # 并发测试配置
        concurrent_users = [1, 5, 10, 20, 50]
        endpoint = '/api/v1/health'
        
        for users in concurrent_users:
            print(f"Testing with {users} concurrent users...")
            
            async with aiohttp.ClientSession() as session:
                tasks = []
                for _ in range(users):
                    for _ in range(10):  # 每个用户发送10个请求
                        task = make_request(session, f"http://test{endpoint}")
                        tasks.append(task)
                
                results = await asyncio.gather(*tasks, return_exceptions=True)
                
                for result in results:
                    if isinstance(result, tuple):
                        response_time, status = result
                        metrics.add_response_time(response_time)
                        if status < 400:
                            metrics.add_success()
                        else:
                            metrics.add_error()
                    else:
                        metrics.add_error()
                
                metrics.record_memory_usage()
                metrics.record_cpu_usage()
        
        metrics.stop_monitoring()
        summary = metrics.get_summary()
        
        # 并发性能断言
        assert summary['avg_response_time'] < 5.0, f"Concurrent average response time too high: {summary['avg_response_time']}s"
        assert summary['error_rate'] < 0.1, f"Concurrent error rate too high: {summary['error_rate']}"
        
        metrics.save_report('tests/reports/concurrent_performance.json')


@pytest.mark.slow
@pytest.mark.performance
class TestDatabasePerformance:
    """数据库性能测试"""
    
    @pytest.mark.asyncio
    async def test_database_query_performance(self, db_session: AsyncSession):
        """测试数据库查询性能"""
        metrics = PerformanceMetrics()
        metrics.start_monitoring()
        
        # 创建测试数据
        test_data = BatchFactory.create_medical_department(
            num_doctors=10, num_patients=100, num_images=500
        )
        
        # 测试不同类型的查询
        queries = [
            # 简单查询
            "SELECT COUNT(*) FROM users",
            "SELECT COUNT(*) FROM patients",
            "SELECT COUNT(*) FROM medical_images",
            
            # 复杂查询
            """SELECT u.full_name, COUNT(p.id) as patient_count 
               FROM users u LEFT JOIN patients p ON u.id = p.created_by 
               GROUP BY u.id, u.full_name""",
            
            """SELECT p.name, COUNT(mi.id) as image_count 
               FROM patients p LEFT JOIN medical_images mi ON p.id = mi.patient_id 
               GROUP BY p.id, p.name""",
            
            # 聚合查询
            """SELECT DATE(mi.uploaded_at) as upload_date, COUNT(*) as daily_uploads 
               FROM medical_images mi 
               GROUP BY DATE(mi.uploaded_at) 
               ORDER BY upload_date DESC"""
        ]
        
        for query in queries:
            for _ in range(5):  # 每个查询执行5次
                start_time = time.time()
                
                try:
                    result = await db_session.execute(text(query))
                    await result.fetchall()
                    
                    query_time = time.time() - start_time
                    metrics.metrics['database_query_times'].append(query_time)
                    metrics.add_success()
                    
                except Exception as e:
                    print(f"Query failed: {query[:50]}... Error: {e}")
                    metrics.add_error()
                
                metrics.record_memory_usage()
        
        metrics.stop_monitoring()
        summary = metrics.get_summary()
        
        # 数据库性能断言
        if metrics.metrics['database_query_times']:
            avg_query_time = statistics.mean(metrics.metrics['database_query_times'])
            assert avg_query_time < 1.0, f"Average database query time too high: {avg_query_time}s"
        
        assert summary['error_rate'] < 0.05, f"Database error rate too high: {summary['error_rate']}"
        
        metrics.save_report('tests/reports/database_performance.json')
    
    @pytest.mark.asyncio
    async def test_database_connection_pool(self, db_session: AsyncSession):
        """测试数据库连接池性能"""
        metrics = PerformanceMetrics()
        metrics.start_monitoring()
        
        async def execute_query(query_id):
            """执行单个查询"""
            start_time = time.time()
            try:
                result = await db_session.execute(text("SELECT 1"))
                await result.fetchone()
                return time.time() - start_time, True
            except Exception as e:
                return time.time() - start_time, False
        
        # 并发数据库连接测试
        concurrent_connections = [5, 10, 20, 50]
        
        for connections in concurrent_connections:
            print(f"Testing with {connections} concurrent database connections...")
            
            tasks = [execute_query(i) for i in range(connections)]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            for result in results:
                if isinstance(result, tuple):
                    query_time, success = result
                    metrics.metrics['database_query_times'].append(query_time)
                    if success:
                        metrics.add_success()
                    else:
                        metrics.add_error()
                else:
                    metrics.add_error()
        
        metrics.stop_monitoring()
        summary = metrics.get_summary()
        
        # 连接池性能断言
        if metrics.metrics['database_query_times']:
            avg_query_time = statistics.mean(metrics.metrics['database_query_times'])
            assert avg_query_time < 0.5, f"Connection pool query time too high: {avg_query_time}s"
        
        metrics.save_report('tests/reports/connection_pool_performance.json')


@pytest.mark.slow
@pytest.mark.performance
class TestAIInferencePerformance:
    """AI推理性能测试"""
    
    @pytest.mark.asyncio
    async def test_ai_inference_speed(self, mock_ai_service):
        """测试AI推理速度"""
        metrics = PerformanceMetrics()
        metrics.start_monitoring()
        
        # 创建测试图像
        test_images = MedicalImageFactory.create_batch(20)
        
        for image in test_images:
            start_time = time.time()
            
            try:
                # 模拟AI推理
                result = await mock_ai_service.process_image(
                    image_path=image['file_path'],
                    model_name='chest_xray_classifier'
                )
                
                inference_time = time.time() - start_time
                metrics.metrics['ai_inference_times'].append(inference_time)
                metrics.add_success()
                
            except Exception as e:
                print(f"AI inference failed: {e}")
                metrics.add_error()
            
            metrics.record_memory_usage()
            metrics.record_cpu_usage()
        
        metrics.stop_monitoring()
        summary = metrics.get_summary()
        
        # AI推理性能断言
        if metrics.metrics['ai_inference_times']:
            avg_inference_time = statistics.mean(metrics.metrics['ai_inference_times'])
            assert avg_inference_time < 10.0, f"Average AI inference time too high: {avg_inference_time}s"
            
            p95_inference_time = np.percentile(metrics.metrics['ai_inference_times'], 95)
            assert p95_inference_time < 30.0, f"95th percentile inference time too high: {p95_inference_time}s"
        
        assert summary['error_rate'] < 0.05, f"AI inference error rate too high: {summary['error_rate']}"
        
        metrics.save_report('tests/reports/ai_inference_performance.json')
    
    @pytest.mark.asyncio
    async def test_concurrent_ai_inference(self, mock_ai_service):
        """测试并发AI推理"""
        metrics = PerformanceMetrics()
        metrics.start_monitoring()
        
        async def process_single_image(image_data):
            """处理单个图像"""
            start_time = time.time()
            try:
                result = await mock_ai_service.process_image(
                    image_path=image_data['file_path'],
                    model_name='chest_xray_classifier'
                )
                return time.time() - start_time, True
            except Exception as e:
                return time.time() - start_time, False
        
        # 创建测试图像
        test_images = MedicalImageFactory.create_batch(50)
        
        # 并发推理测试
        concurrent_tasks = [1, 2, 4, 8]
        
        for task_count in concurrent_tasks:
            print(f"Testing with {task_count} concurrent AI inference tasks...")
            
            # 选择图像子集
            images_subset = test_images[:task_count * 5]
            
            tasks = [process_single_image(image) for image in images_subset]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            for result in results:
                if isinstance(result, tuple):
                    inference_time, success = result
                    metrics.metrics['ai_inference_times'].append(inference_time)
                    if success:
                        metrics.add_success()
                    else:
                        metrics.add_error()
                else:
                    metrics.add_error()
            
            metrics.record_memory_usage()
            metrics.record_cpu_usage()
        
        metrics.stop_monitoring()
        summary = metrics.get_summary()
        
        # 并发推理性能断言
        if metrics.metrics['ai_inference_times']:
            avg_inference_time = statistics.mean(metrics.metrics['ai_inference_times'])
            assert avg_inference_time < 15.0, f"Concurrent AI inference time too high: {avg_inference_time}s"
        
        metrics.save_report('tests/reports/concurrent_ai_performance.json')


@pytest.mark.slow
@pytest.mark.performance
class TestFileOperationPerformance:
    """文件操作性能测试"""
    
    @pytest.mark.asyncio
    async def test_file_upload_performance(self, async_client: AsyncClient, auth_headers_doctor, temp_storage_dir):
        """测试文件上传性能"""
        metrics = PerformanceMetrics()
        metrics.start_monitoring()
        
        # 创建不同大小的测试文件
        file_sizes = [1, 5, 10, 50]  # MB
        
        for size_mb in file_sizes:
            # 创建测试文件
            test_file = os.path.join(temp_storage_dir, f"test_{size_mb}mb.dcm")
            with open(test_file, 'wb') as f:
                f.write(b'\x00' * (size_mb * 1024 * 1024))
            
            # 测试上传
            for _ in range(3):  # 每个大小测试3次
                start_time = time.time()
                
                try:
                    with open(test_file, 'rb') as f:
                        files = {'file': (f'test_{size_mb}mb.dcm', f, 'application/dicom')}
                        data = {
                            'patient_id': 'test_patient_id',
                            'study_type': 'chest_xray',
                            'description': f'Test upload {size_mb}MB'
                        }
                        
                        response = await async_client.post(
                            '/api/v1/images/upload',
                            files=files,
                            data=data,
                            headers=auth_headers_doctor
                        )
                    
                    upload_time = time.time() - start_time
                    metrics.metrics['file_upload_times'].append(upload_time)
                    
                    if response.status_code < 400:
                        metrics.add_success()
                    else:
                        metrics.add_error()
                        
                except Exception as e:
                    print(f"File upload failed: {e}")
                    metrics.add_error()
                
                metrics.record_memory_usage()
            
            # 清理测试文件
            os.unlink(test_file)
        
        metrics.stop_monitoring()
        summary = metrics.get_summary()
        
        # 文件上传性能断言
        if metrics.metrics['file_upload_times']:
            avg_upload_time = statistics.mean(metrics.metrics['file_upload_times'])
            assert avg_upload_time < 30.0, f"Average file upload time too high: {avg_upload_time}s"
        
        assert summary['error_rate'] < 0.1, f"File upload error rate too high: {summary['error_rate']}"
        
        metrics.save_report('tests/reports/file_upload_performance.json')
    
    @pytest.mark.asyncio
    async def test_concurrent_file_operations(self, mock_storage_service):
        """测试并发文件操作"""
        metrics = PerformanceMetrics()
        metrics.start_monitoring()
        
        async def perform_file_operation(operation_type, file_key):
            """执行文件操作"""
            start_time = time.time()
            
            try:
                if operation_type == 'upload':
                    result = await mock_storage_service.upload_file(
                        file_path='/tmp/test_file',
                        key=file_key
                    )
                elif operation_type == 'download':
                    result = await mock_storage_service.download_file(file_key)
                elif operation_type == 'delete':
                    result = await mock_storage_service.delete_file(file_key)
                else:
                    result = await mock_storage_service.file_exists(file_key)
                
                operation_time = time.time() - start_time
                return operation_time, True
                
            except Exception as e:
                return time.time() - start_time, False
        
        # 并发文件操作测试
        operations = ['upload', 'download', 'exists', 'delete']
        concurrent_ops = [5, 10, 20]
        
        for op_count in concurrent_ops:
            print(f"Testing with {op_count} concurrent file operations...")
            
            tasks = []
            for i in range(op_count):
                operation = operations[i % len(operations)]
                file_key = f"test_file_{i}.dcm"
                task = perform_file_operation(operation, file_key)
                tasks.append(task)
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            for result in results:
                if isinstance(result, tuple):
                    op_time, success = result
                    metrics.add_response_time(op_time)
                    if success:
                        metrics.add_success()
                    else:
                        metrics.add_error()
                else:
                    metrics.add_error()
            
            metrics.record_memory_usage()
        
        metrics.stop_monitoring()
        summary = metrics.get_summary()
        
        # 并发文件操作性能断言
        assert summary['avg_response_time'] < 5.0, f"Concurrent file operation time too high: {summary['avg_response_time']}s"
        assert summary['error_rate'] < 0.1, f"File operation error rate too high: {summary['error_rate']}"
        
        metrics.save_report('tests/reports/concurrent_file_performance.json')


@pytest.mark.slow
@pytest.mark.performance
class TestMemoryPerformance:
    """内存性能测试"""
    
    def test_memory_usage_under_load(self):
        """测试负载下的内存使用"""
        metrics = PerformanceMetrics()
        metrics.start_monitoring()
        
        # 创建大量测试数据
        initial_memory = psutil.virtual_memory().used
        
        # 批量创建数据
        large_dataset = BatchFactory.create_medical_department(
            num_doctors=50, num_patients=1000, num_images=5000
        )
        
        metrics.record_memory_usage()
        
        # 模拟数据处理
        for i in range(100):
            # 创建临时数据
            temp_data = {
                'patients': PatientFactory.create_batch(10),
                'images': MedicalImageFactory.create_batch(50),
                'tasks': InferenceTaskFactory.create_batch(20)
            }
            
            # 处理数据
            processed_data = []
            for patient in temp_data['patients']:
                processed_patient = patient.copy()
                processed_patient['processed'] = True
                processed_data.append(processed_patient)
            
            # 记录内存使用
            if i % 10 == 0:
                metrics.record_memory_usage()
                metrics.record_cpu_usage()
            
            # 清理临时数据
            del temp_data
            del processed_data
        
        final_memory = psutil.virtual_memory().used
        memory_increase = (final_memory - initial_memory) / 1024 / 1024  # MB
        
        metrics.stop_monitoring()
        summary = metrics.get_summary()
        
        # 内存性能断言
        assert memory_increase < 500, f"Memory increase too high: {memory_increase}MB"
        
        if metrics.metrics['memory_usage']:
            max_memory = max(metrics.metrics['memory_usage'])
            avg_memory = statistics.mean(metrics.metrics['memory_usage'])
            
            # 确保内存使用在合理范围内
            assert max_memory < 2000, f"Maximum memory usage too high: {max_memory}MB"
        
        metrics.save_report('tests/reports/memory_performance.json')
    
    def test_memory_leak_detection(self):
        """测试内存泄漏检测"""
        import gc
        
        initial_memory = psutil.virtual_memory().used
        memory_samples = []
        
        # 多次创建和销毁对象
        for iteration in range(10):
            # 创建大量对象
            objects = []
            for _ in range(1000):
                obj = {
                    'patient': PatientFactory(),
                    'image': MedicalImageFactory(),
                    'task': InferenceTaskFactory()
                }
                objects.append(obj)
            
            # 记录内存使用
            current_memory = psutil.virtual_memory().used
            memory_samples.append(current_memory)
            
            # 清理对象
            del objects
            gc.collect()
            
            # 等待垃圾回收
            time.sleep(0.1)
        
        final_memory = psutil.virtual_memory().used
        
        # 检查内存是否持续增长（可能的内存泄漏）
        memory_trend = np.polyfit(range(len(memory_samples)), memory_samples, 1)[0]
        memory_increase = (final_memory - initial_memory) / 1024 / 1024  # MB
        
        # 内存泄漏断言
        assert memory_increase < 100, f"Potential memory leak detected: {memory_increase}MB increase"
        assert memory_trend < 1024 * 1024 * 10, f"Memory trend indicates leak: {memory_trend} bytes/iteration"


@pytest.mark.slow
@pytest.mark.performance
class TestCachePerformance:
    """缓存性能测试"""
    
    @pytest.mark.asyncio
    async def test_redis_cache_performance(self, redis_client):
        """测试Redis缓存性能"""
        metrics = PerformanceMetrics()
        metrics.start_monitoring()
        
        # 测试数据
        test_keys = [f"test_key_{i}" for i in range(1000)]
        test_values = [f"test_value_{i}" * 100 for i in range(1000)]  # 较大的值
        
        # 写入性能测试
        write_times = []
        for key, value in zip(test_keys, test_values):
            start_time = time.time()
            redis_client.set(key, value, ex=3600)  # 1小时过期
            write_time = time.time() - start_time
            write_times.append(write_time)
        
        # 读取性能测试
        read_times = []
        cache_hits = 0
        for key in test_keys:
            start_time = time.time()
            value = redis_client.get(key)
            read_time = time.time() - start_time
            read_times.append(read_time)
            
            if value is not None:
                cache_hits += 1
        
        # 批量操作性能测试
        start_time = time.time()
        pipe = redis_client.pipeline()
        for key, value in zip(test_keys[:100], test_values[:100]):
            pipe.set(f"batch_{key}", value)
        pipe.execute()
        batch_write_time = time.time() - start_time
        
        start_time = time.time()
        pipe = redis_client.pipeline()
        for key in test_keys[:100]:
            pipe.get(f"batch_{key}")
        batch_results = pipe.execute()
        batch_read_time = time.time() - start_time
        
        metrics.stop_monitoring()
        
        # 缓存性能断言
        avg_write_time = statistics.mean(write_times)
        avg_read_time = statistics.mean(read_times)
        cache_hit_rate = cache_hits / len(test_keys)
        
        assert avg_write_time < 0.01, f"Redis write time too high: {avg_write_time}s"
        assert avg_read_time < 0.005, f"Redis read time too high: {avg_read_time}s"
        assert cache_hit_rate > 0.95, f"Cache hit rate too low: {cache_hit_rate}"
        assert batch_write_time < 0.1, f"Batch write time too high: {batch_write_time}s"
        assert batch_read_time < 0.05, f"Batch read time too high: {batch_read_time}s"
        
        # 保存缓存性能报告
        cache_report = {
            'avg_write_time': avg_write_time,
            'avg_read_time': avg_read_time,
            'cache_hit_rate': cache_hit_rate,
            'batch_write_time': batch_write_time,
            'batch_read_time': batch_read_time,
            'total_operations': len(test_keys) * 2
        }
        
        with open('tests/reports/cache_performance.json', 'w') as f:
            json.dump(cache_report, f, indent=2)


@pytest.mark.slow
@pytest.mark.performance
class TestLoadTesting:
    """负载测试"""
    
    @pytest.mark.asyncio
    async def test_system_under_load(self, async_client: AsyncClient, auth_headers_doctor):
        """测试系统负载能力"""
        metrics = PerformanceMetrics()
        metrics.start_monitoring()
        
        # 负载测试配置
        load_scenarios = [
            {'users': 10, 'duration': 30, 'ramp_up': 5},
            {'users': 25, 'duration': 60, 'ramp_up': 10},
            {'users': 50, 'duration': 120, 'ramp_up': 20}
        ]
        
        for scenario in load_scenarios:
            print(f"Load test: {scenario['users']} users, {scenario['duration']}s duration")
            
            async def user_simulation(user_id):
                """模拟用户行为"""
                user_metrics = []
                
                # 用户行为序列
                actions = [
                    ('GET', '/api/v1/health'),
                    ('GET', '/api/v1/users/me'),
                    ('GET', '/api/v1/patients'),
                    ('GET', '/api/v1/images'),
                    ('GET', '/api/v1/reports')
                ]
                
                end_time = time.time() + scenario['duration']
                
                while time.time() < end_time:
                    for method, endpoint in actions:
                        start_time = time.time()
                        
                        try:
                            response = await async_client.request(
                                method, endpoint, headers=auth_headers_doctor
                            )
                            
                            response_time = time.time() - start_time
                            user_metrics.append({
                                'user_id': user_id,
                                'endpoint': endpoint,
                                'response_time': response_time,
                                'status_code': response.status_code,
                                'success': response.status_code < 400
                            })
                            
                        except Exception as e:
                            user_metrics.append({
                                'user_id': user_id,
                                'endpoint': endpoint,
                                'response_time': time.time() - start_time,
                                'status_code': 500,
                                'success': False,
                                'error': str(e)
                            })
                        
                        # 用户思考时间
                        await asyncio.sleep(random.uniform(0.5, 2.0))
                
                return user_metrics
            
            # 逐步增加用户负载
            tasks = []
            for i in range(scenario['users']):
                # 分阶段启动用户
                if i > 0:
                    await asyncio.sleep(scenario['ramp_up'] / scenario['users'])
                
                task = asyncio.create_task(user_simulation(i))
                tasks.append(task)
            
            # 等待所有用户完成
            all_results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # 汇总结果
            for user_results in all_results:
                if isinstance(user_results, list):
                    for result in user_results:
                        metrics.add_response_time(result['response_time'])
                        if result['success']:
                            metrics.add_success()
                        else:
                            metrics.add_error()
            
            metrics.record_memory_usage()
            metrics.record_cpu_usage()
        
        metrics.stop_monitoring()
        summary = metrics.get_summary()
        
        # 负载测试断言
        assert summary['avg_response_time'] < 10.0, f"Load test average response time too high: {summary['avg_response_time']}s"
        assert summary['p95_response_time'] < 30.0, f"Load test 95th percentile too high: {summary['p95_response_time']}s"
        assert summary['error_rate'] < 0.15, f"Load test error rate too high: {summary['error_rate']}"
        assert summary['throughput'] > 1.0, f"Load test throughput too low: {summary['throughput']} req/s"
        
        metrics.save_report('tests/reports/load_test_performance.json')
        metrics.plot_metrics('tests/reports/charts/load_test')


if __name__ == "__main__":
    # 运行性能测试
    pytest.main([
        __file__,
        "-v",
        "-s",
        "--tb=short",
        "-m", "performance"
    ])