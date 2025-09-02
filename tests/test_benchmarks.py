# -*- coding: utf-8 -*-
"""
基准测试

本文件包含了医疗AI系统的基准测试：
- 性能基准建立
- 回归测试
- 性能对比
- 基准报告生成
- 性能趋势分析
"""

import pytest
import json
import os
import time
import statistics
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from dataclasses import dataclass, asdict
from pathlib import Path

from conftest import PerformanceTimer
from test_performance import PerformanceMetrics


@dataclass
class BenchmarkResult:
    """基准测试结果"""
    test_name: str
    timestamp: str
    version: str
    environment: str
    metrics: Dict[str, float]
    metadata: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'BenchmarkResult':
        """从字典创建"""
        return cls(**data)


class BenchmarkManager:
    """基准测试管理器"""
    
    def __init__(self, benchmark_dir: str = "tests/benchmarks"):
        self.benchmark_dir = Path(benchmark_dir)
        self.benchmark_dir.mkdir(parents=True, exist_ok=True)
        
        # 基准配置
        self.config = {
            'version': '1.0.0',
            'environment': 'test',
            'retention_days': 90,
            'baseline_percentile': 95,
            'regression_threshold': 0.2  # 20% 性能下降阈值
        }
        
        # 性能基准
        self.baselines = {
            'api_response_time': {'p50': 0.5, 'p95': 2.0, 'p99': 5.0},
            'database_query_time': {'avg': 0.1, 'p95': 0.5},
            'ai_inference_time': {'avg': 5.0, 'p95': 15.0},
            'file_upload_time': {'avg': 10.0, 'p95': 30.0},
            'memory_usage_mb': {'avg': 500, 'max': 1000},
            'cpu_usage_percent': {'avg': 30, 'max': 80},
            'throughput_rps': {'min': 10, 'target': 50},
            'error_rate': {'max': 0.05},
            'cache_hit_rate': {'min': 0.9}
        }
    
    def save_benchmark(self, result: BenchmarkResult):
        """保存基准测试结果"""
        filename = f"{result.test_name}_{result.timestamp}.json"
        filepath = self.benchmark_dir / filename
        
        with open(filepath, 'w') as f:
            json.dump(result.to_dict(), f, indent=2)
    
    def load_benchmarks(self, test_name: str, days: int = 30) -> List[BenchmarkResult]:
        """加载历史基准测试结果"""
        results = []
        cutoff_date = datetime.now() - timedelta(days=days)
        
        for filepath in self.benchmark_dir.glob(f"{test_name}_*.json"):
            try:
                with open(filepath, 'r') as f:
                    data = json.load(f)
                
                result = BenchmarkResult.from_dict(data)
                result_date = datetime.fromisoformat(result.timestamp)
                
                if result_date >= cutoff_date:
                    results.append(result)
                    
            except Exception as e:
                print(f"Error loading benchmark {filepath}: {e}")
        
        return sorted(results, key=lambda x: x.timestamp)
    
    def get_baseline(self, test_name: str, metric_name: str) -> Optional[float]:
        """获取基准值"""
        if test_name in self.baselines and metric_name in self.baselines[test_name]:
            return self.baselines[test_name][metric_name]
        return None
    
    def compare_with_baseline(self, result: BenchmarkResult) -> Dict[str, Any]:
        """与基准比较"""
        comparison = {
            'test_name': result.test_name,
            'timestamp': result.timestamp,
            'passed': True,
            'violations': [],
            'improvements': [],
            'metrics_comparison': {}
        }
        
        for metric_name, value in result.metrics.items():
            baseline = self.get_baseline(result.test_name, metric_name)
            if baseline is None:
                continue
            
            # 计算性能变化
            if 'time' in metric_name or 'usage' in metric_name:
                # 时间和使用率指标：越小越好
                change_ratio = (value - baseline) / baseline
                is_regression = change_ratio > self.config['regression_threshold']
                is_improvement = change_ratio < -0.1  # 10% 改善
            elif 'rate' in metric_name and 'error' in metric_name:
                # 错误率：越小越好
                change_ratio = (value - baseline) / baseline if baseline > 0 else 0
                is_regression = value > baseline
                is_improvement = value < baseline * 0.5  # 50% 改善
            else:
                # 吞吐量等：越大越好
                change_ratio = (baseline - value) / baseline if baseline > 0 else 0
                is_regression = change_ratio > self.config['regression_threshold']
                is_improvement = change_ratio < -0.1
            
            comparison['metrics_comparison'][metric_name] = {
                'current': value,
                'baseline': baseline,
                'change_ratio': change_ratio,
                'is_regression': is_regression,
                'is_improvement': is_improvement
            }
            
            if is_regression:
                comparison['passed'] = False
                comparison['violations'].append({
                    'metric': metric_name,
                    'current': value,
                    'baseline': baseline,
                    'change_ratio': change_ratio
                })
            
            if is_improvement:
                comparison['improvements'].append({
                    'metric': metric_name,
                    'current': value,
                    'baseline': baseline,
                    'change_ratio': change_ratio
                })
        
        return comparison
    
    def generate_trend_report(self, test_name: str, days: int = 30) -> Dict[str, Any]:
        """生成趋势报告"""
        results = self.load_benchmarks(test_name, days)
        
        if not results:
            return {'error': 'No benchmark data found'}
        
        # 提取时间序列数据
        timestamps = [datetime.fromisoformat(r.timestamp) for r in results]
        metrics_data = {}
        
        # 收集所有指标
        all_metrics = set()
        for result in results:
            all_metrics.update(result.metrics.keys())
        
        for metric in all_metrics:
            values = []
            for result in results:
                if metric in result.metrics:
                    values.append(result.metrics[metric])
                else:
                    values.append(None)
            metrics_data[metric] = values
        
        # 计算趋势
        trends = {}
        for metric, values in metrics_data.items():
            # 过滤None值
            valid_data = [(i, v) for i, v in enumerate(values) if v is not None]
            if len(valid_data) < 2:
                continue
            
            indices, vals = zip(*valid_data)
            
            # 线性回归计算趋势
            trend_slope = np.polyfit(indices, vals, 1)[0]
            
            # 计算统计信息
            trends[metric] = {
                'slope': trend_slope,
                'mean': statistics.mean(vals),
                'std': statistics.stdev(vals) if len(vals) > 1 else 0,
                'min': min(vals),
                'max': max(vals),
                'latest': vals[-1],
                'trend_direction': 'improving' if trend_slope < 0 else 'degrading' if trend_slope > 0 else 'stable'
            }
        
        return {
            'test_name': test_name,
            'period': f'{days} days',
            'data_points': len(results),
            'start_date': timestamps[0].isoformat(),
            'end_date': timestamps[-1].isoformat(),
            'trends': trends,
            'raw_data': {
                'timestamps': [t.isoformat() for t in timestamps],
                'metrics': metrics_data
            }
        }
    
    def plot_trends(self, test_name: str, output_dir: str, days: int = 30):
        """绘制趋势图"""
        trend_report = self.generate_trend_report(test_name, days)
        
        if 'error' in trend_report:
            print(f"Cannot plot trends: {trend_report['error']}")
            return
        
        os.makedirs(output_dir, exist_ok=True)
        
        timestamps = [datetime.fromisoformat(ts) for ts in trend_report['raw_data']['timestamps']]
        metrics_data = trend_report['raw_data']['metrics']
        
        # 为每个指标创建图表
        for metric, values in metrics_data.items():
            # 过滤None值
            valid_data = [(t, v) for t, v in zip(timestamps, values) if v is not None]
            if not valid_data:
                continue
            
            valid_timestamps, valid_values = zip(*valid_data)
            
            plt.figure(figsize=(12, 6))
            plt.plot(valid_timestamps, valid_values, marker='o', linewidth=2, markersize=4)
            
            # 添加趋势线
            if len(valid_values) > 1:
                z = np.polyfit(range(len(valid_values)), valid_values, 1)
                p = np.poly1d(z)
                trend_line = p(range(len(valid_values)))
                plt.plot(valid_timestamps, trend_line, "--", alpha=0.7, color='red')
            
            # 添加基准线
            baseline = self.get_baseline(test_name, metric)
            if baseline:
                plt.axhline(y=baseline, color='green', linestyle=':', alpha=0.7, label=f'Baseline: {baseline}')
            
            plt.title(f'{test_name} - {metric} Trend')
            plt.xlabel('Date')
            plt.ylabel(metric)
            plt.xticks(rotation=45)
            plt.grid(True, alpha=0.3)
            plt.legend()
            plt.tight_layout()
            
            # 保存图表
            filename = f"{test_name}_{metric}_trend.png"
            plt.savefig(os.path.join(output_dir, filename), dpi=300, bbox_inches='tight')
            plt.close()
    
    def cleanup_old_benchmarks(self):
        """清理旧的基准测试数据"""
        cutoff_date = datetime.now() - timedelta(days=self.config['retention_days'])
        
        for filepath in self.benchmark_dir.glob("*.json"):
            try:
                # 从文件名提取时间戳
                parts = filepath.stem.split('_')
                if len(parts) >= 2:
                    timestamp_str = '_'.join(parts[-2:])
                    file_date = datetime.fromisoformat(timestamp_str)
                    
                    if file_date < cutoff_date:
                        filepath.unlink()
                        print(f"Deleted old benchmark: {filepath}")
                        
            except Exception as e:
                print(f"Error processing {filepath}: {e}")


@pytest.mark.slow
@pytest.mark.benchmark
class TestBenchmarks:
    """基准测试类"""
    
    def setup_method(self):
        """测试设置"""
        self.benchmark_manager = BenchmarkManager()
    
    @pytest.mark.asyncio
    async def test_api_performance_benchmark(self, async_client, auth_headers_doctor):
        """API性能基准测试"""
        metrics = PerformanceMetrics()
        metrics.start_monitoring()
        
        # 执行标准化的API测试
        endpoints = [
            '/api/v1/health',
            '/api/v1/users/me',
            '/api/v1/patients',
            '/api/v1/images',
            '/api/v1/reports'
        ]
        
        for endpoint in endpoints:
            for _ in range(20):  # 每个端点20次请求
                start_time = time.time()
                
                try:
                    response = await async_client.get(endpoint, headers=auth_headers_doctor)
                    response_time = time.time() - start_time
                    metrics.add_response_time(response_time)
                    
                    if response.status_code < 400:
                        metrics.add_success()
                    else:
                        metrics.add_error()
                        
                except Exception:
                    metrics.add_error()
                
                metrics.record_memory_usage()
                metrics.record_cpu_usage()
        
        metrics.stop_monitoring()
        summary = metrics.get_summary()
        
        # 创建基准结果
        benchmark_result = BenchmarkResult(
            test_name='api_performance',
            timestamp=datetime.now().isoformat(),
            version=self.benchmark_manager.config['version'],
            environment=self.benchmark_manager.config['environment'],
            metrics={
                'avg_response_time': summary['avg_response_time'],
                'p50_response_time': summary['p50_response_time'],
                'p95_response_time': summary['p95_response_time'],
                'p99_response_time': summary['p99_response_time'],
                'throughput_rps': summary['throughput'],
                'error_rate': summary['error_rate'],
                'avg_memory_mb': summary.get('avg_memory_mb', 0),
                'max_memory_mb': summary.get('max_memory_mb', 0),
                'avg_cpu_percent': summary.get('avg_cpu_percent', 0)
            },
            metadata={
                'total_requests': summary['total_requests'],
                'duration': summary['duration'],
                'endpoints_tested': len(endpoints)
            }
        )
        
        # 保存基准结果
        self.benchmark_manager.save_benchmark(benchmark_result)
        
        # 与基准比较
        comparison = self.benchmark_manager.compare_with_baseline(benchmark_result)
        
        # 断言基准要求
        assert comparison['passed'], f"Performance regression detected: {comparison['violations']}"
        
        # 输出改进信息
        if comparison['improvements']:
            print(f"Performance improvements detected: {comparison['improvements']}")
    
    @pytest.mark.asyncio
    async def test_database_performance_benchmark(self, db_session):
        """数据库性能基准测试"""
        from sqlalchemy import text
        
        metrics = PerformanceMetrics()
        metrics.start_monitoring()
        
        # 标准化数据库查询测试
        queries = [
            "SELECT COUNT(*) FROM users",
            "SELECT COUNT(*) FROM patients",
            "SELECT COUNT(*) FROM medical_images",
            """SELECT u.full_name, COUNT(p.id) as patient_count 
               FROM users u LEFT JOIN patients p ON u.id = p.created_by 
               GROUP BY u.id, u.full_name LIMIT 10""",
            """SELECT p.name, COUNT(mi.id) as image_count 
               FROM patients p LEFT JOIN medical_images mi ON p.id = mi.patient_id 
               GROUP BY p.id, p.name LIMIT 10"""
        ]
        
        query_times = []
        for query in queries:
            for _ in range(10):  # 每个查询10次
                start_time = time.time()
                
                try:
                    result = await db_session.execute(text(query))
                    await result.fetchall()
                    
                    query_time = time.time() - start_time
                    query_times.append(query_time)
                    metrics.add_success()
                    
                except Exception:
                    metrics.add_error()
                
                metrics.record_memory_usage()
        
        metrics.stop_monitoring()
        summary = metrics.get_summary()
        
        # 创建基准结果
        benchmark_result = BenchmarkResult(
            test_name='database_performance',
            timestamp=datetime.now().isoformat(),
            version=self.benchmark_manager.config['version'],
            environment=self.benchmark_manager.config['environment'],
            metrics={
                'avg_query_time': statistics.mean(query_times) if query_times else 0,
                'p95_query_time': np.percentile(query_times, 95) if query_times else 0,
                'max_query_time': max(query_times) if query_times else 0,
                'error_rate': summary['error_rate'],
                'avg_memory_mb': summary.get('avg_memory_mb', 0)
            },
            metadata={
                'total_queries': len(query_times),
                'query_types': len(queries),
                'duration': summary['duration']
            }
        )
        
        # 保存和比较基准
        self.benchmark_manager.save_benchmark(benchmark_result)
        comparison = self.benchmark_manager.compare_with_baseline(benchmark_result)
        
        assert comparison['passed'], f"Database performance regression: {comparison['violations']}"
    
    def test_memory_usage_benchmark(self):
        """内存使用基准测试"""
        import psutil
        import gc
        from factories import BatchFactory
        
        initial_memory = psutil.virtual_memory().used
        memory_samples = []
        
        # 标准化内存测试
        for i in range(10):
            # 创建标准数据集
            dataset = BatchFactory.create_medical_department(
                num_doctors=10, num_patients=100, num_images=200
            )
            
            # 记录内存使用
            current_memory = psutil.virtual_memory().used
            memory_mb = current_memory / 1024 / 1024
            memory_samples.append(memory_mb)
            
            # 清理
            del dataset
            gc.collect()
        
        final_memory = psutil.virtual_memory().used
        memory_increase = (final_memory - initial_memory) / 1024 / 1024
        
        # 创建基准结果
        benchmark_result = BenchmarkResult(
            test_name='memory_usage',
            timestamp=datetime.now().isoformat(),
            version=self.benchmark_manager.config['version'],
            environment=self.benchmark_manager.config['environment'],
            metrics={
                'avg_memory_mb': statistics.mean(memory_samples),
                'max_memory_mb': max(memory_samples),
                'memory_increase_mb': memory_increase,
                'memory_std_mb': statistics.stdev(memory_samples) if len(memory_samples) > 1 else 0
            },
            metadata={
                'iterations': len(memory_samples),
                'initial_memory_mb': initial_memory / 1024 / 1024,
                'final_memory_mb': final_memory / 1024 / 1024
            }
        )
        
        # 保存和比较基准
        self.benchmark_manager.save_benchmark(benchmark_result)
        comparison = self.benchmark_manager.compare_with_baseline(benchmark_result)
        
        assert comparison['passed'], f"Memory usage regression: {comparison['violations']}"
    
    def test_generate_benchmark_report(self):
        """生成基准测试报告"""
        report_dir = "tests/reports/benchmarks"
        os.makedirs(report_dir, exist_ok=True)
        
        # 为每个测试生成趋势报告
        test_names = ['api_performance', 'database_performance', 'memory_usage']
        
        full_report = {
            'generated_at': datetime.now().isoformat(),
            'version': self.benchmark_manager.config['version'],
            'environment': self.benchmark_manager.config['environment'],
            'tests': {}
        }
        
        for test_name in test_names:
            try:
                trend_report = self.benchmark_manager.generate_trend_report(test_name, days=30)
                full_report['tests'][test_name] = trend_report
                
                # 生成趋势图
                self.benchmark_manager.plot_trends(
                    test_name, 
                    os.path.join(report_dir, 'charts'),
                    days=30
                )
                
            except Exception as e:
                print(f"Error generating report for {test_name}: {e}")
                full_report['tests'][test_name] = {'error': str(e)}
        
        # 保存完整报告
        report_file = os.path.join(report_dir, 'benchmark_report.json')
        with open(report_file, 'w') as f:
            json.dump(full_report, f, indent=2)
        
        print(f"Benchmark report generated: {report_file}")
    
    def test_cleanup_old_benchmarks(self):
        """清理旧基准数据"""
        self.benchmark_manager.cleanup_old_benchmarks()


@pytest.mark.benchmark
class TestRegressionDetection:
    """回归检测测试"""
    
    def setup_method(self):
        self.benchmark_manager = BenchmarkManager()
    
    def test_performance_regression_detection(self):
        """测试性能回归检测"""
        # 创建模拟的基准结果
        baseline_result = BenchmarkResult(
            test_name='regression_test',
            timestamp=(datetime.now() - timedelta(days=1)).isoformat(),
            version='1.0.0',
            environment='test',
            metrics={
                'avg_response_time': 1.0,
                'error_rate': 0.01,
                'throughput_rps': 50
            },
            metadata={}
        )
        
        # 创建回归结果（性能下降）
        regression_result = BenchmarkResult(
            test_name='regression_test',
            timestamp=datetime.now().isoformat(),
            version='1.0.1',
            environment='test',
            metrics={
                'avg_response_time': 1.5,  # 50% 增加（回归）
                'error_rate': 0.08,        # 错误率增加（回归）
                'throughput_rps': 30       # 吞吐量下降（回归）
            },
            metadata={}
        )
        
        # 保存基准结果
        self.benchmark_manager.save_benchmark(baseline_result)
        self.benchmark_manager.save_benchmark(regression_result)
        
        # 检测回归
        comparison = self.benchmark_manager.compare_with_baseline(regression_result)
        
        # 验证回归检测
        assert not comparison['passed'], "Should detect performance regression"
        assert len(comparison['violations']) > 0, "Should have violation details"
        
        # 检查具体的回归指标
        violations = {v['metric']: v for v in comparison['violations']}
        assert 'avg_response_time' in violations, "Should detect response time regression"
    
    def test_performance_improvement_detection(self):
        """测试性能改善检测"""
        # 创建改善结果（性能提升）
        improvement_result = BenchmarkResult(
            test_name='improvement_test',
            timestamp=datetime.now().isoformat(),
            version='1.0.2',
            environment='test',
            metrics={
                'avg_response_time': 0.3,  # 70% 减少（改善）
                'error_rate': 0.001,       # 错误率大幅下降（改善）
                'throughput_rps': 80       # 吞吐量提升（改善）
            },
            metadata={}
        )
        
        # 检测改善
        comparison = self.benchmark_manager.compare_with_baseline(improvement_result)
        
        # 验证改善检测
        assert comparison['passed'], "Should pass with performance improvements"
        assert len(comparison['improvements']) > 0, "Should detect improvements"
        
        # 检查具体的改善指标
        improvements = {i['metric']: i for i in comparison['improvements']}
        assert 'avg_response_time' in improvements, "Should detect response time improvement"


if __name__ == "__main__":
    # 运行基准测试
    pytest.main([
        __file__,
        "-v",
        "-s",
        "--tb=short",
        "-m", "benchmark"
    ])