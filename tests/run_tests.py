#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试运行脚本

本脚本提供了运行不同类型测试的便捷方法：
- 单元测试
- 集成测试
- 端到端测试
- 性能测试
- 覆盖率测试
- 并行测试

使用方法:
    python run_tests.py --type unit
    python run_tests.py --type integration
    python run_tests.py --type e2e
    python run_tests.py --type all
    python run_tests.py --coverage
    python run_tests.py --parallel
"""

import os
import sys
import argparse
import subprocess
import time
from pathlib import Path
from typing import List, Dict, Any
import json


class TestRunner:
    """测试运行器"""
    
    def __init__(self):
        self.project_root = Path(__file__).parent.parent
        self.tests_dir = Path(__file__).parent
        self.reports_dir = self.tests_dir / "reports"
        self.coverage_dir = self.reports_dir / "coverage"
        
        # 确保报告目录存在
        self.reports_dir.mkdir(exist_ok=True)
        self.coverage_dir.mkdir(exist_ok=True)
    
    def run_command(self, cmd: List[str], cwd: Path = None) -> subprocess.CompletedProcess:
        """运行命令"""
        if cwd is None:
            cwd = self.project_root
        
        print(f"Running: {' '.join(cmd)}")
        print(f"Working directory: {cwd}")
        
        try:
            result = subprocess.run(
                cmd,
                cwd=cwd,
                capture_output=True,
                text=True,
                check=False
            )
            
            if result.stdout:
                print("STDOUT:")
                print(result.stdout)
            
            if result.stderr:
                print("STDERR:")
                print(result.stderr)
            
            return result
        
        except Exception as e:
            print(f"Error running command: {e}")
            return None
    
    def install_dependencies(self):
        """安装测试依赖"""
        print("Installing test dependencies...")
        
        # 安装基础依赖
        deps = [
            "pytest>=7.0.0",
            "pytest-asyncio>=0.21.0",
            "pytest-cov>=4.0.0",
            "pytest-xdist>=3.0.0",
            "pytest-html>=3.1.0",
            "pytest-mock>=3.10.0",
            "httpx>=0.24.0",
            "fakeredis>=2.10.0",
            "moto>=4.1.0",
            "factory-boy>=3.2.0",
            "freezegun>=1.2.0",
            "responses>=0.23.0",
            "aiofiles>=23.0.0",
            "pillow>=9.5.0",
            "numpy>=1.24.0",
            "pandas>=2.0.0",
            "matplotlib>=3.7.0",
            "seaborn>=0.12.0"
        ]
        
        for dep in deps:
            result = self.run_command(["pip", "install", dep])
            if result and result.returncode != 0:
                print(f"Failed to install {dep}")
                return False
        
        print("Dependencies installed successfully!")
        return True
    
    def run_unit_tests(self, verbose: bool = False, coverage: bool = False) -> bool:
        """运行单元测试"""
        print("\n" + "="*50)
        print("Running Unit Tests")
        print("="*50)
        
        cmd = ["python", "-m", "pytest"]
        
        # 添加测试文件
        cmd.extend([
            "tests/test_api.py",
            "-m", "unit",
            "--tb=short"
        ])
        
        if verbose:
            cmd.append("-v")
        
        if coverage:
            cmd.extend([
                "--cov=backend",
                "--cov-report=html:tests/reports/coverage/unit",
                "--cov-report=term-missing",
                "--cov-fail-under=80"
            ])
        
        # 添加 HTML 报告
        cmd.extend([
            "--html=tests/reports/unit_test_report.html",
            "--self-contained-html"
        ])
        
        result = self.run_command(cmd)
        return result and result.returncode == 0
    
    def run_integration_tests(self, verbose: bool = False, coverage: bool = False) -> bool:
        """运行集成测试"""
        print("\n" + "="*50)
        print("Running Integration Tests")
        print("="*50)
        
        cmd = ["python", "-m", "pytest"]
        
        # 添加测试文件
        cmd.extend([
            "tests/test_integration.py",
            "-m", "integration",
            "--tb=short"
        ])
        
        if verbose:
            cmd.append("-v")
        
        if coverage:
            cmd.extend([
                "--cov=backend",
                "--cov-report=html:tests/reports/coverage/integration",
                "--cov-report=term-missing",
                "--cov-fail-under=70"
            ])
        
        # 添加 HTML 报告
        cmd.extend([
            "--html=tests/reports/integration_test_report.html",
            "--self-contained-html"
        ])
        
        result = self.run_command(cmd)
        return result and result.returncode == 0
    
    def run_e2e_tests(self, verbose: bool = False, coverage: bool = False) -> bool:
        """运行端到端测试"""
        print("\n" + "="*50)
        print("Running End-to-End Tests")
        print("="*50)
        
        cmd = ["python", "-m", "pytest"]
        
        # 添加测试文件
        cmd.extend([
            "tests/test_e2e.py",
            "-m", "e2e",
            "--tb=short",
            "-s"  # 不捕获输出，便于调试
        ])
        
        if verbose:
            cmd.append("-v")
        
        if coverage:
            cmd.extend([
                "--cov=backend",
                "--cov-report=html:tests/reports/coverage/e2e",
                "--cov-report=term-missing"
            ])
        
        # 添加 HTML 报告
        cmd.extend([
            "--html=tests/reports/e2e_test_report.html",
            "--self-contained-html"
        ])
        
        result = self.run_command(cmd)
        return result and result.returncode == 0
    
    def run_performance_tests(self, verbose: bool = False) -> bool:
        """运行性能测试"""
        print("\n" + "="*50)
        print("Running Performance Tests")
        print("="*50)
        
        cmd = ["python", "-m", "pytest"]
        
        # 添加测试文件
        cmd.extend([
            "tests/test_e2e.py::TestPerformance",
            "-m", "slow",
            "--tb=short",
            "-s"
        ])
        
        if verbose:
            cmd.append("-v")
        
        # 添加 HTML 报告
        cmd.extend([
            "--html=tests/reports/performance_test_report.html",
            "--self-contained-html"
        ])
        
        result = self.run_command(cmd)
        return result and result.returncode == 0
    
    def run_all_tests(self, verbose: bool = False, coverage: bool = False, parallel: bool = False) -> bool:
        """运行所有测试"""
        print("\n" + "="*50)
        print("Running All Tests")
        print("="*50)
        
        cmd = ["python", "-m", "pytest"]
        
        # 添加所有测试文件
        cmd.extend([
            "tests/",
            "--tb=short"
        ])
        
        if verbose:
            cmd.append("-v")
        
        if parallel:
            cmd.extend(["-n", "auto"])  # 自动检测 CPU 核心数
        
        if coverage:
            cmd.extend([
                "--cov=backend",
                "--cov-report=html:tests/reports/coverage/all",
                "--cov-report=term-missing",
                "--cov-report=json:tests/reports/coverage.json",
                "--cov-fail-under=75"
            ])
        
        # 添加 HTML 报告
        cmd.extend([
            "--html=tests/reports/all_tests_report.html",
            "--self-contained-html"
        ])
        
        result = self.run_command(cmd)
        return result and result.returncode == 0
    
    def run_coverage_report(self) -> bool:
        """生成覆盖率报告"""
        print("\n" + "="*50)
        print("Generating Coverage Report")
        print("="*50)
        
        # 运行带覆盖率的测试
        cmd = ["python", "-m", "pytest"]
        cmd.extend([
            "tests/",
            "--cov=backend",
            "--cov-report=html:tests/reports/coverage",
            "--cov-report=term-missing",
            "--cov-report=json:tests/reports/coverage.json",
            "--cov-report=xml:tests/reports/coverage.xml",
            "--tb=no",
            "-q"
        ])
        
        result = self.run_command(cmd)
        
        if result and result.returncode == 0:
            print(f"\nCoverage report generated:")
            print(f"  HTML: {self.coverage_dir}/index.html")
            print(f"  JSON: {self.reports_dir}/coverage.json")
            print(f"  XML: {self.reports_dir}/coverage.xml")
        
        return result and result.returncode == 0
    
    def run_lint_checks(self) -> bool:
        """运行代码质量检查"""
        print("\n" + "="*50)
        print("Running Code Quality Checks")
        print("="*50)
        
        checks = [
            # Flake8 检查
            {
                "name": "Flake8",
                "cmd": ["flake8", "backend/", "tests/", "--max-line-length=88", "--extend-ignore=E203,W503"]
            },
            # Black 格式检查
            {
                "name": "Black",
                "cmd": ["black", "--check", "--diff", "backend/", "tests/"]
            },
            # isort 导入排序检查
            {
                "name": "isort",
                "cmd": ["isort", "--check-only", "--diff", "backend/", "tests/"]
            },
            # mypy 类型检查
            {
                "name": "mypy",
                "cmd": ["mypy", "backend/", "--ignore-missing-imports"]
            }
        ]
        
        all_passed = True
        
        for check in checks:
            print(f"\nRunning {check['name']}...")
            result = self.run_command(check["cmd"])
            
            if not result or result.returncode != 0:
                print(f"❌ {check['name']} failed")
                all_passed = False
            else:
                print(f"✅ {check['name']} passed")
        
        return all_passed
    
    def run_security_checks(self) -> bool:
        """运行安全检查"""
        print("\n" + "="*50)
        print("Running Security Checks")
        print("="*50)
        
        checks = [
            # Bandit 安全检查
            {
                "name": "Bandit",
                "cmd": ["bandit", "-r", "backend/", "-f", "json", "-o", "tests/reports/bandit_report.json"]
            },
            # Safety 依赖安全检查
            {
                "name": "Safety",
                "cmd": ["safety", "check", "--json", "--output", "tests/reports/safety_report.json"]
            }
        ]
        
        all_passed = True
        
        for check in checks:
            print(f"\nRunning {check['name']}...")
            result = self.run_command(check["cmd"])
            
            if not result or result.returncode != 0:
                print(f"❌ {check['name']} found issues")
                all_passed = False
            else:
                print(f"✅ {check['name']} passed")
        
        return all_passed
    
    def generate_test_summary(self) -> Dict[str, Any]:
        """生成测试摘要"""
        summary = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "reports": {
                "unit_tests": str(self.reports_dir / "unit_test_report.html"),
                "integration_tests": str(self.reports_dir / "integration_test_report.html"),
                "e2e_tests": str(self.reports_dir / "e2e_test_report.html"),
                "all_tests": str(self.reports_dir / "all_tests_report.html"),
                "coverage": str(self.coverage_dir / "index.html"),
                "bandit": str(self.reports_dir / "bandit_report.json"),
                "safety": str(self.reports_dir / "safety_report.json")
            }
        }
        
        # 读取覆盖率数据
        coverage_file = self.reports_dir / "coverage.json"
        if coverage_file.exists():
            try:
                with open(coverage_file, 'r') as f:
                    coverage_data = json.load(f)
                    summary["coverage"] = {
                        "total": coverage_data.get("totals", {}).get("percent_covered", 0),
                        "lines": coverage_data.get("totals", {}).get("num_statements", 0),
                        "covered": coverage_data.get("totals", {}).get("covered_lines", 0)
                    }
            except Exception as e:
                print(f"Error reading coverage data: {e}")
        
        # 保存摘要
        summary_file = self.reports_dir / "test_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        return summary
    
    def print_summary(self, summary: Dict[str, Any]):
        """打印测试摘要"""
        print("\n" + "="*50)
        print("Test Summary")
        print("="*50)
        
        print(f"Timestamp: {summary['timestamp']}")
        
        if "coverage" in summary:
            coverage = summary["coverage"]
            print(f"\nCoverage: {coverage['total']:.1f}%")
            print(f"Lines: {coverage['covered']}/{coverage['lines']}")
        
        print("\nReports generated:")
        for report_type, path in summary["reports"].items():
            if os.path.exists(path):
                print(f"  ✅ {report_type}: {path}")
            else:
                print(f"  ❌ {report_type}: Not generated")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="Medical AI Test Runner")
    
    parser.add_argument(
        "--type",
        choices=["unit", "integration", "e2e", "performance", "all"],
        default="all",
        help="Test type to run"
    )
    
    parser.add_argument(
        "--coverage",
        action="store_true",
        help="Generate coverage report"
    )
    
    parser.add_argument(
        "--parallel",
        action="store_true",
        help="Run tests in parallel"
    )
    
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Verbose output"
    )
    
    parser.add_argument(
        "--install-deps",
        action="store_true",
        help="Install test dependencies"
    )
    
    parser.add_argument(
        "--lint",
        action="store_true",
        help="Run code quality checks"
    )
    
    parser.add_argument(
        "--security",
        action="store_true",
        help="Run security checks"
    )
    
    parser.add_argument(
        "--report-only",
        action="store_true",
        help="Only generate coverage report"
    )
    
    args = parser.parse_args()
    
    runner = TestRunner()
    
    # 安装依赖
    if args.install_deps:
        if not runner.install_dependencies():
            sys.exit(1)
        return
    
    # 只生成报告
    if args.report_only:
        success = runner.run_coverage_report()
        summary = runner.generate_test_summary()
        runner.print_summary(summary)
        sys.exit(0 if success else 1)
    
    # 运行代码质量检查
    if args.lint:
        success = runner.run_lint_checks()
        sys.exit(0 if success else 1)
    
    # 运行安全检查
    if args.security:
        success = runner.run_security_checks()
        sys.exit(0 if success else 1)
    
    # 运行测试
    success = True
    
    if args.type == "unit":
        success = runner.run_unit_tests(args.verbose, args.coverage)
    elif args.type == "integration":
        success = runner.run_integration_tests(args.verbose, args.coverage)
    elif args.type == "e2e":
        success = runner.run_e2e_tests(args.verbose, args.coverage)
    elif args.type == "performance":
        success = runner.run_performance_tests(args.verbose)
    elif args.type == "all":
        success = runner.run_all_tests(args.verbose, args.coverage, args.parallel)
    
    # 生成摘要
    summary = runner.generate_test_summary()
    runner.print_summary(summary)
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()