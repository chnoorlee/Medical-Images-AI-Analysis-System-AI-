@echo off
REM 医疗AI系统测试运行脚本 (Windows批处理版本)
REM 提供多种测试运行选项和报告生成

setlocal enabledelayedexpansion

REM 设置颜色代码
set "RED=[91m"
set "GREEN=[92m"
set "YELLOW=[93m"
set "BLUE=[94m"
set "MAGENTA=[95m"
set "CYAN=[96m"
set "WHITE=[97m"
set "RESET=[0m"

REM 设置项目根目录
set "PROJECT_ROOT=%~dp0.."
set "TEST_DIR=%~dp0"
set "REPORTS_DIR=%TEST_DIR%reports"
set "LOGS_DIR=%TEST_DIR%logs"

REM 创建必要的目录
if not exist "%REPORTS_DIR%" mkdir "%REPORTS_DIR%"
if not exist "%LOGS_DIR%" mkdir "%LOGS_DIR%"
if not exist "%REPORTS_DIR%\coverage" mkdir "%REPORTS_DIR%\coverage"
if not exist "%REPORTS_DIR%\charts" mkdir "%REPORTS_DIR%\charts"
if not exist "%REPORTS_DIR%\benchmarks" mkdir "%REPORTS_DIR%\benchmarks"

REM 显示帮助信息
if "%1"=="help" goto :show_help
if "%1"=="--help" goto :show_help
if "%1"=="-h" goto :show_help

REM 检查Python环境
echo %CYAN%检查Python环境...%RESET%
python --version >nul 2>&1
if errorlevel 1 (
    echo %RED%错误: 未找到Python，请确保Python已安装并添加到PATH%RESET%
    exit /b 1
)

REM 检查pytest
pytest --version >nul 2>&1
if errorlevel 1 (
    echo %YELLOW%警告: 未找到pytest，正在安装测试依赖...%RESET%
    pip install -r "%TEST_DIR%requirements-test.txt"
    if errorlevel 1 (
        echo %RED%错误: 安装测试依赖失败%RESET%
        exit /b 1
    )
)

REM 设置环境变量
set "PYTHONPATH=%PROJECT_ROOT%;%PROJECT_ROOT%\src;%PYTHONPATH%"
set "TESTING=1"
set "TEST_DATABASE_URL=sqlite:///test.db"
set "TEST_REDIS_URL=redis://localhost:6379/1"

REM 解析命令行参数
set "TEST_TYPE=%1"
if "%TEST_TYPE%"=="" set "TEST_TYPE=all"

REM 设置测试选项
set "PYTEST_ARGS=-v --tb=short --durations=10"
set "COVERAGE_ARGS=--cov=src --cov-report=html:%REPORTS_DIR%\coverage --cov-report=xml:%REPORTS_DIR%\coverage.xml --cov-report=term-missing"
set "REPORT_ARGS=--html=%REPORTS_DIR%\report.html --self-contained-html --junitxml=%REPORTS_DIR%\junit.xml"

echo %BLUE%========================================%RESET%
echo %BLUE%     医疗AI系统测试运行器%RESET%
echo %BLUE%========================================%RESET%
echo.
echo %CYAN%测试类型: %TEST_TYPE%%RESET%
echo %CYAN%项目根目录: %PROJECT_ROOT%%RESET%
echo %CYAN%测试目录: %TEST_DIR%%RESET%
echo %CYAN%报告目录: %REPORTS_DIR%%RESET%
echo.

REM 根据测试类型执行相应的测试
if "%TEST_TYPE%"=="unit" goto :run_unit_tests
if "%TEST_TYPE%"=="integration" goto :run_integration_tests
if "%TEST_TYPE%"=="e2e" goto :run_e2e_tests
if "%TEST_TYPE%"=="performance" goto :run_performance_tests
if "%TEST_TYPE%"=="benchmark" goto :run_benchmark_tests
if "%TEST_TYPE%"=="api" goto :run_api_tests
if "%TEST_TYPE%"=="security" goto :run_security_tests
if "%TEST_TYPE%"=="smoke" goto :run_smoke_tests
if "%TEST_TYPE%"=="regression" goto :run_regression_tests
if "%TEST_TYPE%"=="fast" goto :run_fast_tests
if "%TEST_TYPE%"=="slow" goto :run_slow_tests
if "%TEST_TYPE%"=="coverage" goto :run_coverage_tests
if "%TEST_TYPE%"=="quality" goto :run_quality_checks
if "%TEST_TYPE%"=="security-scan" goto :run_security_scan
if "%TEST_TYPE%"=="install-deps" goto :install_dependencies
if "%TEST_TYPE%"=="clean" goto :clean_reports
if "%TEST_TYPE%"=="all" goto :run_all_tests

echo %RED%错误: 未知的测试类型 '%TEST_TYPE%'%RESET%
echo %YELLOW%使用 'run_tests.bat help' 查看可用选项%RESET%
exit /b 1

:show_help
echo %GREEN%医疗AI系统测试运行器%RESET%
echo.
echo %CYAN%用法:%RESET%
echo   run_tests.bat [TEST_TYPE]
echo.
echo %CYAN%可用的测试类型:%RESET%
echo   %YELLOW%unit%RESET%           - 运行单元测试
echo   %YELLOW%integration%RESET%    - 运行集成测试
echo   %YELLOW%e2e%RESET%            - 运行端到端测试
echo   %YELLOW%performance%RESET%    - 运行性能测试
echo   %YELLOW%benchmark%RESET%      - 运行基准测试
echo   %YELLOW%api%RESET%            - 运行API测试
echo   %YELLOW%security%RESET%       - 运行安全测试
echo   %YELLOW%smoke%RESET%          - 运行冒烟测试
echo   %YELLOW%regression%RESET%     - 运行回归测试
echo   %YELLOW%fast%RESET%           - 运行快速测试
echo   %YELLOW%slow%RESET%           - 运行慢速测试
echo   %YELLOW%coverage%RESET%       - 运行覆盖率测试
echo   %YELLOW%all%RESET%            - 运行所有测试 (默认)
echo.
echo %CYAN%实用工具:%RESET%
echo   %YELLOW%quality%RESET%        - 运行代码质量检查
echo   %YELLOW%security-scan%RESET%  - 运行安全扫描
echo   %YELLOW%install-deps%RESET%   - 安装测试依赖
echo   %YELLOW%clean%RESET%          - 清理测试报告
echo   %YELLOW%help%RESET%           - 显示此帮助信息
echo.
echo %CYAN%示例:%RESET%
echo   run_tests.bat unit
echo   run_tests.bat performance
echo   run_tests.bat all
exit /b 0

:run_unit_tests
echo %GREEN%运行单元测试...%RESET%
pytest %PYTEST_ARGS% %COVERAGE_ARGS% %REPORT_ARGS% -m "unit and not slow" "%TEST_DIR%"
goto :test_complete

:run_integration_tests
echo %GREEN%运行集成测试...%RESET%
pytest %PYTEST_ARGS% %COVERAGE_ARGS% %REPORT_ARGS% -m "integration" "%TEST_DIR%"
goto :test_complete

:run_e2e_tests
echo %GREEN%运行端到端测试...%RESET%
pytest %PYTEST_ARGS% %COVERAGE_ARGS% %REPORT_ARGS% -m "e2e" "%TEST_DIR%"
goto :test_complete

:run_performance_tests
echo %GREEN%运行性能测试...%RESET%
pytest %PYTEST_ARGS% -m "performance" --benchmark-only --benchmark-sort=mean "%TEST_DIR%"
goto :test_complete

:run_benchmark_tests
echo %GREEN%运行基准测试...%RESET%
pytest %PYTEST_ARGS% -m "benchmark" "%TEST_DIR%"
goto :test_complete

:run_api_tests
echo %GREEN%运行API测试...%RESET%
pytest %PYTEST_ARGS% %COVERAGE_ARGS% %REPORT_ARGS% -m "api" "%TEST_DIR%"
goto :test_complete

:run_security_tests
echo %GREEN%运行安全测试...%RESET%
pytest %PYTEST_ARGS% %COVERAGE_ARGS% %REPORT_ARGS% -m "security" "%TEST_DIR%"
goto :test_complete

:run_smoke_tests
echo %GREEN%运行冒烟测试...%RESET%
pytest %PYTEST_ARGS% %COVERAGE_ARGS% %REPORT_ARGS% -m "smoke" "%TEST_DIR%"
goto :test_complete

:run_regression_tests
echo %GREEN%运行回归测试...%RESET%
pytest %PYTEST_ARGS% %COVERAGE_ARGS% %REPORT_ARGS% -m "regression" "%TEST_DIR%"
goto :test_complete

:run_fast_tests
echo %GREEN%运行快速测试...%RESET%
pytest %PYTEST_ARGS% %COVERAGE_ARGS% %REPORT_ARGS% -m "fast" "%TEST_DIR%"
goto :test_complete

:run_slow_tests
echo %GREEN%运行慢速测试...%RESET%
pytest %PYTEST_ARGS% %COVERAGE_ARGS% %REPORT_ARGS% -m "slow" "%TEST_DIR%"
goto :test_complete

:run_coverage_tests
echo %GREEN%运行覆盖率测试...%RESET%
pytest %PYTEST_ARGS% %COVERAGE_ARGS% %REPORT_ARGS% --cov-fail-under=80 "%TEST_DIR%"
echo.
echo %CYAN%覆盖率报告已生成:%RESET%
echo   HTML: %REPORTS_DIR%\coverage\index.html
echo   XML:  %REPORTS_DIR%\coverage.xml
goto :test_complete

:run_all_tests
echo %GREEN%运行所有测试...%RESET%
echo.
echo %CYAN%第1步: 单元测试%RESET%
pytest %PYTEST_ARGS% -m "unit and not slow" "%TEST_DIR%"
if errorlevel 1 (
    echo %RED%单元测试失败%RESET%
    goto :test_failed
)

echo.
echo %CYAN%第2步: 集成测试%RESET%
pytest %PYTEST_ARGS% -m "integration and not slow" "%TEST_DIR%"
if errorlevel 1 (
    echo %RED%集成测试失败%RESET%
    goto :test_failed
)

echo.
echo %CYAN%第3步: API测试%RESET%
pytest %PYTEST_ARGS% -m "api and not slow" "%TEST_DIR%"
if errorlevel 1 (
    echo %RED%API测试失败%RESET%
    goto :test_failed
)

echo.
echo %CYAN%第4步: 生成最终覆盖率报告%RESET%
pytest %PYTEST_ARGS% %COVERAGE_ARGS% %REPORT_ARGS% --cov-fail-under=70 "%TEST_DIR%"
goto :test_complete

:run_quality_checks
echo %GREEN%运行代码质量检查...%RESET%
echo.

echo %CYAN%检查 Flake8...%RESET%
flake8 "%PROJECT_ROOT%\src" --max-line-length=88 --extend-ignore=E203,W503 --output-file="%REPORTS_DIR%\flake8.txt"
if errorlevel 1 (
    echo %YELLOW%Flake8 发现问题，详情请查看 %REPORTS_DIR%\flake8.txt%RESET%
) else (
    echo %GREEN%Flake8 检查通过%RESET%
)

echo.
echo %CYAN%检查 Black 格式...%RESET%
black --check --diff "%PROJECT_ROOT%\src" > "%REPORTS_DIR%\black.txt" 2>&1
if errorlevel 1 (
    echo %YELLOW%Black 发现格式问题，详情请查看 %REPORTS_DIR%\black.txt%RESET%
    echo %CYAN%运行 'black src' 自动修复格式问题%RESET%
) else (
    echo %GREEN%Black 格式检查通过%RESET%
)

echo.
echo %CYAN%检查 isort 导入排序...%RESET%
isort --check-only --diff "%PROJECT_ROOT%\src" > "%REPORTS_DIR%\isort.txt" 2>&1
if errorlevel 1 (
    echo %YELLOW%isort 发现导入排序问题，详情请查看 %REPORTS_DIR%\isort.txt%RESET%
    echo %CYAN%运行 'isort src' 自动修复导入排序%RESET%
) else (
    echo %GREEN%isort 检查通过%RESET%
)

echo.
echo %CYAN%检查 MyPy 类型注解...%RESET%
mypy "%PROJECT_ROOT%\src" --ignore-missing-imports --output "%REPORTS_DIR%\mypy.txt"
if errorlevel 1 (
    echo %YELLOW%MyPy 发现类型问题，详情请查看 %REPORTS_DIR%\mypy.txt%RESET%
) else (
    echo %GREEN%MyPy 类型检查通过%RESET%
)

goto :test_complete

:run_security_scan
echo %GREEN%运行安全扫描...%RESET%
echo.

echo %CYAN%运行 Bandit 安全扫描...%RESET%
bandit -r "%PROJECT_ROOT%\src" -f json -o "%REPORTS_DIR%\bandit.json"
if errorlevel 1 (
    echo %YELLOW%Bandit 发现安全问题，详情请查看 %REPORTS_DIR%\bandit.json%RESET%
) else (
    echo %GREEN%Bandit 安全扫描通过%RESET%
)

echo.
echo %CYAN%运行 Safety 依赖安全检查...%RESET%
safety check --json --output "%REPORTS_DIR%\safety.json"
if errorlevel 1 (
    echo %YELLOW%Safety 发现依赖安全问题，详情请查看 %REPORTS_DIR%\safety.json%RESET%
) else (
    echo %GREEN%Safety 依赖检查通过%RESET%
)

goto :test_complete

:install_dependencies
echo %GREEN%安装测试依赖...%RESET%
pip install -r "%TEST_DIR%requirements-test.txt"
if errorlevel 1 (
    echo %RED%安装依赖失败%RESET%
    exit /b 1
) else (
    echo %GREEN%依赖安装成功%RESET%
)
exit /b 0

:clean_reports
echo %GREEN%清理测试报告...%RESET%
if exist "%REPORTS_DIR%" (
    rmdir /s /q "%REPORTS_DIR%"
    mkdir "%REPORTS_DIR%"
    mkdir "%REPORTS_DIR%\coverage"
    mkdir "%REPORTS_DIR%\charts"
    mkdir "%REPORTS_DIR%\benchmarks"
    echo %GREEN%测试报告已清理%RESET%
) else (
    echo %YELLOW%报告目录不存在%RESET%
)

if exist "%LOGS_DIR%" (
    rmdir /s /q "%LOGS_DIR%"
    mkdir "%LOGS_DIR%"
    echo %GREEN%测试日志已清理%RESET%
)

if exist "%PROJECT_ROOT%\.coverage" (
    del "%PROJECT_ROOT%\.coverage"
    echo %GREEN%覆盖率数据已清理%RESET%
)

if exist "%PROJECT_ROOT%\.pytest_cache" (
    rmdir /s /q "%PROJECT_ROOT%\.pytest_cache"
    echo %GREEN%Pytest缓存已清理%RESET%
)

exit /b 0

:test_complete
set "EXIT_CODE=%ERRORLEVEL%"
echo.
echo %BLUE%========================================%RESET%
if %EXIT_CODE%==0 (
    echo %GREEN%     测试完成 - 全部通过!%RESET%
    echo %BLUE%========================================%RESET%
    echo.
    echo %CYAN%报告文件:%RESET%
    if exist "%REPORTS_DIR%\report.html" echo   HTML报告: %REPORTS_DIR%\report.html
    if exist "%REPORTS_DIR%\coverage\index.html" echo   覆盖率报告: %REPORTS_DIR%\coverage\index.html
    if exist "%REPORTS_DIR%\junit.xml" echo   JUnit报告: %REPORTS_DIR%\junit.xml
    if exist "%REPORTS_DIR%\coverage.xml" echo   覆盖率XML: %REPORTS_DIR%\coverage.xml
) else (
    echo %RED%     测试完成 - 有失败!%RESET%
    echo %BLUE%========================================%RESET%
    echo.
    echo %YELLOW%请检查测试输出和报告文件以获取详细信息%RESET%
)
echo.
exit /b %EXIT_CODE%

:test_failed
echo.
echo %BLUE%========================================%RESET%
echo %RED%     测试失败 - 提前终止!%RESET%
echo %BLUE%========================================%RESET%
echo.
echo %YELLOW%请检查上面的错误信息%RESET%
exit /b 1