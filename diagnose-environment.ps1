# 医疗AI项目环境诊断脚本
# Medical AI Project Environment Diagnostic Script

Write-Host "=== 医疗AI项目环境诊断 ===" -ForegroundColor Cyan
Write-Host "Medical AI Project Environment Diagnostic" -ForegroundColor Cyan
Write-Host ""

# 检查Node.js
Write-Host "1. 检查 Node.js 安装..." -ForegroundColor Yellow
try {
    $nodeVersion = node --version 2>$null
    if ($nodeVersion) {
        Write-Host "   ✓ Node.js 已安装: $nodeVersion" -ForegroundColor Green
    } else {
        Write-Host "   ✗ Node.js 未安装或不在PATH中" -ForegroundColor Red
        Write-Host "     请访问 https://nodejs.org/ 下载并安装 Node.js" -ForegroundColor Yellow
    }
} catch {
    Write-Host "   ✗ Node.js 未安装" -ForegroundColor Red
    Write-Host "     请访问 https://nodejs.org/ 下载并安装 Node.js" -ForegroundColor Yellow
}

# 检查npm
Write-Host "\n2. 检查 npm 包管理器..." -ForegroundColor Yellow
try {
    $npmVersion = npm --version 2>$null
    if ($npmVersion) {
        Write-Host "   ✓ npm 已安装: $npmVersion" -ForegroundColor Green
    } else {
        Write-Host "   ✗ npm 未安装" -ForegroundColor Red
    }
} catch {
    Write-Host "   ✗ npm 未安装" -ForegroundColor Red
}

# 检查项目目录
Write-Host "\n3. 检查项目文件..." -ForegroundColor Yellow
$projectFiles = @(
    "package.json",
    "tsconfig.json", 
    "vite.config.ts",
    "src"
)

foreach ($file in $projectFiles) {
    if (Test-Path $file) {
        Write-Host "   ✓ $file 存在" -ForegroundColor Green
    } else {
        Write-Host "   ✗ $file 缺失" -ForegroundColor Red
    }
}

# 检查node_modules
Write-Host "\n4. 检查依赖安装..." -ForegroundColor Yellow
if (Test-Path "node_modules") {
    Write-Host "   ✓ node_modules 文件夹存在" -ForegroundColor Green
    
    # 检查关键依赖
    $keyDeps = @("react", "typescript", "vite", "antd")
    foreach ($dep in $keyDeps) {
        if (Test-Path "node_modules\$dep") {
            Write-Host "   ✓ $dep 已安装" -ForegroundColor Green
        } else {
            Write-Host "   ✗ $dep 未安装" -ForegroundColor Red
        }
    }
} else {
    Write-Host "   ✗ node_modules 文件夹不存在" -ForegroundColor Red
    Write-Host "     需要运行: npm install" -ForegroundColor Yellow
}

# 检查TypeScript配置
Write-Host "\n5. 检查 TypeScript 配置..." -ForegroundColor Yellow
if (Test-Path "tsconfig.json") {
    try {
        $tsconfig = Get-Content "tsconfig.json" | ConvertFrom-Json
        if ($tsconfig.compilerOptions.paths) {
            Write-Host "   ✓ 路径别名配置存在" -ForegroundColor Green
        } else {
            Write-Host "   ⚠ 路径别名配置可能缺失" -ForegroundColor Yellow
        }
    } catch {
        Write-Host "   ⚠ tsconfig.json 格式可能有问题" -ForegroundColor Yellow
    }
}

# 提供解决方案
Write-Host "\n=== 解决方案建议 ===" -ForegroundColor Cyan

if (-not (Get-Command node -ErrorAction SilentlyContinue)) {
    Write-Host "\n🔧 安装 Node.js:" -ForegroundColor Magenta
    Write-Host "   1. 访问 https://nodejs.org/"
    Write-Host "   2. 下载 LTS 版本 (推荐 18.x 或 20.x)"
    Write-Host "   3. 运行安装程序，确保勾选 'Add to PATH'"
    Write-Host "   4. 重启命令行工具"
}

if (-not (Test-Path "node_modules")) {
    Write-Host "\n🔧 安装项目依赖:" -ForegroundColor Magenta
    Write-Host "   运行以下命令之一:"
    Write-Host "   • .\setup-frontend.bat"
    Write-Host "   • npm install"
}

Write-Host "\n🔧 完成安装后:" -ForegroundColor Magenta
Write-Host "   1. 重启您的代码编辑器 (VS Code/WebStorm 等)"
Write-Host "   2. 运行: npm run dev"
Write-Host "   3. 访问: http://localhost:3000"

Write-Host "\n=== 诊断完成 ===" -ForegroundColor Cyan
Write-Host "如果问题仍然存在，请查看 NODE_SETUP_GUIDE.md 获取详细说明" -ForegroundColor Gray

Pause