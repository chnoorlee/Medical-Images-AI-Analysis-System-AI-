# Node.js 环境安装指南

## 问题诊断

您的项目文件显示红色是因为缺少 Node.js 环境和项目依赖包。这会导致：
- TypeScript 类型检查失败
- 路径别名无法解析
- IDE 无法正确识别模块导入
- 代码智能提示功能受限

## 解决方案

### 步骤 1: 安装 Node.js

1. **下载 Node.js**
   - 访问官网：https://nodejs.org/
   - 下载 LTS 版本（推荐 18.x 或 20.x）
   - 选择 Windows Installer (.msi)

2. **安装 Node.js**
   - 运行下载的 .msi 文件
   - 按照安装向导完成安装
   - 确保勾选 "Add to PATH" 选项

3. **验证安装**
   ```powershell
   node --version
   npm --version
   ```

### 步骤 2: 安装项目依赖

1. **打开 PowerShell 或命令提示符**
   - 导航到项目目录：`cd "F:\medical AI"`

2. **运行安装脚本**
   ```powershell
   # 方法 1: 使用提供的批处理文件
   .\setup-frontend.bat
   
   # 方法 2: 直接使用 npm
   npm install
   ```

3. **等待安装完成**
   - 安装过程可能需要几分钟
   - 完成后会生成 `node_modules` 文件夹

### 步骤 3: 验证环境

1. **检查依赖安装**
   ```powershell
   # 查看已安装的包
   npm list --depth=0
   
   # 运行类型检查
   npx tsc --noEmit
   ```

2. **启动开发服务器**
   ```powershell
   npm run dev
   ```

### 步骤 4: IDE 配置

1. **重启 IDE**
   - 关闭并重新打开您的代码编辑器
   - 让 IDE 重新索引项目文件

2. **检查 TypeScript 服务**
   - 确保 IDE 使用项目本地的 TypeScript 版本
   - 检查是否有 TypeScript 错误提示

## 常见问题解决

### 问题 1: npm 命令不识别
**解决方案：**
- 重新安装 Node.js，确保勾选 "Add to PATH"
- 重启命令行工具
- 手动添加 Node.js 到系统 PATH

### 问题 2: 安装依赖失败
**解决方案：**
```powershell
# 清理缓存
npm cache clean --force

# 删除 node_modules 和 package-lock.json
Remove-Item -Recurse -Force node_modules
Remove-Item package-lock.json

# 重新安装
npm install
```

### 问题 3: 权限错误
**解决方案：**
- 以管理员身份运行 PowerShell
- 或者配置 npm 全局目录到用户文件夹

### 问题 4: 网络问题
**解决方案：**
```powershell
# 使用淘宝镜像
npm config set registry https://registry.npmmirror.com/

# 或者使用 cnpm
npm install -g cnpm --registry=https://registry.npmmirror.com/
cnpm install
```

## 项目结构说明

安装完成后，您的项目应该包含：
```
F:\medical AI\
├── node_modules/          # 依赖包文件夹
├── src/                   # 源代码
├── package.json           # 项目配置
├── tsconfig.json          # TypeScript 配置
├── vite.config.ts         # Vite 构建配置
└── .eslintrc.json         # ESLint 配置
```

## 验证清单

- [ ] Node.js 已安装并可在命令行中使用
- [ ] npm 命令可正常执行
- [ ] `node_modules` 文件夹已生成
- [ ] `npm run dev` 可以启动开发服务器
- [ ] IDE 中的红色错误提示消失
- [ ] TypeScript 类型检查正常工作
- [ ] 路径别名（如 `@/` `@components/`）可以正确解析

## 下一步

环境配置完成后，您可以：
1. 运行 `npm run dev` 启动开发服务器
2. 访问 http://localhost:3000 查看应用
3. 开始开发和调试代码
4. 使用 `npm run build` 构建生产版本

如果仍有问题，请检查：
- Node.js 版本是否兼容（建议 18.x+）
- 网络连接是否正常
- 防火墙或杀毒软件是否阻止了安装
- 磁盘空间是否充足