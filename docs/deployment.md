# Medical AI 部署指南

## 概述

本文档详细介绍了 Medical AI 系统在生产环境中的部署方法，包括 Kubernetes 集群部署、Docker Compose 部署以及相关的配置和优化建议。

## 部署架构

### 推荐架构

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Load Balancer │    │   Web Tier      │    │   App Tier      │
│   (Nginx/ALB)   │────│   (Frontend)    │────│   (Backend API) │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                                        │
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   AI Tier       │    │   Cache Tier    │    │   Data Tier     │
│   (Inference)   │────│   (Redis)       │────│   (PostgreSQL)  │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                                        │
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Storage Tier  │    │   Message Queue │    │   Monitoring    │
│   (MinIO/S3)    │────│   (RabbitMQ)    │────│   (Prometheus)  │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## 环境要求

### 硬件要求

#### 最小配置
- **CPU**: 8 核心
- **内存**: 32GB RAM
- **存储**: 500GB SSD
- **GPU**: NVIDIA GTX 1080 或同等性能（可选）

#### 推荐配置
- **CPU**: 16+ 核心
- **内存**: 64GB+ RAM
- **存储**: 1TB+ NVMe SSD
- **GPU**: NVIDIA RTX 3080 或更高（推荐）
- **网络**: 1Gbps+ 带宽

#### 生产环境配置
- **CPU**: 32+ 核心
- **内存**: 128GB+ RAM
- **存储**: 2TB+ NVMe SSD + 10TB+ HDD
- **GPU**: NVIDIA A100 或 V100（多卡）
- **网络**: 10Gbps+ 带宽

### 软件要求

- **操作系统**: Ubuntu 20.04+ / CentOS 8+ / RHEL 8+
- **Docker**: 20.10+
- **Docker Compose**: 2.0+
- **Kubernetes**: 1.25+
- **Helm**: 3.8+
- **NVIDIA Docker**: 2.0+（如使用GPU）

## Kubernetes 部署

### 1. 准备工作

#### 1.1 创建命名空间

```bash
kubectl apply -f k8s/namespace.yaml
```

#### 1.2 配置存储类

```bash
# 检查可用存储类
kubectl get storageclass

# 如果需要，创建自定义存储类
kubectl apply -f k8s/storage.yaml
```

#### 1.3 创建密钥

```bash
# 编辑密钥文件，替换占位符
cp k8s/secrets.yaml k8s/secrets-prod.yaml
vim k8s/secrets-prod.yaml

# 应用密钥
kubectl apply -f k8s/secrets-prod.yaml
```

### 2. 部署基础服务

#### 2.1 部署数据库服务

```bash
# 部署 PostgreSQL, Redis, MinIO
kubectl apply -f k8s/database.yaml

# 等待服务就绪
kubectl wait --for=condition=ready pod -l app=postgres -n medical-ai --timeout=300s
kubectl wait --for=condition=ready pod -l app=redis -n medical-ai --timeout=300s
kubectl wait --for=condition=ready pod -l app=minio -n medical-ai --timeout=300s
```

#### 2.2 初始化数据库

```bash
# 连接到 PostgreSQL 并运行初始化脚本
kubectl exec -it deployment/postgres -n medical-ai -- psql -U postgres -d medical_ai -f /docker-entrypoint-initdb.d/init.sql
```

### 3. 部署应用服务

#### 3.1 部署后端服务

```bash
# 应用配置映射
kubectl apply -f k8s/configmap.yaml

# 部署后端应用
kubectl apply -f k8s/backend.yaml

# 检查部署状态
kubectl get pods -n medical-ai -l app=medical-ai-backend
kubectl logs -f deployment/medical-ai-backend -n medical-ai
```

#### 3.2 部署前端服务

```bash
# 部署前端和 Nginx
kubectl apply -f k8s/frontend.yaml

# 检查服务状态
kubectl get svc -n medical-ai
```

### 4. 配置网络和安全

#### 4.1 应用网络策略

```bash
kubectl apply -f k8s/security.yaml
```

#### 4.2 配置 Ingress

```bash
# 安装 Nginx Ingress Controller（如果未安装）
helm repo add ingress-nginx https://kubernetes.github.io/ingress-nginx
helm install ingress-nginx ingress-nginx/ingress-nginx

# 应用 Ingress 配置
kubectl apply -f k8s/frontend.yaml
```

### 5. 部署监控和日志

#### 5.1 部署监控系统

```bash
# 部署 Prometheus 和 Grafana
kubectl apply -f k8s/monitoring.yaml
kubectl apply -f k8s/monitoring-config.yaml

# 等待服务就绪
kubectl wait --for=condition=ready pod -l app=prometheus -n medical-ai --timeout=300s
kubectl wait --for=condition=ready pod -l app=grafana -n medical-ai --timeout=300s
```

#### 5.2 部署日志系统

```bash
# 部署 ELK Stack
kubectl apply -f k8s/logging.yaml

# 等待 Elasticsearch 集群就绪
kubectl wait --for=condition=ready pod -l app=elasticsearch -n medical-ai --timeout=600s
```

### 6. 验证部署

#### 6.1 检查所有服务状态

```bash
# 查看所有 Pod 状态
kubectl get pods -n medical-ai

# 查看服务状态
kubectl get svc -n medical-ai

# 查看 Ingress 状态
kubectl get ingress -n medical-ai
```

#### 6.2 健康检查

```bash
# 检查后端 API 健康状态
kubectl exec -it deployment/medical-ai-backend -n medical-ai -- curl http://localhost:8000/health

# 检查数据库连接
kubectl exec -it deployment/postgres -n medical-ai -- pg_isready -U postgres
```

## Docker Compose 部署

### 1. 准备环境

#### 1.1 克隆项目

```bash
git clone <repository-url>
cd medical-ai
```

#### 1.2 配置环境变量

```bash
# 复制环境变量模板
cp .env.example .env

# 编辑环境变量
vim .env
```

**环境变量示例**:
```bash
# 数据库配置
POSTGRES_DB=medical_ai
POSTGRES_USER=postgres
POSTGRES_PASSWORD=your_secure_password
DATABASE_URL=postgresql://postgres:your_secure_password@postgres:5432/medical_ai

# Redis 配置
REDIS_URL=redis://redis:6379/0

# JWT 配置
JWT_SECRET_KEY=your_jwt_secret_key
JWT_ALGORITHM=HS256
JWT_ACCESS_TOKEN_EXPIRE_MINUTES=30

# MinIO 配置
MINIO_ROOT_USER=minioadmin
MINIO_ROOT_PASSWORD=your_minio_password
MINIO_ENDPOINT=http://minio:9000

# 应用配置
ENVIRONMENT=production
DEBUG=false
ALLOWED_HOSTS=your-domain.com,localhost
```

### 2. 构建和启动服务

```bash
# 构建镜像
docker-compose build

# 启动所有服务
docker-compose up -d

# 查看服务状态
docker-compose ps
```

### 3. 初始化数据库

```bash
# 等待数据库启动
docker-compose exec postgres pg_isready -U postgres

# 运行数据库迁移
docker-compose exec backend python -m alembic upgrade head

# 创建初始用户
docker-compose exec backend python scripts/create_admin_user.py
```

### 4. 验证部署

```bash
# 检查所有容器状态
docker-compose ps

# 查看日志
docker-compose logs -f backend

# 测试 API
curl http://localhost:8000/health
```

## SSL/TLS 配置

### 1. 使用 Let's Encrypt

#### 1.1 安装 Certbot

```bash
sudo apt-get update
sudo apt-get install certbot python3-certbot-nginx
```

#### 1.2 获取证书

```bash
# 为域名获取证书
sudo certbot --nginx -d your-domain.com -d www.your-domain.com

# 设置自动续期
sudo crontab -e
# 添加以下行：
# 0 12 * * * /usr/bin/certbot renew --quiet
```

### 2. 使用自签名证书（开发环境）

```bash
# 生成私钥
openssl genrsa -out ssl/private.key 2048

# 生成证书
openssl req -new -x509 -key ssl/private.key -out ssl/certificate.crt -days 365
```

## 性能优化

### 1. 数据库优化

#### 1.1 PostgreSQL 配置

```sql
-- 在 postgresql.conf 中设置
shared_buffers = 256MB
effective_cache_size = 1GB
work_mem = 4MB
maintenance_work_mem = 64MB
max_connections = 100

-- 创建索引
CREATE INDEX CONCURRENTLY idx_images_patient_id ON images(patient_id);
CREATE INDEX CONCURRENTLY idx_reports_created_at ON reports(created_at);
CREATE INDEX CONCURRENTLY idx_ai_tasks_status ON ai_inference_tasks(status);
```

#### 1.2 Redis 配置

```bash
# 在 redis.conf 中设置
maxmemory 2gb
maxmemory-policy allkeys-lru
save 900 1
save 300 10
save 60 10000
```

### 2. 应用优化

#### 2.1 后端优化

```python
# 在 config.py 中设置
DATABASE_POOL_SIZE = 20
DATABASE_MAX_OVERFLOW = 30
REDIS_CONNECTION_POOL_SIZE = 50

# 启用 Gzip 压缩
from fastapi.middleware.gzip import GZipMiddleware
app.add_middleware(GZipMiddleware, minimum_size=1000)
```

#### 2.2 前端优化

```javascript
// 在 vite.config.ts 中配置
export default defineConfig({
  build: {
    rollupOptions: {
      output: {
        manualChunks: {
          vendor: ['react', 'react-dom'],
          antd: ['antd'],
          cornerstone: ['cornerstone-core', 'cornerstone-tools']
        }
      }
    },
    chunkSizeWarningLimit: 1000
  }
});
```

### 3. 缓存策略

#### 3.1 API 缓存

```python
# 使用 Redis 缓存 API 响应
from fastapi_cache import FastAPICache
from fastapi_cache.backends.redis import RedisBackend

@app.on_event("startup")
async def startup():
    redis = aioredis.from_url("redis://localhost")
    FastAPICache.init(RedisBackend(redis), prefix="medical-ai-cache")

@cache(expire=300)  # 缓存 5 分钟
async def get_patient_list():
    # API 逻辑
    pass
```

#### 3.2 静态资源缓存

```nginx
# 在 nginx.conf 中配置
location ~* \.(js|css|png|jpg|jpeg|gif|ico|svg)$ {
    expires 1y;
    add_header Cache-Control "public, immutable";
    add_header Vary Accept-Encoding;
    gzip_static on;
}
```

## 监控和告警

### 1. Prometheus 配置

#### 1.1 自定义指标

```python
# 在应用中添加自定义指标
from prometheus_client import Counter, Histogram, Gauge

api_requests_total = Counter('api_requests_total', 'Total API requests', ['method', 'endpoint'])
api_request_duration = Histogram('api_request_duration_seconds', 'API request duration')
active_users = Gauge('active_users_total', 'Number of active users')
```

#### 1.2 告警规则

```yaml
# 在 alert_rules.yml 中添加
groups:
- name: medical-ai-critical
  rules:
  - alert: HighErrorRate
    expr: rate(api_requests_total{status=~"5.."}[5m]) > 0.1
    for: 2m
    labels:
      severity: critical
    annotations:
      summary: "High error rate detected"
      description: "Error rate is {{ $value }} errors per second"
```

### 2. Grafana 仪表板

#### 2.1 导入预配置仪表板

```bash
# 复制仪表板配置
kubectl create configmap grafana-dashboards \
  --from-file=monitoring/grafana/dashboards/ \
  -n medical-ai
```

#### 2.2 关键指标监控

- **系统指标**: CPU、内存、磁盘、网络
- **应用指标**: 请求量、响应时间、错误率
- **业务指标**: 活跃用户、诊断完成率、AI 准确率
- **基础设施**: 数据库连接、缓存命中率、队列长度

## 备份和恢复

### 1. 数据库备份

#### 1.1 自动备份脚本

```bash
#!/bin/bash
# backup_database.sh

DATE=$(date +%Y%m%d_%H%M%S)
BACKUP_DIR="/backups/postgres"
DB_NAME="medical_ai"

# 创建备份目录
mkdir -p $BACKUP_DIR

# 执行备份
kubectl exec deployment/postgres -n medical-ai -- pg_dump -U postgres $DB_NAME | gzip > $BACKUP_DIR/backup_$DATE.sql.gz

# 保留最近 30 天的备份
find $BACKUP_DIR -name "backup_*.sql.gz" -mtime +30 -delete

echo "Database backup completed: backup_$DATE.sql.gz"
```

#### 1.2 设置定时备份

```bash
# 添加到 crontab
0 2 * * * /path/to/backup_database.sh
```

### 2. 文件存储备份

```bash
#!/bin/bash
# backup_files.sh

DATE=$(date +%Y%m%d_%H%M%S)
BACKUP_DIR="/backups/files"
SOURCE_DIR="/data/medical-ai"

# 使用 rsync 同步文件
rsync -av --delete $SOURCE_DIR/ $BACKUP_DIR/latest/

# 创建快照
cp -al $BACKUP_DIR/latest $BACKUP_DIR/snapshot_$DATE

echo "File backup completed: snapshot_$DATE"
```

### 3. 恢复流程

#### 3.1 数据库恢复

```bash
# 恢复数据库
gunzip -c /backups/postgres/backup_20240120_020000.sql.gz | \
kubectl exec -i deployment/postgres -n medical-ai -- psql -U postgres medical_ai
```

#### 3.2 文件恢复

```bash
# 恢复文件
rsync -av /backups/files/snapshot_20240120_020000/ /data/medical-ai/
```

## 故障排除

### 1. 常见问题

#### 1.1 Pod 启动失败

```bash
# 查看 Pod 状态
kubectl describe pod <pod-name> -n medical-ai

# 查看日志
kubectl logs <pod-name> -n medical-ai --previous

# 检查资源限制
kubectl top pods -n medical-ai
```

#### 1.2 数据库连接问题

```bash
# 测试数据库连接
kubectl exec -it deployment/postgres -n medical-ai -- psql -U postgres -c "SELECT 1;"

# 检查网络策略
kubectl get networkpolicy -n medical-ai

# 测试网络连通性
kubectl exec -it deployment/medical-ai-backend -n medical-ai -- nc -zv postgres 5432
```

#### 1.3 AI 推理服务问题

```bash
# 检查 GPU 资源
kubectl describe node | grep nvidia.com/gpu

# 查看 AI 服务日志
kubectl logs -f deployment/ai-inference -n medical-ai

# 检查模型文件
kubectl exec -it deployment/ai-inference -n medical-ai -- ls -la /models/
```

### 2. 性能问题诊断

#### 2.1 资源使用分析

```bash
# 查看资源使用情况
kubectl top pods -n medical-ai
kubectl top nodes

# 查看详细资源指标
kubectl exec -it deployment/prometheus -n medical-ai -- \
  promtool query instant 'rate(container_cpu_usage_seconds_total[5m])'
```

#### 2.2 网络延迟分析

```bash
# 测试服务间延迟
kubectl exec -it deployment/medical-ai-backend -n medical-ai -- \
  curl -w "@curl-format.txt" -o /dev/null -s http://postgres:5432
```

## 安全加固

### 1. 网络安全

```bash
# 启用网络策略
kubectl apply -f k8s/security.yaml

# 配置防火墙规则
sudo ufw allow 22/tcp
sudo ufw allow 80/tcp
sudo ufw allow 443/tcp
sudo ufw deny 8000/tcp  # 禁止直接访问后端
```

### 2. 容器安全

```yaml
# 在 deployment 中配置安全上下文
securityContext:
  runAsNonRoot: true
  runAsUser: 1000
  readOnlyRootFilesystem: true
  allowPrivilegeEscalation: false
  capabilities:
    drop:
    - ALL
```

### 3. 密钥管理

```bash
# 使用 Kubernetes Secrets
kubectl create secret generic medical-ai-secrets \
  --from-literal=database-password=your-secure-password \
  --from-literal=jwt-secret=your-jwt-secret \
  -n medical-ai

# 启用密钥加密
echo "encryption-provider-config.yaml" | \
sudo tee /etc/kubernetes/encryption-provider-config.yaml
```

## 扩展和升级

### 1. 水平扩展

```bash
# 扩展后端服务
kubectl scale deployment medical-ai-backend --replicas=5 -n medical-ai

# 配置自动扩展
kubectl autoscale deployment medical-ai-backend \
  --cpu-percent=70 --min=2 --max=10 -n medical-ai
```

### 2. 滚动升级

```bash
# 更新镜像
kubectl set image deployment/medical-ai-backend \
  backend=medical-ai:v2.0.0 -n medical-ai

# 查看升级状态
kubectl rollout status deployment/medical-ai-backend -n medical-ai

# 回滚（如需要）
kubectl rollout undo deployment/medical-ai-backend -n medical-ai
```

## 维护计划

### 1. 定期维护任务

- **每日**: 检查系统状态、备份验证
- **每周**: 安全更新、性能分析
- **每月**: 容量规划、灾难恢复演练
- **每季度**: 安全审计、架构评估

### 2. 监控检查清单

- [ ] 所有服务运行正常
- [ ] 资源使用率在合理范围内
- [ ] 备份任务执行成功
- [ ] 安全告警无异常
- [ ] 性能指标符合预期
- [ ] 日志系统正常工作

---

如有部署相关问题，请参考故障排除章节或联系技术支持团队。