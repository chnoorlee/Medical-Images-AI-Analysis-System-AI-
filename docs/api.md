# Medical AI API 文档

## 概述

Medical AI 系统提供完整的 RESTful API，支持医疗影像分析、用户管理、诊断报告等功能。所有API都遵循REST设计原则，使用JSON格式进行数据交换。

## 基础信息

- **Base URL**: `http://localhost:8000/api/v1`
- **认证方式**: JWT Bearer Token
- **内容类型**: `application/json`
- **字符编码**: UTF-8

## 认证

### 获取访问令牌

```http
POST /auth/login
Content-Type: application/json

{
  "username": "doctor@example.com",
  "password": "password123"
}
```

**响应示例**:
```json
{
  "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "token_type": "bearer",
  "expires_in": 3600,
  "refresh_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "user": {
    "id": "123e4567-e89b-12d3-a456-426614174000",
    "username": "doctor@example.com",
    "role": "doctor",
    "permissions": ["read:images", "write:reports"]
  }
}
```

### 刷新令牌

```http
POST /auth/refresh
Content-Type: application/json

{
  "refresh_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9..."
}
```

### 使用令牌

所有需要认证的API请求都需要在请求头中包含访问令牌：

```http
Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...
```

## 用户管理

### 获取当前用户信息

```http
GET /users/me
Authorization: Bearer {token}
```

**响应示例**:
```json
{
  "id": "123e4567-e89b-12d3-a456-426614174000",
  "username": "doctor@example.com",
  "email": "doctor@example.com",
  "full_name": "Dr. John Smith",
  "role": "doctor",
  "department": "Radiology",
  "license_number": "MD123456",
  "created_at": "2024-01-15T10:30:00Z",
  "last_login": "2024-01-20T14:25:00Z",
  "is_active": true
}
```

### 更新用户信息

```http
PUT /users/me
Authorization: Bearer {token}
Content-Type: application/json

{
  "full_name": "Dr. John Smith Jr.",
  "department": "Emergency Medicine"
}
```

### 修改密码

```http
POST /users/change-password
Authorization: Bearer {token}
Content-Type: application/json

{
  "current_password": "old_password",
  "new_password": "new_password123"
}
```

## 患者管理

### 创建患者

```http
POST /patients
Authorization: Bearer {token}
Content-Type: application/json

{
  "patient_id": "P123456",
  "name": "张三",
  "gender": "male",
  "birth_date": "1980-05-15",
  "phone": "13800138000",
  "email": "zhangsan@example.com",
  "address": "北京市朝阳区",
  "emergency_contact": {
    "name": "李四",
    "phone": "13900139000",
    "relationship": "spouse"
  }
}
```

### 获取患者列表

```http
GET /patients?page=1&size=20&search=张三
Authorization: Bearer {token}
```

**响应示例**:
```json
{
  "items": [
    {
      "id": "456e7890-e89b-12d3-a456-426614174000",
      "patient_id": "P123456",
      "name": "张三",
      "gender": "male",
      "age": 44,
      "phone": "13800138000",
      "last_visit": "2024-01-20T09:30:00Z",
      "total_studies": 5
    }
  ],
  "total": 1,
  "page": 1,
  "size": 20,
  "pages": 1
}
```

### 获取患者详情

```http
GET /patients/{patient_id}
Authorization: Bearer {token}
```

## 医疗影像管理

### 上传影像

```http
POST /images/upload
Authorization: Bearer {token}
Content-Type: multipart/form-data

{
  "file": <DICOM文件>,
  "patient_id": "P123456",
  "study_type": "chest_xray",
  "description": "胸部X光检查",
  "metadata": {
    "acquisition_date": "2024-01-20T10:00:00Z",
    "modality": "CR",
    "body_part": "CHEST"
  }
}
```

**响应示例**:
```json
{
  "id": "789e0123-e89b-12d3-a456-426614174000",
  "filename": "chest_xray_20240120.dcm",
  "patient_id": "P123456",
  "study_type": "chest_xray",
  "file_size": 2048576,
  "upload_status": "completed",
  "created_at": "2024-01-20T10:05:00Z",
  "download_url": "/api/v1/images/789e0123-e89b-12d3-a456-426614174000/download"
}
```

### 获取影像列表

```http
GET /images?patient_id=P123456&study_type=chest_xray&page=1&size=10
Authorization: Bearer {token}
```

### 获取影像详情

```http
GET /images/{image_id}
Authorization: Bearer {token}
```

### 下载影像文件

```http
GET /images/{image_id}/download
Authorization: Bearer {token}
```

### 获取影像缩略图

```http
GET /images/{image_id}/thumbnail?size=256
Authorization: Bearer {token}
```

## AI 推理

### 提交推理任务

```http
POST /ai/inference
Authorization: Bearer {token}
Content-Type: application/json

{
  "image_id": "789e0123-e89b-12d3-a456-426614174000",
  "model_name": "chest_xray_classifier",
  "parameters": {
    "confidence_threshold": 0.8,
    "enable_heatmap": true
  }
}
```

**响应示例**:
```json
{
  "task_id": "abc12345-e89b-12d3-a456-426614174000",
  "status": "pending",
  "created_at": "2024-01-20T10:10:00Z",
  "estimated_completion": "2024-01-20T10:12:00Z"
}
```

### 获取推理结果

```http
GET /ai/inference/{task_id}
Authorization: Bearer {token}
```

**响应示例**:
```json
{
  "task_id": "abc12345-e89b-12d3-a456-426614174000",
  "status": "completed",
  "model_name": "chest_xray_classifier",
  "image_id": "789e0123-e89b-12d3-a456-426614174000",
  "results": {
    "predictions": [
      {
        "class": "pneumonia",
        "confidence": 0.92,
        "bbox": [100, 150, 300, 400]
      },
      {
        "class": "normal",
        "confidence": 0.08
      }
    ],
    "heatmap_url": "/api/v1/ai/inference/abc12345/heatmap",
    "processing_time": 2.5
  },
  "created_at": "2024-01-20T10:10:00Z",
  "completed_at": "2024-01-20T10:12:30Z"
}
```

### 获取可用模型列表

```http
GET /ai/models
Authorization: Bearer {token}
```

**响应示例**:
```json
{
  "models": [
    {
      "name": "chest_xray_classifier",
      "version": "1.2.0",
      "description": "胸部X光分类模型",
      "input_types": ["chest_xray"],
      "output_classes": ["normal", "pneumonia", "tuberculosis"],
      "accuracy": 0.94,
      "status": "active"
    }
  ]
}
```

## 诊断报告

### 创建报告

```http
POST /reports
Authorization: Bearer {token}
Content-Type: application/json

{
  "patient_id": "P123456",
  "image_id": "789e0123-e89b-12d3-a456-426614174000",
  "study_type": "chest_xray",
  "findings": "右下肺野可见片状阴影，考虑肺炎可能",
  "impression": "右下肺炎",
  "recommendations": "建议抗感染治疗，1周后复查",
  "ai_assistance": {
    "task_id": "abc12345-e89b-12d3-a456-426614174000",
    "confidence": 0.92,
    "used_suggestions": true
  }
}
```

### 获取报告列表

```http
GET /reports?patient_id=P123456&status=draft&page=1&size=10
Authorization: Bearer {token}
```

### 获取报告详情

```http
GET /reports/{report_id}
Authorization: Bearer {token}
```

### 更新报告

```http
PUT /reports/{report_id}
Authorization: Bearer {token}
Content-Type: application/json

{
  "findings": "更新后的发现",
  "impression": "更新后的印象",
  "status": "completed"
}
```

### 导出报告

```http
GET /reports/{report_id}/export?format=pdf
Authorization: Bearer {token}
```

## 工作列表

### 获取待处理任务

```http
GET /worklist?status=pending&priority=high&assigned_to=me
Authorization: Bearer {token}
```

**响应示例**:
```json
{
  "items": [
    {
      "id": "task123",
      "patient_id": "P123456",
      "patient_name": "张三",
      "study_type": "chest_xray",
      "priority": "high",
      "status": "pending",
      "assigned_to": "doctor@example.com",
      "created_at": "2024-01-20T08:00:00Z",
      "due_date": "2024-01-20T18:00:00Z"
    }
  ],
  "total": 1,
  "page": 1,
  "size": 10
}
```

### 分配任务

```http
POST /worklist/{task_id}/assign
Authorization: Bearer {token}
Content-Type: application/json

{
  "assigned_to": "doctor2@example.com",
  "priority": "medium",
  "notes": "请优先处理"
}
```

## 通知管理

### 获取通知列表

```http
GET /notifications?unread_only=true&page=1&size=20
Authorization: Bearer {token}
```

### 标记通知为已读

```http
POST /notifications/{notification_id}/read
Authorization: Bearer {token}
```

### 创建通知

```http
POST /notifications
Authorization: Bearer {token}
Content-Type: application/json

{
  "recipient_id": "doctor2@example.com",
  "type": "urgent_case",
  "title": "紧急病例需要会诊",
  "message": "患者张三的胸部X光显示异常，需要紧急会诊",
  "data": {
    "patient_id": "P123456",
    "image_id": "789e0123-e89b-12d3-a456-426614174000"
  }
}
```

## 统计分析

### 获取仪表板数据

```http
GET /analytics/dashboard?period=7d
Authorization: Bearer {token}
```

**响应示例**:
```json
{
  "period": "7d",
  "metrics": {
    "total_patients": 150,
    "total_studies": 320,
    "completed_reports": 280,
    "pending_reports": 40,
    "ai_accuracy": 0.94,
    "average_processing_time": 3.2
  },
  "trends": {
    "daily_studies": [45, 52, 38, 61, 47, 55, 42],
    "ai_usage": [0.85, 0.88, 0.91, 0.89, 0.92, 0.94, 0.93]
  }
}
```

### 获取性能报告

```http
GET /analytics/performance?start_date=2024-01-01&end_date=2024-01-31
Authorization: Bearer {token}
```

## 系统管理

### 获取系统状态

```http
GET /system/health
Authorization: Bearer {token}
```

**响应示例**:
```json
{
  "status": "healthy",
  "timestamp": "2024-01-20T10:30:00Z",
  "services": {
    "database": {
      "status": "healthy",
      "response_time": 12
    },
    "redis": {
      "status": "healthy",
      "response_time": 3
    },
    "ai_service": {
      "status": "healthy",
      "response_time": 150,
      "gpu_usage": 0.65
    }
  }
}
```

### 获取系统配置

```http
GET /system/config
Authorization: Bearer {token}
```

## 错误处理

### 错误响应格式

所有错误响应都遵循统一格式：

```json
{
  "error": {
    "code": "VALIDATION_ERROR",
    "message": "请求参数验证失败",
    "details": [
      {
        "field": "email",
        "message": "邮箱格式不正确"
      }
    ],
    "request_id": "req_123456789"
  }
}
```

### 常见错误码

| 状态码 | 错误码 | 描述 |
|--------|--------|------|
| 400 | VALIDATION_ERROR | 请求参数验证失败 |
| 401 | UNAUTHORIZED | 未授权访问 |
| 403 | FORBIDDEN | 权限不足 |
| 404 | NOT_FOUND | 资源不存在 |
| 409 | CONFLICT | 资源冲突 |
| 422 | UNPROCESSABLE_ENTITY | 无法处理的实体 |
| 429 | RATE_LIMIT_EXCEEDED | 请求频率超限 |
| 500 | INTERNAL_ERROR | 服务器内部错误 |
| 503 | SERVICE_UNAVAILABLE | 服务不可用 |

## 限流和配额

### 请求限制

- **认证接口**: 每分钟最多 10 次请求
- **文件上传**: 每分钟最多 5 次请求
- **AI推理**: 每分钟最多 20 次请求
- **其他接口**: 每分钟最多 100 次请求

### 文件大小限制

- **单个DICOM文件**: 最大 100MB
- **批量上传**: 最大 500MB
- **报告导出**: 最大 50MB

## WebSocket 实时通信

### 连接地址

```
ws://localhost:8000/ws?token={jwt_token}
```

### 消息格式

```json
{
  "type": "notification",
  "data": {
    "id": "notif123",
    "title": "新的推理结果",
    "message": "您的AI推理任务已完成",
    "timestamp": "2024-01-20T10:30:00Z"
  }
}
```

### 支持的消息类型

- `notification`: 系统通知
- `task_update`: 任务状态更新
- `ai_result`: AI推理结果
- `chat_message`: 会诊聊天消息

## SDK 和示例

### Python SDK 示例

```python
from medical_ai_sdk import MedicalAIClient

# 初始化客户端
client = MedicalAIClient(
    base_url="http://localhost:8000/api/v1",
    username="doctor@example.com",
    password="password123"
)

# 上传影像
with open("chest_xray.dcm", "rb") as f:
    image = client.upload_image(
        file=f,
        patient_id="P123456",
        study_type="chest_xray"
    )

# 提交AI推理
task = client.submit_inference(
    image_id=image.id,
    model_name="chest_xray_classifier"
)

# 等待结果
result = client.wait_for_result(task.task_id)
print(f"预测结果: {result.predictions}")
```

### JavaScript SDK 示例

```javascript
import { MedicalAIClient } from 'medical-ai-sdk';

// 初始化客户端
const client = new MedicalAIClient({
  baseURL: 'http://localhost:8000/api/v1',
  username: 'doctor@example.com',
  password: 'password123'
});

// 获取患者列表
const patients = await client.getPatients({
  page: 1,
  size: 20,
  search: '张三'
});

// 创建诊断报告
const report = await client.createReport({
  patientId: 'P123456',
  imageId: 'img123',
  findings: '右下肺野可见片状阴影',
  impression: '右下肺炎'
});
```

## 版本控制

API版本通过URL路径进行控制：

- **v1**: `/api/v1/` - 当前稳定版本
- **v2**: `/api/v2/` - 下一个主要版本（开发中）

### 版本兼容性

- 向后兼容的更改会在同一版本内发布
- 破坏性更改会发布新的主要版本
- 旧版本会维护至少 12 个月

## 更多信息

- **API测试工具**: 访问 `/docs` 查看交互式API文档
- **Postman集合**: 下载 [Medical AI API.postman_collection.json](./Medical_AI_API.postman_collection.json)
- **OpenAPI规范**: 下载 [openapi.json](./openapi.json)

---

如有任何问题或建议，请联系开发团队或提交 GitHub Issue。