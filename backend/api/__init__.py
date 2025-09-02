"""API模块

提供RESTful API接口，包括：
- 用户认证和授权
- 图像管理和处理
- AI模型管理和推理
- 质量控制和评估
- 数据管理和统计
"""

from .auth import router as auth_router
from .images import router as images_router
from .models import router as models_router
from .quality import router as quality_router
from .admin import router as admin_router
from .health import router as health_router

__all__ = [
    'auth_router',
    'images_router', 
    'models_router',
    'quality_router',
    'admin_router',
    'health_router'
]