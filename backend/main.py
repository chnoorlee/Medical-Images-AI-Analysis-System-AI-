from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from contextlib import asynccontextmanager
import logging
import logging.config
import time
import uuid
from typing import Callable

from backend.core.config import settings, LOGGING_CONFIG
from backend.core.database import init_database, check_database_connection
from backend.core.init_data import init_base_data

# 配置日志
logging.config.dictConfig(LOGGING_CONFIG)
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    # 启动时执行
    logger.info("启动 Medical AI Platform...")
    
    try:
        # 初始化数据库
        logger.info("初始化数据库...")
        init_database()
        
        # 检查数据库连接
        if not check_database_connection():
            raise Exception("数据库连接失败")
        
        # 初始化基础数据
        logger.info("初始化基础数据...")
        init_base_data()
        
        logger.info("Medical AI Platform 启动完成")
        
    except Exception as e:
        logger.error(f"应用启动失败: {e}")
        raise
    
    yield
    
    # 关闭时执行
    logger.info("关闭 Medical AI Platform...")

# 创建FastAPI应用
app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="医学AI平台后端API",
    docs_url="/docs" if settings.debug else None,
    redoc_url="/redoc" if settings.debug else None,
    lifespan=lifespan
)

# 添加CORS中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=settings.cors_credentials,
    allow_methods=settings.cors_methods,
    allow_headers=settings.cors_headers,
)

# 添加受信任主机中间件
if settings.environment == "production":
    app.add_middleware(
        TrustedHostMiddleware,
        allowed_hosts=settings.allowed_hosts
    )

# 请求ID中间件
@app.middleware("http")
async def add_request_id(request: Request, call_next: Callable):
    """为每个请求添加唯一ID"""
    request_id = str(uuid.uuid4())
    request.state.request_id = request_id
    
    # 添加到响应头
    response = await call_next(request)
    response.headers["X-Request-ID"] = request_id
    
    return response

# 请求日志中间件
@app.middleware("http")
async def log_requests(request: Request, call_next: Callable):
    """记录请求日志"""
    start_time = time.time()
    request_id = getattr(request.state, 'request_id', 'unknown')
    
    # 记录请求开始
    print(f"[REQUEST START] ID: {request_id}, Method: {request.method}, URL: {request.url}, Client: {request.client.host if request.client else 'unknown'}")
    logger.info(
        f"Request started - ID: {request_id}, Method: {request.method}, "
        f"URL: {request.url}, Client: {request.client.host if request.client else 'unknown'}"
    )
    
    try:
        response = await call_next(request)
        
        # 计算处理时间
        process_time = time.time() - start_time
        
        # 记录请求完成
        print(f"[REQUEST END] ID: {request_id}, Status: {response.status_code}, Time: {process_time:.3f}s")
        logger.info(
            f"Request completed - ID: {request_id}, Status: {response.status_code}, "
            f"Time: {process_time:.3f}s"
        )
        
        # 添加处理时间到响应头
        response.headers["X-Process-Time"] = str(process_time)
        
        return response
        
    except Exception as e:
        # 记录请求错误
        process_time = time.time() - start_time
        logger.error(
            f"Request failed - ID: {request_id}, Error: {str(e)}, "
            f"Time: {process_time:.3f}s"
        )
        raise

# 全局异常处理器
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """HTTP异常处理"""
    request_id = getattr(request.state, 'request_id', 'unknown')
    
    logger.warning(
        f"HTTP Exception - ID: {request_id}, Status: {exc.status_code}, "
        f"Detail: {exc.detail}"
    )
    
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": {
                "code": exc.status_code,
                "message": exc.detail,
                "request_id": request_id,
                "timestamp": time.time()
            }
        }
    )

@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """通用异常处理"""
    request_id = getattr(request.state, 'request_id', 'unknown')
    
    logger.error(
        f"Unhandled Exception - ID: {request_id}, Type: {type(exc).__name__}, "
        f"Message: {str(exc)}",
        exc_info=True
    )
    
    return JSONResponse(
        status_code=500,
        content={
            "error": {
                "code": 500,
                "message": "内部服务器错误" if settings.environment == "production" else str(exc),
                "request_id": request_id,
                "timestamp": time.time()
            }
        }
    )

# 健康检查端点
@app.get("/health")
async def health_check():
    """健康检查"""
    try:
        # 检查数据库连接
        db_status = check_database_connection()
        
        return {
            "status": "healthy" if db_status else "unhealthy",
            "timestamp": time.time(),
            "version": settings.app_version,
            "environment": settings.environment,
            "database": "connected" if db_status else "disconnected"
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return JSONResponse(
            status_code=503,
            content={
                "status": "unhealthy",
                "timestamp": time.time(),
                "error": str(e)
            }
        )

# 根路径
@app.get("/")
async def root():
    """根路径"""
    return {
        "message": "Welcome to Medical AI Platform API",
        "version": settings.app_version,
        "docs": "/docs" if settings.debug else "Documentation not available in production",
        "health": "/health"
    }

# API信息端点
@app.get("/api/info")
async def api_info():
    """API信息"""
    return {
        "name": settings.app_name,
        "version": settings.app_version,
        "environment": settings.environment,
        "debug": settings.debug,
        "timestamp": time.time()
    }

# 系统信息端点（前端需要）
@app.get("/api/system/info")
async def system_info():
    """系统信息"""
    return {
        "name": settings.app_name,
        "version": settings.app_version,
        "environment": settings.environment,
        "debug": settings.debug,
        "timestamp": time.time(),
        "status": "running"
    }

# 挂载静态文件（如果需要）
if settings.debug:
    try:
        app.mount("/static", StaticFiles(directory="static"), name="static")
    except RuntimeError:
        # 静态目录不存在时忽略
        pass

# 添加API路由
from backend.api.auth import router as auth_router
from backend.api.admin import router as admin_router
from backend.api.health import router as health_router
from backend.api.images import router as images_router
from backend.api.models import router as models_router
from backend.api.quality import router as quality_router

app.include_router(auth_router, prefix="/api", tags=["认证"])
app.include_router(admin_router, prefix="/api", tags=["系统管理"])
app.include_router(health_router, prefix="/api", tags=["健康检查"])
app.include_router(images_router, prefix="/api", tags=["图像管理"])
app.include_router(models_router, prefix="/api", tags=["模型管理"])
app.include_router(quality_router, prefix="/api", tags=["质量控制"])

if __name__ == "__main__":
    import uvicorn
    
    # 运行应用
    uvicorn.run(
        "backend.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.reload,
        log_config=LOGGING_CONFIG,
        access_log=True
    )