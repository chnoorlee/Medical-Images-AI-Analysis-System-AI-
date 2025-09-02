from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel
from typing import Dict, Any, Optional
from datetime import datetime
import logging
import asyncio
import time

from backend.core.database import get_db_context
from backend.core.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

router = APIRouter(prefix="/health", tags=["健康检查"])

# Pydantic模型
class HealthResponse(BaseModel):
    """健康检查响应模型"""
    status: str
    timestamp: datetime
    version: str
    uptime: Optional[float]
    checks: Dict[str, Dict[str, Any]]

class ComponentHealth(BaseModel):
    """组件健康状态模型"""
    status: str
    response_time: Optional[float]
    message: Optional[str]
    details: Optional[Dict[str, Any]]

# 启动时间记录
start_time = time.time()

@router.get("/", response_model=HealthResponse, summary="基础健康检查")
async def health_check():
    """基础健康检查
    
    返回系统的基本健康状态。
    """
    try:
        current_time = datetime.now()
        uptime = time.time() - start_time
        
        return HealthResponse(
            status="healthy",
            timestamp=current_time,
            version=settings.app_version,
            uptime=uptime,
            checks={}
        )
        
    except Exception as e:
        logger.error(f"健康检查失败: {e}")
        return HealthResponse(
            status="unhealthy",
            timestamp=datetime.now(),
            version=settings.app_version,
            uptime=time.time() - start_time,
            checks={
                "error": {
                    "status": "error",
                    "message": str(e)
                }
            }
        )

@router.get("/detailed", response_model=HealthResponse, summary="详细健康检查")
async def detailed_health_check():
    """详细健康检查
    
    检查所有系统组件的健康状态。
    """
    current_time = datetime.now()
    uptime = time.time() - start_time
    checks = {}
    overall_status = "healthy"
    
    # 数据库健康检查
    db_check = await _check_database()
    checks["database"] = db_check
    if db_check["status"] != "healthy":
        overall_status = "degraded"
    
    # Redis健康检查
    redis_check = await _check_redis()
    checks["redis"] = redis_check
    if redis_check["status"] != "healthy":
        overall_status = "degraded"
    
    # 存储健康检查
    storage_check = await _check_storage()
    checks["storage"] = storage_check
    if storage_check["status"] != "healthy":
        overall_status = "degraded"
    
    # AI模型健康检查
    model_check = await _check_ai_models()
    checks["ai_models"] = model_check
    if model_check["status"] != "healthy":
        overall_status = "degraded"
    
    # 外部服务健康检查
    external_check = await _check_external_services()
    checks["external_services"] = external_check
    if external_check["status"] != "healthy":
        overall_status = "degraded"
    
    return HealthResponse(
        status=overall_status,
        timestamp=current_time,
        version=settings.app_version,
        uptime=uptime,
        checks=checks
    )

@router.get("/database", summary="数据库健康检查")
async def database_health():
    """数据库健康检查
    
    专门检查数据库连接和性能。
    """
    check_result = await _check_database()
    
    if check_result["status"] == "healthy":
        return {"success": True, **check_result}
    else:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=check_result
        )

@router.get("/redis", summary="Redis健康检查")
async def redis_health():
    """Redis健康检查
    
    专门检查Redis连接和性能。
    """
    check_result = await _check_redis()
    
    if check_result["status"] == "healthy":
        return {"success": True, **check_result}
    else:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=check_result
        )

@router.get("/storage", summary="存储健康检查")
async def storage_health():
    """存储健康检查
    
    检查文件存储系统的健康状态。
    """
    check_result = await _check_storage()
    
    if check_result["status"] == "healthy":
        return {"success": True, **check_result}
    else:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=check_result
        )

@router.get("/models", summary="AI模型健康检查")
async def models_health():
    """AI模型健康检查
    
    检查AI模型服务的健康状态。
    """
    check_result = await _check_ai_models()
    
    if check_result["status"] == "healthy":
        return {"success": True, **check_result}
    else:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=check_result
        )

@router.get("/readiness", summary="就绪检查")
async def readiness_check():
    """就绪检查
    
    检查系统是否准备好接收请求。
    """
    try:
        # 检查关键组件
        db_check = await _check_database()
        redis_check = await _check_redis()
        
        if db_check["status"] == "healthy" and redis_check["status"] == "healthy":
            return {
                "status": "ready",
                "timestamp": datetime.now().isoformat(),
                "message": "系统已准备就绪"
            }
        else:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail={
                    "status": "not_ready",
                    "timestamp": datetime.now().isoformat(),
                    "message": "系统尚未准备就绪",
                    "checks": {
                        "database": db_check,
                        "redis": redis_check
                    }
                }
            )
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"就绪检查失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={
                "status": "error",
                "timestamp": datetime.now().isoformat(),
                "message": "就绪检查失败",
                "error": str(e)
            }
        )

@router.get("/liveness", summary="存活检查")
async def liveness_check():
    """存活检查
    
    检查应用程序是否仍在运行。
    """
    try:
        return {
            "status": "alive",
            "timestamp": datetime.now().isoformat(),
            "uptime": time.time() - start_time,
            "message": "应用程序正在运行"
        }
        
    except Exception as e:
        logger.error(f"存活检查失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={
                "status": "dead",
                "timestamp": datetime.now().isoformat(),
                "message": "应用程序存活检查失败",
                "error": str(e)
            }
        )

# 辅助函数
async def _check_database() -> Dict[str, Any]:
    """检查数据库健康状态"""
    start_time_check = time.time()
    
    try:
        with get_db_context() as db:
            # 执行简单查询测试连接
            result = db.execute("SELECT 1 as test")
            test_value = result.fetchone()[0]
            
            if test_value == 1:
                response_time = time.time() - start_time_check
                return {
                    "status": "healthy",
                    "response_time": round(response_time * 1000, 2),  # 毫秒
                    "message": "数据库连接正常",
                    "details": {
                        "connection_pool": "active",
                        "query_test": "passed"
                    }
                }
            else:
                return {
                    "status": "unhealthy",
                    "message": "数据库查询测试失败",
                    "details": {"query_result": test_value}
                }
                
    except Exception as e:
        return {
            "status": "unhealthy",
            "message": f"数据库连接失败: {str(e)}",
            "details": {"error_type": type(e).__name__}
        }

async def _check_redis() -> Dict[str, Any]:
    """检查Redis健康状态"""
    start_time_check = time.time()
    
    try:
        from backend.core.redis import redis_client
        
        # 测试Redis连接
        await redis_client.ping()
        
        # 测试读写操作
        test_key = "health_check_test"
        test_value = "test_value"
        
        await redis_client.set(test_key, test_value, ex=10)  # 10秒过期
        retrieved_value = await redis_client.get(test_key)
        
        if retrieved_value == test_value:
            response_time = time.time() - start_time_check
            return {
                "status": "healthy",
                "response_time": round(response_time * 1000, 2),
                "message": "Redis连接正常",
                "details": {
                    "ping_test": "passed",
                    "read_write_test": "passed"
                }
            }
        else:
            return {
                "status": "unhealthy",
                "message": "Redis读写测试失败",
                "details": {
                    "expected": test_value,
                    "actual": retrieved_value
                }
            }
            
    except Exception as e:
        return {
            "status": "unhealthy",
            "message": f"Redis连接失败: {str(e)}",
            "details": {"error_type": type(e).__name__}
        }

async def _check_storage() -> Dict[str, Any]:
    """检查存储健康状态"""
    start_time_check = time.time()
    
    try:
        from backend.services.storage_service import StorageService
        
        storage_service = StorageService()
        
        # 获取存储信息
        storage_info = await storage_service.get_storage_info()
        
        response_time = time.time() - start_time_check
        
        return {
            "status": "healthy",
            "response_time": round(response_time * 1000, 2),
            "message": "存储服务正常",
            "details": {
                "storage_type": storage_info.get("type", "unknown"),
                "available_space": storage_info.get("available_space"),
                "total_space": storage_info.get("total_space")
            }
        }
        
    except Exception as e:
        return {
            "status": "unhealthy",
            "message": f"存储服务检查失败: {str(e)}",
            "details": {"error_type": type(e).__name__}
        }

async def _check_ai_models() -> Dict[str, Any]:
    """检查AI模型健康状态"""
    start_time_check = time.time()
    
    try:
        from backend.services.model_service import ModelService
        
        model_service = ModelService()
        
        # 获取已加载的模型
        loaded_models = await model_service.get_loaded_models()
        
        response_time = time.time() - start_time_check
        
        return {
            "status": "healthy",
            "response_time": round(response_time * 1000, 2),
            "message": "AI模型服务正常",
            "details": {
                "loaded_models_count": len(loaded_models),
                "loaded_models": [model.get("name") for model in loaded_models]
            }
        }
        
    except Exception as e:
        return {
            "status": "unhealthy",
            "message": f"AI模型服务检查失败: {str(e)}",
            "details": {"error_type": type(e).__name__}
        }

async def _check_external_services() -> Dict[str, Any]:
    """检查外部服务健康状态"""
    start_time_check = time.time()
    
    try:
        # 这里可以检查外部API、第三方服务等
        # 目前返回模拟结果
        
        response_time = time.time() - start_time_check
        
        return {
            "status": "healthy",
            "response_time": round(response_time * 1000, 2),
            "message": "外部服务正常",
            "details": {
                "checked_services": [],
                "all_available": True
            }
        }
        
    except Exception as e:
        return {
            "status": "unhealthy",
            "message": f"外部服务检查失败: {str(e)}",
            "details": {"error_type": type(e).__name__}
        }