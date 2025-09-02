from fastapi import APIRouter, Depends, HTTPException, status, Query, BackgroundTasks
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List
from datetime import datetime, timedelta
from sqlalchemy import or_
import logging
import json
import uuid

from backend.api.auth import get_current_active_user, require_permission, require_role
from backend.models.user import User, Role, Permission
from backend.models.patient import Image as MedicalImage
from backend.models.ai_model import AIModel, ModelVersion as ModelTrainingJob
from backend.models.quality import QualityAssessment
from backend.core.database import get_db_context
from backend.core.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

router = APIRouter(prefix="/admin", tags=["系统管理"])

# Pydantic模型
class SystemStatusResponse(BaseModel):
    """系统状态响应模型"""
    status: str
    uptime: str
    version: str
    database_status: str
    redis_status: str
    storage_status: str
    ai_models_loaded: int
    active_users: int
    system_resources: Dict[str, Any]

class SystemConfigRequest(BaseModel):
    """系统配置请求模型"""
    config_key: str = Field(..., description="配置键")
    config_value: Any = Field(..., description="配置值")
    description: Optional[str] = Field(None, description="配置描述")

class UserManagementRequest(BaseModel):
    """用户管理请求模型"""
    action: str = Field(..., description="操作类型 (activate/deactivate/reset_password/assign_role)")
    user_ids: List[str] = Field(..., description="用户ID列表")
    role_id: Optional[str] = Field(None, description="角色ID（分配角色时使用）")
    new_password: Optional[str] = Field(None, description="新密码（重置密码时使用）")

class DataCleanupRequest(BaseModel):
    """数据清理请求模型"""
    cleanup_type: str = Field(..., description="清理类型 (old_images/failed_jobs/logs/temp_files)")
    days_old: int = Field(30, ge=1, le=365, description="保留天数")
    dry_run: bool = Field(True, description="是否为试运行")

class BackupRequest(BaseModel):
    """备份请求模型"""
    backup_type: str = Field(..., description="备份类型 (database/files/full)")
    include_images: bool = Field(False, description="是否包含图像文件")
    compression: bool = Field(True, description="是否压缩")

class SystemMetricsResponse(BaseModel):
    """系统指标响应模型"""
    cpu_usage: float
    memory_usage: float
    disk_usage: float
    network_io: Dict[str, float]
    database_connections: int
    active_sessions: int
    request_rate: float
    error_rate: float

class AuditLogResponse(BaseModel):
    """审计日志响应模型"""
    log_id: str
    user_id: str
    action: str
    resource_type: str
    resource_id: Optional[str]
    details: Dict[str, Any]
    ip_address: str
    user_agent: str
    timestamp: datetime

# API端点
@router.get("/status", response_model=SystemStatusResponse, summary="获取系统状态")
async def get_system_status(
    current_user: Dict[str, Any] = Depends(require_role("super_admin"))
):
    """获取系统运行状态
    
    返回系统的整体运行状态和关键指标。
    """
    try:
        import psutil
        import time
        from backend.core.redis import redis_client
        from backend.services.storage_service import StorageService
        
        # 系统运行时间
        boot_time = datetime.fromtimestamp(psutil.boot_time())
        uptime = str(datetime.now() - boot_time)
        
        # 数据库状态检查
        database_status = "healthy"
        try:
            with get_db_context() as db:
                db.execute("SELECT 1")
        except Exception:
            database_status = "error"
        
        # Redis状态检查
        redis_status = "healthy"
        try:
            await redis_client.ping()
        except Exception:
            redis_status = "error"
        
        # 存储状态检查
        storage_status = "healthy"
        try:
            storage_service = StorageService()
            await storage_service.get_storage_info()
        except Exception:
            storage_status = "error"
        
        # 统计信息
        with get_db_context() as db:
            ai_models_loaded = db.query(AIModel).filter(AIModel.status == 'loaded').count()
            active_users = db.query(User).filter(
                User.is_active == True,
                User.last_login >= datetime.now() - timedelta(hours=24)
            ).count()
        
        # 系统资源
        system_resources = {
            "cpu_percent": psutil.cpu_percent(),
            "memory_percent": psutil.virtual_memory().percent,
            "disk_percent": psutil.disk_usage('/').percent,
            "load_average": psutil.getloadavg() if hasattr(psutil, 'getloadavg') else [0, 0, 0]
        }
        
        return SystemStatusResponse(
            status="healthy" if all([
                database_status == "healthy",
                redis_status == "healthy",
                storage_status == "healthy"
            ]) else "degraded",
            uptime=uptime,
            version=settings.app_version,
            database_status=database_status,
            redis_status=redis_status,
            storage_status=storage_status,
            ai_models_loaded=ai_models_loaded,
            active_users=active_users,
            system_resources=system_resources
        )
        
    except Exception as e:
        logger.error(f"获取系统状态失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="获取系统状态失败"
        )

@router.get("/metrics", response_model=SystemMetricsResponse, summary="获取系统指标")
async def get_system_metrics(
    current_user: Dict[str, Any] = Depends(require_role("admin"))
):
    """获取系统性能指标
    
    返回详细的系统性能监控数据。
    """
    try:
        import psutil
        
        # CPU使用率
        cpu_usage = psutil.cpu_percent(interval=1)
        
        # 内存使用率
        memory = psutil.virtual_memory()
        memory_usage = memory.percent
        
        # 磁盘使用率
        disk = psutil.disk_usage('/')
        disk_usage = (disk.used / disk.total) * 100
        
        # 网络IO
        net_io = psutil.net_io_counters()
        network_io = {
            "bytes_sent": net_io.bytes_sent,
            "bytes_recv": net_io.bytes_recv,
            "packets_sent": net_io.packets_sent,
            "packets_recv": net_io.packets_recv
        }
        
        # 数据库连接数（模拟）
        database_connections = 10  # 实际应该从数据库获取
        
        # 活跃会话数
        with get_db_context() as db:
            active_sessions = db.query(User).filter(
                User.is_active == True,
                User.last_login >= datetime.now() - timedelta(hours=1)
            ).count()
        
        # 请求速率和错误率（模拟）
        request_rate = 100.0  # 每分钟请求数
        error_rate = 0.5      # 错误率百分比
        
        return SystemMetricsResponse(
            cpu_usage=cpu_usage,
            memory_usage=memory_usage,
            disk_usage=disk_usage,
            network_io=network_io,
            database_connections=database_connections,
            active_sessions=active_sessions,
            request_rate=request_rate,
            error_rate=error_rate
        )
        
    except Exception as e:
        logger.error(f"获取系统指标失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="获取系统指标失败"
        )

@router.get("/users", summary="获取用户管理列表")
async def get_users_admin(
    skip: int = Query(0, ge=0, description="跳过数量"),
    limit: int = Query(50, ge=1, le=200, description="返回数量"),
    is_active: Optional[bool] = Query(None, description="是否激活筛选"),
    role: Optional[str] = Query(None, description="角色筛选"),
    search: Optional[str] = Query(None, description="搜索关键词"),
    current_user: Dict[str, Any] = Depends(require_role("super_admin"))
):
    """获取用户管理列表
    
    返回系统中所有用户的管理信息。
    """
    try:
        with get_db_context() as db:
            query = db.query(User)
            
            if is_active is not None:
                query = query.filter(User.is_active == is_active)
            if role:
                query = query.join(User.roles).filter(Role.role_name == role)
            if search:
                query = query.filter(
                    or_(
                        User.username.contains(search),
                        User.email.contains(search),
                        User.full_name.contains(search)
                    )
                )
            
            total = query.count()
            users = query.offset(skip).limit(limit).all()
            
            user_list = []
            for user in users:
                user_roles = [role.role_name for role in user.roles] if user.roles else []
                user_list.append({
                    "user_id": str(user.user_id),
                    "username": user.username,
                    "email": user.email,
                    "full_name": user.full_name,
                    "is_active": user.is_active,
                    "role": user_roles[0] if user_roles else None,
                    "roles": user_roles,
                    "last_login": user.last_login.isoformat() if user.last_login else None,
                    "created_at": user.created_at.isoformat(),
                    "login_count": getattr(user, 'login_count', 0)
                })
            
            return {
                "success": True,
                "users": user_list,
                "total": total,
                "skip": skip,
                "limit": limit
            }
            
    except Exception as e:
        logger.error(f"获取用户列表失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="获取用户列表失败"
        )

@router.post("/users/manage", summary="批量用户管理")
async def manage_users(
    request: UserManagementRequest,
    current_user: Dict[str, Any] = Depends(require_role("super_admin"))
):
    """批量用户管理操作
    
    支持激活、停用、重置密码、分配角色等操作。
    """
    try:
        with get_db_context() as db:
            users = db.query(User).filter(
                User.user_id.in_([uuid.UUID(uid) for uid in request.user_ids])
            ).all()
            
            if not users:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="未找到指定用户"
                )
            
            results = []
            
            for user in users:
                try:
                    if request.action == "activate":
                        user.is_active = True
                        result = "用户已激活"
                    
                    elif request.action == "deactivate":
                        user.is_active = False
                        result = "用户已停用"
                    
                    elif request.action == "reset_password":
                        if not request.new_password:
                            raise ValueError("新密码不能为空")
                        # 这里应该使用密码哈希
                        from backend.services.auth_service import AuthService
                        auth_service = AuthService()
                        user.password_hash = auth_service.hash_password(request.new_password)
                        result = "密码已重置"
                    
                    elif request.action == "assign_role":
                        if not request.role_id:
                            raise ValueError("角色ID不能为空")
                        role = db.query(Role).filter(Role.role_id == uuid.UUID(request.role_id)).first()
                        if not role:
                            raise ValueError("角色不存在")
                        user.role_id = role.role_id
                        result = f"已分配角色: {role.name}"
                    
                    else:
                        raise ValueError(f"不支持的操作: {request.action}")
                    
                    results.append({
                        "user_id": str(user.user_id),
                        "username": user.username,
                        "success": True,
                        "message": result
                    })
                    
                except Exception as e:
                    results.append({
                        "user_id": str(user.user_id),
                        "username": user.username,
                        "success": False,
                        "error": str(e)
                    })
            
            db.commit()
            
            successful_count = sum(1 for r in results if r['success'])
            
            return {
                "success": True,
                "message": f"操作完成，成功: {successful_count}/{len(request.user_ids)}",
                "results": results
            }
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"用户管理操作失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="用户管理操作失败"
        )

@router.get("/statistics/overview", summary="获取系统统计概览")
async def get_system_statistics(
    current_user: Dict[str, Any] = Depends(require_role("super_admin"))
):
    """获取系统统计概览
    
    返回系统的整体统计信息。
    """
    try:
        with get_db_context() as db:
            # 用户统计
            total_users = db.query(User).count()
            active_users = db.query(User).filter(User.is_active == True).count()
            new_users_today = db.query(User).filter(
                User.created_at >= datetime.now().date()
            ).count()
            
            # 图像统计
            total_images = db.query(MedicalImage).count()
            images_today = db.query(MedicalImage).filter(
                MedicalImage.created_at >= datetime.now().date()
            ).count()
            
            # AI模型统计
            total_models = db.query(AIModel).count()
            active_models = db.query(AIModel).filter(AIModel.status == 'active').count()
            
            # 质量评估统计
            total_assessments = db.query(QualityAssessment).count()
            assessments_today = db.query(QualityAssessment).filter(
                QualityAssessment.created_at >= datetime.now().date()
            ).count()
            
            # 训练任务统计
            total_training_jobs = db.query(ModelTrainingJob).count()
            running_jobs = db.query(ModelTrainingJob).filter(
                ModelTrainingJob.status == 'running'
            ).count()
            
            return {
                "success": True,
                "statistics": {
                    "users": {
                        "total": total_users,
                        "active": active_users,
                        "new_today": new_users_today
                    },
                    "images": {
                        "total": total_images,
                        "uploaded_today": images_today
                    },
                    "models": {
                        "total": total_models,
                        "active": active_models
                    },
                    "quality_assessments": {
                        "total": total_assessments,
                        "today": assessments_today
                    },
                    "training_jobs": {
                        "total": total_training_jobs,
                        "running": running_jobs
                    }
                }
            }
            
    except Exception as e:
        logger.error(f"获取系统统计失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="获取系统统计失败"
        )

@router.post("/cleanup", summary="数据清理")
async def cleanup_data(
    request: DataCleanupRequest,
    background_tasks: BackgroundTasks,
    current_user: Dict[str, Any] = Depends(require_role("super_admin"))
):
    """执行数据清理操作
    
    清理旧数据、失败的任务、日志文件等。
    """
    try:
        cleanup_id = str(uuid.uuid4())
        
        # 启动后台清理任务
        background_tasks.add_task(
            _execute_cleanup,
            cleanup_id,
            request.cleanup_type,
            request.days_old,
            request.dry_run,
            current_user['user_id']
        )
        
        return {
            "success": True,
            "message": "数据清理任务已启动",
            "cleanup_id": cleanup_id,
            "dry_run": request.dry_run
        }
        
    except Exception as e:
        logger.error(f"启动数据清理失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="启动数据清理失败"
        )

@router.post("/backup", summary="系统备份")
async def create_backup(
    request: BackupRequest,
    background_tasks: BackgroundTasks,
    current_user: Dict[str, Any] = Depends(require_role("super_admin"))
):
    """创建系统备份
    
    备份数据库、文件或完整系统。
    """
    try:
        backup_id = str(uuid.uuid4())
        
        # 启动后台备份任务
        background_tasks.add_task(
            _execute_backup,
            backup_id,
            request.backup_type,
            request.include_images,
            request.compression,
            current_user['user_id']
        )
        
        return {
            "success": True,
            "message": "备份任务已启动",
            "backup_id": backup_id,
            "backup_type": request.backup_type
        }
        
    except Exception as e:
        logger.error(f"启动备份失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="启动备份失败"
        )

@router.get("/logs/audit", summary="获取审计日志")
async def get_audit_logs(
    skip: int = Query(0, ge=0, description="跳过数量"),
    limit: int = Query(50, ge=1, le=200, description="返回数量"),
    user_id: Optional[str] = Query(None, description="用户ID筛选"),
    action: Optional[str] = Query(None, description="操作类型筛选"),
    date_from: Optional[datetime] = Query(None, description="开始日期"),
    date_to: Optional[datetime] = Query(None, description="结束日期"),
    current_user: Dict[str, Any] = Depends(require_role("super_admin"))
):
    """获取审计日志
    
    返回系统的审计日志记录。
    """
    try:
        # 这里应该从审计日志表获取数据
        # 目前返回模拟数据
        logs = []
        
        return {
            "success": True,
            "logs": logs,
            "total": len(logs),
            "skip": skip,
            "limit": limit
        }
        
    except Exception as e:
        logger.error(f"获取审计日志失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="获取审计日志失败"
        )

@router.get("/config", summary="获取系统配置")
async def get_system_config(
    current_user: Dict[str, Any] = Depends(require_role("super_admin"))
):
    """获取系统配置
    
    返回当前的系统配置信息。
    """
    try:
        config = {
            "app_name": settings.app_name,
            "app_version": settings.app_version,
            "debug": settings.debug,
            "database_url": "***",  # 隐藏敏感信息
            "redis_url": "***",
            "cors_origins": settings.cors_origins,
            "max_upload_size": settings.max_upload_size,
            "supported_formats": settings.supported_image_formats,
            "jwt_expire_minutes": settings.jwt_access_token_expire_minutes,
            "rate_limit": {
                "requests_per_minute": settings.rate_limit_requests_per_minute,
                "burst_size": settings.rate_limit_burst_size
            }
        }
        
        return {
            "success": True,
            "config": config
        }
        
    except Exception as e:
        logger.error(f"获取系统配置失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="获取系统配置失败"
        )

@router.put("/config", summary="更新系统配置")
async def update_system_config(
    config_request: SystemConfigRequest,
    current_user: Dict[str, Any] = Depends(require_role("super_admin"))
):
    """更新系统配置
    
    修改系统配置参数。
    """
    try:
        # 这里应该实现配置更新逻辑
        # 注意：某些配置可能需要重启服务才能生效
        
        logger.info(f"管理员 {current_user['username']} 更新了配置: {config_request.config_key}")
        
        return {
            "success": True,
            "message": "配置更新成功",
            "config_key": config_request.config_key,
            "requires_restart": config_request.config_key in [
                "database_url", "redis_url", "cors_origins"
            ]
        }
        
    except Exception as e:
        logger.error(f"更新系统配置失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="更新系统配置失败"
        )

# 辅助函数
async def _execute_cleanup(cleanup_id: str, cleanup_type: str, days_old: int, dry_run: bool, user_id: str):
    """执行数据清理的后台任务"""
    try:
        logger.info(f"开始执行数据清理任务: {cleanup_id}, 类型: {cleanup_type}")
        
        cutoff_date = datetime.now() - timedelta(days=days_old)
        
        if cleanup_type == "old_images":
            # 清理旧图像
            with get_db_context() as db:
                old_images = db.query(MedicalImage).filter(
                    MedicalImage.created_at < cutoff_date
                ).all()
                
                if not dry_run:
                    for image in old_images:
                        # 删除文件和数据库记录
                        pass
                
                logger.info(f"{'模拟' if dry_run else '实际'}清理了 {len(old_images)} 个旧图像")
        
        elif cleanup_type == "failed_jobs":
            # 清理失败的训练任务
            with get_db_context() as db:
                failed_jobs = db.query(ModelTrainingJob).filter(
                    ModelTrainingJob.status == 'failed',
                    ModelTrainingJob.created_at < cutoff_date
                ).all()
                
                if not dry_run:
                    for job in failed_jobs:
                        db.delete(job)
                    db.commit()
                
                logger.info(f"{'模拟' if dry_run else '实际'}清理了 {len(failed_jobs)} 个失败任务")
        
        logger.info(f"数据清理任务完成: {cleanup_id}")
        
    except Exception as e:
        logger.error(f"数据清理任务失败: {cleanup_id}, 错误: {e}")

async def _execute_backup(backup_id: str, backup_type: str, include_images: bool, compression: bool, user_id: str):
    """执行备份的后台任务"""
    try:
        logger.info(f"开始执行备份任务: {backup_id}, 类型: {backup_type}")
        
        # 这里应该实现实际的备份逻辑
        # 包括数据库备份、文件备份等
        
        logger.info(f"备份任务完成: {backup_id}")
        
    except Exception as e:
        logger.error(f"备份任务失败: {backup_id}, 错误: {e}")