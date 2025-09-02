from fastapi import APIRouter, Depends, HTTPException, status, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, EmailStr, Field
from typing import Optional, Dict, Any, List
from datetime import datetime, timedelta
import logging
import uuid

from backend.services.auth_service import AuthService
from backend.models.user import User, Role
from backend.core.database import get_db_context

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/auth", tags=["认证"])
security = HTTPBearer()
auth_service = AuthService()

# Pydantic模型
class UserRegister(BaseModel):
    """用户注册请求模型"""
    username: str = Field(..., min_length=3, max_length=50, description="用户名")
    email: EmailStr = Field(..., description="邮箱地址")
    password: str = Field(..., min_length=6, max_length=128, description="密码")
    full_name: str = Field(..., min_length=1, max_length=100, description="姓名")
    department: Optional[str] = Field(None, max_length=100, description="科室")
    role_name: str = Field("viewer", description="角色名称")
    phone: Optional[str] = Field(None, max_length=20, description="电话")

class UserLogin(BaseModel):
    """用户登录请求模型"""
    username: str = Field(..., description="用户名或邮箱")
    password: str = Field(..., description="密码")
    remember_me: bool = Field(False, description="记住我")

class PasswordChange(BaseModel):
    """密码修改请求模型"""
    old_password: str = Field(..., description="旧密码")
    new_password: str = Field(..., min_length=6, max_length=128, description="新密码")

class PasswordReset(BaseModel):
    """密码重置请求模型"""
    email: EmailStr = Field(..., description="邮箱地址")

class PasswordResetConfirm(BaseModel):
    """密码重置确认请求模型"""
    token: str = Field(..., description="重置令牌")
    new_password: str = Field(..., min_length=6, max_length=128, description="新密码")

class UserProfile(BaseModel):
    """用户资料更新请求模型"""
    full_name: Optional[str] = Field(None, max_length=100, description="姓名")
    email: Optional[EmailStr] = Field(None, description="邮箱地址")
    department: Optional[str] = Field(None, max_length=100, description="科室")
    phone: Optional[str] = Field(None, max_length=20, description="电话")
    bio: Optional[str] = Field(None, max_length=500, description="个人简介")

class TokenResponse(BaseModel):
    """令牌响应模型"""
    access_token: str
    token_type: str = "bearer"
    expires_in: int
    refresh_token: Optional[str] = None
    user_info: Dict[str, Any]

class UserResponse(BaseModel):
    """用户信息响应模型"""
    user_id: str
    username: str
    email: str
    full_name: str
    department: Optional[str]
    phone: Optional[str]
    bio: Optional[str]
    role: str
    permissions: List[str]
    is_active: bool
    created_at: datetime
    last_login: Optional[datetime]

# 依赖注入
async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)) -> dict:
    """获取当前用户"""
    try:
        token = credentials.credentials
        print(f"[DEBUG] 验证令牌: {token[:20]}...")
        logger.info(f"验证令牌: {token[:20]}...")
        user_info = await auth_service.verify_token(token)
        
        print(f"[DEBUG] 令牌验证结果: {user_info}")
        logger.info(f"令牌验证结果: {user_info}")
        
        if not user_info or not user_info.get('valid'):
            print(f"[DEBUG] 令牌验证失败: {user_info.get('error', '无效的访问令牌')}")
            logger.error(f"令牌验证失败: {user_info.get('error', '无效的访问令牌')}")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail=user_info.get('error', '无效的访问令牌'),
                headers={"WWW-Authenticate": "Bearer"},
            )
        
        # 直接返回验证结果中的用户信息，避免会话绑定问题
        logger.info(f"用户认证成功: {user_info['username']}")
        return user_info['user_data']
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"用户认证失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="认证失败",
            headers={"WWW-Authenticate": "Bearer"},
        )

async def get_current_active_user(current_user: dict = Depends(get_current_user)) -> dict:
    """获取当前活跃用户"""
    if not current_user.get('is_active'):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="用户账户已被禁用"
        )
    return current_user

def require_permission(permission: str):
    """权限检查装饰器"""
    async def permission_checker(current_user: Dict[str, Any] = Depends(get_current_active_user)):
        user_permissions = current_user.get('permissions', [])
        if permission not in user_permissions and 'admin' not in user_permissions:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"需要权限: {permission}"
            )
        return current_user
    return permission_checker

def require_role(role: str):
    """角色检查装饰器"""
    async def role_checker(current_user: Dict[str, Any] = Depends(get_current_active_user)):
        user_role = current_user.get('role')
        if user_role != role and user_role != 'super_admin':
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"需要角色: {role}"
            )
        return current_user
    return role_checker

# API端点
@router.post("/register", response_model=Dict[str, Any], summary="用户注册")
async def register(user_data: UserRegister, request: Request):
    """用户注册
    
    创建新用户账户，需要提供用户名、邮箱、密码等基本信息。
    默认角色为viewer，可以指定其他角色。
    """
    try:
        # 获取客户端IP
        client_ip = request.client.host
        
        result = await auth_service.register_user(
            username=user_data.username,
            email=user_data.email,
            password=user_data.password,
            full_name=user_data.full_name,
            department=user_data.department,
            role_name=user_data.role_name,
            phone=user_data.phone,
            client_ip=client_ip
        )
        
        if result['success']:
            return {
                "success": True,
                "message": "用户注册成功",
                "user_id": result['user_id']
            }
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=result['error']
            )
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"用户注册失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="注册失败，请稍后重试"
        )

@router.post("/login", response_model=TokenResponse, summary="用户登录")
async def login(user_data: UserLogin, request: Request):
    """用户登录
    
    使用用户名/邮箱和密码进行登录，返回访问令牌。
    """
    print(f"[LOGIN ENDPOINT] Received login request for username: {user_data.username}")
    try:
        # 获取客户端IP和User-Agent
        client_ip = request.client.host
        user_agent = request.headers.get("user-agent", "")
        
        result = await auth_service.login(
            username=user_data.username,
            password=user_data.password,
            client_ip=client_ip,
            user_agent=user_agent,
            remember_me=user_data.remember_me
        )
        
        if result['success']:
            return TokenResponse(
                access_token=result['access_token'],
                expires_in=result['expires_in'],
                refresh_token=result.get('refresh_token'),
                user_info=result['user']
            )
        else:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail=result['error']
            )
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"用户登录失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="登录失败，请稍后重试"
        )

@router.post("/logout", summary="用户登出")
async def logout(current_user: Dict[str, Any] = Depends(get_current_active_user)):
    """用户登出
    
    注销当前用户会话，使访问令牌失效。
    """
    try:
        result = await auth_service.logout(current_user['user_id'])
        
        if result['success']:
            return {
                "success": True,
                "message": "登出成功"
            }
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=result['error']
            )
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"用户登出失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="登出失败，请稍后重试"
        )

@router.get("/me", response_model=UserResponse, summary="获取当前用户信息")
async def get_current_user_info(current_user: dict = Depends(get_current_active_user)):
    """获取当前用户信息
    
    返回当前登录用户的详细信息。
    """
    print("[DEBUG] /me endpoint called!")
    print(f"[DEBUG] current_user: {current_user}")
    try:
        from datetime import datetime, timezone
        return UserResponse(
            user_id=current_user.get('user_id'),
            username=current_user.get('username'),
            email=current_user.get('email'),
            full_name=current_user.get('full_name'),
            department=current_user.get('department'),
            phone=current_user.get('phone'),
            bio=current_user.get('bio'),
            role="admin",  # 临时硬编码，后续需要从token中获取
            permissions=[],  # 临时空列表，后续需要实现权限查询
            is_active=current_user.get('is_active'),
                created_at=datetime.now(timezone.utc),  # 临时使用当前时间
                 last_login=None   # 可选字段，可以为None
            )
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"获取用户信息失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="获取用户信息失败"
        )

@router.put("/me", response_model=Dict[str, Any], summary="更新用户资料")
async def update_profile(profile_data: UserProfile, 
                        current_user: Dict[str, Any] = Depends(get_current_active_user)):
    """更新用户资料
    
    更新当前用户的个人信息。
    """
    try:
        result = await auth_service.update_user_profile(
            user_id=current_user['user_id'],
            profile_data=profile_data.dict(exclude_unset=True)
        )
        
        if result['success']:
            return {
                "success": True,
                "message": "用户资料更新成功"
            }
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=result['error']
            )
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"更新用户资料失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="更新用户资料失败"
        )

@router.post("/change-password", summary="修改密码")
async def change_password(password_data: PasswordChange,
                         current_user: Dict[str, Any] = Depends(get_current_active_user)):
    """修改密码
    
    修改当前用户的登录密码。
    """
    try:
        result = await auth_service.change_password(
            user_id=current_user['user_id'],
            old_password=password_data.old_password,
            new_password=password_data.new_password
        )
        
        if result['success']:
            return {
                "success": True,
                "message": "密码修改成功"
            }
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=result['error']
            )
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"修改密码失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="修改密码失败"
        )

@router.post("/reset-password", summary="请求密码重置")
async def reset_password(reset_data: PasswordReset):
    """请求密码重置
    
    发送密码重置邮件到指定邮箱。
    """
    try:
        result = await auth_service.request_password_reset(reset_data.email)
        
        if result['success']:
            return {
                "success": True,
                "message": "密码重置邮件已发送"
            }
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=result['error']
            )
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"请求密码重置失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="请求密码重置失败"
        )

@router.post("/reset-password/confirm", summary="确认密码重置")
async def confirm_password_reset(reset_data: PasswordResetConfirm):
    """确认密码重置
    
    使用重置令牌设置新密码。
    """
    try:
        result = await auth_service.confirm_password_reset(
            token=reset_data.token,
            new_password=reset_data.new_password
        )
        
        if result['success']:
            return {
                "success": True,
                "message": "密码重置成功"
            }
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=result['error']
            )
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"确认密码重置失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="确认密码重置失败"
        )

@router.post("/refresh", response_model=TokenResponse, summary="刷新访问令牌")
async def refresh_token(refresh_token: str):
    """刷新访问令牌
    
    使用刷新令牌获取新的访问令牌。
    """
    try:
        result = await auth_service.refresh_access_token(refresh_token)
        
        if result['success']:
            return TokenResponse(
                access_token=result['access_token'],
                expires_in=result['expires_in'],
                refresh_token=result.get('refresh_token'),
                user_info=result['user_info']
            )
        else:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail=result['error']
            )
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"刷新令牌失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="刷新令牌失败"
        )

@router.get("/permissions", summary="获取用户权限")
async def get_user_permissions(current_user: Dict[str, Any] = Depends(get_current_active_user)):
    """获取用户权限
    
    返回当前用户的所有权限列表。
    """
    try:
        permissions = await auth_service.get_user_permissions(current_user['user_id'])
        
        return {
            "success": True,
            "permissions": permissions,
            "role": current_user.get('role')
        }
        
    except Exception as e:
        logger.error(f"获取用户权限失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="获取用户权限失败"
        )

@router.get("/sessions", summary="获取用户会话")
async def get_user_sessions(current_user: Dict[str, Any] = Depends(get_current_active_user)):
    """获取用户会话
    
    返回当前用户的所有活跃会话。
    """
    try:
        sessions = await auth_service.get_user_sessions(current_user['user_id'])
        
        return {
            "success": True,
            "sessions": sessions
        }
        
    except Exception as e:
        logger.error(f"获取用户会话失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="获取用户会话失败"
        )

@router.delete("/sessions/{session_id}", summary="删除用户会话")
async def delete_user_session(session_id: str,
                             current_user: Dict[str, Any] = Depends(get_current_active_user)):
    """删除用户会话
    
    删除指定的用户会话。
    """
    try:
        result = await auth_service.delete_user_session(
            user_id=current_user['user_id'],
            session_id=session_id
        )
        
        if result['success']:
            return {
                "success": True,
                "message": "会话删除成功"
            }
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=result['error']
            )
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"删除用户会话失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="删除用户会话失败"
        )

# 管理员专用端点
@router.get("/users", summary="获取用户列表")
async def get_users(skip: int = 0, limit: int = 100,
                   current_user: Dict[str, Any] = Depends(require_permission("user_management"))):
    """获取用户列表
    
    管理员获取系统中所有用户的列表。
    """
    try:
        with get_db_context() as db:
            users = db.query(User).offset(skip).limit(limit).all()
            total = db.query(User).count()
            
            user_list = []
            for user in users:
                user_list.append({
                    "user_id": str(user.user_id),
                    "username": user.username,
                    "email": user.email,
                    "full_name": user.full_name,
                    "department": user.department,
                    "role": user.role.name if user.role else None,
                    "is_active": user.is_active,
                    "created_at": user.created_at.isoformat(),
                    "last_login": user.last_login.isoformat() if user.last_login else None
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

@router.put("/users/{user_id}/status", summary="更新用户状态")
async def update_user_status(user_id: str, is_active: bool,
                           current_user: Dict[str, Any] = Depends(require_permission("user_management"))):
    """更新用户状态
    
    管理员启用或禁用用户账户。
    """
    try:
        result = await auth_service.update_user_status(user_id, is_active)
        
        if result['success']:
            return {
                "success": True,
                "message": f"用户状态已更新为{'启用' if is_active else '禁用'}"
            }
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=result['error']
            )
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"更新用户状态失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="更新用户状态失败"
        )

@router.put("/users/{user_id}/role", summary="更新用户角色")
async def update_user_role(user_id: str, role_name: str,
                         current_user: Dict[str, Any] = Depends(require_permission("user_management"))):
    """更新用户角色
    
    管理员修改用户的角色。
    """
    try:
        result = await auth_service.update_user_role(user_id, role_name)
        
        if result['success']:
            return {
                "success": True,
                "message": f"用户角色已更新为{role_name}"
            }
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=result['error']
            )
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"更新用户角色失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="更新用户角色失败"
        )