import jwt
import bcrypt
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Any, Tuple
import logging
import secrets
import hashlib
from pathlib import Path
import json
import uuid
from sqlalchemy.orm import Session
from sqlalchemy import and_, or_

from backend.models.user import User, Role, Permission, UserSession
from backend.core.database import get_db_context
from backend.core.config import settings

logger = logging.getLogger(__name__)

class AuthService:
    """认证服务
    
    提供用户认证、授权和会话管理功能，包括：
    - 用户注册和登录
    - 密码加密和验证
    - JWT令牌生成和验证
    - 权限检查
    - 会话管理
    - 密码重置
    """
    
    def __init__(self):
        self.secret_key = settings.secret_key
        self.algorithm = settings.algorithm
        self.access_token_expire_minutes = settings.access_token_expire_minutes
        self.refresh_token_expire_days = settings.refresh_token_expire_days
        self.password_min_length = 8
        self.max_login_attempts = 5
        self.lockout_duration_minutes = 30
    
    # 用户注册和登录
    async def register_user(self, username: str, email: str, password: str, 
                          first_name: str, last_name: str, 
                          role_name: str = 'viewer') -> Dict[str, Any]:
        """注册新用户
        
        Args:
            username: 用户名
            email: 邮箱
            password: 密码
            first_name: 名
            last_name: 姓
            role_name: 角色名称
            
        Returns:
            注册结果
        """
        try:
            # 验证输入
            validation_result = self._validate_registration_data(
                username, email, password, first_name, last_name
            )
            if not validation_result['valid']:
                return {
                    'success': False,
                    'error': validation_result['error']
                }
            
            with get_db_context() as db:
                # 检查用户是否已存在
                existing_user = db.query(User).filter(
                    or_(User.username == username, User.email == email)
                ).first()
                
                if existing_user:
                    return {
                        'success': False,
                        'error': '用户名或邮箱已存在'
                    }
                
                # 获取角色
                role = db.query(Role).filter(Role.name == role_name).first()
                if not role:
                    return {
                        'success': False,
                        'error': f'角色 {role_name} 不存在'
                    }
                
                # 加密密码
                password_hash = self._hash_password(password)
                
                # 创建用户
                user = User(
                    user_id=uuid.uuid4(),
                    username=username,
                    email=email,
                    hashed_password=password_hash,
                    full_name=f"{first_name} {last_name}",
                    is_active=True,
                    is_verified=False,
                    created_at=datetime.now(timezone.utc)
                )
                
                db.add(user)
                db.commit()
                db.refresh(user)
                
                logger.info(f"用户注册成功: {username}")
                return {
                    'success': True,
                    'user_id': str(user.user_id),
                    'message': '用户注册成功'
                }
                
        except Exception as e:
            logger.error(f"用户注册失败: {e}")
            return {
                'success': False,
                'error': '注册失败，请稍后重试'
            }
    
    async def login(self, username: str, password: str, 
                   client_ip: Optional[str] = None,
                   user_agent: Optional[str] = None,
                   remember_me: bool = False) -> Dict[str, Any]:
        """用户登录
        
        Args:
            username: 用户名或邮箱
            password: 密码
            client_ip: 客户端IP地址
            user_agent: 用户代理
            remember_me: 是否记住登录状态
            
        Returns:
            登录结果
        """
        return await self.login_user(username, password, client_ip, user_agent)
    
    async def login_user(self, username: str, password: str, 
                        ip_address: Optional[str] = None,
                        user_agent: Optional[str] = None) -> Dict[str, Any]:
        """用户登录
        
        Args:
            username: 用户名或邮箱
            password: 密码
            ip_address: IP地址
            user_agent: 用户代理
            
        Returns:
            登录结果
        """
        try:
            with get_db_context() as db:
                # 查找用户
                user = db.query(User).filter(
                    or_(User.username == username, User.email == username)
                ).first()
                
                if not user:
                    return {
                        'success': False,
                        'error': '用户名或密码错误'
                    }
                
                # 检查账户状态
                if not user.is_active:
                    return {
                        'success': False,
                        'error': '账户已被禁用'
                    }
                
                # 检查账户锁定状态
                if self._is_account_locked(user):
                    return {
                        'success': False,
                        'error': f'账户已锁定，请在{self.lockout_duration_minutes}分钟后重试'
                    }
                
                # 验证密码
                if not self._verify_password(password, user.hashed_password):
                    # 记录失败尝试
                    self._record_login_attempt(user, False, ip_address)
                    return {
                        'success': False,
                        'error': '用户名或密码错误'
                    }
                
                # 生成令牌
                access_token = self._generate_access_token(user)
                refresh_token = self._generate_refresh_token(user)
                
                # 创建会话
                session = UserSession(
                    session_id=uuid.uuid4(),
                    user_id=user.user_id,
                    access_token=access_token,
                    refresh_token=refresh_token,
                    ip_address=ip_address,
                    user_agent=user_agent,
                    created_at=datetime.now(timezone.utc),
                    expires_at=datetime.now(timezone.utc) + timedelta(minutes=self.access_token_expire_minutes),
                    is_active=True
                )
                
                db.add(session)
                
                # 更新用户最后登录时间
                user.last_login = datetime.now(timezone.utc)
                user.failed_login_attempts = 0  # 重置失败次数
                
                # 记录成功登录
                self._record_login_attempt(user, True, ip_address)
                
                db.commit()
                
                logger.info(f"用户登录成功: {user.username}")
                return {
                    'success': True,
                    'access_token': access_token,
                    'refresh_token': refresh_token,
                    'token_type': 'bearer',
                    'expires_in': self.access_token_expire_minutes * 60,
                    'user': {
                        'user_id': str(user.user_id),
                        'username': user.username,
                        'email': user.email,
                        'full_name': user.full_name,
                        'roles': [role.role_name for role in user.roles] if user.roles else []
                    }
                }
                
        except Exception as e:
            logger.error(f"用户登录失败: {e}")
            return {
                'success': False,
                'error': '登录失败，请稍后重试'
            }
    
    async def logout_user(self, access_token: str) -> Dict[str, Any]:
        """用户登出
        
        Args:
            access_token: 访问令牌
            
        Returns:
            登出结果
        """
        try:
            with get_db_context() as db:
                # 查找会话
                session = db.query(UserSession).filter(
                    UserSession.access_token == access_token,
                    UserSession.is_active == True
                ).first()
                
                if session:
                    session.is_active = False
                    session.logged_out_at = datetime.now(timezone.utc)
                    db.commit()
                    
                    logger.info(f"用户登出成功: {session.user.username}")
                
                return {
                    'success': True,
                    'message': '登出成功'
                }
                
        except Exception as e:
            logger.error(f"用户登出失败: {e}")
            return {
                'success': False,
                'error': '登出失败'
            }
    
    # 令牌管理
    def _generate_access_token(self, user: User) -> str:
        """生成访问令牌"""
        try:
            now = datetime.now(timezone.utc)
            exp_time = now + timedelta(minutes=self.access_token_expire_minutes)
            
            payload = {
                'user_id': str(user.user_id),
                'username': user.username,
                'role': user.roles[0].role_name if user.roles else None,
                'exp': int(exp_time.timestamp()),
                'iat': int(now.timestamp()),
                'type': 'access'
            }
            
            return jwt.encode(payload, self.secret_key, algorithm=self.algorithm)
            
        except Exception as e:
            logger.error(f"生成访问令牌失败: {e}")
            raise
    
    def _generate_refresh_token(self, user: User) -> str:
        """生成刷新令牌"""
        try:
            now = datetime.now(timezone.utc)
            exp_time = now + timedelta(days=self.refresh_token_expire_days)
            
            payload = {
                'user_id': str(user.user_id),
                'exp': int(exp_time.timestamp()),
                'iat': int(now.timestamp()),
                'type': 'refresh'
            }
            
            return jwt.encode(payload, self.secret_key, algorithm=self.algorithm)
            
        except Exception as e:
            logger.error(f"生成刷新令牌失败: {e}")
            raise
    
    async def verify_token(self, token: str) -> Dict[str, Any]:
        """验证令牌
        
        Args:
            token: JWT令牌
            
        Returns:
            验证结果
        """
        try:
            # 解码令牌
            payload = jwt.decode(token, self.secret_key, algorithms=[self.algorithm])
            
            # 检查令牌类型
            if payload.get('type') != 'access':
                return {
                    'valid': False,
                    'error': '无效的令牌类型'
                }
            
            # 检查用户是否存在且活跃
            with get_db_context() as db:
                user = db.query(User).filter(
                    User.user_id == uuid.UUID(payload['user_id']),
                    User.is_active == True
                ).first()
                
                if not user:
                    return {
                        'valid': False,
                        'error': '用户不存在或已被禁用'
                    }
                
                # 检查会话是否有效
                session = db.query(UserSession).filter(
                    UserSession.access_token == token,
                    UserSession.is_active == True
                ).first()
                
                if not session:
                    return {
                        'valid': False,
                        'error': '会话无效或已过期'
                    }
                
                # 获取用户角色和权限
                user_roles = [role.role_name for role in user.roles] if user.roles else []
                user_permissions = []
                for role in user.roles:
                    for permission in role.permissions:
                        if permission.permission_name not in user_permissions:
                            user_permissions.append(permission.permission_name)
                
                return {
                    'valid': True,
                    'user_id': payload['user_id'],
                    'username': payload['username'],
                    'role': payload['role'],
                    'user_data': {
                        'user_id': str(user.user_id),
                        'username': user.username,
                        'email': user.email,
                        'full_name': user.full_name,
                        'is_active': user.is_active,
                        'is_verified': user.is_verified,
                        'role': user_roles[0] if user_roles else None,
                        'roles': user_roles,
                        'permissions': user_permissions
                    }
                }
                
        except jwt.ExpiredSignatureError:
            return {
                'valid': False,
                'error': '令牌已过期'
            }
        except jwt.InvalidTokenError:
            return {
                'valid': False,
                'error': '无效的令牌'
            }
        except Exception as e:
            logger.error(f"令牌验证失败: {e}")
            return {
                'valid': False,
                'error': '令牌验证失败'
            }
    
    async def refresh_token(self, refresh_token: str) -> Dict[str, Any]:
        """刷新访问令牌
        
        Args:
            refresh_token: 刷新令牌
            
        Returns:
            刷新结果
        """
        try:
            # 解码刷新令牌
            payload = jwt.decode(refresh_token, self.secret_key, algorithms=[self.algorithm])
            
            # 检查令牌类型
            if payload.get('type') != 'refresh':
                return {
                    'success': False,
                    'error': '无效的刷新令牌'
                }
            
            with get_db_context() as db:
                # 查找用户
                user = db.query(User).filter(
                    User.user_id == uuid.UUID(payload['user_id']),
                    User.is_active == True
                ).first()
                
                if not user:
                    return {
                        'success': False,
                        'error': '用户不存在或已被禁用'
                    }
                
                # 查找会话
                session = db.query(UserSession).filter(
                    UserSession.refresh_token == refresh_token,
                    UserSession.is_active == True
                ).first()
                
                if not session:
                    return {
                        'success': False,
                        'error': '刷新令牌无效'
                    }
                
                # 生成新的访问令牌
                new_access_token = self._generate_access_token(user)
                
                # 更新会话
                session.access_token = new_access_token
                session.expires_at = datetime.now(timezone.utc) + timedelta(minutes=self.access_token_expire_minutes)
                
                db.commit()
                
                return {
                    'success': True,
                    'access_token': new_access_token,
                    'token_type': 'bearer',
                    'expires_in': self.access_token_expire_minutes * 60
                }
                
        except jwt.ExpiredSignatureError:
            return {
                'success': False,
                'error': '刷新令牌已过期'
            }
        except jwt.InvalidTokenError:
            return {
                'success': False,
                'error': '无效的刷新令牌'
            }
        except Exception as e:
            logger.error(f"令牌刷新失败: {e}")
            return {
                'success': False,
                'error': '令牌刷新失败'
            }
    
    # 权限管理
    async def check_permission(self, user_id: str, permission_name: str) -> bool:
        """检查用户权限
        
        Args:
            user_id: 用户ID
            permission_name: 权限名称
            
        Returns:
            是否有权限
        """
        try:
            with get_db_context() as db:
                user = db.query(User).filter(
                    User.user_id == uuid.UUID(user_id)
                ).first()
                
                if not user or not user.is_active:
                    return False
                
                return user.has_permission(permission_name)
                
        except Exception as e:
            logger.error(f"权限检查失败: {e}")
            return False
    
    async def get_user_permissions(self, user_id: str) -> List[str]:
        """获取用户权限列表
        
        Args:
            user_id: 用户ID
            
        Returns:
            权限列表
        """
        try:
            with get_db_context() as db:
                user = db.query(User).filter(
                    User.user_id == uuid.UUID(user_id)
                ).first()
                
                if not user or not user.roles:
                    return []
                
                permissions = db.query(Permission).join(
                    Permission.roles
                ).filter(
                    Role.role_id.in_([role.role_id for role in user.roles])
                ).all()
                
                return [permission.name for permission in permissions]
                
        except Exception as e:
            logger.error(f"获取用户权限失败: {e}")
            return []
    
    # 密码管理
    def _hash_password(self, password: str) -> str:
        """加密密码"""
        salt = bcrypt.gensalt()
        return bcrypt.hashpw(password.encode('utf-8'), salt).decode('utf-8')
    
    def _verify_password(self, password: str, password_hash: str) -> bool:
        """验证密码"""
        try:
            return bcrypt.checkpw(password.encode('utf-8'), password_hash.encode('utf-8'))
        except Exception:
            return False
    
    async def change_password(self, user_id: str, old_password: str, 
                            new_password: str) -> Dict[str, Any]:
        """修改密码
        
        Args:
            user_id: 用户ID
            old_password: 旧密码
            new_password: 新密码
            
        Returns:
            修改结果
        """
        try:
            # 验证新密码
            if not self._validate_password(new_password):
                return {
                    'success': False,
                    'error': f'密码长度至少{self.password_min_length}位，且包含字母和数字'
                }
            
            with get_db_context() as db:
                user = db.query(User).filter(
                    User.user_id == uuid.UUID(user_id)
                ).first()
                
                if not user:
                    return {
                        'success': False,
                        'error': '用户不存在'
                    }
                
                # 验证旧密码
                if not self._verify_password(old_password, user.hashed_password):
                    return {
                        'success': False,
                        'error': '旧密码错误'
                    }
                
                # 更新密码
                user.hashed_password = self._hash_password(new_password)
                user.password_changed_at = datetime.now(timezone.utc)
                
                # 使所有会话失效
                db.query(UserSession).filter(
                    UserSession.user_id == user.user_id,
                    UserSession.is_active == True
                ).update({'is_active': False})
                
                db.commit()
                
                logger.info(f"用户密码修改成功: {user.username}")
                return {
                    'success': True,
                    'message': '密码修改成功，请重新登录'
                }
                
        except Exception as e:
            logger.error(f"密码修改失败: {e}")
            return {
                'success': False,
                'error': '密码修改失败'
            }
    
    async def reset_password(self, email: str) -> Dict[str, Any]:
        """重置密码
        
        Args:
            email: 邮箱地址
            
        Returns:
            重置结果
        """
        try:
            with get_db_context() as db:
                user = db.query(User).filter(User.email == email).first()
                
                if not user:
                    # 为了安全，即使用户不存在也返回成功
                    return {
                        'success': True,
                        'message': '如果邮箱存在，重置链接已发送'
                    }
                
                # 生成重置令牌
                reset_token = self._generate_reset_token(user)
                
                # 保存重置令牌（这里简化处理，实际应该保存到数据库）
                user.password_reset_token = reset_token
                user.password_reset_expires = datetime.now(timezone.utc) + timedelta(hours=1)
                
                db.commit()
                
                # TODO: 发送重置邮件
                logger.info(f"密码重置令牌已生成: {user.username}")
                
                return {
                    'success': True,
                    'message': '重置链接已发送到您的邮箱',
                    'reset_token': reset_token  # 实际部署时不应返回
                }
                
        except Exception as e:
            logger.error(f"密码重置失败: {e}")
            return {
                'success': False,
                'error': '密码重置失败'
            }
    
    def _generate_reset_token(self, user: User) -> str:
        """生成密码重置令牌"""
        payload = {
            'user_id': str(user.user_id),
            'email': user.email,
            'exp': datetime.now(timezone.utc) + timedelta(hours=1),
            'type': 'reset'
        }
        return jwt.encode(payload, self.secret_key, algorithm=self.algorithm)
    
    # 会话管理
    async def get_active_sessions(self, user_id: str) -> List[Dict[str, Any]]:
        """获取用户活跃会话
        
        Args:
            user_id: 用户ID
            
        Returns:
            会话列表
        """
        try:
            with get_db_context() as db:
                sessions = db.query(UserSession).filter(
                    UserSession.user_id == uuid.UUID(user_id),
                    UserSession.is_active == True,
                    UserSession.expires_at > datetime.now(timezone.utc)
                ).all()
                
                return [
                    {
                        'session_id': str(session.session_id),
                        'ip_address': session.ip_address,
                        'user_agent': session.user_agent,
                        'created_at': session.created_at.isoformat(),
                        'expires_at': session.expires_at.isoformat()
                    }
                    for session in sessions
                ]
                
        except Exception as e:
            logger.error(f"获取活跃会话失败: {e}")
            return []
    
    async def revoke_session(self, session_id: str, user_id: str) -> Dict[str, Any]:
        """撤销会话
        
        Args:
            session_id: 会话ID
            user_id: 用户ID
            
        Returns:
            撤销结果
        """
        try:
            with get_db_context() as db:
                session = db.query(UserSession).filter(
                    UserSession.session_id == uuid.UUID(session_id),
                    UserSession.user_id == uuid.UUID(user_id),
                    UserSession.is_active == True
                ).first()
                
                if session:
                    session.is_active = False
                    session.logged_out_at = datetime.now(timezone.utc)
                    db.commit()
                    
                    return {
                        'success': True,
                        'message': '会话已撤销'
                    }
                else:
                    return {
                        'success': False,
                        'error': '会话不存在或已失效'
                    }
                    
        except Exception as e:
            logger.error(f"撤销会话失败: {e}")
            return {
                'success': False,
                'error': '撤销会话失败'
            }
    
    async def cleanup_expired_sessions(self) -> int:
        """清理过期会话
        
        Returns:
            清理的会话数量
        """
        try:
            with get_db_context() as db:
                expired_sessions = db.query(UserSession).filter(
                    UserSession.expires_at < datetime.now(timezone.utc),
                    UserSession.is_active == True
                ).all()
                
                count = len(expired_sessions)
                
                for session in expired_sessions:
                    session.is_active = False
                    session.logged_out_at = datetime.now(timezone.utc)
                
                db.commit()
                
                logger.info(f"清理了{count}个过期会话")
                return count
                
        except Exception as e:
            logger.error(f"清理过期会话失败: {e}")
            return 0
    
    # 安全功能
    def _is_account_locked(self, user: User) -> bool:
        """检查账户是否被锁定"""
        if user.failed_login_attempts >= self.max_login_attempts:
            if user.last_failed_login:
                lockout_until = user.last_failed_login + timedelta(minutes=self.lockout_duration_minutes)
                return datetime.now(timezone.utc) < lockout_until
        return False
    
    def _record_login_attempt(self, user: User, success: bool, ip_address: Optional[str] = None):
        """记录登录尝试"""
        try:
            with get_db_context() as db:
                if success:
                    user.failed_login_attempts = 0
                    user.last_login = datetime.now(timezone.utc)
                else:
                    user.failed_login_attempts += 1
                    user.last_failed_login = datetime.now(timezone.utc)
                
                db.commit()
                
        except Exception as e:
            logger.error(f"记录登录尝试失败: {e}")
    
    # 验证方法
    def _validate_registration_data(self, username: str, email: str, password: str, 
                                  first_name: str, last_name: str) -> Dict[str, Any]:
        """验证注册数据"""
        errors = []
        
        # 验证用户名
        if not username or len(username) < 3:
            errors.append('用户名长度至少3位')
        
        # 验证邮箱
        if not email or '@' not in email:
            errors.append('邮箱格式不正确')
        
        # 验证密码
        if not self._validate_password(password):
            errors.append(f'密码长度至少{self.password_min_length}位，且包含字母和数字')
        
        # 验证姓名
        if not first_name or not last_name:
            errors.append('姓名不能为空')
        
        return {
            'valid': len(errors) == 0,
            'error': '; '.join(errors) if errors else None
        }
    
    def _validate_password(self, password: str) -> bool:
        """验证密码强度"""
        if len(password) < self.password_min_length:
            return False
        
        has_letter = any(c.isalpha() for c in password)
        has_digit = any(c.isdigit() for c in password)
        
        return has_letter and has_digit
    
    # 用户管理
    async def get_user_profile(self, user_id: str) -> Dict[str, Any]:
        """获取用户资料
        
        Args:
            user_id: 用户ID
            
        Returns:
            用户资料
        """
        try:
            with get_db_context() as db:
                user = db.query(User).filter(
                    User.user_id == uuid.UUID(user_id)
                ).first()
                
                if not user:
                    return {
                        'success': False,
                        'error': '用户不存在'
                    }
                
                return {
                    'success': True,
                    'user': {
                        'user_id': str(user.user_id),
                        'username': user.username,
                        'email': user.email,
                        'full_name': user.full_name,
                        'roles': [role.role_name for role in user.roles] if user.roles else [],
                        'is_active': user.is_active,
                        'is_verified': user.is_verified,
                        'created_at': user.created_at.isoformat(),
                        'last_login': user.last_login.isoformat() if user.last_login else None
                    }
                }
                
        except Exception as e:
            logger.error(f"获取用户资料失败: {e}")
            return {
                'success': False,
                'error': '获取用户资料失败'
            }
    
    async def update_user_profile(self, user_id: str, **kwargs) -> Dict[str, Any]:
        """更新用户资料
        
        Args:
            user_id: 用户ID
            **kwargs: 更新字段
            
        Returns:
            更新结果
        """
        try:
            with get_db_context() as db:
                user = db.query(User).filter(
                    User.user_id == uuid.UUID(user_id)
                ).first()
                
                if not user:
                    return {
                        'success': False,
                        'error': '用户不存在'
                    }
                
                # 允许更新的字段
                allowed_fields = ['first_name', 'last_name', 'email']
                
                for field, value in kwargs.items():
                    if field in allowed_fields and hasattr(user, field):
                        setattr(user, field, value)
                
                user.updated_at = datetime.now(timezone.utc)
                db.commit()
                
                return {
                    'success': True,
                    'message': '用户资料更新成功'
                }
                
        except Exception as e:
            logger.error(f"更新用户资料失败: {e}")
            return {
                'success': False,
                'error': '更新用户资料失败'
            }