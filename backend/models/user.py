from sqlalchemy import Column, String, DateTime, Boolean, ForeignKey, Text, Table, Integer
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy import JSON
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
import uuid
from .base import Base

# 用户角色关联表
user_roles = Table(
    'user_roles',
    Base.metadata,
    Column('user_id', UUID(as_uuid=True), ForeignKey('users.user_id'), primary_key=True),
    Column('role_id', UUID(as_uuid=True), ForeignKey('roles.role_id'), primary_key=True),
    Column('assigned_at', DateTime(timezone=True), server_default=func.now()),
    Column('assigned_by', UUID(as_uuid=True), ForeignKey('users.user_id'))
)

# 角色权限关联表
role_permissions = Table(
    'role_permissions',
    Base.metadata,
    Column('role_id', UUID(as_uuid=True), ForeignKey('roles.role_id'), primary_key=True),
    Column('permission_id', UUID(as_uuid=True), ForeignKey('permissions.permission_id'), primary_key=True),
    Column('granted_at', DateTime(timezone=True), server_default=func.now())
)

class User(Base):
    """用户模型"""
    __tablename__ = 'users'
    
    user_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    username = Column(String(50), unique=True, nullable=False, index=True)
    email = Column(String(100), unique=True, nullable=False, index=True)
    hashed_password = Column(String(255), nullable=False)
    full_name = Column(String(100), nullable=False)
    department = Column(String(100))                      # 科室
    title = Column(String(100))                           # 职称
    license_number = Column(String(50))                   # 执业证书号
    specialization = Column(String(100))                  # 专业方向
    experience_years = Column(String(20))                 # 工作经验年限
    phone = Column(String(20))
    is_active = Column(Boolean, default=True)
    is_verified = Column(Boolean, default=False)          # 是否已验证身份
    last_login = Column(DateTime(timezone=True))
    login_count = Column(String(20), default='0')
    failed_login_attempts = Column(Integer, default=0)  # 失败登录尝试次数
    preferences = Column(JSON)                           # 用户偏好设置
    profile_image_url = Column(String(500))               # 头像URL
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    # 关联关系
    roles = relationship("Role", secondary=user_roles, back_populates="users", 
                        primaryjoin="User.user_id == user_roles.c.user_id",
                        secondaryjoin="Role.role_id == user_roles.c.role_id")
    annotations = relationship("Annotation", back_populates="annotator")
    assigned_tasks = relationship("AnnotationTask", foreign_keys="AnnotationTask.assigned_to", back_populates="assignee")
    review_tasks = relationship("AnnotationTask", foreign_keys="AnnotationTask.reviewer_id", back_populates="reviewer")
    audit_logs = relationship("AuditLog", back_populates="user")
    
    def __repr__(self):
        return f"<User(username='{self.username}', full_name='{self.full_name}', department='{self.department}')>"
    
    def has_permission(self, permission_name):
        """检查用户是否具有指定权限"""
        for role in self.roles:
            if role.has_permission(permission_name):
                return True
        return False
    
    def has_role(self, role_name):
        """检查用户是否具有指定角色"""
        return any(role.role_name == role_name for role in self.roles)

class Role(Base):
    """角色模型"""
    __tablename__ = 'roles'
    
    role_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    role_name = Column(String(50), unique=True, nullable=False, index=True)
    role_display_name = Column(String(100), nullable=False)  # 角色显示名称
    description = Column(Text)
    is_system_role = Column(Boolean, default=False)          # 是否为系统角色
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    # 关联关系
    users = relationship("User", secondary=user_roles, back_populates="roles",
                        primaryjoin="Role.role_id == user_roles.c.role_id",
                        secondaryjoin="User.user_id == user_roles.c.user_id")
    permissions = relationship("Permission", secondary=role_permissions, back_populates="roles")
    
    def __repr__(self):
        return f"<Role(role_name='{self.role_name}', display_name='{self.role_display_name}')>"
    
    def has_permission(self, permission_name):
        """检查角色是否具有指定权限"""
        return any(perm.permission_name == permission_name for perm in self.permissions)

class Permission(Base):
    """权限模型"""
    __tablename__ = 'permissions'
    
    permission_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    permission_name = Column(String(100), unique=True, nullable=False, index=True)
    permission_display_name = Column(String(100), nullable=False)  # 权限显示名称
    description = Column(Text)
    resource = Column(String(50))                            # 资源类型：patient, image, annotation, model等
    action = Column(String(50))                              # 操作类型：create, read, update, delete, execute等
    is_system_permission = Column(Boolean, default=False)    # 是否为系统权限
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    # 关联关系
    roles = relationship("Role", secondary=role_permissions, back_populates="permissions")
    
    def __repr__(self):
        return f"<Permission(permission_name='{self.permission_name}', resource='{self.resource}', action='{self.action}')>"

class UserSession(Base):
    """用户会话模型"""
    __tablename__ = 'user_sessions'
    
    session_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey('users.user_id'), nullable=False)
    access_token = Column(String(500), nullable=False, unique=True)
    refresh_token = Column(String(500), nullable=False, unique=True)
    ip_address = Column(String(45))                          # 支持IPv6
    user_agent = Column(Text)
    is_active = Column(Boolean, default=True)
    expires_at = Column(DateTime(timezone=True), nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    last_accessed_at = Column(DateTime(timezone=True), server_default=func.now())
    
    # 关联关系
    user = relationship("User")
    
    def __repr__(self):
        return f"<UserSession(session_id='{self.session_id}', user_id='{self.user_id}', is_active={self.is_active})>"
    
    def is_expired(self):
        """检查会话是否已过期"""
        from datetime import datetime, timezone
        return datetime.now(timezone.utc) > self.expires_at