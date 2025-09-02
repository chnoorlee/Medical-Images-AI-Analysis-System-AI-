from sqlalchemy import Column, String, DateTime, Text, ForeignKey, Integer
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy import JSON
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
import uuid
from .base import Base

class AuditLog(Base):
    """审计日志模型"""
    __tablename__ = 'audit_logs'
    
    log_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey('users.user_id'))
    session_id = Column(UUID(as_uuid=True), ForeignKey('user_sessions.session_id'))
    action = Column(String(100), nullable=False)             # 操作类型：login, logout, create, update, delete等
    resource_type = Column(String(50))                       # 资源类型：patient, image, annotation, model等
    resource_id = Column(UUID(as_uuid=True))                # 资源ID
    operation_details = Column(JSON)                        # 操作详情
    old_values = Column(JSON)                               # 修改前的值
    new_values = Column(JSON)                               # 修改后的值
    ip_address = Column(String(45))                                # IP地址
    user_agent = Column(Text)                                # 用户代理
    request_method = Column(String(10))                      # HTTP方法
    request_url = Column(String(500))                        # 请求URL
    request_params = Column(JSON)                           # 请求参数
    response_status = Column(Integer)                        # 响应状态码
    processing_time_ms = Column(Integer)                     # 处理时间（毫秒）
    severity = Column(String(20), default='info')           # 严重级别：debug, info, warning, error, critical
    category = Column(String(50))                            # 分类：security, data_access, system, user_action
    compliance_relevant = Column(String(10), default='no')   # 是否与合规相关：yes, no
    retention_period_days = Column(Integer, default=2555)    # 保留期限（天）
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    # 关联关系
    user = relationship("User", back_populates="audit_logs")
    session = relationship("UserSession")
    
    def __repr__(self):
        return f"<AuditLog(log_id='{self.log_id}', action='{self.action}', resource_type='{self.resource_type}')>"

class SecurityEvent(Base):
    """安全事件模型"""
    __tablename__ = 'security_events'
    
    event_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    event_type = Column(String(50), nullable=False)          # failed_login, suspicious_activity, data_breach等
    severity = Column(String(20), nullable=False)            # low, medium, high, critical
    source_ip = Column(String(45))
    target_user_id = Column(UUID(as_uuid=True), ForeignKey('users.user_id'))
    event_details = Column(JSON, nullable=False)
    detection_method = Column(String(100))                   # 检测方法
    risk_score = Column(Integer)                             # 风险评分 0-100
    status = Column(String(20), default='open')             # open, investigating, resolved, false_positive
    assigned_to = Column(UUID(as_uuid=True), ForeignKey('users.user_id'))
    resolution_notes = Column(Text)
    automated_response = Column(JSON)                       # 自动响应措施
    manual_actions = Column(JSON)                           # 手动处理措施
    related_events = Column(JSON)                           # 相关事件ID列表
    compliance_impact = Column(Text)                         # 合规影响评估
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    resolved_at = Column(DateTime(timezone=True))
    
    # 关联关系
    target_user = relationship("User", foreign_keys=[target_user_id])
    assigned_user = relationship("User", foreign_keys=[assigned_to])
    
    def __repr__(self):
        return f"<SecurityEvent(event_id='{self.event_id}', type='{self.event_type}', severity='{self.severity}')>"

class DataAccessLog(Base):
    """数据访问日志模型"""
    __tablename__ = 'data_access_logs'
    
    access_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey('users.user_id'), nullable=False)
    patient_id = Column(UUID(as_uuid=True), ForeignKey('patients.patient_id'))
    image_id = Column(UUID(as_uuid=True), ForeignKey('images.image_id'))
    study_id = Column(UUID(as_uuid=True), ForeignKey('studies.study_id'))
    access_type = Column(String(50), nullable=False)         # view, download, export, print, share
    access_purpose = Column(String(100))                     # 访问目的：diagnosis, research, teaching, quality_control
    data_sensitivity = Column(String(20))                    # 数据敏感级别：public, internal, confidential, restricted
    access_duration_seconds = Column(Integer)                # 访问持续时间
    data_volume_mb = Column(Integer)                         # 访问的数据量（MB）
    export_format = Column(String(20))                       # 导出格式：dicom, jpeg, pdf等
    export_destination = Column(String(200))                 # 导出目标
    justification = Column(Text)                             # 访问理由
    approval_required = Column(String(10), default='no')     # 是否需要审批
    approved_by = Column(UUID(as_uuid=True), ForeignKey('users.user_id'))
    approval_status = Column(String(20), default='auto_approved') # auto_approved, pending, approved, denied
    ip_address = Column(String(45))
    device_info = Column(JSON)                              # 设备信息
    location_info = Column(JSON)                            # 位置信息
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    # 关联关系
    user = relationship("User", foreign_keys=[user_id])
    approver = relationship("User", foreign_keys=[approved_by])
    patient = relationship("Patient")
    image = relationship("Image")
    study = relationship("Study")
    
    def __repr__(self):
        return f"<DataAccessLog(access_id='{self.access_id}', access_type='{self.access_type}', user_id='{self.user_id}')>"

class ComplianceReport(Base):
    """合规报告模型"""
    __tablename__ = 'compliance_reports'
    
    report_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    report_type = Column(String(50), nullable=False)         # hipaa, gdpr, fda, internal_audit
    report_period_start = Column(DateTime(timezone=True), nullable=False)
    report_period_end = Column(DateTime(timezone=True), nullable=False)
    generated_by = Column(UUID(as_uuid=True), ForeignKey('users.user_id'))
    report_data = Column(JSON, nullable=False)              # 报告数据
    summary_statistics = Column(JSON)                       # 汇总统计
    compliance_score = Column(Integer)                       # 合规评分 0-100
    violations_found = Column(Integer, default=0)            # 发现的违规数量
    recommendations = Column(JSON)                          # 改进建议
    action_items = Column(JSON)                             # 行动项目
    status = Column(String(20), default='draft')            # draft, final, submitted, approved
    file_path = Column(String(500))                          # 报告文件路径
    digital_signature = Column(String(500))                  # 数字签名
    reviewed_by = Column(UUID(as_uuid=True), ForeignKey('users.user_id'))
    approved_by = Column(UUID(as_uuid=True), ForeignKey('users.user_id'))
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    reviewed_at = Column(DateTime(timezone=True))
    approved_at = Column(DateTime(timezone=True))
    
    # 关联关系
    generator = relationship("User", foreign_keys=[generated_by])
    reviewer = relationship("User", foreign_keys=[reviewed_by])
    approver = relationship("User", foreign_keys=[approved_by])
    
    def __repr__(self):
        return f"<ComplianceReport(report_id='{self.report_id}', type='{self.report_type}', status='{self.status}')>"