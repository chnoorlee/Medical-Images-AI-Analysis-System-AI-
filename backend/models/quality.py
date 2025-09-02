from sqlalchemy import Column, String, DateTime, Text, ForeignKey, Integer, DECIMAL, Boolean
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy import JSON
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
import uuid
from .base import Base

class QualityMetrics(Base):
    """质量指标模型"""
    __tablename__ = 'quality_metrics'
    
    metric_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    metric_name = Column(String(100), nullable=False, unique=True)
    metric_category = Column(String(50), nullable=False)     # technical, clinical, annotation, model_performance
    description = Column(Text)
    measurement_unit = Column(String(20))                    # 测量单位
    data_type = Column(String(20), nullable=False)          # numeric, percentage, boolean, categorical
    min_value = Column(DECIMAL(10, 4))                       # 最小值
    max_value = Column(DECIMAL(10, 4))                       # 最大值
    target_value = Column(DECIMAL(10, 4))                    # 目标值
    threshold_excellent = Column(DECIMAL(10, 4))             # 优秀阈值
    threshold_good = Column(DECIMAL(10, 4))                  # 良好阈值
    threshold_acceptable = Column(DECIMAL(10, 4))            # 可接受阈值
    calculation_method = Column(Text)                        # 计算方法描述
    automation_level = Column(String(20))                    # manual, semi_automated, automated
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    # 关联关系
    assessments = relationship("QualityAssessment", back_populates="metric")
    
    def __repr__(self):
        return f"<QualityMetrics(metric_name='{self.metric_name}', category='{self.metric_category}')>"
    
    def evaluate_score(self, value):
        """根据阈值评估质量等级"""
        if value >= self.threshold_excellent:
            return 'excellent'
        elif value >= self.threshold_good:
            return 'good'
        elif value >= self.threshold_acceptable:
            return 'acceptable'
        else:
            return 'poor'

class QualityAssessment(Base):
    """质量评估模型"""
    __tablename__ = 'quality_assessments'
    
    assessment_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    target_type = Column(String(50), nullable=False)         # image, annotation, model, dataset
    target_id = Column(UUID(as_uuid=True), nullable=False)   # 目标对象ID
    metric_id = Column(UUID(as_uuid=True), ForeignKey('quality_metrics.metric_id'), nullable=False)
    assessor_id = Column(UUID(as_uuid=True), ForeignKey('users.user_id'))
    assessment_method = Column(String(50))                   # automated, manual, hybrid
    measured_value = Column(DECIMAL(10, 4))                  # 测量值
    quality_grade = Column(String(20))                       # excellent, good, acceptable, poor
    confidence_score = Column(DECIMAL(3, 2))                # 置信度 0.00-1.00
    assessment_details = Column(JSON)                       # 评估详情
    issues_identified = Column(JSON)                        # 识别的问题
    recommendations = Column(JSON)                          # 改进建议
    validation_status = Column(String(20), default='pending') # pending, validated, rejected
    validated_by = Column(UUID(as_uuid=True), ForeignKey('users.user_id'))
    notes = Column(Text)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    validated_at = Column(DateTime(timezone=True))
    
    # 关联关系
    metric = relationship("QualityMetrics", back_populates="assessments")
    assessor = relationship("User", foreign_keys=[assessor_id])
    validator = relationship("User", foreign_keys=[validated_by])
    
    def __repr__(self):
        return f"<QualityAssessment(assessment_id='{self.assessment_id}', target_type='{self.target_type}', grade='{self.quality_grade}')>"

class QualityControlRule(Base):
    """质量控制规则模型"""
    __tablename__ = 'quality_control_rules'
    
    rule_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    rule_name = Column(String(100), nullable=False)
    rule_category = Column(String(50), nullable=False)       # data_validation, image_quality, annotation_quality
    target_type = Column(String(50), nullable=False)         # image, annotation, dataset, model
    rule_description = Column(Text)
    condition_logic = Column(JSON, nullable=False)          # 规则条件逻辑
    action_on_pass = Column(JSON)                           # 通过时的动作
    action_on_fail = Column(JSON)                           # 失败时的动作
    severity = Column(String(20), default='medium')         # low, medium, high, critical
    is_blocking = Column(Boolean, default=False)            # 是否为阻塞性规则
    auto_remediation = Column(JSON)                         # 自动修复措施
    notification_config = Column(JSON)                      # 通知配置
    is_active = Column(Boolean, default=True)
    execution_order = Column(Integer, default=100)          # 执行顺序
    created_by = Column(UUID(as_uuid=True), ForeignKey('users.user_id'))
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    # 关联关系
    creator = relationship("User")
    executions = relationship("QualityControlExecution", back_populates="rule")
    
    def __repr__(self):
        return f"<QualityControlRule(rule_name='{self.rule_name}', category='{self.rule_category}', is_active={self.is_active})>"

class QualityControlExecution(Base):
    """质量控制执行记录模型"""
    __tablename__ = 'quality_control_executions'
    
    execution_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    rule_id = Column(UUID(as_uuid=True), ForeignKey('quality_control_rules.rule_id'), nullable=False)
    target_type = Column(String(50), nullable=False)
    target_id = Column(UUID(as_uuid=True), nullable=False)
    execution_trigger = Column(String(50))                   # scheduled, event_driven, manual
    execution_context = Column(JSON)                        # 执行上下文
    input_data = Column(JSON)                               # 输入数据
    rule_result = Column(String(20))                         # pass, fail, error, skipped
    result_details = Column(JSON)                           # 结果详情
    measured_values = Column(JSON)                          # 测量值
    actions_taken = Column(JSON)                            # 执行的动作
    processing_time_ms = Column(Integer)                     # 处理时间（毫秒）
    error_message = Column(Text)                             # 错误信息
    notifications_sent = Column(JSON)                       # 发送的通知
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    # 关联关系
    rule = relationship("QualityControlRule", back_populates="executions")
    
    def __repr__(self):
        return f"<QualityControlExecution(execution_id='{self.execution_id}', result='{self.rule_result}')>"

class DataQualityReport(Base):
    """数据质量报告模型"""
    __tablename__ = 'data_quality_reports'
    
    report_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    report_name = Column(String(200), nullable=False)
    report_type = Column(String(50), nullable=False)         # daily, weekly, monthly, ad_hoc
    scope = Column(String(50))                               # dataset, model, system, department
    scope_details = Column(JSON)                            # 范围详情
    report_period_start = Column(DateTime(timezone=True))
    report_period_end = Column(DateTime(timezone=True))
    overall_quality_score = Column(DECIMAL(5, 2))           # 总体质量评分
    quality_dimensions = Column(JSON)                       # 质量维度评分
    key_findings = Column(JSON)                             # 关键发现
    quality_trends = Column(JSON)                           # 质量趋势
    issues_summary = Column(JSON)                           # 问题汇总
    improvement_recommendations = Column(JSON)              # 改进建议
    action_plan = Column(JSON)                              # 行动计划
    stakeholder_notifications = Column(JSON)                # 利益相关者通知
    report_data = Column(JSON)                              # 详细报告数据
    generated_by = Column(UUID(as_uuid=True), ForeignKey('users.user_id'))
    status = Column(String(20), default='draft')            # draft, final, published
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    published_at = Column(DateTime(timezone=True))
    
    # 关联关系
    generator = relationship("User")
    
    def __repr__(self):
        return f"<DataQualityReport(report_name='{self.report_name}', type='{self.report_type}', status='{self.status}')>"

class QualityImprovement(Base):
    """质量改进模型"""
    __tablename__ = 'quality_improvements'
    
    improvement_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    title = Column(String(200), nullable=False)
    description = Column(Text)
    category = Column(String(50))                            # process, technology, training, policy
    priority = Column(String(20), default='medium')         # low, medium, high, critical
    target_area = Column(String(50))                        # data_collection, annotation, model_training, deployment
    current_state = Column(Text)                             # 当前状态描述
    target_state = Column(Text)                              # 目标状态描述
    success_criteria = Column(JSON)                         # 成功标准
    implementation_plan = Column(JSON)                      # 实施计划
    resource_requirements = Column(JSON)                    # 资源需求
    timeline = Column(JSON)                                 # 时间线
    responsible_person = Column(UUID(as_uuid=True), ForeignKey('users.user_id'))
    stakeholders = Column(JSON)                             # 利益相关者
    status = Column(String(20), default='proposed')         # proposed, approved, in_progress, completed, cancelled
    progress_percent = Column(DECIMAL(5, 2), default=0.0)   # 进度百分比
    actual_impact = Column(JSON)                            # 实际影响
    lessons_learned = Column(Text)                           # 经验教训
    created_by = Column(UUID(as_uuid=True), ForeignKey('users.user_id'))
    approved_by = Column(UUID(as_uuid=True), ForeignKey('users.user_id'))
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    approved_at = Column(DateTime(timezone=True))
    completed_at = Column(DateTime(timezone=True))
    
    # 关联关系
    responsible = relationship("User", foreign_keys=[responsible_person])
    creator = relationship("User", foreign_keys=[created_by])
    approver = relationship("User", foreign_keys=[approved_by])
    
    def __repr__(self):
        return f"<QualityImprovement(title='{self.title}', status='{self.status}', progress={self.progress_percent}%)>"