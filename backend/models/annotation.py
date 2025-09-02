from sqlalchemy import Column, String, DateTime, DECIMAL, ForeignKey, Text, Boolean
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy import JSON
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
import uuid
from .base import Base

class Annotation(Base):
    """标注模型"""
    __tablename__ = 'annotations'
    
    annotation_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    image_id = Column(UUID(as_uuid=True), ForeignKey('images.image_id'), nullable=False)
    annotator_id = Column(UUID(as_uuid=True), ForeignKey('users.user_id'), nullable=False)
    annotation_type = Column(String(50), nullable=False)  # classification, segmentation, detection, measurement
    annotation_data = Column(JSON, nullable=False)       # 标注数据（坐标、分类、分割掩码等）
    confidence_score = Column(DECIMAL(3, 2))              # 置信度评分 0.00-1.00
    validation_status = Column(String(20), default='pending')  # pending, validated, rejected, consensus
    annotation_time_seconds = Column(DECIMAL(8, 2))       # 标注耗时（秒）
    annotation_version = Column(String(20), default='1.0') # 标注版本
    notes = Column(Text)                                   # 标注备注
    is_ground_truth = Column(Boolean, default=False)      # 是否为金标准
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    # 关联关系
    image = relationship("Image", back_populates="annotations")
    annotator = relationship("User", back_populates="annotations")
    consensus_participations = relationship("AnnotationConsensus", 
                                          secondary="annotation_consensus_participants",
                                          back_populates="participating_annotations")
    
    def __repr__(self):
        return f"<Annotation(annotation_id='{self.annotation_id}', type='{self.annotation_type}', status='{self.validation_status}')>"

class AnnotationConsensus(Base):
    """标注共识模型"""
    __tablename__ = 'annotation_consensus'
    
    consensus_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    image_id = Column(UUID(as_uuid=True), ForeignKey('images.image_id'), nullable=False)
    final_annotation = Column(JSON, nullable=False)       # 最终共识标注
    agreement_score = Column(DECIMAL(3, 2))               # 一致性评分 0.00-1.00
    participating_annotators = Column(JSON)        # 参与共识的标注者ID列表
    resolution_method = Column(String(50))                # 共识解决方法：majority_vote, expert_decision, discussion
    consensus_reached_at = Column(DateTime(timezone=True))
    quality_metrics = Column(JSON)                       # 共识质量指标
    notes = Column(Text)                                  # 共识备注
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    # 关联关系
    image = relationship("Image")
    participating_annotations = relationship("Annotation",
                                           secondary="annotation_consensus_participants",
                                           back_populates="consensus_participations")
    
    def __repr__(self):
        return f"<AnnotationConsensus(consensus_id='{self.consensus_id}', agreement_score={self.agreement_score})>"

class AnnotationConsensusParticipant(Base):
    """标注共识参与者关联表"""
    __tablename__ = 'annotation_consensus_participants'
    
    consensus_id = Column(UUID(as_uuid=True), ForeignKey('annotation_consensus.consensus_id'), primary_key=True)
    annotation_id = Column(UUID(as_uuid=True), ForeignKey('annotations.annotation_id'), primary_key=True)
    weight = Column(DECIMAL(3, 2), default=1.0)          # 该标注在共识中的权重
    contribution_score = Column(DECIMAL(3, 2))            # 贡献度评分
    created_at = Column(DateTime(timezone=True), server_default=func.now())

class AnnotationTask(Base):
    """标注任务模型"""
    __tablename__ = 'annotation_tasks'
    
    task_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    task_name = Column(String(200), nullable=False)
    task_description = Column(Text)
    task_type = Column(String(50), nullable=False)        # classification, segmentation, detection
    priority = Column(String(20), default='medium')       # low, medium, high, urgent
    status = Column(String(20), default='created')        # created, assigned, in_progress, completed, reviewed
    assigned_to = Column(UUID(as_uuid=True), ForeignKey('users.user_id'))
    reviewer_id = Column(UUID(as_uuid=True), ForeignKey('users.user_id'))
    deadline = Column(DateTime(timezone=True))
    estimated_time_hours = Column(DECIMAL(5, 2))          # 预估耗时（小时）
    actual_time_hours = Column(DECIMAL(5, 2))             # 实际耗时（小时）
    quality_requirements = Column(JSON)                   # 质量要求
    guidelines = Column(Text)                              # 标注指南
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    # 关联关系
    assignee = relationship("User", foreign_keys=[assigned_to], back_populates="assigned_tasks")
    reviewer = relationship("User", foreign_keys=[reviewer_id], back_populates="review_tasks")
    
    def __repr__(self):
        return f"<AnnotationTask(task_id='{self.task_id}', name='{self.task_name}', status='{self.status}')>"