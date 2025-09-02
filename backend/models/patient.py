from sqlalchemy import Column, String, Integer, DateTime, Text, BigInteger, DECIMAL, ForeignKey, JSON
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy import JSON
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
import uuid
from .base import Base

class Patient(Base):
    """患者信息模型"""
    __tablename__ = 'patients'
    
    patient_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    anonymized_id = Column(String(50), unique=True, nullable=False, index=True)
    age_group = Column(String(20))  # 年龄组：儿童、青年、中年、老年
    gender = Column(String(10))     # 性别：男、女、未知
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    # 关联关系
    studies = relationship("Study", back_populates="patient", cascade="all, delete-orphan")
    
    def __repr__(self):
        return f"<Patient(anonymized_id='{self.anonymized_id}', age_group='{self.age_group}')>"

class Study(Base):
    """检查研究模型"""
    __tablename__ = 'studies'
    
    study_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    patient_id = Column(UUID(as_uuid=True), ForeignKey('patients.patient_id'), nullable=False)
    study_date = Column(DateTime)
    modality = Column(String(10), nullable=False)  # CT, MRI, X-Ray, US等
    body_part = Column(String(50))                 # 检查部位
    study_description = Column(Text)
    referring_physician = Column(String(100))
    study_status = Column(String(20), default='pending')  # pending, completed, reviewed
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    # 关联关系
    patient = relationship("Patient", back_populates="studies")
    series = relationship("Series", back_populates="study", cascade="all, delete-orphan")
    
    def __repr__(self):
        return f"<Study(study_id='{self.study_id}', modality='{self.modality}', body_part='{self.body_part}')>"

class Series(Base):
    """序列模型"""
    __tablename__ = 'series'
    
    series_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    study_id = Column(UUID(as_uuid=True), ForeignKey('studies.study_id'), nullable=False)
    series_number = Column(Integer)
    series_description = Column(Text)
    image_count = Column(Integer, default=0)
    acquisition_parameters = Column(JSON)  # 采集参数
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    # 关联关系
    study = relationship("Study", back_populates="series")
    images = relationship("Image", back_populates="series", cascade="all, delete-orphan")
    
    def __repr__(self):
        return f"<Series(series_id='{self.series_id}', series_number={self.series_number})>"

class Image(Base):
    """图像模型"""
    __tablename__ = 'images'
    
    image_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    series_id = Column(UUID(as_uuid=True), ForeignKey('series.series_id'), nullable=False)
    instance_number = Column(Integer)
    file_path = Column(String(500), nullable=False)  # 文件存储路径
    file_size = Column(BigInteger)                   # 文件大小（字节）
    image_hash = Column(String(64), unique=True)     # 文件哈希值
    quality_score = Column(DECIMAL(3, 2))            # 质量评分 0.00-1.00
    processing_status = Column(String(20), default='pending')  # pending, processed, failed
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    # 关联关系
    series = relationship("Series", back_populates="images")
    image_metadata = relationship("ImageMetadata", back_populates="image", uselist=False, cascade="all, delete-orphan")
    annotations = relationship("Annotation", back_populates="image", cascade="all, delete-orphan")
    
    def __repr__(self):
        return f"<Image(image_id='{self.image_id}', instance_number={self.instance_number})>"

class ImageMetadata(Base):
    """图像元数据模型"""
    __tablename__ = 'image_metadata'
    
    image_id = Column(UUID(as_uuid=True), ForeignKey('images.image_id'), primary_key=True)
    dicom_tags = Column(JSON)              # DICOM标签信息
    technical_parameters = Column(JSON)     # 技术参数
    quality_metrics = Column(JSON)          # 质量指标
    extracted_features = Column(JSON)       # 提取的特征
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    # 关联关系
    image = relationship("Image", back_populates="image_metadata")
    
    def __repr__(self):
        return f"<ImageMetadata(image_id='{self.image_id}')>"