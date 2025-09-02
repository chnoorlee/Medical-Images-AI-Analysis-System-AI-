# -*- coding: utf-8 -*-
"""
测试数据工厂

本文件使用 Factory Boy 库来生成测试数据：
- 用户数据工厂
- 患者数据工厂
- 医学影像数据工厂
- 诊断报告数据工厂
- 推理任务数据工厂
- 通知数据工厂
"""

import factory
import factory.fuzzy
from datetime import datetime, date, timedelta
from typing import Dict, Any, List
import uuid
import random
import string
from faker import Faker
from faker.providers import BaseProvider

fake = Faker(['zh_CN', 'en_US'])


class MedicalProvider(BaseProvider):
    """医学相关数据提供者"""
    
    # 医学科室
    departments = [
        "放射科", "影像科", "内科", "外科", "骨科", "神经科",
        "心血管科", "呼吸科", "消化科", "泌尿科", "妇科", "儿科"
    ]
    
    # 影像检查类型
    study_types = [
        "chest_xray", "brain_mri", "abdominal_ct", "spine_mri",
        "knee_mri", "cardiac_ct", "lung_ct", "liver_mri",
        "breast_mammography", "pelvic_ultrasound"
    ]
    
    # 影像模态
    modalities = [
        "CR", "DR", "CT", "MR", "US", "MG", "NM", "PET", "RF", "XA"
    ]
    
    # 身体部位
    body_parts = [
        "CHEST", "ABDOMEN", "PELVIS", "HEAD", "NECK", "SPINE",
        "EXTREMITY", "HEART", "LUNG", "LIVER", "KIDNEY", "BRAIN"
    ]
    
    # 诊断结果
    diagnoses = [
        "正常", "肺炎", "肺结节", "骨折", "肿瘤", "炎症",
        "积液", "出血", "梗塞", "增生", "萎缩", "钙化"
    ]
    
    # 医生执照号前缀
    license_prefixes = ["DOC", "MD", "DR", "PHY", "RAD"]
    
    def department(self):
        """生成科室名称"""
        return self.random_element(self.departments)
    
    def study_type(self):
        """生成检查类型"""
        return self.random_element(self.study_types)
    
    def modality(self):
        """生成影像模态"""
        return self.random_element(self.modalities)
    
    def body_part(self):
        """生成身体部位"""
        return self.random_element(self.body_parts)
    
    def diagnosis(self):
        """生成诊断结果"""
        return self.random_element(self.diagnoses)
    
    def license_number(self):
        """生成医生执照号"""
        prefix = self.random_element(self.license_prefixes)
        number = ''.join(random.choices(string.digits, k=6))
        return f"{prefix}{number}"
    
    def patient_id(self):
        """生成患者ID"""
        return f"P{random.randint(100000, 999999)}"
    
    def dicom_filename(self):
        """生成DICOM文件名"""
        study = self.study_type()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"{study}_{timestamp}.dcm"


# 注册自定义提供者
fake.add_provider(MedicalProvider)


class UserFactory(factory.Factory):
    """用户数据工厂"""
    
    class Meta:
        model = dict
    
    id = factory.LazyFunction(lambda: str(uuid.uuid4()))
    username = factory.Sequence(lambda n: f"user{n}@test.com")
    email = factory.LazyAttribute(lambda obj: obj.username)
    full_name = factory.Faker('name', locale='zh_CN')
    password = factory.Faker('password', length=12)
    hashed_password = factory.LazyAttribute(lambda obj: f"hashed_{obj.password}")
    role = factory.fuzzy.FuzzyChoice(["doctor", "radiologist", "admin", "technician"])
    department = factory.Faker('department')
    license_number = factory.Maybe(
        'is_medical_staff',
        yes_declaration=factory.Faker('license_number'),
        no_declaration=None
    )
    phone = factory.Faker('phone_number', locale='zh_CN')
    is_active = True
    is_verified = True
    created_at = factory.Faker('date_time_between', start_date='-1y', end_date='now')
    updated_at = factory.LazyAttribute(lambda obj: obj.created_at + timedelta(days=random.randint(0, 30)))
    last_login = factory.Faker('date_time_between', start_date='-30d', end_date='now')
    
    @factory.lazy_attribute
    def is_medical_staff(self):
        return self.role in ["doctor", "radiologist"]


class DoctorFactory(UserFactory):
    """医生用户工厂"""
    
    role = "doctor"
    department = factory.Faker('department')
    license_number = factory.Faker('license_number')
    is_medical_staff = True


class RadiologistFactory(UserFactory):
    """放射科医生工厂"""
    
    role = "radiologist"
    department = "放射科"
    license_number = factory.Faker('license_number')
    is_medical_staff = True


class AdminFactory(UserFactory):
    """管理员工厂"""
    
    role = "admin"
    department = "IT部门"
    license_number = None
    is_medical_staff = False


class PatientFactory(factory.Factory):
    """患者数据工厂"""
    
    class Meta:
        model = dict
    
    id = factory.LazyFunction(lambda: str(uuid.uuid4()))
    patient_id = factory.Faker('patient_id')
    name = factory.Faker('name', locale='zh_CN')
    gender = factory.fuzzy.FuzzyChoice(["male", "female"])
    birth_date = factory.Faker('date_of_birth', minimum_age=18, maximum_age=90)
    age = factory.LazyAttribute(lambda obj: (date.today() - obj.birth_date).days // 365)
    phone = factory.Faker('phone_number', locale='zh_CN')
    email = factory.Faker('email')
    address = factory.Faker('address', locale='zh_CN')
    emergency_contact = factory.Faker('name', locale='zh_CN')
    emergency_phone = factory.Faker('phone_number', locale='zh_CN')
    id_number = factory.Faker('ssn', locale='zh_CN')
    insurance_number = factory.Sequence(lambda n: f"INS{n:08d}")
    medical_history = factory.List([
        factory.Faker('diagnosis') for _ in range(random.randint(0, 3))
    ])
    allergies = factory.List([
        factory.Faker('word') for _ in range(random.randint(0, 2))
    ])
    created_at = factory.Faker('date_time_between', start_date='-2y', end_date='now')
    updated_at = factory.LazyAttribute(lambda obj: obj.created_at + timedelta(days=random.randint(0, 60)))
    created_by = factory.LazyFunction(lambda: str(uuid.uuid4()))


class MedicalImageFactory(factory.Factory):
    """医学影像数据工厂"""
    
    class Meta:
        model = dict
    
    id = factory.LazyFunction(lambda: str(uuid.uuid4()))
    filename = factory.Faker('dicom_filename')
    original_filename = factory.LazyAttribute(lambda obj: f"original_{obj.filename}")
    file_path = factory.LazyAttribute(lambda obj: f"/storage/images/{obj.filename}")
    file_size = factory.fuzzy.FuzzyInteger(500000, 10000000)  # 500KB - 10MB
    file_hash = factory.LazyFunction(lambda: fake.sha256())
    study_type = factory.Faker('study_type')
    patient_id = factory.LazyFunction(lambda: str(uuid.uuid4()))
    study_id = factory.LazyFunction(lambda: str(uuid.uuid4()))
    series_id = factory.LazyFunction(lambda: str(uuid.uuid4()))
    instance_id = factory.LazyFunction(lambda: str(uuid.uuid4()))
    description = factory.LazyAttribute(lambda obj: f"{obj.study_type}检查")
    acquisition_date = factory.Faker('date_time_between', start_date='-1y', end_date='now')
    modality = factory.Faker('modality')
    body_part = factory.Faker('body_part')
    view_position = factory.fuzzy.FuzzyChoice(["AP", "PA", "LAT", "OBL", "AXI", "SAG", "COR"])
    slice_thickness = factory.fuzzy.FuzzyFloat(0.5, 5.0)
    pixel_spacing = factory.List([factory.fuzzy.FuzzyFloat(0.1, 1.0) for _ in range(2)])
    image_dimensions = factory.List([factory.fuzzy.FuzzyInteger(256, 2048) for _ in range(2)])
    bits_allocated = factory.fuzzy.FuzzyChoice([8, 16])
    contrast_used = factory.Faker('boolean', chance_of_getting_true=30)
    radiation_dose = factory.Maybe(
        'is_ct_or_xray',
        yes_declaration=factory.fuzzy.FuzzyFloat(0.1, 50.0),
        no_declaration=None
    )
    quality_score = factory.fuzzy.FuzzyFloat(0.7, 1.0)
    is_processed = factory.Faker('boolean', chance_of_getting_true=80)
    processing_status = factory.LazyAttribute(
        lambda obj: "completed" if obj.is_processed else "pending"
    )
    uploaded_by = factory.LazyFunction(lambda: str(uuid.uuid4()))
    uploaded_at = factory.Faker('date_time_between', start_date='-6m', end_date='now')
    
    @factory.lazy_attribute
    def is_ct_or_xray(self):
        return self.modality in ["CT", "CR", "DR"]


class InferenceTaskFactory(factory.Factory):
    """推理任务数据工厂"""
    
    class Meta:
        model = dict
    
    id = factory.LazyFunction(lambda: str(uuid.uuid4()))
    image_id = factory.LazyFunction(lambda: str(uuid.uuid4()))
    model_name = factory.fuzzy.FuzzyChoice([
        "chest_xray_classifier", "brain_mri_segmentation", "lung_nodule_detector",
        "bone_fracture_detector", "cardiac_analyzer", "liver_segmentation"
    ])
    model_version = factory.fuzzy.FuzzyChoice(["1.0.0", "1.1.0", "1.2.0", "2.0.0"])
    status = factory.fuzzy.FuzzyChoice(["pending", "running", "completed", "failed"])
    priority = factory.fuzzy.FuzzyChoice(["low", "normal", "high", "urgent"])
    
    # 推理结果
    predictions = factory.LazyAttribute(lambda obj: [
        {
            "class": fake.diagnosis(),
            "confidence": round(random.uniform(0.1, 0.99), 3),
            "bbox": [random.randint(0, 500), random.randint(0, 500), 
                    random.randint(100, 200), random.randint(100, 200)] if random.choice([True, False]) else None
        }
        for _ in range(random.randint(1, 5))
    ])
    
    confidence_score = factory.LazyAttribute(
        lambda obj: max([p["confidence"] for p in obj.predictions]) if obj.predictions else 0.0
    )
    
    processing_time = factory.fuzzy.FuzzyFloat(0.5, 30.0)
    gpu_memory_used = factory.fuzzy.FuzzyInteger(1000, 8000)  # MB
    
    # 可视化数据
    heatmap_data = factory.LazyFunction(lambda: fake.text(max_nb_chars=100))
    attention_maps = factory.List([
        factory.LazyFunction(lambda: fake.text(max_nb_chars=50)) for _ in range(random.randint(1, 3))
    ])
    
    # 质量评估
    quality_metrics = factory.LazyAttribute(lambda obj: {
        "image_quality": round(random.uniform(0.7, 1.0), 3),
        "model_confidence": round(random.uniform(0.6, 0.95), 3),
        "consistency_score": round(random.uniform(0.8, 1.0), 3)
    })
    
    error_message = factory.Maybe(
        'is_failed',
        yes_declaration=factory.Faker('sentence'),
        no_declaration=None
    )
    
    created_at = factory.Faker('date_time_between', start_date='-3m', end_date='now')
    started_at = factory.LazyAttribute(
        lambda obj: obj.created_at + timedelta(seconds=random.randint(1, 300))
    )
    completed_at = factory.Maybe(
        'is_completed',
        yes_declaration=factory.LazyAttribute(
            lambda obj: obj.started_at + timedelta(seconds=random.randint(1, 1800))
        ),
        no_declaration=None
    )
    
    created_by = factory.LazyFunction(lambda: str(uuid.uuid4()))
    
    @factory.lazy_attribute
    def is_failed(self):
        return self.status == "failed"
    
    @factory.lazy_attribute
    def is_completed(self):
        return self.status in ["completed", "failed"]


class DiagnosticReportFactory(factory.Factory):
    """诊断报告数据工厂"""
    
    class Meta:
        model = dict
    
    id = factory.LazyFunction(lambda: str(uuid.uuid4()))
    patient_id = factory.LazyFunction(lambda: str(uuid.uuid4()))
    image_id = factory.LazyFunction(lambda: str(uuid.uuid4()))
    inference_task_id = factory.LazyFunction(lambda: str(uuid.uuid4()))
    report_number = factory.Sequence(lambda n: f"RPT{n:08d}")
    
    # 报告内容
    title = factory.LazyAttribute(lambda obj: f"{fake.study_type()}检查报告")
    clinical_history = factory.Faker('text', max_nb_chars=200)
    examination_description = factory.Faker('text', max_nb_chars=300)
    findings = factory.Faker('text', max_nb_chars=500)
    impression = factory.Faker('text', max_nb_chars=200)
    recommendations = factory.Faker('text', max_nb_chars=300)
    
    # AI 辅助诊断
    ai_findings = factory.LazyAttribute(lambda obj: [
        {
            "finding": fake.diagnosis(),
            "confidence": round(random.uniform(0.7, 0.99), 3),
            "location": fake.body_part(),
            "severity": random.choice(["mild", "moderate", "severe"])
        }
        for _ in range(random.randint(1, 4))
    ])
    
    ai_confidence = factory.LazyAttribute(
        lambda obj: max([f["confidence"] for f in obj.ai_findings]) if obj.ai_findings else 0.0
    )
    
    # 报告状态
    status = factory.fuzzy.FuzzyChoice(["draft", "pending_review", "reviewed", "finalized", "amended"])
    priority = factory.fuzzy.FuzzyChoice(["routine", "urgent", "stat"])
    
    # 医生信息
    radiologist_id = factory.LazyFunction(lambda: str(uuid.uuid4()))
    reviewing_doctor_id = factory.Maybe(
        'is_reviewed',
        yes_declaration=factory.LazyFunction(lambda: str(uuid.uuid4())),
        no_declaration=None
    )
    
    # 时间戳
    created_at = factory.Faker('date_time_between', start_date='-2m', end_date='now')
    dictated_at = factory.LazyAttribute(
        lambda obj: obj.created_at + timedelta(minutes=random.randint(5, 60))
    )
    reviewed_at = factory.Maybe(
        'is_reviewed',
        yes_declaration=factory.LazyAttribute(
            lambda obj: obj.dictated_at + timedelta(hours=random.randint(1, 24))
        ),
        no_declaration=None
    )
    finalized_at = factory.Maybe(
        'is_finalized',
        yes_declaration=factory.LazyAttribute(
            lambda obj: (obj.reviewed_at or obj.dictated_at) + timedelta(hours=random.randint(1, 12))
        ),
        no_declaration=None
    )
    
    # 质量指标
    turnaround_time = factory.LazyAttribute(
        lambda obj: (obj.finalized_at - obj.created_at).total_seconds() / 3600 if obj.finalized_at else None
    )
    
    word_count = factory.LazyAttribute(
        lambda obj: len(obj.findings.split()) + len(obj.impression.split())
    )
    
    @factory.lazy_attribute
    def is_reviewed(self):
        return self.status in ["reviewed", "finalized", "amended"]
    
    @factory.lazy_attribute
    def is_finalized(self):
        return self.status in ["finalized", "amended"]


class NotificationFactory(factory.Factory):
    """通知数据工厂"""
    
    class Meta:
        model = dict
    
    id = factory.LazyFunction(lambda: str(uuid.uuid4()))
    user_id = factory.LazyFunction(lambda: str(uuid.uuid4()))
    title = factory.Faker('sentence', nb_words=6)
    message = factory.Faker('text', max_nb_chars=200)
    type = factory.fuzzy.FuzzyChoice([
        "info", "warning", "error", "success", "urgent", "system"
    ])
    category = factory.fuzzy.FuzzyChoice([
        "task_completed", "new_assignment", "system_alert", "report_ready",
        "quality_issue", "maintenance", "security"
    ])
    priority = factory.fuzzy.FuzzyChoice(["low", "normal", "high", "urgent"])
    
    # 相关资源
    related_resource_type = factory.fuzzy.FuzzyChoice([
        "patient", "image", "report", "task", "user", "system"
    ])
    related_resource_id = factory.LazyFunction(lambda: str(uuid.uuid4()))
    
    # 状态
    is_read = factory.Faker('boolean', chance_of_getting_true=60)
    is_archived = factory.Faker('boolean', chance_of_getting_true=20)
    
    # 时间戳
    created_at = factory.Faker('date_time_between', start_date='-7d', end_date='now')
    read_at = factory.Maybe(
        'is_read',
        yes_declaration=factory.LazyAttribute(
            lambda obj: obj.created_at + timedelta(hours=random.randint(1, 48))
        ),
        no_declaration=None
    )
    
    # 元数据
    metadata = factory.LazyAttribute(lambda obj: {
        "source": random.choice(["system", "user", "ai", "external"]),
        "action_required": random.choice([True, False]),
        "expires_at": (obj.created_at + timedelta(days=random.randint(1, 30))).isoformat() if random.choice([True, False]) else None
    })


class WorklistItemFactory(factory.Factory):
    """工作列表项数据工厂"""
    
    class Meta:
        model = dict
    
    id = factory.LazyFunction(lambda: str(uuid.uuid4()))
    patient_id = factory.LazyFunction(lambda: str(uuid.uuid4()))
    image_id = factory.LazyFunction(lambda: str(uuid.uuid4()))
    assigned_to = factory.LazyFunction(lambda: str(uuid.uuid4()))
    
    # 任务信息
    task_type = factory.fuzzy.FuzzyChoice([
        "initial_reading", "second_opinion", "quality_review", "urgent_review",
        "teaching_case", "research_case"
    ])
    priority = factory.fuzzy.FuzzyChoice(["routine", "urgent", "stat", "callback"])
    status = factory.fuzzy.FuzzyChoice(["pending", "in_progress", "completed", "cancelled"])
    
    # 时间信息
    scheduled_date = factory.Faker('date_time_between', start_date='-1d', end_date='+7d')
    due_date = factory.LazyAttribute(
        lambda obj: obj.scheduled_date + timedelta(hours=random.randint(2, 48))
    )
    started_at = factory.Maybe(
        'is_started',
        yes_declaration=factory.Faker('date_time_between', start_date='-1d', end_date='now'),
        no_declaration=None
    )
    completed_at = factory.Maybe(
        'is_completed',
        yes_declaration=factory.LazyAttribute(
            lambda obj: obj.started_at + timedelta(minutes=random.randint(10, 120)) if obj.started_at else None
        ),
        no_declaration=None
    )
    
    # 元数据
    notes = factory.Faker('text', max_nb_chars=100)
    tags = factory.List([
        factory.fuzzy.FuzzyChoice(["urgent", "teaching", "research", "quality", "callback"])
        for _ in range(random.randint(0, 3))
    ])
    
    created_at = factory.Faker('date_time_between', start_date='-7d', end_date='now')
    created_by = factory.LazyFunction(lambda: str(uuid.uuid4()))
    
    @factory.lazy_attribute
    def is_started(self):
        return self.status in ["in_progress", "completed"]
    
    @factory.lazy_attribute
    def is_completed(self):
        return self.status == "completed"


# 批量生成工厂
class BatchFactory:
    """批量数据生成工厂"""
    
    @staticmethod
    def create_medical_department(num_doctors=5, num_patients=20, num_images=50):
        """创建完整的医学科室数据"""
        # 创建医生
        doctors = DoctorFactory.create_batch(num_doctors)
        radiologists = RadiologistFactory.create_batch(2)
        admin = AdminFactory()
        
        all_users = doctors + radiologists + [admin]
        
        # 创建患者
        patients = PatientFactory.create_batch(
            num_patients,
            created_by=factory.Iterator([u['id'] for u in all_users])
        )
        
        # 创建影像
        images = MedicalImageFactory.create_batch(
            num_images,
            patient_id=factory.Iterator([p['id'] for p in patients]),
            uploaded_by=factory.Iterator([u['id'] for u in all_users])
        )
        
        # 创建推理任务
        inference_tasks = InferenceTaskFactory.create_batch(
            num_images // 2,
            image_id=factory.Iterator([i['id'] for i in images]),
            created_by=factory.Iterator([u['id'] for u in all_users])
        )
        
        # 创建诊断报告
        reports = DiagnosticReportFactory.create_batch(
            num_images // 3,
            patient_id=factory.Iterator([p['id'] for p in patients]),
            image_id=factory.Iterator([i['id'] for i in images]),
            radiologist_id=factory.Iterator([r['id'] for r in radiologists])
        )
        
        # 创建工作列表
        worklist_items = WorklistItemFactory.create_batch(
            num_images // 4,
            patient_id=factory.Iterator([p['id'] for p in patients]),
            image_id=factory.Iterator([i['id'] for i in images]),
            assigned_to=factory.Iterator([u['id'] for u in all_users])
        )
        
        # 创建通知
        notifications = NotificationFactory.create_batch(
            num_doctors * 5,
            user_id=factory.Iterator([u['id'] for u in all_users])
        )
        
        return {
            'users': all_users,
            'doctors': doctors,
            'radiologists': radiologists,
            'admin': admin,
            'patients': patients,
            'images': images,
            'inference_tasks': inference_tasks,
            'reports': reports,
            'worklist_items': worklist_items,
            'notifications': notifications
        }
    
    @staticmethod
    def create_test_scenario(scenario_type="normal"):
        """创建特定测试场景的数据"""
        if scenario_type == "high_volume":
            return BatchFactory.create_medical_department(
                num_doctors=10, num_patients=100, num_images=500
            )
        elif scenario_type == "emergency":
            # 创建紧急情况数据
            patients = PatientFactory.create_batch(5)
            images = MedicalImageFactory.create_batch(
                10,
                patient_id=factory.Iterator([p['id'] for p in patients]),
                study_type="chest_xray"  # 急诊常见检查
            )
            tasks = InferenceTaskFactory.create_batch(
                10,
                image_id=factory.Iterator([i['id'] for i in images]),
                priority="urgent",
                status="pending"
            )
            return {
                'patients': patients,
                'images': images,
                'tasks': tasks
            }
        elif scenario_type == "quality_control":
            # 创建质量控制数据
            images = MedicalImageFactory.create_batch(
                20,
                quality_score=factory.Iterator([0.5, 0.6, 0.7, 0.8, 0.9, 0.95])
            )
            tasks = InferenceTaskFactory.create_batch(
                20,
                image_id=factory.Iterator([i['id'] for i in images]),
                status="completed"
            )
            return {
                'images': images,
                'tasks': tasks
            }
        else:
            return BatchFactory.create_medical_department()


# 导出主要工厂类
__all__ = [
    'UserFactory', 'DoctorFactory', 'RadiologistFactory', 'AdminFactory',
    'PatientFactory', 'MedicalImageFactory', 'InferenceTaskFactory',
    'DiagnosticReportFactory', 'NotificationFactory', 'WorklistItemFactory',
    'BatchFactory', 'MedicalProvider'
]