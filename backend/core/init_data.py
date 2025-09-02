from sqlalchemy.orm import Session
from backend.core.database import get_db_context
from backend.models.user import User, Role, Permission, user_roles, role_permissions
from backend.models.quality import QualityMetrics
from passlib.context import CryptContext
from datetime import datetime, timezone
import uuid

# 密码加密上下文
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

def hash_password(password: str) -> str:
    """加密密码"""
    return pwd_context.hash(password)

def init_permissions(db: Session):
    """初始化权限数据"""
    permissions_data = [
        # 患者管理权限
        {"permission_name": "patient.create", "permission_display_name": "创建患者", "description": "创建新患者记录", "resource": "patient", "action": "create"},
        {"permission_name": "patient.read", "permission_display_name": "查看患者", "description": "查看患者信息", "resource": "patient", "action": "read"},
        {"permission_name": "patient.update", "permission_display_name": "更新患者", "description": "更新患者信息", "resource": "patient", "action": "update"},
        {"permission_name": "patient.delete", "permission_display_name": "删除患者", "description": "删除患者记录", "resource": "patient", "action": "delete"},
        
        # 图像管理权限
        {"permission_name": "image.create", "permission_display_name": "上传图像", "description": "上传医学图像", "resource": "image", "action": "create"},
        {"permission_name": "image.read", "permission_display_name": "查看图像", "description": "查看医学图像", "resource": "image", "action": "read"},
        {"permission_name": "image.update", "permission_display_name": "更新图像", "description": "更新图像信息", "resource": "image", "action": "update"},
        {"permission_name": "image.delete", "permission_display_name": "删除图像", "description": "删除医学图像", "resource": "image", "action": "delete"},
        {"permission_name": "image.download", "permission_display_name": "下载图像", "description": "下载医学图像", "resource": "image", "action": "download"},
        
        # 标注管理权限
        {"permission_name": "annotation.create", "permission_display_name": "创建标注", "description": "创建图像标注", "resource": "annotation", "action": "create"},
        {"permission_name": "annotation.read", "permission_display_name": "查看标注", "description": "查看图像标注", "resource": "annotation", "action": "read"},
        {"permission_name": "annotation.update", "permission_display_name": "更新标注", "description": "更新图像标注", "resource": "annotation", "action": "update"},
        {"permission_name": "annotation.delete", "permission_display_name": "删除标注", "description": "删除图像标注", "resource": "annotation", "action": "delete"},
        {"permission_name": "annotation.validate", "permission_display_name": "验证标注", "description": "验证标注质量", "resource": "annotation", "action": "validate"},
        
        # AI模型权限
        {"permission_name": "model.create", "permission_display_name": "创建模型", "description": "创建AI模型", "resource": "model", "action": "create"},
        {"permission_name": "model.read", "permission_display_name": "查看模型", "description": "查看AI模型", "resource": "model", "action": "read"},
        {"permission_name": "model.update", "permission_display_name": "更新模型", "description": "更新AI模型", "resource": "model", "action": "update"},
        {"permission_name": "model.delete", "permission_display_name": "删除模型", "description": "删除AI模型", "resource": "model", "action": "delete"},
        {"permission_name": "model.execute", "permission_display_name": "执行推理", "description": "执行模型推理", "resource": "model", "action": "execute"},
        {"permission_name": "model.train", "permission_display_name": "训练模型", "description": "训练AI模型", "resource": "model", "action": "train"},
        {"permission_name": "model.deploy", "permission_display_name": "部署模型", "description": "部署AI模型", "resource": "model", "action": "deploy"},
        
        # API端点权限
        {"permission_name": "model_view", "permission_display_name": "模型查看权限", "description": "查看模型列表和详情的API权限", "resource": "model", "action": "view"},
        {"permission_name": "model_management", "permission_display_name": "模型管理权限", "description": "管理模型的API权限", "resource": "model", "action": "management"},
        {"permission_name": "model_inference", "permission_display_name": "模型推理权限", "description": "执行模型推理的API权限", "resource": "model", "action": "inference"},
        {"permission_name": "model_training", "permission_display_name": "模型训练权限", "description": "训练模型的API权限", "resource": "model", "action": "training"},
        
        # 用户管理权限
        {"permission_name": "user.create", "permission_display_name": "创建用户", "description": "创建新用户", "resource": "user", "action": "create"},
        {"permission_name": "user.read", "permission_display_name": "查看用户", "description": "查看用户信息", "resource": "user", "action": "read"},
        {"permission_name": "user.update", "permission_display_name": "更新用户", "description": "更新用户信息", "resource": "user", "action": "update"},
        {"permission_name": "user.delete", "permission_display_name": "删除用户", "description": "删除用户", "resource": "user", "action": "delete"},
        
        # 系统管理权限
        {"permission_name": "system.admin", "permission_display_name": "系统管理", "description": "系统管理员权限", "resource": "system", "action": "admin"},
        {"permission_name": "system.audit", "permission_display_name": "审计查看", "description": "查看审计日志", "resource": "system", "action": "audit"},
        {"permission_name": "system.backup", "permission_display_name": "数据备份", "description": "执行数据备份", "resource": "system", "action": "backup"},
        
        # 质量控制权限
        {"permission_name": "quality.assess", "permission_display_name": "质量评估", "description": "执行质量评估", "resource": "quality", "action": "assess"},
        {"permission_name": "quality.control", "permission_display_name": "质量控制", "description": "执行质量控制", "resource": "quality", "action": "control"},
        {"permission_name": "quality.report", "permission_display_name": "质量报告", "description": "生成质量报告", "resource": "quality", "action": "report"},
    ]
    
    for perm_data in permissions_data:
        existing_perm = db.query(Permission).filter(
            Permission.permission_name == perm_data["permission_name"]
        ).first()
        
        if not existing_perm:
            permission = Permission(
                permission_id=uuid.uuid4(),
                **perm_data,
                is_system_permission=True
            )
            db.add(permission)
    
    db.commit()
    print("权限数据初始化完成")

def init_roles(db: Session):
    """初始化角色数据"""
    roles_data = [
        {
            "role_name": "super_admin",
            "role_display_name": "超级管理员",
            "description": "系统超级管理员，拥有所有权限",
            "permissions": [
                "system.admin", "system.audit", "system.backup",
                "user.create", "user.read", "user.update", "user.delete",
                "patient.create", "patient.read", "patient.update", "patient.delete",
                "image.create", "image.read", "image.update", "image.delete", "image.download",
                "annotation.create", "annotation.read", "annotation.update", "annotation.delete", "annotation.validate",
                "model.create", "model.read", "model.update", "model.delete", "model.execute", "model.train", "model.deploy",
                "model_view", "model_management", "model_inference", "model_training",
                "quality.assess", "quality.control", "quality.report"
            ]
        },
        {
            "role_name": "radiologist",
            "role_display_name": "放射科医生",
            "description": "放射科医生，负责图像诊断和标注",
            "permissions": ["patient.read", "image.read", "image.download", "annotation.create", "annotation.read", "annotation.update", "model.execute"]
        },
        {
            "role_name": "technician",
            "role_display_name": "技师",
            "description": "医学技师，负责图像采集和上传",
            "permissions": ["patient.create", "patient.read", "patient.update", "image.create", "image.read", "image.update"]
        },
        {
            "role_name": "researcher",
            "role_display_name": "研究员",
            "description": "AI研究员，负责模型开发和训练",
            "permissions": ["image.read", "annotation.read", "model.create", "model.read", "model.update", "model.train", "model.execute"]
        },
        {
            "role_name": "quality_controller",
            "role_display_name": "质量控制员",
            "description": "质量控制专员，负责数据质量管理",
            "permissions": ["image.read", "annotation.read", "annotation.validate", "quality.assess", "quality.control", "quality.report"]
        },
        {
            "role_name": "data_manager",
            "role_display_name": "数据管理员",
            "description": "数据管理员，负责数据管理和维护",
            "permissions": ["patient.read", "patient.update", "image.read", "image.update", "image.delete", "annotation.read", "system.backup"]
        },
        {
            "role_name": "viewer",
            "role_display_name": "查看者",
            "description": "只读用户，只能查看数据",
            "permissions": ["patient.read", "image.read", "annotation.read", "model.read"]
        }
    ]
    
    for role_data in roles_data:
        existing_role = db.query(Role).filter(
            Role.role_name == role_data["role_name"]
        ).first()
        
        if not existing_role:
            role = Role(
                role_id=uuid.uuid4(),
                role_name=role_data["role_name"],
                role_display_name=role_data["role_display_name"],
                description=role_data["description"],
                is_system_role=True
            )
            db.add(role)
            db.flush()  # 确保角色被保存并获得ID
        else:
            role = existing_role
            # 清除现有权限关联
            role.permissions.clear()
            
        # 分配权限给角色
        for perm_name in role_data["permissions"]:
            permission = db.query(Permission).filter(
                Permission.permission_name == perm_name
            ).first()
            if permission:
                role.permissions.append(permission)
    
    db.commit()
    print("角色数据初始化完成")

def init_admin_user(db: Session):
    """初始化管理员用户"""
    admin_username = "admin"
    admin_email = "admin@medical-ai.com"
    
    existing_admin = db.query(User).filter(
        User.username == admin_username
    ).first()
    
    if not existing_admin:
        # 创建管理员用户
        admin_user = User(
            user_id=uuid.uuid4(),
            username=admin_username,
            email=admin_email,
            hashed_password=hash_password("admin123456"),  # 默认密码，生产环境需要修改
            full_name="系统管理员",
            department="信息科",
            title="系统管理员",
            is_active=True,
            is_verified=True
        )
        db.add(admin_user)
        db.flush()
        
        # 分配超级管理员角色
        super_admin_role = db.query(Role).filter(
            Role.role_name == "super_admin"
        ).first()
        
        if super_admin_role:
            admin_user.roles.append(super_admin_role)
        
        db.commit()
        print(f"管理员用户创建完成: {admin_username} / admin123456")
    else:
        print("管理员用户已存在")

def init_quality_metrics(db: Session):
    """初始化质量指标"""
    metrics_data = [
        # 图像技术质量指标
        {
            "metric_name": "image_resolution",
            "metric_category": "technical",
            "description": "图像分辨率质量评估",
            "measurement_unit": "pixels",
            "data_type": "numeric",
            "min_value": 0,
            "max_value": 4096,
            "target_value": 512,
            "threshold_excellent": 1024,
            "threshold_good": 512,
            "threshold_acceptable": 256,
            "calculation_method": "基于图像像素尺寸计算",
            "automation_level": "automated"
        },
        {
            "metric_name": "signal_noise_ratio",
            "metric_category": "technical",
            "description": "信噪比评估",
            "measurement_unit": "dB",
            "data_type": "numeric",
            "min_value": 0,
            "max_value": 100,
            "target_value": 30,
            "threshold_excellent": 40,
            "threshold_good": 30,
            "threshold_acceptable": 20,
            "calculation_method": "信号功率与噪声功率比值的对数",
            "automation_level": "automated"
        },
        {
            "metric_name": "contrast_quality",
            "metric_category": "technical",
            "description": "对比度质量评估",
            "measurement_unit": "ratio",
            "data_type": "numeric",
            "min_value": 0,
            "max_value": 1,
            "target_value": 0.8,
            "threshold_excellent": 0.9,
            "threshold_good": 0.8,
            "threshold_acceptable": 0.6,
            "calculation_method": "基于图像对比度统计分析",
            "automation_level": "automated"
        },
        
        # 标注质量指标
        {
            "metric_name": "annotation_accuracy",
            "metric_category": "annotation",
            "description": "标注准确性评估",
            "measurement_unit": "percentage",
            "data_type": "percentage",
            "min_value": 0,
            "max_value": 100,
            "target_value": 95,
            "threshold_excellent": 98,
            "threshold_good": 95,
            "threshold_acceptable": 90,
            "calculation_method": "与金标准对比的准确率",
            "automation_level": "semi_automated"
        },
        {
            "metric_name": "annotation_consistency",
            "metric_category": "annotation",
            "description": "标注一致性评估",
            "measurement_unit": "kappa",
            "data_type": "numeric",
            "min_value": 0,
            "max_value": 1,
            "target_value": 0.8,
            "threshold_excellent": 0.9,
            "threshold_good": 0.8,
            "threshold_acceptable": 0.6,
            "calculation_method": "多标注者间Kappa一致性系数",
            "automation_level": "automated"
        },
        
        # 模型性能指标
        {
            "metric_name": "model_accuracy",
            "metric_category": "model_performance",
            "description": "模型准确率",
            "measurement_unit": "percentage",
            "data_type": "percentage",
            "min_value": 0,
            "max_value": 100,
            "target_value": 90,
            "threshold_excellent": 95,
            "threshold_good": 90,
            "threshold_acceptable": 85,
            "calculation_method": "正确预测样本数/总样本数",
            "automation_level": "automated"
        },
        {
            "metric_name": "model_sensitivity",
            "metric_category": "model_performance",
            "description": "模型敏感性（召回率）",
            "measurement_unit": "percentage",
            "data_type": "percentage",
            "min_value": 0,
            "max_value": 100,
            "target_value": 90,
            "threshold_excellent": 95,
            "threshold_good": 90,
            "threshold_acceptable": 85,
            "calculation_method": "真阳性/(真阳性+假阴性)",
            "automation_level": "automated"
        },
        {
            "metric_name": "model_specificity",
            "metric_category": "model_performance",
            "description": "模型特异性",
            "measurement_unit": "percentage",
            "data_type": "percentage",
            "min_value": 0,
            "max_value": 100,
            "target_value": 90,
            "threshold_excellent": 95,
            "threshold_good": 90,
            "threshold_acceptable": 85,
            "calculation_method": "真阴性/(真阴性+假阳性)",
            "automation_level": "automated"
        },
        
        # 临床质量指标
        {
            "metric_name": "diagnostic_confidence",
            "metric_category": "clinical",
            "description": "诊断置信度",
            "measurement_unit": "score",
            "data_type": "numeric",
            "min_value": 1,
            "max_value": 5,
            "target_value": 4,
            "threshold_excellent": 5,
            "threshold_good": 4,
            "threshold_acceptable": 3,
            "calculation_method": "医生主观评分1-5分",
            "automation_level": "manual"
        }
    ]
    
    for metric_data in metrics_data:
        existing_metric = db.query(QualityMetrics).filter(
            QualityMetrics.metric_name == metric_data["metric_name"]
        ).first()
        
        if not existing_metric:
            metric = QualityMetrics(
                metric_id=uuid.uuid4(),
                **metric_data
            )
            db.add(metric)
    
    db.commit()
    print("质量指标数据初始化完成")

def init_base_data():
    """初始化所有基础数据"""
    try:
        with get_db_context() as db:
            print("开始初始化基础数据...")
            
            # 初始化权限
            init_permissions(db)
            
            # 初始化角色
            init_roles(db)
            
            # 初始化管理员用户
            init_admin_user(db)
            
            # 初始化质量指标
            init_quality_metrics(db)
            
            print("基础数据初始化完成")
            
    except Exception as e:
        print(f"基础数据初始化失败: {e}")
        raise

if __name__ == "__main__":
    init_base_data()