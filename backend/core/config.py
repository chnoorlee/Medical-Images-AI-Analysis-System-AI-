import os
from typing import Optional, List
from pydantic import validator
from pydantic_settings import BaseSettings
from functools import lru_cache

class Settings(BaseSettings):
    """应用配置类"""
    
    # 应用基础配置
    app_name: str = "Medical AI Platform"
    app_version: str = "1.0.0"
    debug: bool = False
    environment: str = "development"
    
    # 服务器配置
    host: str = "0.0.0.0"
    port: int = 8000
    reload: bool = True
    
    # 数据库配置
    database_url: str = "sqlite:///./medical_ai.db"
    database_echo: bool = False
    database_pool_size: int = 10
    database_max_overflow: int = 20
    database_pool_timeout: int = 30
    database_pool_recycle: int = 3600
    
    # 测试数据库配置
    test_database_url: str = "sqlite:///./test_medical_ai.db"
    
    # Redis配置
    redis_url: str = "redis://localhost:6379/0"
    redis_password: Optional[str] = None
    redis_db: int = 0
    
    # JWT配置
    secret_key: str = "your-secret-key-change-in-production"
    algorithm: str = "HS256"
    access_token_expire_minutes: int = 30
    refresh_token_expire_days: int = 7
    
    # 安全配置
    allowed_hosts: List[str] = ["*"]
    cors_origins: List[str] = ["http://localhost:3000", "http://localhost:5173"]
    cors_credentials: bool = True
    cors_methods: List[str] = ["*"]
    cors_headers: List[str] = ["*"]
    
    # 文件存储配置
    upload_dir: str = "./uploads"
    max_file_size: int = 100 * 1024 * 1024  # 100MB
    allowed_image_types: List[str] = [".dcm", ".jpg", ".jpeg", ".png", ".tiff", ".nii", ".nii.gz"]
    
    # MinIO/S3配置
    minio_endpoint: str = "localhost:9000"
    minio_access_key: str = "minioadmin"
    minio_secret_key: str = "minioadmin"
    minio_bucket_name: str = "medical-images"
    minio_secure: bool = False
    
    # AI模型配置
    model_storage_path: str = "./models"
    inference_timeout: int = 300  # 5分钟
    max_concurrent_inferences: int = 5
    
    # 日志配置
    log_level: str = "INFO"
    log_file: str = "./logs/app.log"
    log_max_size: int = 10 * 1024 * 1024  # 10MB
    log_backup_count: int = 5
    
    # 监控配置
    enable_metrics: bool = True
    metrics_port: int = 8001
    
    # 邮件配置
    smtp_server: str = "smtp.gmail.com"
    smtp_port: int = 587
    smtp_username: Optional[str] = None
    smtp_password: Optional[str] = None
    smtp_use_tls: bool = True
    
    # 质量控制配置
    quality_check_enabled: bool = True
    auto_quality_assessment: bool = True
    quality_threshold_warning: float = 0.7
    quality_threshold_error: float = 0.5
    
    # 数据备份配置
    backup_enabled: bool = True
    backup_schedule: str = "0 2 * * *"  # 每天凌晨2点
    backup_retention_days: int = 30
    backup_storage_path: str = "./backups"
    
    # 缓存配置
    cache_ttl: int = 3600  # 1小时
    cache_max_size: int = 1000
    
    # API限流配置
    rate_limit_enabled: bool = True
    rate_limit_requests: int = 100
    rate_limit_window: int = 60  # 60秒
    
    # 健康检查配置
    health_check_interval: int = 30  # 30秒
    
    @validator('database_url')
    def validate_database_url(cls, v):
        """验证数据库URL"""
        if not v:
            raise ValueError('数据库URL不能为空')
        return v
    
    @validator('secret_key')
    def validate_secret_key(cls, v, values):
        """验证密钥"""
        if values.get('environment') == 'production' and v == 'your-secret-key-change-in-production':
            raise ValueError('生产环境必须设置安全的密钥')
        if len(v) < 32:
            raise ValueError('密钥长度至少32个字符')
        return v
    
    @validator('upload_dir')
    def validate_upload_dir(cls, v):
        """验证上传目录"""
        os.makedirs(v, exist_ok=True)
        return v
    
    @validator('model_storage_path')
    def validate_model_storage_path(cls, v):
        """验证模型存储路径"""
        os.makedirs(v, exist_ok=True)
        return v
    
    @validator('backup_storage_path')
    def validate_backup_storage_path(cls, v):
        """验证备份存储路径"""
        os.makedirs(v, exist_ok=True)
        return v
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False

class DevelopmentSettings(Settings):
    """开发环境配置"""
    debug: bool = True
    environment: str = "development"
    database_echo: bool = True
    log_level: str = "DEBUG"
    reload: bool = True

class ProductionSettings(Settings):
    """生产环境配置"""
    debug: bool = False
    environment: str = "production"
    database_echo: bool = False
    log_level: str = "INFO"
    reload: bool = False
    
    # 生产环境安全配置
    allowed_hosts: List[str] = ["your-domain.com"]
    cors_origins: List[str] = ["https://your-frontend-domain.com"]

class TestingSettings(Settings):
    """测试环境配置"""
    debug: bool = True
    environment: str = "testing"
    database_url: str = "sqlite:///./test_medical_ai.db"
    database_echo: bool = False
    log_level: str = "DEBUG"
    
    # 测试环境禁用某些功能
    backup_enabled: bool = False
    enable_metrics: bool = False
    rate_limit_enabled: bool = False

@lru_cache()
def get_settings() -> Settings:
    """获取配置实例（单例模式）"""
    environment = os.getenv("ENVIRONMENT", "development").lower()
    
    if environment == "production":
        return ProductionSettings()
    elif environment == "testing":
        return TestingSettings()
    else:
        return DevelopmentSettings()

# 全局配置实例
settings = get_settings()

# 数据库配置字典
DATABASE_CONFIG = {
    "development": {
        "url": "sqlite:///./medical_ai.db",
        "echo": True,
        "pool_size": 5,
        "max_overflow": 10
    },
    "testing": {
        "url": "sqlite:///./test_medical_ai.db",
        "echo": False,
        "pool_size": 1,
        "max_overflow": 0
    },
    "production": {
        "url": os.getenv("DATABASE_URL", "postgresql://user:password@localhost/medical_ai"),
        "echo": False,
        "pool_size": 20,
        "max_overflow": 30
    }
}

# 日志配置
LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "default": {
            "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            "datefmt": "%Y-%m-%d %H:%M:%S"
        },
        "detailed": {
            "format": "%(asctime)s - %(name)s - %(levelname)s - %(module)s - %(funcName)s - %(lineno)d - %(message)s",
            "datefmt": "%Y-%m-%d %H:%M:%S"
        }
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "level": settings.log_level,
            "formatter": "default",
            "stream": "ext://sys.stdout"
        },
        "file": {
            "class": "logging.handlers.RotatingFileHandler",
            "level": settings.log_level,
            "formatter": "detailed",
            "filename": settings.log_file,
            "maxBytes": settings.log_max_size,
            "backupCount": settings.log_backup_count,
            "encoding": "utf-8"
        }
    },
    "loggers": {
        "": {
            "level": settings.log_level,
            "handlers": ["console", "file"],
            "propagate": False
        },
        "uvicorn": {
            "level": "INFO",
            "handlers": ["console"],
            "propagate": False
        },
        "sqlalchemy": {
            "level": "WARNING",
            "handlers": ["console"],
            "propagate": False
        }
    }
}

# 确保日志目录存在
os.makedirs(os.path.dirname(settings.log_file), exist_ok=True)