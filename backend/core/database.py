from sqlalchemy import create_engine, MetaData, text
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import StaticPool
from typing import Generator
import os
from contextlib import contextmanager

# 数据库配置
DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "sqlite:///./medical_ai.db"
)

# 测试数据库配置
TEST_DATABASE_URL = os.getenv(
    "TEST_DATABASE_URL",
    "sqlite:///./test_medical_ai.db"
)

# 创建数据库引擎
engine = create_engine(
    DATABASE_URL,
    pool_pre_ping=True,
    pool_recycle=300,
    pool_size=20,
    max_overflow=30,
    echo=os.getenv("SQL_ECHO", "false").lower() == "true"
)

# 测试数据库引擎
test_engine = create_engine(
    TEST_DATABASE_URL,
    pool_pre_ping=True,
    poolclass=StaticPool,
    echo=False
)

# 创建会话工厂
SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine
)

# 测试会话工厂
TestSessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=test_engine
)

# 导入共享的Base类
from backend.models.base import Base

# 元数据
metadata = MetaData()

def get_db() -> Generator[Session, None, None]:
    """获取数据库会话"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def get_test_db() -> Generator[Session, None, None]:
    """获取测试数据库会话"""
    db = TestSessionLocal()
    try:
        yield db
    finally:
        db.close()

@contextmanager
def get_db_context():
    """数据库会话上下文管理器"""
    db = SessionLocal()
    try:
        yield db
        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()

def create_tables():
    """创建所有数据库表"""
    # 导入所有模型以确保它们被注册
    from backend.models.patient import Patient, Study, Series, Image, ImageMetadata
    from backend.models.annotation import (
        Annotation, AnnotationConsensus, AnnotationConsensusParticipant, AnnotationTask
    )
    from backend.models.user import User, Role, Permission, UserSession
    from backend.models.ai_model import AIModel, ModelVersion, Inference, ModelTrainingJob
    from backend.models.audit import AuditLog, SecurityEvent, DataAccessLog, ComplianceReport
    from backend.models.quality import (
        QualityMetrics, QualityAssessment, QualityControlRule, 
        QualityControlExecution, DataQualityReport, QualityImprovement
    )
    
    # 创建所有表
    Base.metadata.create_all(bind=engine)
    print("数据库表创建完成")

def drop_tables():
    """删除所有数据库表"""
    Base.metadata.drop_all(bind=engine)
    print("数据库表删除完成")

def create_test_tables():
    """创建测试数据库表"""
    # 导入所有模型
    from backend.models.patient import Patient, Study, Series, Image, ImageMetadata
    from backend.models.annotation import (
        Annotation, AnnotationConsensus, AnnotationConsensusParticipant, AnnotationTask
    )
    from backend.models.user import User, Role, Permission, UserSession
    from backend.models.ai_model import AIModel, ModelVersion, Inference, ModelTrainingJob
    from backend.models.audit import AuditLog, SecurityEvent, DataAccessLog, ComplianceReport
    from backend.models.quality import (
        QualityMetrics, QualityAssessment, QualityControlRule, 
        QualityControlExecution, DataQualityReport, QualityImprovement
    )
    
    # 创建测试表
    Base.metadata.create_all(bind=test_engine)
    print("测试数据库表创建完成")

def init_database():
    """初始化数据库"""
    try:
        # 创建表
        create_tables()
        
        # 初始化基础数据
        from backend.core.init_data import init_base_data
        init_base_data()
        
        print("数据库初始化完成")
    except Exception as e:
        print(f"数据库初始化失败: {e}")
        raise

def check_database_connection():
    """检查数据库连接"""
    try:
        with engine.connect() as connection:
            result = connection.execute(text("SELECT 1"))
            return result.fetchone()[0] == 1
    except Exception as e:
        print(f"数据库连接检查失败: {e}")
        return False

def get_database_info():
    """获取数据库信息"""
    try:
        with engine.connect() as connection:
            # 获取数据库版本
            version_result = connection.execute("SELECT version()")
            version = version_result.fetchone()[0]
            
            # 获取当前数据库名
            db_result = connection.execute("SELECT current_database()")
            database_name = db_result.fetchone()[0]
            
            # 获取当前用户
            user_result = connection.execute("SELECT current_user")
            current_user = user_result.fetchone()[0]
            
            return {
                "version": version,
                "database_name": database_name,
                "current_user": current_user,
                "url": DATABASE_URL.split('@')[1] if '@' in DATABASE_URL else DATABASE_URL
            }
    except Exception as e:
        print(f"获取数据库信息失败: {e}")
        return None

class DatabaseManager:
    """数据库管理器"""
    
    def __init__(self):
        self.engine = engine
        self.SessionLocal = SessionLocal
    
    def create_session(self) -> Session:
        """创建数据库会话"""
        return self.SessionLocal()
    
    def execute_raw_sql(self, sql: str, params: dict = None):
        """执行原始SQL"""
        with self.engine.connect() as connection:
            if params:
                result = connection.execute(sql, params)
            else:
                result = connection.execute(sql)
            return result.fetchall()
    
    def backup_database(self, backup_path: str):
        """备份数据库"""
        import subprocess
        import os
        
        # 从DATABASE_URL解析连接信息
        # postgresql://user:password@host:port/database
        url_parts = DATABASE_URL.replace('postgresql://', '').split('/')
        db_name = url_parts[1]
        user_host = url_parts[0].split('@')
        user_pass = user_host[0].split(':')
        host_port = user_host[1].split(':')
        
        username = user_pass[0]
        password = user_pass[1] if len(user_pass) > 1 else ''
        host = host_port[0]
        port = host_port[1] if len(host_port) > 1 else '5432'
        
        # 设置环境变量
        env = os.environ.copy()
        if password:
            env['PGPASSWORD'] = password
        
        # 执行pg_dump
        cmd = [
            'pg_dump',
            '-h', host,
            '-p', port,
            '-U', username,
            '-d', db_name,
            '-f', backup_path,
            '--verbose'
        ]
        
        try:
            subprocess.run(cmd, env=env, check=True)
            print(f"数据库备份成功: {backup_path}")
            return True
        except subprocess.CalledProcessError as e:
            print(f"数据库备份失败: {e}")
            return False
    
    def restore_database(self, backup_path: str):
        """恢复数据库"""
        import subprocess
        import os
        
        # 解析连接信息（同备份方法）
        url_parts = DATABASE_URL.replace('postgresql://', '').split('/')
        db_name = url_parts[1]
        user_host = url_parts[0].split('@')
        user_pass = user_host[0].split(':')
        host_port = user_host[1].split(':')
        
        username = user_pass[0]
        password = user_pass[1] if len(user_pass) > 1 else ''
        host = host_port[0]
        port = host_port[1] if len(host_port) > 1 else '5432'
        
        # 设置环境变量
        env = os.environ.copy()
        if password:
            env['PGPASSWORD'] = password
        
        # 执行psql恢复
        cmd = [
            'psql',
            '-h', host,
            '-p', port,
            '-U', username,
            '-d', db_name,
            '-f', backup_path,
            '--verbose'
        ]
        
        try:
            subprocess.run(cmd, env=env, check=True)
            print(f"数据库恢复成功: {backup_path}")
            return True
        except subprocess.CalledProcessError as e:
            print(f"数据库恢复失败: {e}")
            return False

# 全局数据库管理器实例
db_manager = DatabaseManager()