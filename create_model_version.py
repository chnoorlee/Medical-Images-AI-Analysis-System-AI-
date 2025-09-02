#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
为现有模型创建版本记录的脚本
"""

import sys
import os

# 添加项目根目录到Python路径
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

from backend.core.database import get_db
from backend.models.ai_model import AIModel, ModelVersion
from sqlalchemy.orm import Session
import uuid
from datetime import datetime

def create_model_version():
    """为现有模型创建版本记录"""
    db: Session = next(get_db())
    
    try:
        # 查找所有没有版本的模型
        models = db.query(AIModel).all()
        
        for model in models:
            # 检查是否已有版本
            existing_version = db.query(ModelVersion).filter(
                ModelVersion.model_id == model.model_id
            ).first()
            
            if not existing_version:
                print(f"为模型 {model.model_name} 创建版本记录...")
                
                # 创建默认版本
                model_version = ModelVersion(
                    version_id=uuid.uuid4(),
                    model_id=model.model_id,
                    version_number="1.0.0",
                    version_name="Initial Version",
                    description="初始版本",
                    model_file_path=f"models/{model.model_name}/model.pth",
                    is_production=True,
                    deployment_status="deployed",
                    deployment_config={
                        "input_size": [224, 224],
                        "num_classes": 2,
                        "architecture": model.architecture
                    }
                )
                
                db.add(model_version)
                print(f"✓ 已为模型 {model.model_name} 创建版本 1.0.0")
            else:
                print(f"模型 {model.model_name} 已有版本记录")
        
        db.commit()
        print("\n所有模型版本记录创建完成！")
        
    except Exception as e:
        print(f"创建模型版本时出错: {e}")
        db.rollback()
    finally:
        db.close()

if __name__ == "__main__":
    create_model_version()