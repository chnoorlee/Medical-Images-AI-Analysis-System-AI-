#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from backend.core.init_data import init_permissions, init_roles, init_admin_user
from backend.core.database import get_db_context

def main():
    """重新初始化数据库权限和角色"""
    try:
        with get_db_context() as db:
            print("开始初始化权限...")
            init_permissions(db)
            
            print("开始初始化角色...")
            init_roles(db)
            
            print("开始初始化管理员用户...")
            init_admin_user(db)
            
            print("数据初始化完成！")
    except Exception as e:
        print(f"初始化失败: {e}")
        raise

if __name__ == "__main__":
    main()