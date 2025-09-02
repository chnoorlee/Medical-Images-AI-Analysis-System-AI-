#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from backend.core.database import get_db_context
from backend.models.user import User, Role, Permission

def check_admin_permissions():
    """检查admin用户的权限配置"""
    try:
        with get_db_context() as db:
            # 查找admin用户
            admin = db.query(User).filter(User.username == 'admin').first()
            
            if not admin:
                print("Admin用户不存在")
                return
            
            print(f"Admin用户: {admin.username}")
            print(f"用户ID: {admin.user_id}")
            print(f"是否激活: {admin.is_active}")
            print(f"角色数量: {len(admin.roles)}")
            
            if admin.roles:
                for role in admin.roles:
                    print(f"\n角色: {role.role_name} ({role.role_display_name})")
                    print(f"权限数量: {len(role.permissions)}")
                    
                    for perm in role.permissions:
                        print(f"  - {perm.permission_name}: {perm.permission_display_name}")
            else:
                print("用户没有分配任何角色")
            
            # 检查特定权限
            all_permissions = []
            for role in admin.roles:
                for perm in role.permissions:
                    if perm.permission_name not in all_permissions:
                        all_permissions.append(perm.permission_name)
            
            print(f"\n总权限数量: {len(all_permissions)}")
            print(f"是否有model.read权限: {'model.read' in all_permissions}")
            print(f"是否有model.create权限: {'model.create' in all_permissions}")
            
    except Exception as e:
        print(f"检查权限失败: {e}")

if __name__ == "__main__":
    check_admin_permissions()