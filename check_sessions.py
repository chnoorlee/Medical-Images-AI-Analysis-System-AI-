from backend.core.database import get_db_context
from backend.models.user import UserSession

# 新生成的token
new_token = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ1c2VyX2lkIjoiOTA5NDk1NWUtM2NjYi00MWJhLWEwYTgtYWY2M2Q5N2M3ZmEzIiwidXNlcm5hbWUiOiJhZG1pbiIsInJvbGUiOiJzdXBlcl9hZG1pbiIsImV4cCI6MTc1NjgxMDUyMiwiaWF0IjoxNzU2ODA4NzIyLCJ0eXBlIjoiYWNjZXNzIn0.AdTC9iX6NqL0-Gh_AN-flx4_OmQZaqYwIn9MEdWLqew"

with get_db_context() as db:
    # 查找匹配的会话
    matching_session = db.query(UserSession).filter(
        UserSession.access_token == new_token,
        UserSession.is_active == True
    ).first()
    
    if matching_session:
        print(f"找到匹配的会话: {matching_session.session_id}")
        print(f"用户ID: {matching_session.user_id}")
        print(f"创建时间: {matching_session.created_at}")
        print(f"过期时间: {matching_session.expires_at}")
        print(f"是否活跃: {matching_session.is_active}")
    else:
        print("未找到匹配的会话")
        
        # 显示最近的几个会话
        recent_sessions = db.query(UserSession).filter(
            UserSession.is_active == True
        ).order_by(UserSession.created_at.desc()).limit(3).all()
        
        print("\n最近的3个活跃会话:")
        for session in recent_sessions:
            token_preview = session.access_token[:50] + '...' if session.access_token else 'None'
            print(f"会话ID: {session.session_id}")
            print(f"Token前50位: {token_preview}")
            print(f"创建时间: {session.created_at}")
            print("---")