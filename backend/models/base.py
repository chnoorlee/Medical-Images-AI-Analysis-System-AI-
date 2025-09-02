from sqlalchemy.ext.declarative import declarative_base

# 共享的Base类，所有模型都应该继承这个Base
Base = declarative_base()