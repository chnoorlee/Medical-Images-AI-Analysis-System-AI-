#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import requests
import json

def test_image_upload():
    """测试图像上传功能"""
    base_url = "http://localhost:8000"
    
    # 1. 登录获取token
    login_data = {
        "username": "admin",
        "password": "admin123456"
    }
    
    login_response = requests.post(
        f"{base_url}/api/auth/login",
        json=login_data
    )
    
    if login_response.status_code != 200:
        print(f"登录失败: {login_response.text}")
        return
    
    token = login_response.json()["access_token"]
    print(f"登录成功，获取token: {token[:20]}...")
    
    # 2. 上传图像
    headers = {
        "Authorization": f"Bearer {token}"
    }
    
    files = {
        "file": ("test_image.jpg", open("test_image.jpg", "rb"), "image/jpeg")
    }
    
    data = {
        "auto_process": "true"
    }
    
    upload_response = requests.post(
        f"{base_url}/api/images/upload",
        headers=headers,
        files=files,
        data=data
    )
    
    print(f"上传响应状态码: {upload_response.status_code}")
    print(f"上传响应内容: {upload_response.text}")
    
    if upload_response.status_code == 200:
        result = upload_response.json()
        print(f"图像上传成功! 图像ID: {result.get('image_id')}")
    else:
        print(f"图像上传失败: {upload_response.text}")

if __name__ == "__main__":
    test_image_upload()