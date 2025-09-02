import React from 'react'
import { Result, Button } from 'antd'
import { useNavigate } from 'react-router-dom'
import { HomeOutlined } from '@ant-design/icons'

const NotFoundPage: React.FC = () => {
  const navigate = useNavigate()

  const handleBackHome = () => {
    navigate('/')
  }

  const handleGoBack = () => {
    navigate(-1)
  }

  return (
    <div style={{ 
      display: 'flex', 
      justifyContent: 'center', 
      alignItems: 'center', 
      minHeight: '100vh',
      background: '#f5f5f5'
    }}>
      <Result
        status="404"
        title="404"
        subTitle="抱歉，您访问的页面不存在。"
        extra={
          <div style={{ display: 'flex', gap: '12px', justifyContent: 'center' }}>
            <Button type="primary" icon={<HomeOutlined />} onClick={handleBackHome}>
              返回首页
            </Button>
            <Button onClick={handleGoBack}>
              返回上一页
            </Button>
          </div>
        }
      />
    </div>
  )
}

export default NotFoundPage