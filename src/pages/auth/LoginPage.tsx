import React, { useState, useEffect } from 'react'
import { Form, Input, Button, Card, Alert, Checkbox, Divider } from 'antd'
import { UserOutlined, LockOutlined, EyeInvisibleOutlined, EyeTwoTone } from '@ant-design/icons'
import { useNavigate, useLocation } from 'react-router-dom'
import { useDispatch, useSelector } from 'react-redux'
import styled from 'styled-components'
import { RootState } from '../../store'
import { login } from '../../store/slices/authSlice'

const LoginContainer = styled.div`
  min-height: 100vh;
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  display: flex;
  align-items: center;
  justify-content: center;
  padding: 20px;
`

const LoginCard = styled(Card)`
  width: 100%;
  max-width: 400px;
  box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
  border-radius: 12px;
  
  .ant-card-body {
    padding: 40px;
  }
`

const Logo = styled.div`
  text-align: center;
  margin-bottom: 32px;
  
  h1 {
    color: #1890ff;
    font-size: 28px;
    font-weight: bold;
    margin: 0;
  }
  
  p {
    color: #666;
    margin: 8px 0 0 0;
    font-size: 14px;
  }
`

const StyledForm = styled(Form)`
  .ant-form-item {
    margin-bottom: 20px;
  }
  
  .ant-btn {
    height: 44px;
    font-size: 16px;
    border-radius: 6px;
  }
`

interface LoginFormData {
  username: string
  password: string
  remember: boolean
}

const LoginPage: React.FC = () => {
  const [form] = Form.useForm()
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  
  const navigate = useNavigate()
  const location = useLocation()
  const dispatch = useDispatch()
  const { isAuthenticated } = useSelector((state: RootState) => state.auth)
  
  const from = (location.state as any)?.from?.pathname || '/'

  useEffect(() => {
    if (isAuthenticated) {
      navigate(from, { replace: true })
    }
  }, [isAuthenticated, navigate, from])

  const handleSubmit = async (values: LoginFormData) => {
    try {
      setLoading(true)
      setError(null)
      
      // Dispatch login action
      const result = await dispatch(login({
        username: values.username,
        password: values.password,
        remember: values.remember
      }))
      
      if (login.fulfilled.match(result)) {
        // Login successful, navigation will be handled by useEffect
      } else {
        // Login failed
        setError(result.payload as string || '登录失败，请检查用户名和密码')
      }
    } catch (err) {
      setError('登录过程中发生错误，请稍后重试')
    } finally {
      setLoading(false)
    }
  }

  const handleDemoLogin = () => {
    form.setFieldsValue({
      username: 'demo_doctor',
      password: 'demo123',
      remember: true
    })
  }

  return (
    <LoginContainer>
      <LoginCard>
        <Logo>
          <h1>Medical AI</h1>
          <p>医学影像AI分析系统</p>
        </Logo>
        
        {error && (
          <Alert
            message={error}
            type="error"
            showIcon
            closable
            onClose={() => setError(null)}
            style={{ marginBottom: 24 }}
          />
        )}
        
        <StyledForm
          form={form}
          name="login"
          onFinish={handleSubmit}
          autoComplete="off"
          size="large"
        >
          <Form.Item
            name="username"
            rules={[
              { required: true, message: '请输入用户名' },
              { min: 3, message: '用户名至少3个字符' }
            ]}
          >
            <Input
              prefix={<UserOutlined />}
              placeholder="用户名"
              autoComplete="username"
            />
          </Form.Item>
          
          <Form.Item
            name="password"
            rules={[
              { required: true, message: '请输入密码' },
              { min: 6, message: '密码至少6个字符' }
            ]}
          >
            <Input.Password
              prefix={<LockOutlined />}
              placeholder="密码"
              autoComplete="current-password"
              iconRender={(visible) => (visible ? <EyeTwoTone /> : <EyeInvisibleOutlined />)}
            />
          </Form.Item>
          
          <Form.Item name="remember" valuePropName="checked">
            <Checkbox>记住我</Checkbox>
          </Form.Item>
          
          <Form.Item>
            <Button
              type="primary"
              htmlType="submit"
              loading={loading}
              block
            >
              登录
            </Button>
          </Form.Item>
          
          <Divider>或</Divider>
          
          <Form.Item>
            <Button
              type="default"
              onClick={handleDemoLogin}
              block
            >
              使用演示账户
            </Button>
          </Form.Item>
        </StyledForm>
        
        <div style={{ textAlign: 'center', marginTop: 16, color: '#666', fontSize: '12px' }}>
          <p>演示账户: demo_doctor / demo123</p>
          <p>管理员账户: admin / admin123</p>
        </div>
      </LoginCard>
    </LoginContainer>
  )
}

export default LoginPage