import React, { useState, useEffect } from 'react'
import {
  Card,
  Form,
  Input,
  Switch,
  Select,
  Button,
  Space,
  Divider,
  message,
  Row,
  Col,
  Typography,
  Tabs,
  InputNumber,
  Radio,
  Upload,
  Avatar
} from 'antd'
import {
  UserOutlined,
  SettingOutlined,
  BellOutlined,
  SecurityScanOutlined,
  UploadOutlined
} from '@ant-design/icons'
import { useSelector, useDispatch } from 'react-redux'
import { RootState } from '../../store'
import { 
  updateUserPreferences,
  updateSystemSettings,
  updateAISettings,
  updateNotificationSettings
} from '../../store/slices/settingsSlice'

const { Title, Text } = Typography
const { Option } = Select
const { TabPane } = Tabs

interface SettingsFormData {
  // 个人信息
  name: string
  email: string
  phone: string
  department: string
  avatar?: string
  
  // 系统设置
  language: string
  theme: string
  autoSave: boolean
  autoLogout: number
  
  // 通知设置
  emailNotifications: boolean
  pushNotifications: boolean
  smsNotifications: boolean
  notificationSound: boolean
  
  // 工作站设置
  defaultLayout: string
  imageQuality: string
  autoAnalysis: boolean
  showAnnotations: boolean
  
  // 安全设置
  twoFactorAuth: boolean
  sessionTimeout: number
  passwordExpiry: number
}

const SettingsPage: React.FC = () => {
  const dispatch = useDispatch()
  const { user } = useSelector((state: RootState) => state.auth)
  const { settings } = useSelector((state: RootState) => state.settings)
  const [form] = Form.useForm()
  const [loading, setLoading] = useState(false)
  const [activeTab, setActiveTab] = useState('profile')

  useEffect(() => {
    // 初始化表单数据
    if (user && settings) {
      form.setFieldsValue({
        name: user.name,
        email: user.email,
        phone: user.phone,
        department: user.department,
        ...settings
      })
    }
  }, [user, settings, form])

  const handleSave = async (values: SettingsFormData) => {
    setLoading(true)
    try {
      // 分离用户信息和设置信息
      const { name, email, phone, department, avatar, ...settingsData } = values
      
      // 更新设置 - 分类处理不同类型的设置
      const {
        language, theme, autoSave, autoLogout,
        emailNotifications, pushNotifications, smsNotifications, notificationSound,
        defaultLayout, imageQuality, autoAnalysis, showAnnotations,
        twoFactorAuth, sessionTimeout, passwordExpiry,
        ...otherSettings
      } = settingsData
      
      // 更新用户偏好设置
      if (language || theme || autoSave !== undefined) {
        dispatch(updateUserPreferences({
          language: language as 'zh-CN' | 'en-US',
          theme: theme as 'light' | 'dark' | 'auto',
          autoSave
        }))
      }
      
      // 更新通知设置
      if (emailNotifications !== undefined || pushNotifications !== undefined || 
          smsNotifications !== undefined || notificationSound !== undefined) {
        dispatch(updateUserPreferences({
          notifications: {
            email: emailNotifications,
            push: pushNotifications,
            sound: notificationSound,
            desktop: false // 默认值
          }
        }))
      }
      
      // 更新AI设置
      if (autoAnalysis !== undefined) {
        dispatch(updateAISettings({
          autoAnalysis
        }))
      }
      
      // 更新系统设置
      if (sessionTimeout !== undefined) {
        dispatch(updateSystemSettings({
          sessionTimeout
        }))
      }
      
      // 这里应该调用API更新用户信息
      // await updateUserProfile({ name, email, phone, department, avatar })
      
      message.success('设置保存成功')
    } catch (error) {
      message.error('设置保存失败')
    } finally {
      setLoading(false)
    }
  }

  const handleAvatarUpload = (info: any) => {
    if (info.file.status === 'done') {
      message.success('头像上传成功')
      form.setFieldsValue({ avatar: info.file.response.url })
    } else if (info.file.status === 'error') {
      message.error('头像上传失败')
    }
  }

  return (
    <div style={{ padding: '24px' }}>
      <Title level={2}>
        <SettingOutlined /> 系统设置
      </Title>
      
      <Card>
        <Tabs activeKey={activeTab} onChange={setActiveTab}>
          {/* 个人资料 */}
          <TabPane
            tab={
              <span>
                <UserOutlined />
                个人资料
              </span>
            }
            key="profile"
          >
            <Form
              form={form}
              layout="vertical"
              onFinish={handleSave}
              initialValues={{
                language: 'zh-CN',
                theme: 'light',
                autoSave: true,
                autoLogout: 30,
                emailNotifications: true,
                pushNotifications: true,
                defaultLayout: 'grid',
                imageQuality: 'high'
              }}
            >
              <Row gutter={24}>
                <Col span={8}>
                  <Form.Item label="头像">
                    <div style={{ textAlign: 'center' }}>
                      <Avatar
                        size={100}
                        src={form.getFieldValue('avatar')}
                        icon={<UserOutlined />}
                        style={{ marginBottom: 16 }}
                      />
                      <br />
                      <Upload
                        name="avatar"
                        action="/api/upload/avatar"
                        showUploadList={false}
                        onChange={handleAvatarUpload}
                      >
                        <Button icon={<UploadOutlined />}>更换头像</Button>
                      </Upload>
                    </div>
                  </Form.Item>
                </Col>
                <Col span={16}>
                  <Row gutter={16}>
                    <Col span={12}>
                      <Form.Item
                        label="姓名"
                        name="name"
                        rules={[{ required: true, message: '请输入姓名' }]}
                      >
                        <Input placeholder="请输入姓名" />
                      </Form.Item>
                    </Col>
                    <Col span={12}>
                      <Form.Item
                        label="邮箱"
                        name="email"
                        rules={[
                          { required: true, message: '请输入邮箱' },
                          { type: 'email', message: '请输入有效的邮箱地址' }
                        ]}
                      >
                        <Input placeholder="请输入邮箱" />
                      </Form.Item>
                    </Col>
                    <Col span={12}>
                      <Form.Item
                        label="电话"
                        name="phone"
                        rules={[{ required: true, message: '请输入电话号码' }]}
                      >
                        <Input placeholder="请输入电话号码" />
                      </Form.Item>
                    </Col>
                    <Col span={12}>
                      <Form.Item
                        label="科室"
                        name="department"
                        rules={[{ required: true, message: '请选择科室' }]}
                      >
                        <Select placeholder="请选择科室">
                          <Option value="radiology">放射科</Option>
                          <Option value="cardiology">心内科</Option>
                          <Option value="neurology">神经科</Option>
                          <Option value="orthopedics">骨科</Option>
                          <Option value="oncology">肿瘤科</Option>
                        </Select>
                      </Form.Item>
                    </Col>
                  </Row>
                </Col>
              </Row>
            </Form>
          </TabPane>

          {/* 系统设置 */}
          <TabPane
            tab={
              <span>
                <SettingOutlined />
                系统设置
              </span>
            }
            key="system"
          >
            <Form form={form} layout="vertical" onFinish={handleSave}>
              <Row gutter={24}>
                <Col span={12}>
                  <Form.Item label="语言" name="language">
                    <Select>
                      <Option value="zh-CN">简体中文</Option>
                      <Option value="en-US">English</Option>
                    </Select>
                  </Form.Item>
                </Col>
                <Col span={12}>
                  <Form.Item label="主题" name="theme">
                    <Radio.Group>
                      <Radio value="light">浅色主题</Radio>
                      <Radio value="dark">深色主题</Radio>
                    </Radio.Group>
                  </Form.Item>
                </Col>
                <Col span={12}>
                  <Form.Item label="自动保存" name="autoSave" valuePropName="checked">
                    <Switch />
                  </Form.Item>
                </Col>
                <Col span={12}>
                  <Form.Item label="自动登出时间(分钟)" name="autoLogout">
                    <InputNumber min={5} max={120} style={{ width: '100%' }} />
                  </Form.Item>
                </Col>
              </Row>
            </Form>
          </TabPane>

          {/* 通知设置 */}
          <TabPane
            tab={
              <span>
                <BellOutlined />
                通知设置
              </span>
            }
            key="notifications"
          >
            <Form form={form} layout="vertical" onFinish={handleSave}>
              <Row gutter={24}>
                <Col span={12}>
                  <Form.Item label="邮件通知" name="emailNotifications" valuePropName="checked">
                    <Switch />
                  </Form.Item>
                </Col>
                <Col span={12}>
                  <Form.Item label="推送通知" name="pushNotifications" valuePropName="checked">
                    <Switch />
                  </Form.Item>
                </Col>
                <Col span={12}>
                  <Form.Item label="短信通知" name="smsNotifications" valuePropName="checked">
                    <Switch />
                  </Form.Item>
                </Col>
                <Col span={12}>
                  <Form.Item label="通知声音" name="notificationSound" valuePropName="checked">
                    <Switch />
                  </Form.Item>
                </Col>
              </Row>
            </Form>
          </TabPane>

          {/* 工作站设置 */}
          <TabPane
            tab={
              <span>
                <SettingOutlined />
                工作站设置
              </span>
            }
            key="workstation"
          >
            <Form form={form} layout="vertical" onFinish={handleSave}>
              <Row gutter={24}>
                <Col span={12}>
                  <Form.Item label="默认布局" name="defaultLayout">
                    <Select>
                      <Option value="grid">网格布局</Option>
                      <Option value="list">列表布局</Option>
                      <Option value="tile">瓦片布局</Option>
                    </Select>
                  </Form.Item>
                </Col>
                <Col span={12}>
                  <Form.Item label="图像质量" name="imageQuality">
                    <Select>
                      <Option value="high">高质量</Option>
                      <Option value="medium">中等质量</Option>
                      <Option value="low">低质量</Option>
                    </Select>
                  </Form.Item>
                </Col>
                <Col span={12}>
                  <Form.Item label="自动分析" name="autoAnalysis" valuePropName="checked">
                    <Switch />
                  </Form.Item>
                </Col>
                <Col span={12}>
                  <Form.Item label="显示标注" name="showAnnotations" valuePropName="checked">
                    <Switch />
                  </Form.Item>
                </Col>
              </Row>
            </Form>
          </TabPane>

          {/* 安全设置 */}
          <TabPane
            tab={
              <span>
                <SecurityScanOutlined />
                安全设置
              </span>
            }
            key="security"
          >
            <Form form={form} layout="vertical" onFinish={handleSave}>
              <Row gutter={24}>
                <Col span={12}>
                  <Form.Item label="双因子认证" name="twoFactorAuth" valuePropName="checked">
                    <Switch />
                  </Form.Item>
                </Col>
                <Col span={12}>
                  <Form.Item label="会话超时(分钟)" name="sessionTimeout">
                    <InputNumber min={5} max={480} style={{ width: '100%' }} />
                  </Form.Item>
                </Col>
                <Col span={24}>
                  <Form.Item label="密码过期天数" name="passwordExpiry">
                    <InputNumber min={30} max={365} style={{ width: '100%' }} />
                  </Form.Item>
                </Col>
              </Row>
            </Form>
          </TabPane>
        </Tabs>

        <Divider />
        
        <div style={{ textAlign: 'right' }}>
          <Space>
            <Button onClick={() => form.resetFields()}>
              重置
            </Button>
            <Button 
              type="primary" 
              loading={loading}
              onClick={() => form.submit()}
            >
              保存设置
            </Button>
          </Space>
        </div>
      </Card>
    </div>
  )
}

export default SettingsPage