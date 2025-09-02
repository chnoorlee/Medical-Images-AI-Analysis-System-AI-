import React, { useState, useEffect } from 'react'
import { Card, Row, Col, Table, Button, Modal, Form, Input, Select, Switch, Tag, Space, message, Tabs, Descriptions, Statistic, Progress, List, Avatar, Tooltip, Popconfirm, Upload, Alert } from 'antd'
import {
  UserOutlined,
  TeamOutlined,
  SettingOutlined,
  SecurityScanOutlined,
  PlusOutlined,
  EditOutlined,
  DeleteOutlined,
  SearchOutlined,
  ReloadOutlined,
  UploadOutlined,
  DownloadOutlined,
  ExclamationCircleOutlined,
  CheckCircleOutlined,
  CloseCircleOutlined,
  EyeOutlined,
  KeyOutlined,
  DatabaseOutlined,
  CloudServerOutlined,
  MonitorOutlined,
  BugOutlined,
  FileTextOutlined,
  MailOutlined,
  BellOutlined
} from '@ant-design/icons'
import styled from 'styled-components'
import type { ColumnsType } from 'antd/es/table'
import dayjs from 'dayjs'

const { TabPane } = Tabs
const { Option } = Select
const { TextArea } = Input
const { confirm } = Modal

const SystemManageContainer = styled.div`
  .page-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: 24px;
    
    h1 {
      margin: 0;
      color: #262626;
      font-size: 24px;
      font-weight: 600;
    }
    
    .header-actions {
      display: flex;
      gap: 8px;
    }
  }
  
  .stats-cards {
    margin-bottom: 24px;
    
    .ant-card {
      text-align: center;
      
      .ant-statistic-title {
        color: #8c8c8c;
        font-size: 14px;
      }
      
      .ant-statistic-content {
        color: #262626;
      }
    }
  }
  
  .table-toolbar {
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: 16px;
    
    .toolbar-left {
      display: flex;
      gap: 8px;
    }
    
    .toolbar-right {
      display: flex;
      gap: 8px;
    }
  }
  
  .user-avatar {
    display: flex;
    align-items: center;
    gap: 8px;
    
    .user-info {
      .user-name {
        font-weight: 500;
        color: #262626;
      }
      
      .user-email {
        font-size: 12px;
        color: #8c8c8c;
      }
    }
  }
  
  .role-permissions {
    .permission-group {
      margin-bottom: 16px;
      
      .group-title {
        font-weight: 600;
        margin-bottom: 8px;
        color: #262626;
      }
      
      .permission-list {
        display: flex;
        flex-wrap: wrap;
        gap: 8px;
      }
    }
  }
  
  .system-status {
    .status-item {
      display: flex;
      align-items: center;
      justify-content: space-between;
      padding: 12px 0;
      border-bottom: 1px solid #f0f0f0;
      
      &:last-child {
        border-bottom: none;
      }
      
      .status-info {
        display: flex;
        align-items: center;
        gap: 8px;
        
        .status-name {
          font-weight: 500;
        }
        
        .status-description {
          font-size: 12px;
          color: #8c8c8c;
        }
      }
    }
  }
  
  .log-entry {
    padding: 12px;
    border: 1px solid #f0f0f0;
    border-radius: 6px;
    margin-bottom: 8px;
    
    .log-header {
      display: flex;
      align-items: center;
      justify-content: space-between;
      margin-bottom: 4px;
      
      .log-level {
        font-weight: 600;
      }
      
      .log-time {
        font-size: 12px;
        color: #8c8c8c;
      }
    }
    
    .log-message {
      color: #595959;
      font-size: 14px;
    }
  }
`

interface User {
  id: string
  username: string
  email: string
  name: string
  avatar?: string
  role: string
  status: 'active' | 'inactive' | 'locked'
  lastLogin: string
  createdAt: string
}

interface Role {
  id: string
  name: string
  description: string
  permissions: string[]
  userCount: number
  createdAt: string
}

interface SystemConfig {
  key: string
  name: string
  value: string
  type: 'string' | 'number' | 'boolean' | 'json'
  description: string
  category: string
}

interface SystemLog {
  id: string
  level: 'info' | 'warn' | 'error' | 'debug'
  message: string
  timestamp: string
  user?: string
  module: string
}

const SystemManagePage: React.FC = () => {
  const [activeTab, setActiveTab] = useState('users')
  const [loading, setLoading] = useState(false)
  const [users, setUsers] = useState<User[]>([])
  const [roles, setRoles] = useState<Role[]>([])
  const [configs, setConfigs] = useState<SystemConfig[]>([])
  const [logs, setLogs] = useState<SystemLog[]>([])
  const [userModalVisible, setUserModalVisible] = useState(false)
  const [roleModalVisible, setRoleModalVisible] = useState(false)
  const [configModalVisible, setConfigModalVisible] = useState(false)
  const [editingUser, setEditingUser] = useState<User | null>(null)
  const [editingRole, setEditingRole] = useState<Role | null>(null)
  const [editingConfig, setEditingConfig] = useState<SystemConfig | null>(null)
  const [userForm] = Form.useForm()
  const [roleForm] = Form.useForm()
  const [configForm] = Form.useForm()
  
  // Mock data
  const mockUsers: User[] = [
    {
      id: '1',
      username: 'admin',
      email: 'admin@hospital.com',
      name: '系统管理员',
      role: 'admin',
      status: 'active',
      lastLogin: '2024-01-20T10:30:00Z',
      createdAt: '2024-01-01T00:00:00Z'
    },
    {
      id: '2',
      username: 'doctor1',
      email: 'doctor1@hospital.com',
      name: '李医生',
      role: 'doctor',
      status: 'active',
      lastLogin: '2024-01-20T09:15:00Z',
      createdAt: '2024-01-05T00:00:00Z'
    },
    {
      id: '3',
      username: 'technician1',
      email: 'tech1@hospital.com',
      name: '王技师',
      role: 'technician',
      status: 'active',
      lastLogin: '2024-01-20T08:45:00Z',
      createdAt: '2024-01-10T00:00:00Z'
    }
  ]
  
  const mockRoles: Role[] = [
    {
      id: '1',
      name: 'admin',
      description: '系统管理员',
      permissions: ['user:read', 'user:write', 'user:delete', 'system:config', 'system:logs', 'report:all'],
      userCount: 1,
      createdAt: '2024-01-01T00:00:00Z'
    },
    {
      id: '2',
      name: 'doctor',
      description: '医生',
      permissions: ['patient:read', 'patient:write', 'image:read', 'image:analyze', 'report:read', 'report:write'],
      userCount: 5,
      createdAt: '2024-01-01T00:00:00Z'
    },
    {
      id: '3',
      name: 'technician',
      description: '技师',
      permissions: ['image:upload', 'image:read', 'patient:read'],
      userCount: 3,
      createdAt: '2024-01-01T00:00:00Z'
    }
  ]
  
  const mockConfigs: SystemConfig[] = [
    {
      key: 'system.name',
      name: '系统名称',
      value: '医疗影像AI分析系统',
      type: 'string',
      description: '系统显示名称',
      category: '基础设置'
    },
    {
      key: 'ai.model.confidence_threshold',
      name: 'AI置信度阈值',
      value: '0.8',
      type: 'number',
      description: 'AI分析结果的最低置信度要求',
      category: 'AI设置'
    },
    {
      key: 'security.session_timeout',
      name: '会话超时时间',
      value: '3600',
      type: 'number',
      description: '用户会话超时时间（秒）',
      category: '安全设置'
    },
    {
      key: 'notification.email_enabled',
      name: '邮件通知',
      value: 'true',
      type: 'boolean',
      description: '是否启用邮件通知功能',
      category: '通知设置'
    }
  ]
  
  const mockLogs: SystemLog[] = [
    {
      id: '1',
      level: 'info',
      message: '用户登录成功',
      timestamp: '2024-01-20T10:30:00Z',
      user: 'admin',
      module: 'auth'
    },
    {
      id: '2',
      level: 'warn',
      message: 'AI模型响应时间较长',
      timestamp: '2024-01-20T10:25:00Z',
      module: 'ai'
    },
    {
      id: '3',
      level: 'error',
      message: '数据库连接失败',
      timestamp: '2024-01-20T10:20:00Z',
      module: 'database'
    }
  ]
  
  useEffect(() => {
    loadData()
  }, [])
  
  const loadData = async () => {
    setLoading(true)
    // Simulate API calls
    setTimeout(() => {
      setUsers(mockUsers)
      setRoles(mockRoles)
      setConfigs(mockConfigs)
      setLogs(mockLogs)
      setLoading(false)
    }, 1000)
  }
  
  const handleAddUser = () => {
    setEditingUser(null)
    userForm.resetFields()
    setUserModalVisible(true)
  }
  
  const handleEditUser = (user: User) => {
    setEditingUser(user)
    userForm.setFieldsValue(user)
    setUserModalVisible(true)
  }
  
  const handleDeleteUser = (userId: string) => {
    confirm({
      title: '确认删除',
      content: '确定要删除这个用户吗？此操作不可恢复。',
      onOk: () => {
        setUsers(prev => prev.filter(u => u.id !== userId))
        message.success('用户已删除')
      }
    })
  }
  
  const handleUserSubmit = async () => {
    try {
      const values = await userForm.validateFields()
      if (editingUser) {
        // Update user
        setUsers(prev => prev.map(u => u.id === editingUser.id ? { ...u, ...values } : u))
        message.success('用户信息已更新')
      } else {
        // Add new user
        const newUser: User = {
          id: Date.now().toString(),
          ...values,
          createdAt: new Date().toISOString(),
          lastLogin: '从未登录'
        }
        setUsers(prev => [...prev, newUser])
        message.success('用户已添加')
      }
      setUserModalVisible(false)
    } catch (error) {
      message.error('保存失败，请检查输入内容')
    }
  }
  
  const handleAddRole = () => {
    setEditingRole(null)
    roleForm.resetFields()
    setRoleModalVisible(true)
  }
  
  const handleEditRole = (role: Role) => {
    setEditingRole(role)
    roleForm.setFieldsValue(role)
    setRoleModalVisible(true)
  }
  
  const handleDeleteRole = (roleId: string) => {
    confirm({
      title: '确认删除',
      content: '确定要删除这个角色吗？此操作不可恢复。',
      onOk: () => {
        setRoles(prev => prev.filter(r => r.id !== roleId))
        message.success('角色已删除')
      }
    })
  }
  
  const handleRoleSubmit = async () => {
    try {
      const values = await roleForm.validateFields()
      if (editingRole) {
        // Update role
        setRoles(prev => prev.map(r => r.id === editingRole.id ? { ...r, ...values } : r))
        message.success('角色信息已更新')
      } else {
        // Add new role
        const newRole: Role = {
          id: Date.now().toString(),
          ...values,
          userCount: 0,
          createdAt: new Date().toISOString()
        }
        setRoles(prev => [...prev, newRole])
        message.success('角色已添加')
      }
      setRoleModalVisible(false)
    } catch (error) {
      message.error('保存失败，请检查输入内容')
    }
  }
  
  const handleEditConfig = (config: SystemConfig) => {
    setEditingConfig(config)
    configForm.setFieldsValue(config)
    setConfigModalVisible(true)
  }
  
  const handleConfigSubmit = async () => {
    try {
      const values = await configForm.validateFields()
      if (editingConfig) {
        setConfigs(prev => prev.map(c => c.key === editingConfig.key ? { ...c, ...values } : c))
        message.success('配置已更新')
      }
      setConfigModalVisible(false)
    } catch (error) {
      message.error('保存失败，请检查输入内容')
    }
  }
  
  const getStatusColor = (status: string) => {
    switch (status) {
      case 'active': return 'green'
      case 'inactive': return 'orange'
      case 'locked': return 'red'
      default: return 'default'
    }
  }
  
  const getStatusText = (status: string) => {
    switch (status) {
      case 'active': return '正常'
      case 'inactive': return '停用'
      case 'locked': return '锁定'
      default: return '未知'
    }
  }
  
  const getLogLevelColor = (level: string) => {
    switch (level) {
      case 'info': return '#1890ff'
      case 'warn': return '#fa8c16'
      case 'error': return '#f5222d'
      case 'debug': return '#722ed1'
      default: return '#8c8c8c'
    }
  }
  
  const userColumns: ColumnsType<User> = [
    {
      title: '用户',
      dataIndex: 'name',
      key: 'name',
      render: (text, record) => (
        <div className="user-avatar">
          <Avatar icon={<UserOutlined />} />
          <div className="user-info">
            <div className="user-name">{text}</div>
            <div className="user-email">{record.email}</div>
          </div>
        </div>
      )
    },
    {
      title: '用户名',
      dataIndex: 'username',
      key: 'username'
    },
    {
      title: '角色',
      dataIndex: 'role',
      key: 'role',
      render: (role) => {
        const roleInfo = roles.find(r => r.name === role)
        return <Tag color="blue">{roleInfo?.description || role}</Tag>
      }
    },
    {
      title: '状态',
      dataIndex: 'status',
      key: 'status',
      render: (status) => (
        <Tag color={getStatusColor(status)}>
          {getStatusText(status)}
        </Tag>
      )
    },
    {
      title: '最后登录',
      dataIndex: 'lastLogin',
      key: 'lastLogin',
      render: (time) => time === '从未登录' ? time : dayjs(time).format('YYYY-MM-DD HH:mm')
    },
    {
      title: '操作',
      key: 'actions',
      render: (_, record) => (
        <Space>
          <Button 
            type="link" 
            icon={<EditOutlined />} 
            onClick={() => handleEditUser(record)}
          >
            编辑
          </Button>
          <Popconfirm
            title="确定要删除这个用户吗？"
            onConfirm={() => handleDeleteUser(record.id)}
          >
            <Button type="link" danger icon={<DeleteOutlined />}>
              删除
            </Button>
          </Popconfirm>
        </Space>
      )
    }
  ]
  
  const roleColumns: ColumnsType<Role> = [
    {
      title: '角色名称',
      dataIndex: 'name',
      key: 'name',
      render: (text, record) => (
        <div>
          <div style={{ fontWeight: 500 }}>{record.description}</div>
          <div style={{ fontSize: '12px', color: '#8c8c8c' }}>{text}</div>
        </div>
      )
    },
    {
      title: '权限数量',
      dataIndex: 'permissions',
      key: 'permissions',
      render: (permissions) => permissions.length
    },
    {
      title: '用户数量',
      dataIndex: 'userCount',
      key: 'userCount'
    },
    {
      title: '创建时间',
      dataIndex: 'createdAt',
      key: 'createdAt',
      render: (time) => dayjs(time).format('YYYY-MM-DD')
    },
    {
      title: '操作',
      key: 'actions',
      render: (_, record) => (
        <Space>
          <Button 
            type="link" 
            icon={<EyeOutlined />}
            onClick={() => {
              Modal.info({
                title: `角色权限 - ${record.description}`,
                width: 600,
                content: (
                  <div style={{ marginTop: 16 }}>
                    <div style={{ marginBottom: 12 }}>权限列表：</div>
                    <div style={{ display: 'flex', flexWrap: 'wrap', gap: 8 }}>
                      {record.permissions.map(permission => (
                        <Tag key={permission} color="blue">{permission}</Tag>
                      ))}
                    </div>
                  </div>
                )
              })
            }}
          >
            查看
          </Button>
          <Button 
            type="link" 
            icon={<EditOutlined />} 
            onClick={() => handleEditRole(record)}
          >
            编辑
          </Button>
          <Popconfirm
            title="确定要删除这个角色吗？"
            onConfirm={() => handleDeleteRole(record.id)}
          >
            <Button type="link" danger icon={<DeleteOutlined />}>
              删除
            </Button>
          </Popconfirm>
        </Space>
      )
    }
  ]
  
  const configColumns: ColumnsType<SystemConfig> = [
    {
      title: '配置项',
      dataIndex: 'name',
      key: 'name',
      render: (text, record) => (
        <div>
          <div style={{ fontWeight: 500 }}>{text}</div>
          <div style={{ fontSize: '12px', color: '#8c8c8c' }}>{record.key}</div>
        </div>
      )
    },
    {
      title: '当前值',
      dataIndex: 'value',
      key: 'value',
      render: (value, record) => {
        if (record.type === 'boolean') {
          return <Tag color={value === 'true' ? 'green' : 'red'}>
            {value === 'true' ? '启用' : '禁用'}
          </Tag>
        }
        return <code>{value}</code>
      }
    },
    {
      title: '类型',
      dataIndex: 'type',
      key: 'type',
      render: (type) => <Tag>{type}</Tag>
    },
    {
      title: '分类',
      dataIndex: 'category',
      key: 'category',
      render: (category) => <Tag color="blue">{category}</Tag>
    },
    {
      title: '操作',
      key: 'actions',
      render: (_, record) => (
        <Button 
          type="link" 
          icon={<EditOutlined />} 
          onClick={() => handleEditConfig(record)}
        >
          编辑
        </Button>
      )
    }
  ]
  
  return (
    <SystemManageContainer>
      <div className="page-header">
        <h1>系统管理</h1>
        <div className="header-actions">
          <Button icon={<ReloadOutlined />} onClick={loadData}>
            刷新
          </Button>
        </div>
      </div>
      
      {/* 统计卡片 */}
      <Row gutter={[16, 16]} className="stats-cards">
        <Col span={6}>
          <Card>
            <Statistic
              title="总用户数"
              value={users.length}
              prefix={<UserOutlined />}
              valueStyle={{ color: '#1890ff' }}
            />
          </Card>
        </Col>
        <Col span={6}>
          <Card>
            <Statistic
              title="角色数量"
              value={roles.length}
              prefix={<TeamOutlined />}
              valueStyle={{ color: '#52c41a' }}
            />
          </Card>
        </Col>
        <Col span={6}>
          <Card>
            <Statistic
              title="在线用户"
              value={users.filter(u => u.status === 'active').length}
              prefix={<CheckCircleOutlined />}
              valueStyle={{ color: '#fa8c16' }}
            />
          </Card>
        </Col>
        <Col span={6}>
          <Card>
            <Statistic
              title="系统配置"
              value={configs.length}
              prefix={<SettingOutlined />}
              valueStyle={{ color: '#722ed1' }}
            />
          </Card>
        </Col>
      </Row>
      
      <Tabs activeKey={activeTab} onChange={setActiveTab}>
        {/* 用户管理 */}
        <TabPane tab={<span><UserOutlined />用户管理</span>} key="users">
          <Card>
            <div className="table-toolbar">
              <div className="toolbar-left">
                <Button type="primary" icon={<PlusOutlined />} onClick={handleAddUser}>
                  添加用户
                </Button>
              </div>
              <div className="toolbar-right">
                <Input.Search
                  placeholder="搜索用户"
                  style={{ width: 200 }}
                  onSearch={(value) => console.log('搜索:', value)}
                />
              </div>
            </div>
            <Table
              columns={userColumns}
              dataSource={users}
              rowKey="id"
              loading={loading}
              pagination={{
                showSizeChanger: true,
                showQuickJumper: true,
                showTotal: (total) => `共 ${total} 条记录`
              }}
            />
          </Card>
        </TabPane>
        
        {/* 角色管理 */}
        <TabPane tab={<span><TeamOutlined />角色管理</span>} key="roles">
          <Card>
            <div className="table-toolbar">
              <div className="toolbar-left">
                <Button type="primary" icon={<PlusOutlined />} onClick={handleAddRole}>
                  添加角色
                </Button>
              </div>
            </div>
            <Table
              columns={roleColumns}
              dataSource={roles}
              rowKey="id"
              loading={loading}
              pagination={{
                showSizeChanger: true,
                showQuickJumper: true,
                showTotal: (total) => `共 ${total} 条记录`
              }}
            />
          </Card>
        </TabPane>
        
        {/* 系统配置 */}
        <TabPane tab={<span><SettingOutlined />系统配置</span>} key="configs">
          <Card>
            <Alert
              message="配置修改提醒"
              description="修改系统配置可能会影响系统运行，请谨慎操作。"
              type="warning"
              showIcon
              style={{ marginBottom: 16 }}
            />
            <Table
              columns={configColumns}
              dataSource={configs}
              rowKey="key"
              loading={loading}
              pagination={{
                showSizeChanger: true,
                showQuickJumper: true,
                showTotal: (total) => `共 ${total} 条记录`
              }}
            />
          </Card>
        </TabPane>
        
        {/* 系统监控 */}
        <TabPane tab={<span><MonitorOutlined />系统监控</span>} key="monitor">
          <Row gutter={[16, 16]}>
            <Col span={12}>
              <Card title="系统状态">
                <div className="system-status">
                  <div className="status-item">
                    <div className="status-info">
                      <DatabaseOutlined style={{ color: '#52c41a' }} />
                      <div>
                        <div className="status-name">数据库</div>
                        <div className="status-description">连接正常</div>
                      </div>
                    </div>
                    <Tag color="green">正常</Tag>
                  </div>
                  <div className="status-item">
                    <div className="status-info">
                      <CloudServerOutlined style={{ color: '#1890ff' }} />
                      <div>
                        <div className="status-name">AI服务</div>
                        <div className="status-description">运行中</div>
                      </div>
                    </div>
                    <Tag color="blue">运行中</Tag>
                  </div>
                  <div className="status-item">
                    <div className="status-info">
                      <MailOutlined style={{ color: '#fa8c16' }} />
                      <div>
                        <div className="status-name">邮件服务</div>
                        <div className="status-description">部分异常</div>
                      </div>
                    </div>
                    <Tag color="orange">警告</Tag>
                  </div>
                </div>
              </Card>
            </Col>
            <Col span={12}>
              <Card title="性能指标">
                <div style={{ marginBottom: 16 }}>
                  <div style={{ marginBottom: 8 }}>CPU使用率</div>
                  <Progress percent={45} status="active" />
                </div>
                <div style={{ marginBottom: 16 }}>
                  <div style={{ marginBottom: 8 }}>内存使用率</div>
                  <Progress percent={68} status="active" strokeColor="#fa8c16" />
                </div>
                <div style={{ marginBottom: 16 }}>
                  <div style={{ marginBottom: 8 }}>磁盘使用率</div>
                  <Progress percent={32} status="active" strokeColor="#52c41a" />
                </div>
                <div>
                  <div style={{ marginBottom: 8 }}>网络带宽</div>
                  <Progress percent={23} status="active" strokeColor="#722ed1" />
                </div>
              </Card>
            </Col>
          </Row>
        </TabPane>
        
        {/* 系统日志 */}
        <TabPane tab={<span><FileTextOutlined />系统日志</span>} key="logs">
          <Card>
            <div className="table-toolbar">
              <div className="toolbar-left">
                <Select defaultValue="all" style={{ width: 120 }}>
                  <Option value="all">全部级别</Option>
                  <Option value="info">信息</Option>
                  <Option value="warn">警告</Option>
                  <Option value="error">错误</Option>
                  <Option value="debug">调试</Option>
                </Select>
                <Select defaultValue="all" style={{ width: 120, marginLeft: 8 }}>
                  <Option value="all">全部模块</Option>
                  <Option value="auth">认证</Option>
                  <Option value="ai">AI</Option>
                  <Option value="database">数据库</Option>
                  <Option value="api">API</Option>
                </Select>
              </div>
              <div className="toolbar-right">
                <Button icon={<DownloadOutlined />}>
                  导出日志
                </Button>
              </div>
            </div>
            <div style={{ maxHeight: 400, overflowY: 'auto' }}>
              {logs.map(log => (
                <div key={log.id} className="log-entry">
                  <div className="log-header">
                    <span 
                      className="log-level" 
                      style={{ color: getLogLevelColor(log.level) }}
                    >
                      [{log.level.toUpperCase()}]
                    </span>
                    <span className="log-time">
                      {dayjs(log.timestamp).format('YYYY-MM-DD HH:mm:ss')}
                    </span>
                  </div>
                  <div className="log-message">
                    [{log.module}] {log.message}
                    {log.user && ` - 用户: ${log.user}`}
                  </div>
                </div>
              ))}
            </div>
          </Card>
        </TabPane>
      </Tabs>
      
      {/* 用户编辑模态框 */}
      <Modal
        title={editingUser ? '编辑用户' : '添加用户'}
        open={userModalVisible}
        onOk={handleUserSubmit}
        onCancel={() => setUserModalVisible(false)}
        width={600}
      >
        <Form form={userForm} layout="vertical">
          <Row gutter={16}>
            <Col span={12}>
              <Form.Item
                name="username"
                label="用户名"
                rules={[{ required: true, message: '请输入用户名' }]}
              >
                <Input placeholder="请输入用户名" />
              </Form.Item>
            </Col>
            <Col span={12}>
              <Form.Item
                name="name"
                label="姓名"
                rules={[{ required: true, message: '请输入姓名' }]}
              >
                <Input placeholder="请输入姓名" />
              </Form.Item>
            </Col>
          </Row>
          <Row gutter={16}>
            <Col span={12}>
              <Form.Item
                name="email"
                label="邮箱"
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
                name="role"
                label="角色"
                rules={[{ required: true, message: '请选择角色' }]}
              >
                <Select placeholder="请选择角色">
                  {roles.map(role => (
                    <Option key={role.name} value={role.name}>
                      {role.description}
                    </Option>
                  ))}
                </Select>
              </Form.Item>
            </Col>
          </Row>
          <Form.Item
            name="status"
            label="状态"
            rules={[{ required: true, message: '请选择状态' }]}
          >
            <Select placeholder="请选择状态">
              <Option value="active">正常</Option>
              <Option value="inactive">停用</Option>
              <Option value="locked">锁定</Option>
            </Select>
          </Form.Item>
        </Form>
      </Modal>
      
      {/* 角色编辑模态框 */}
      <Modal
        title={editingRole ? '编辑角色' : '添加角色'}
        open={roleModalVisible}
        onOk={handleRoleSubmit}
        onCancel={() => setRoleModalVisible(false)}
        width={800}
      >
        <Form form={roleForm} layout="vertical">
          <Row gutter={16}>
            <Col span={12}>
              <Form.Item
                name="name"
                label="角色标识"
                rules={[{ required: true, message: '请输入角色标识' }]}
              >
                <Input placeholder="请输入角色标识" />
              </Form.Item>
            </Col>
            <Col span={12}>
              <Form.Item
                name="description"
                label="角色名称"
                rules={[{ required: true, message: '请输入角色名称' }]}
              >
                <Input placeholder="请输入角色名称" />
              </Form.Item>
            </Col>
          </Row>
          <Form.Item
            name="permissions"
            label="权限"
            rules={[{ required: true, message: '请选择权限' }]}
          >
            <Select
              mode="multiple"
              placeholder="请选择权限"
              style={{ width: '100%' }}
            >
              <Option value="user:read">用户查看</Option>
              <Option value="user:write">用户编辑</Option>
              <Option value="user:delete">用户删除</Option>
              <Option value="patient:read">患者查看</Option>
              <Option value="patient:write">患者编辑</Option>
              <Option value="image:read">影像查看</Option>
              <Option value="image:upload">影像上传</Option>
              <Option value="image:analyze">影像分析</Option>
              <Option value="report:read">报告查看</Option>
              <Option value="report:write">报告编辑</Option>
              <Option value="report:all">报告管理</Option>
              <Option value="system:config">系统配置</Option>
              <Option value="system:logs">系统日志</Option>
            </Select>
          </Form.Item>
        </Form>
      </Modal>
      
      {/* 配置编辑模态框 */}
      <Modal
        title="编辑配置"
        open={configModalVisible}
        onOk={handleConfigSubmit}
        onCancel={() => setConfigModalVisible(false)}
        width={600}
      >
        <Form form={configForm} layout="vertical">
          <Form.Item label="配置项">
            <Input value={editingConfig?.name} disabled />
          </Form.Item>
          <Form.Item label="配置键">
            <Input value={editingConfig?.key} disabled />
          </Form.Item>
          <Form.Item
            name="value"
            label="配置值"
            rules={[{ required: true, message: '请输入配置值' }]}
          >
            {editingConfig?.type === 'boolean' ? (
              <Select>
                <Option value="true">启用</Option>
                <Option value="false">禁用</Option>
              </Select>
            ) : editingConfig?.type === 'number' ? (
              <Input type="number" placeholder="请输入数值" />
            ) : (
              <Input placeholder="请输入配置值" />
            )}
          </Form.Item>
          <Form.Item label="说明">
            <Input value={editingConfig?.description} disabled />
          </Form.Item>
        </Form>
      </Modal>
    </SystemManageContainer>
  )
}

export default SystemManagePage