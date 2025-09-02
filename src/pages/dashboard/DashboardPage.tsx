import React, { useState, useEffect } from 'react'
import { Row, Col, Card, Statistic, Progress, Table, List, Avatar, Tag, Button, Space, DatePicker, Select } from 'antd'
import {
  UserOutlined,
  FileImageOutlined,
  ExperimentOutlined,
  CheckCircleOutlined,
  ClockCircleOutlined,
  WarningOutlined,
  TrophyOutlined,
  RiseOutlined,
  FallOutlined
} from '@ant-design/icons'
import { LineChart, Line, AreaChart, Area, BarChart, Bar, PieChart, Pie, Cell, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts'
import styled from 'styled-components'
import { useSelector } from 'react-redux'
import { RootState } from '../../store'
import dayjs from 'dayjs'

const { RangePicker } = DatePicker
const { Option } = Select

const DashboardContainer = styled.div`
  .dashboard-header {
    margin-bottom: 24px;
    
    h1 {
      margin: 0;
      color: #262626;
      font-size: 24px;
      font-weight: 600;
    }
    
    .welcome-text {
      color: #8c8c8c;
      margin-top: 4px;
    }
  }
  
  .stat-card {
    .ant-card-body {
      padding: 20px;
    }
    
    .stat-icon {
      font-size: 32px;
      margin-bottom: 12px;
    }
  }
  
  .chart-card {
    .ant-card-head {
      border-bottom: 1px solid #f0f0f0;
    }
    
    .chart-container {
      height: 300px;
      padding: 16px 0;
    }
  }
  
  .activity-item {
    padding: 12px 0;
    border-bottom: 1px solid #f5f5f5;
    
    &:last-child {
      border-bottom: none;
    }
  }
`

interface DashboardStats {
  totalPatients: number
  totalImages: number
  pendingAnalysis: number
  completedToday: number
  accuracyRate: number
  processingTime: number
}

interface ChartData {
  date: string
  images: number
  analysis: number
  accuracy: number
}

interface RecentActivity {
  id: string
  type: 'upload' | 'analysis' | 'report'
  title: string
  description: string
  time: string
  status: 'success' | 'processing' | 'warning'
}

const DashboardPage: React.FC = () => {
  const { user } = useSelector((state: RootState) => state.auth)
  const [loading, setLoading] = useState(true)
  const [dateRange, setDateRange] = useState<[dayjs.Dayjs, dayjs.Dayjs]>([
    dayjs().subtract(7, 'day'),
    dayjs()
  ])
  const [timeFilter, setTimeFilter] = useState('7d')
  
  // Mock data - in real app, this would come from API
  const [stats, setStats] = useState<DashboardStats>({
    totalPatients: 1248,
    totalImages: 5632,
    pendingAnalysis: 23,
    completedToday: 156,
    accuracyRate: 94.8,
    processingTime: 2.3
  })
  
  const chartData: ChartData[] = [
    { date: '2024-01-15', images: 45, analysis: 42, accuracy: 94.2 },
    { date: '2024-01-16', images: 52, analysis: 48, accuracy: 95.1 },
    { date: '2024-01-17', images: 38, analysis: 36, accuracy: 93.8 },
    { date: '2024-01-18', images: 61, analysis: 58, accuracy: 96.2 },
    { date: '2024-01-19', images: 47, analysis: 44, accuracy: 94.7 },
    { date: '2024-01-20', images: 55, analysis: 52, accuracy: 95.5 },
    { date: '2024-01-21', images: 49, analysis: 47, accuracy: 94.9 }
  ]
  
  const pieData = [
    { name: '正常', value: 65, color: '#52c41a' },
    { name: '异常', value: 25, color: '#ff4d4f' },
    { name: '待确认', value: 10, color: '#faad14' }
  ]
  
  const recentActivities: RecentActivity[] = [
    {
      id: '1',
      type: 'upload',
      title: '新增CT影像',
      description: '患者张三的胸部CT影像已上传',
      time: '5分钟前',
      status: 'success'
    },
    {
      id: '2',
      type: 'analysis',
      title: 'AI分析完成',
      description: '肺结节检测分析已完成，发现2个可疑结节',
      time: '12分钟前',
      status: 'warning'
    },
    {
      id: '3',
      type: 'report',
      title: '报告生成',
      description: '患者李四的影像报告已生成',
      time: '25分钟前',
      status: 'success'
    },
    {
      id: '4',
      type: 'analysis',
      title: '批量分析中',
      description: '正在处理15张MRI影像',
      time: '1小时前',
      status: 'processing'
    }
  ]
  
  useEffect(() => {
    // Simulate loading
    const timer = setTimeout(() => {
      setLoading(false)
    }, 1000)
    
    return () => clearTimeout(timer)
  }, [])
  
  const getActivityIcon = (type: string) => {
    switch (type) {
      case 'upload':
        return <FileImageOutlined style={{ color: '#1890ff' }} />
      case 'analysis':
        return <ExperimentOutlined style={{ color: '#722ed1' }} />
      case 'report':
        return <CheckCircleOutlined style={{ color: '#52c41a' }} />
      default:
        return <ClockCircleOutlined />
    }
  }
  
  const getStatusTag = (status: string) => {
    switch (status) {
      case 'success':
        return <Tag color="success">完成</Tag>
      case 'processing':
        return <Tag color="processing">处理中</Tag>
      case 'warning':
        return <Tag color="warning">需关注</Tag>
      default:
        return <Tag>未知</Tag>
    }
  }
  
  return (
    <DashboardContainer>
      <div className="dashboard-header">
        <h1>仪表板</h1>
        <p className="welcome-text">
          欢迎回来，{user?.name || user?.username}！今天是 {dayjs().format('YYYY年MM月DD日')}
        </p>
      </div>
      
      {/* 统计卡片 */}
      <Row gutter={[16, 16]} style={{ marginBottom: 24 }}>
        <Col xs={24} sm={12} lg={6}>
          <Card className="stat-card" loading={loading}>
            <Statistic
              title="总患者数"
              value={stats.totalPatients}
              prefix={<UserOutlined className="stat-icon" style={{ color: '#1890ff' }} />}
              suffix="人"
            />
          </Card>
        </Col>
        <Col xs={24} sm={12} lg={6}>
          <Card className="stat-card" loading={loading}>
            <Statistic
              title="总影像数"
              value={stats.totalImages}
              prefix={<FileImageOutlined className="stat-icon" style={{ color: '#52c41a' }} />}
              suffix="张"
            />
          </Card>
        </Col>
        <Col xs={24} sm={12} lg={6}>
          <Card className="stat-card" loading={loading}>
            <Statistic
              title="待分析"
              value={stats.pendingAnalysis}
              prefix={<ClockCircleOutlined className="stat-icon" style={{ color: '#faad14' }} />}
              suffix="项"
            />
          </Card>
        </Col>
        <Col xs={24} sm={12} lg={6}>
          <Card className="stat-card" loading={loading}>
            <Statistic
              title="今日完成"
              value={stats.completedToday}
              prefix={<TrophyOutlined className="stat-icon" style={{ color: '#722ed1' }} />}
              suffix="项"
            />
          </Card>
        </Col>
      </Row>
      
      {/* 性能指标 */}
      <Row gutter={[16, 16]} style={{ marginBottom: 24 }}>
        <Col xs={24} lg={12}>
          <Card title="准确率" loading={loading}>
            <div style={{ textAlign: 'center' }}>
              <Progress
                type="circle"
                percent={stats.accuracyRate}
                format={percent => `${percent}%`}
                strokeColor={{
                  '0%': '#108ee9',
                  '100%': '#87d068'
                }}
                size={120}
              />
              <p style={{ marginTop: 16, color: '#8c8c8c' }}>AI分析准确率</p>
            </div>
          </Card>
        </Col>
        <Col xs={24} lg={12}>
          <Card title="平均处理时间" loading={loading}>
            <div style={{ textAlign: 'center' }}>
              <Statistic
                value={stats.processingTime}
                suffix="秒"
                precision={1}
                valueStyle={{ fontSize: '36px', color: '#3f8600' }}
              />
              <p style={{ marginTop: 16, color: '#8c8c8c' }}>每张影像平均处理时间</p>
            </div>
          </Card>
        </Col>
      </Row>
      
      {/* 图表区域 */}
      <Row gutter={[16, 16]} style={{ marginBottom: 24 }}>
        <Col xs={24} lg={16}>
          <Card 
            title="影像处理趋势" 
            className="chart-card"
            loading={loading}
            extra={
              <Space>
                <Select value={timeFilter} onChange={setTimeFilter} size="small">
                  <Option value="7d">最近7天</Option>
                  <Option value="30d">最近30天</Option>
                  <Option value="90d">最近90天</Option>
                </Select>
              </Space>
            }
          >
            <div className="chart-container">
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={chartData}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="date" />
                  <YAxis />
                  <Tooltip />
                  <Legend />
                  <Line type="monotone" dataKey="images" stroke="#1890ff" name="上传影像" />
                  <Line type="monotone" dataKey="analysis" stroke="#52c41a" name="完成分析" />
                </LineChart>
              </ResponsiveContainer>
            </div>
          </Card>
        </Col>
        <Col xs={24} lg={8}>
          <Card title="分析结果分布" className="chart-card" loading={loading}>
            <div className="chart-container">
              <ResponsiveContainer width="100%" height="100%">
                <PieChart>
                  <Pie
                    data={pieData}
                    cx="50%"
                    cy="50%"
                    innerRadius={40}
                    outerRadius={80}
                    paddingAngle={5}
                    dataKey="value"
                    label={({ name, percent }) => `${name} ${(percent * 100).toFixed(0)}%`}
                  >
                    {pieData.map((entry, index) => (
                      <Cell key={`cell-${index}`} fill={entry.color} />
                    ))}
                  </Pie>
                  <Tooltip />
                </PieChart>
              </ResponsiveContainer>
            </div>
          </Card>
        </Col>
      </Row>
      
      {/* 最近活动 */}
      <Row gutter={[16, 16]}>
        <Col xs={24} lg={12}>
          <Card title="最近活动" loading={loading}>
            <List
              dataSource={recentActivities}
              renderItem={(item) => (
                <List.Item className="activity-item">
                  <List.Item.Meta
                    avatar={<Avatar icon={getActivityIcon(item.type)} />}
                    title={
                      <Space>
                        {item.title}
                        {getStatusTag(item.status)}
                      </Space>
                    }
                    description={
                      <div>
                        <p style={{ margin: 0, marginBottom: 4 }}>{item.description}</p>
                        <small style={{ color: '#8c8c8c' }}>{item.time}</small>
                      </div>
                    }
                  />
                </List.Item>
              )}
            />
          </Card>
        </Col>
        <Col xs={24} lg={12}>
          <Card title="快速操作" loading={loading}>
            <Space direction="vertical" style={{ width: '100%' }} size="middle">
              <Button type="primary" size="large" block icon={<FileImageOutlined />}>
                上传新影像
              </Button>
              <Button size="large" block icon={<ExperimentOutlined />}>
                开始批量分析
              </Button>
              <Button size="large" block icon={<UserOutlined />}>
                添加新患者
              </Button>
              <Button size="large" block icon={<CheckCircleOutlined />}>
                查看待审核报告
              </Button>
            </Space>
          </Card>
        </Col>
      </Row>
    </DashboardContainer>
  )
}

export default DashboardPage