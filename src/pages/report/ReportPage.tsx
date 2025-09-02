import React, { useState, useEffect } from 'react'
import { useNavigate } from 'react-router-dom'
import { Card, Row, Col, Table, Button, Input, Select, DatePicker, Tag, Space, Modal, Form, message, Tooltip, Progress, Statistic, Typography, Divider } from 'antd'
import {
  SearchOutlined,
  PlusOutlined,
  EyeOutlined,
  EditOutlined,
  DeleteOutlined,
  DownloadOutlined,
  PrinterOutlined,
  ShareAltOutlined,
  FileTextOutlined,
  CalendarOutlined,
  UserOutlined,
  MedicineBoxOutlined,
  AlertOutlined,
  CheckCircleOutlined,
  ClockCircleOutlined,
  ExclamationCircleOutlined
} from '@ant-design/icons'
import styled from 'styled-components'
import dayjs, { Dayjs } from 'dayjs'

const { RangePicker } = DatePicker
const { Option } = Select
const { TextArea } = Input
const { Title, Text, Paragraph } = Typography
const { confirm } = Modal

const ReportPageContainer = styled.div`
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
  }
  
  .filter-bar {
    display: flex;
    align-items: center;
    gap: 16px;
    margin-bottom: 16px;
    padding: 16px;
    background: #fafafa;
    border-radius: 8px;
    
    .filter-item {
      display: flex;
      align-items: center;
      gap: 8px;
      
      label {
        white-space: nowrap;
        color: #595959;
        font-weight: 500;
      }
    }
  }
  
  .report-table {
    .status-tag {
      &.draft { background: #f0f0f0; color: #595959; }
      &.pending { background: #fff7e6; color: #fa8c16; }
      &.completed { background: #f6ffed; color: #52c41a; }
      &.reviewed { background: #e6f7ff; color: #1890ff; }
    }
    
    .priority-tag {
      &.high { background: #fff2f0; color: #ff4d4f; }
      &.medium { background: #fff7e6; color: #fa8c16; }
      &.low { background: #f6ffed; color: #52c41a; }
    }
  }
  
  .report-actions {
    display: flex;
    gap: 8px;
  }
`

interface Report {
  id: string
  reportNumber: string
  patientId: string
  patientName: string
  studyType: string
  studyDate: string
  reportDate: string
  status: 'draft' | 'pending' | 'completed' | 'reviewed'
  priority: 'high' | 'medium' | 'low'
  doctor: string
  reviewer?: string
  findings: string
  impression: string
  recommendations: string[]
  imageCount: number
  abnormalFindings: number
  confidence: number
}

const ReportPage: React.FC = () => {
  const navigate = useNavigate()
  const [loading, setLoading] = useState(false)
  const [reports, setReports] = useState<Report[]>([])
  const [filteredReports, setFilteredReports] = useState<Report[]>([])
  const [searchText, setSearchText] = useState('')
  const [statusFilter, setStatusFilter] = useState<string>('')
  const [priorityFilter, setPriorityFilter] = useState<string>('')
  const [dateRange, setDateRange] = useState<[Dayjs | null, Dayjs | null] | null>(null)
  const [selectedRowKeys, setSelectedRowKeys] = useState<React.Key[]>([])
  const [createModalVisible, setCreateModalVisible] = useState(false)
  const [form] = Form.useForm()
  
  // Mock data
  const mockReports: Report[] = [
    {
      id: '1',
      reportNumber: 'RPT-2024-001',
      patientId: 'P001',
      patientName: '张三',
      studyType: '胸部CT',
      studyDate: '2024-01-20T14:20:00Z',
      reportDate: '2024-01-20T16:30:00Z',
      status: 'completed',
      priority: 'high',
      doctor: '李医生',
      reviewer: '王主任',
      findings: '右上肺发现8mm结节，边界清晰，密度均匀。左下肺见点状钙化灶。',
      impression: '右上肺结节，建议进一步随访观察。',
      recommendations: ['建议3个月后复查胸部CT', '如有症状变化请及时就诊'],
      imageCount: 120,
      abnormalFindings: 2,
      confidence: 0.92
    },
    {
      id: '2',
      reportNumber: 'RPT-2024-002',
      patientId: 'P002',
      patientName: '李四',
      studyType: '腹部CT',
      studyDate: '2024-01-19T10:15:00Z',
      reportDate: '2024-01-19T15:45:00Z',
      status: 'reviewed',
      priority: 'medium',
      doctor: '张医生',
      reviewer: '王主任',
      findings: '肝脏形态正常，密度均匀。胆囊壁略厚。',
      impression: '胆囊壁增厚，建议进一步检查。',
      recommendations: ['建议行胆囊超声检查', '注意饮食调节'],
      imageCount: 80,
      abnormalFindings: 1,
      confidence: 0.85
    },
    {
      id: '3',
      reportNumber: 'RPT-2024-003',
      patientId: 'P003',
      patientName: '王五',
      studyType: '头部MRI',
      studyDate: '2024-01-18T16:30:00Z',
      reportDate: '2024-01-19T09:20:00Z',
      status: 'pending',
      priority: 'low',
      doctor: '赵医生',
      findings: '脑实质未见明显异常信号。',
      impression: '头部MRI未见明显异常。',
      recommendations: ['定期体检', '保持健康生活方式'],
      imageCount: 60,
      abnormalFindings: 0,
      confidence: 0.95
    },
    {
      id: '4',
      reportNumber: 'RPT-2024-004',
      patientId: 'P004',
      patientName: '赵六',
      studyType: '胸部X光',
      studyDate: '2024-01-17T14:00:00Z',
      reportDate: '',
      status: 'draft',
      priority: 'medium',
      doctor: '孙医生',
      findings: '双肺纹理清晰，心影大小正常。',
      impression: '',
      recommendations: [],
      imageCount: 2,
      abnormalFindings: 0,
      confidence: 0.88
    }
  ]
  
  useEffect(() => {
    loadReports()
  }, [])
  
  useEffect(() => {
    filterReports()
  }, [reports, searchText, statusFilter, priorityFilter, dateRange])
  
  const loadReports = async () => {
    setLoading(true)
    // Simulate API call
    setTimeout(() => {
      setReports(mockReports)
      setLoading(false)
    }, 1000)
  }
  
  const filterReports = () => {
    let filtered = [...reports]
    
    // 文本搜索
    if (searchText) {
      filtered = filtered.filter(report => 
        report.reportNumber.toLowerCase().includes(searchText.toLowerCase()) ||
        report.patientName.toLowerCase().includes(searchText.toLowerCase()) ||
        report.studyType.toLowerCase().includes(searchText.toLowerCase()) ||
        report.doctor.toLowerCase().includes(searchText.toLowerCase())
      )
    }
    
    // 状态筛选
    if (statusFilter) {
      filtered = filtered.filter(report => report.status === statusFilter)
    }
    
    // 优先级筛选
    if (priorityFilter) {
      filtered = filtered.filter(report => report.priority === priorityFilter)
    }
    
    // 日期范围筛选
    if (dateRange && dateRange[0] && dateRange[1]) {
      filtered = filtered.filter(report => {
        const reportDate = dayjs(report.studyDate)
        return reportDate.isAfter(dateRange[0]) && reportDate.isBefore(dateRange[1])
      })
    }
    
    setFilteredReports(filtered)
  }
  
  const handleViewReport = (record: Report) => {
    navigate(`/reports/${record.id}`)
  }
  
  const handleEditReport = (record: Report) => {
    // 编辑报告逻辑
    message.info('编辑功能开发中')
  }
  
  const handleDeleteReport = (record: Report) => {
    confirm({
      title: '确认删除',
      content: `确定要删除报告 ${record.reportNumber} 吗？`,
      okText: '确定',
      cancelText: '取消',
      onOk() {
        setReports(prev => prev.filter(r => r.id !== record.id))
        message.success('报告已删除')
      }
    })
  }
  
  const handleDownloadReport = (record: Report) => {
    message.info('下载功能开发中')
  }
  
  const handlePrintReport = (record: Report) => {
    message.info('打印功能开发中')
  }
  
  const handleShareReport = (record: Report) => {
    message.info('分享功能开发中')
  }
  
  const handleCreateReport = () => {
    setCreateModalVisible(true)
  }
  
  const handleSaveReport = async (values: any) => {
    try {
      const newReport: Report = {
        id: Date.now().toString(),
        reportNumber: `RPT-${dayjs().format('YYYY')}-${String(reports.length + 1).padStart(3, '0')}`,
        ...values,
        studyDate: values.studyDate.toISOString(),
        reportDate: dayjs().toISOString(),
        status: 'draft',
        imageCount: 0,
        abnormalFindings: 0,
        confidence: 0
      }
      
      setReports(prev => [newReport, ...prev])
      setCreateModalVisible(false)
      form.resetFields()
      message.success('报告已创建')
    } catch (error) {
      message.error('创建失败，请重试')
    }
  }
  
  const getStatusColor = (status: string) => {
    switch (status) {
      case 'draft': return 'default'
      case 'pending': return 'warning'
      case 'completed': return 'success'
      case 'reviewed': return 'processing'
      default: return 'default'
    }
  }
  
  const getStatusText = (status: string) => {
    switch (status) {
      case 'draft': return '草稿'
      case 'pending': return '待审核'
      case 'completed': return '已完成'
      case 'reviewed': return '已审核'
      default: return '未知'
    }
  }
  
  const getPriorityColor = (priority: string) => {
    switch (priority) {
      case 'high': return 'red'
      case 'medium': return 'orange'
      case 'low': return 'green'
      default: return 'default'
    }
  }
  
  const getPriorityText = (priority: string) => {
    switch (priority) {
      case 'high': return '高'
      case 'medium': return '中'
      case 'low': return '低'
      default: return '未知'
    }
  }
  
  const columns = [
    {
      title: '报告编号',
      dataIndex: 'reportNumber',
      key: 'reportNumber',
      width: 140,
      render: (text: string, record: Report) => (
        <Button type="link" onClick={() => handleViewReport(record)}>
          {text}
        </Button>
      )
    },
    {
      title: '患者信息',
      key: 'patient',
      width: 150,
      render: (record: Report) => (
        <div>
          <div style={{ fontWeight: 500 }}>{record.patientName}</div>
          <div style={{ fontSize: '12px', color: '#8c8c8c' }}>{record.patientId}</div>
        </div>
      )
    },
    {
      title: '检查类型',
      dataIndex: 'studyType',
      key: 'studyType',
      width: 120
    },
    {
      title: '检查日期',
      dataIndex: 'studyDate',
      key: 'studyDate',
      width: 120,
      render: (date: string) => dayjs(date).format('MM-DD HH:mm')
    },
    {
      title: '状态',
      dataIndex: 'status',
      key: 'status',
      width: 100,
      render: (status: string) => (
        <Tag color={getStatusColor(status)}>
          {getStatusText(status)}
        </Tag>
      )
    },
    {
      title: '优先级',
      dataIndex: 'priority',
      key: 'priority',
      width: 80,
      render: (priority: string) => (
        <Tag color={getPriorityColor(priority)}>
          {getPriorityText(priority)}
        </Tag>
      )
    },
    {
      title: '医生',
      dataIndex: 'doctor',
      key: 'doctor',
      width: 100
    },
    {
      title: '异常发现',
      dataIndex: 'abnormalFindings',
      key: 'abnormalFindings',
      width: 100,
      render: (count: number) => (
        <span style={{ color: count > 0 ? '#ff4d4f' : '#52c41a' }}>
          {count} 项
        </span>
      )
    },
    {
      title: '置信度',
      dataIndex: 'confidence',
      key: 'confidence',
      width: 100,
      render: (confidence: number) => (
        <Progress 
          percent={confidence * 100} 
          size="small" 
          showInfo={false}
          strokeColor={confidence > 0.8 ? '#52c41a' : confidence > 0.6 ? '#fa8c16' : '#ff4d4f'}
        />
      )
    },
    {
      title: '操作',
      key: 'actions',
      width: 200,
      render: (record: Report) => (
        <div className="report-actions">
          <Tooltip title="查看">
            <Button 
              type="text" 
              size="small" 
              icon={<EyeOutlined />} 
              onClick={() => handleViewReport(record)}
            />
          </Tooltip>
          <Tooltip title="编辑">
            <Button 
              type="text" 
              size="small" 
              icon={<EditOutlined />} 
              onClick={() => handleEditReport(record)}
            />
          </Tooltip>
          <Tooltip title="下载">
            <Button 
              type="text" 
              size="small" 
              icon={<DownloadOutlined />} 
              onClick={() => handleDownloadReport(record)}
            />
          </Tooltip>
          <Tooltip title="打印">
            <Button 
              type="text" 
              size="small" 
              icon={<PrinterOutlined />} 
              onClick={() => handlePrintReport(record)}
            />
          </Tooltip>
          <Tooltip title="分享">
            <Button 
              type="text" 
              size="small" 
              icon={<ShareAltOutlined />} 
              onClick={() => handleShareReport(record)}
            />
          </Tooltip>
          <Tooltip title="删除">
            <Button 
              type="text" 
              size="small" 
              icon={<DeleteOutlined />} 
              danger
              onClick={() => handleDeleteReport(record)}
            />
          </Tooltip>
        </div>
      )
    }
  ]
  
  const rowSelection = {
    selectedRowKeys,
    onChange: setSelectedRowKeys
  }
  
  // 统计数据
  const stats = {
    total: reports.length,
    draft: reports.filter(r => r.status === 'draft').length,
    pending: reports.filter(r => r.status === 'pending').length,
    completed: reports.filter(r => r.status === 'completed').length,
    reviewed: reports.filter(r => r.status === 'reviewed').length,
    highPriority: reports.filter(r => r.priority === 'high').length,
    abnormal: reports.filter(r => r.abnormalFindings > 0).length
  }
  
  return (
    <ReportPageContainer>
      <div className="page-header">
        <h1>报告管理</h1>
        <Button type="primary" icon={<PlusOutlined />} onClick={handleCreateReport}>
          创建报告
        </Button>
      </div>
      
      {/* 统计卡片 */}
      <Row gutter={[16, 16]} style={{ marginBottom: 24 }}>
        <Col span={6}>
          <Card>
            <Statistic
              title="总报告数"
              value={stats.total}
              prefix={<FileTextOutlined />}
            />
          </Card>
        </Col>
        <Col span={6}>
          <Card>
            <Statistic
              title="待审核"
              value={stats.pending}
              prefix={<ClockCircleOutlined />}
              valueStyle={{ color: '#fa8c16' }}
            />
          </Card>
        </Col>
        <Col span={6}>
          <Card>
            <Statistic
              title="高优先级"
              value={stats.highPriority}
              prefix={<ExclamationCircleOutlined />}
              valueStyle={{ color: '#ff4d4f' }}
            />
          </Card>
        </Col>
        <Col span={6}>
          <Card>
            <Statistic
              title="异常发现"
              value={stats.abnormal}
              prefix={<AlertOutlined />}
              valueStyle={{ color: '#ff4d4f' }}
            />
          </Card>
        </Col>
      </Row>
      
      {/* 筛选栏 */}
      <Card className="filter-bar">
        <div className="filter-item">
          <label>搜索:</label>
          <Input
            placeholder="报告编号、患者姓名、检查类型、医生"
            prefix={<SearchOutlined />}
            value={searchText}
            onChange={e => setSearchText(e.target.value)}
            style={{ width: 250 }}
            allowClear
          />
        </div>
        
        <div className="filter-item">
          <label>状态:</label>
          <Select
            placeholder="选择状态"
            value={statusFilter}
            onChange={setStatusFilter}
            style={{ width: 120 }}
            allowClear
          >
            <Option value="draft">草稿</Option>
            <Option value="pending">待审核</Option>
            <Option value="completed">已完成</Option>
            <Option value="reviewed">已审核</Option>
          </Select>
        </div>
        
        <div className="filter-item">
          <label>优先级:</label>
          <Select
            placeholder="选择优先级"
            value={priorityFilter}
            onChange={setPriorityFilter}
            style={{ width: 100 }}
            allowClear
          >
            <Option value="high">高</Option>
            <Option value="medium">中</Option>
            <Option value="low">低</Option>
          </Select>
        </div>
        
        <div className="filter-item">
          <label>日期范围:</label>
          <RangePicker
            value={dateRange}
            onChange={setDateRange}
            format="YYYY-MM-DD"
          />
        </div>
        
        <Button onClick={() => {
          setSearchText('')
          setStatusFilter('')
          setPriorityFilter('')
          setDateRange(null)
        }}>
          重置
        </Button>
      </Card>
      
      {/* 报告表格 */}
      <Card>
        <Table
          className="report-table"
          columns={columns}
          dataSource={filteredReports}
          rowKey="id"
          loading={loading}
          rowSelection={rowSelection}
          pagination={{
            total: filteredReports.length,
            pageSize: 10,
            showSizeChanger: true,
            showQuickJumper: true,
            showTotal: (total, range) => `第 ${range[0]}-${range[1]} 条，共 ${total} 条`
          }}
          scroll={{ x: 1200 }}
        />
      </Card>
      
      {/* 创建报告弹窗 */}
      <Modal
        title="创建新报告"
        open={createModalVisible}
        onCancel={() => setCreateModalVisible(false)}
        onOk={() => form.submit()}
        width={600}
      >
        <Form
          form={form}
          layout="vertical"
          onFinish={handleSaveReport}
        >
          <Row gutter={16}>
            <Col span={12}>
              <Form.Item
                label="患者ID"
                name="patientId"
                rules={[{ required: true, message: '请输入患者ID' }]}
              >
                <Input placeholder="例如: P001" />
              </Form.Item>
            </Col>
            <Col span={12}>
              <Form.Item
                label="患者姓名"
                name="patientName"
                rules={[{ required: true, message: '请输入患者姓名' }]}
              >
                <Input placeholder="例如: 张三" />
              </Form.Item>
            </Col>
          </Row>
          
          <Row gutter={16}>
            <Col span={12}>
              <Form.Item
                label="检查类型"
                name="studyType"
                rules={[{ required: true, message: '请选择检查类型' }]}
              >
                <Select placeholder="选择检查类型">
                  <Option value="胸部CT">胸部CT</Option>
                  <Option value="腹部CT">腹部CT</Option>
                  <Option value="头部MRI">头部MRI</Option>
                  <Option value="胸部X光">胸部X光</Option>
                  <Option value="腹部超声">腹部超声</Option>
                </Select>
              </Form.Item>
            </Col>
            <Col span={12}>
              <Form.Item
                label="检查日期"
                name="studyDate"
                rules={[{ required: true, message: '请选择检查日期' }]}
              >
                <DatePicker 
                  style={{ width: '100%' }}
                  showTime
                  format="YYYY-MM-DD HH:mm"
                />
              </Form.Item>
            </Col>
          </Row>
          
          <Row gutter={16}>
            <Col span={12}>
              <Form.Item
                label="优先级"
                name="priority"
                rules={[{ required: true, message: '请选择优先级' }]}
                initialValue="medium"
              >
                <Select>
                  <Option value="high">高</Option>
                  <Option value="medium">中</Option>
                  <Option value="low">低</Option>
                </Select>
              </Form.Item>
            </Col>
            <Col span={12}>
              <Form.Item
                label="医生"
                name="doctor"
                rules={[{ required: true, message: '请输入医生姓名' }]}
              >
                <Input placeholder="例如: 李医生" />
              </Form.Item>
            </Col>
          </Row>
          
          <Form.Item
            label="检查所见"
            name="findings"
          >
            <TextArea 
              rows={4} 
              placeholder="描述检查发现的异常或正常表现"
            />
          </Form.Item>
          
          <Form.Item
            label="诊断印象"
            name="impression"
          >
            <TextArea 
              rows={3} 
              placeholder="基于检查所见的诊断结论"
            />
          </Form.Item>
          
          <Form.Item
            label="建议"
            name="recommendations"
          >
            <Select 
              mode="tags" 
              placeholder="输入建议内容，按回车添加"
              style={{ width: '100%' }}
            >
              <Option value="建议进一步检查">建议进一步检查</Option>
              <Option value="定期随访">定期随访</Option>
              <Option value="药物治疗">药物治疗</Option>
              <Option value="手术治疗">手术治疗</Option>
            </Select>
          </Form.Item>
        </Form>
      </Modal>
    </ReportPageContainer>
  )
}

export default ReportPage