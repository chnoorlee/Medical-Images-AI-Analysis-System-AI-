import React, { useState, useEffect } from 'react'
import { Table, Card, Button, Input, Select, Space, Tag, Avatar, Modal, Form, Row, Col, DatePicker, Pagination, Tooltip, Dropdown, message } from 'antd'
import {
  PlusOutlined,
  SearchOutlined,
  EyeOutlined,
  EditOutlined,
  DeleteOutlined,
  MoreOutlined,
  UserOutlined,
  ManOutlined,
  WomanOutlined,
  FileImageOutlined,
  CalendarOutlined,
  PhoneOutlined,
  EnvironmentOutlined
} from '@ant-design/icons'
import { useNavigate } from 'react-router-dom'
import styled from 'styled-components'
import dayjs from 'dayjs'

const { Search } = Input
const { Option } = Select
const { RangePicker } = DatePicker

const PatientListContainer = styled.div`
  .page-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 24px;
    
    h1 {
      margin: 0;
      color: #262626;
      font-size: 24px;
      font-weight: 600;
    }
  }
  
  .filter-bar {
    background: #fafafa;
    padding: 16px;
    border-radius: 8px;
    margin-bottom: 16px;
    
    .filter-row {
      display: flex;
      gap: 16px;
      align-items: center;
      flex-wrap: wrap;
    }
  }
  
  .patient-avatar {
    display: flex;
    align-items: center;
    gap: 12px;
    
    .patient-info {
      .patient-name {
        font-weight: 500;
        color: #262626;
      }
      
      .patient-id {
        font-size: 12px;
        color: #8c8c8c;
      }
    }
  }
  
  .stats-cards {
    margin-bottom: 24px;
    
    .stat-card {
      text-align: center;
      
      .stat-number {
        font-size: 24px;
        font-weight: 600;
        color: #1890ff;
        margin-bottom: 4px;
      }
      
      .stat-label {
        color: #8c8c8c;
        font-size: 14px;
      }
    }
  }
`

interface Patient {
  id: string
  name: string
  patientId: string
  gender: 'male' | 'female'
  age: number
  phone: string
  address: string
  birthDate: string
  createTime: string
  lastVisit?: string
  imageCount: number
  status: 'active' | 'inactive'
  diagnosis?: string
  doctor: string
}

const PatientListPage: React.FC = () => {
  const navigate = useNavigate()
  const [loading, setLoading] = useState(false)
  const [patients, setPatients] = useState<Patient[]>([])
  const [filteredPatients, setFilteredPatients] = useState<Patient[]>([])
  const [searchText, setSearchText] = useState('')
  const [genderFilter, setGenderFilter] = useState<string>('')
  const [statusFilter, setStatusFilter] = useState<string>('')
  const [dateRange, setDateRange] = useState<[dayjs.Dayjs, dayjs.Dayjs] | null>(null)
  const [currentPage, setCurrentPage] = useState(1)
  const [pageSize, setPageSize] = useState(10)
  const [addModalVisible, setAddModalVisible] = useState(false)
  const [editModalVisible, setEditModalVisible] = useState(false)
  const [selectedPatient, setSelectedPatient] = useState<Patient | null>(null)
  const [form] = Form.useForm()
  
  // Mock data
  const mockPatients: Patient[] = [
    {
      id: '1',
      name: '张三',
      patientId: 'P001',
      gender: 'male',
      age: 45,
      phone: '13800138001',
      address: '北京市朝阳区',
      birthDate: '1978-05-15',
      createTime: '2024-01-15T10:30:00Z',
      lastVisit: '2024-01-20T14:20:00Z',
      imageCount: 12,
      status: 'active',
      diagnosis: '肺结节',
      doctor: '李医生'
    },
    {
      id: '2',
      name: '李四',
      patientId: 'P002',
      gender: 'female',
      age: 32,
      phone: '13800138002',
      address: '上海市浦东新区',
      birthDate: '1991-08-22',
      createTime: '2024-01-16T09:15:00Z',
      lastVisit: '2024-01-19T16:45:00Z',
      imageCount: 8,
      status: 'active',
      diagnosis: '脑肿瘤',
      doctor: '王医生'
    },
    {
      id: '3',
      name: '王五',
      patientId: 'P003',
      gender: 'male',
      age: 67,
      phone: '13800138003',
      address: '广州市天河区',
      birthDate: '1956-12-03',
      createTime: '2024-01-17T11:20:00Z',
      imageCount: 15,
      status: 'active',
      diagnosis: '骨折',
      doctor: '张医生'
    },
    {
      id: '4',
      name: '赵六',
      patientId: 'P004',
      gender: 'female',
      age: 28,
      phone: '13800138004',
      address: '深圳市南山区',
      birthDate: '1995-03-18',
      createTime: '2024-01-18T15:30:00Z',
      lastVisit: '2024-01-21T10:15:00Z',
      imageCount: 6,
      status: 'inactive',
      doctor: '陈医生'
    }
  ]
  
  useEffect(() => {
    loadPatients()
  }, [])
  
  useEffect(() => {
    filterPatients()
  }, [patients, searchText, genderFilter, statusFilter, dateRange])
  
  const loadPatients = async () => {
    setLoading(true)
    // Simulate API call
    setTimeout(() => {
      setPatients(mockPatients)
      setLoading(false)
    }, 1000)
  }
  
  const filterPatients = () => {
    let filtered = [...patients]
    
    // Text search
    if (searchText) {
      filtered = filtered.filter(patient => 
        patient.name.toLowerCase().includes(searchText.toLowerCase()) ||
        patient.patientId.toLowerCase().includes(searchText.toLowerCase()) ||
        patient.phone.includes(searchText)
      )
    }
    
    // Gender filter
    if (genderFilter) {
      filtered = filtered.filter(patient => patient.gender === genderFilter)
    }
    
    // Status filter
    if (statusFilter) {
      filtered = filtered.filter(patient => patient.status === statusFilter)
    }
    
    // Date range filter
    if (dateRange) {
      filtered = filtered.filter(patient => {
        const createDate = dayjs(patient.createTime)
        return createDate.isAfter(dateRange[0]) && createDate.isBefore(dateRange[1])
      })
    }
    
    setFilteredPatients(filtered)
    setCurrentPage(1)
  }
  
  const handleAddPatient = () => {
    setSelectedPatient(null)
    form.resetFields()
    setAddModalVisible(true)
  }
  
  const handleEditPatient = (patient: Patient) => {
    setSelectedPatient(patient)
    form.setFieldsValue({
      ...patient,
      birthDate: dayjs(patient.birthDate)
    })
    setEditModalVisible(true)
  }
  
  const handleDeletePatient = (patient: Patient) => {
    Modal.confirm({
      title: '确认删除',
      content: `确定要删除患者 ${patient.name} 的信息吗？此操作不可恢复。`,
      okText: '确定',
      cancelText: '取消',
      onOk: () => {
        setPatients(prev => prev.filter(p => p.id !== patient.id))
        message.success('患者信息已删除')
      }
    })
  }
  
  const handleViewPatient = (patient: Patient) => {
    navigate(`/patients/${patient.id}`)
  }
  
  const handleSavePatient = async (values: any) => {
    try {
      if (selectedPatient) {
        // Update existing patient
        setPatients(prev => prev.map(p => 
          p.id === selectedPatient.id 
            ? { ...p, ...values, birthDate: values.birthDate.format('YYYY-MM-DD') }
            : p
        ))
        message.success('患者信息已更新')
        setEditModalVisible(false)
      } else {
        // Add new patient
        const newPatient: Patient = {
          id: Date.now().toString(),
          patientId: `P${String(patients.length + 1).padStart(3, '0')}`,
          ...values,
          birthDate: values.birthDate.format('YYYY-MM-DD'),
          createTime: new Date().toISOString(),
          imageCount: 0,
          status: 'active',
          doctor: '当前医生'
        }
        setPatients(prev => [newPatient, ...prev])
        message.success('患者信息已添加')
        setAddModalVisible(false)
      }
      form.resetFields()
    } catch (error) {
      message.error('操作失败，请重试')
    }
  }
  
  const columns = [
    {
      title: '患者信息',
      key: 'patient',
      width: 250,
      render: (record: Patient) => (
        <div className="patient-avatar">
          <Avatar 
            size={40} 
            icon={record.gender === 'male' ? <ManOutlined /> : <WomanOutlined />}
            style={{ backgroundColor: record.gender === 'male' ? '#1890ff' : '#eb2f96' }}
          />
          <div className="patient-info">
            <div className="patient-name">{record.name}</div>
            <div className="patient-id">ID: {record.patientId}</div>
          </div>
        </div>
      )
    },
    {
      title: '性别/年龄',
      key: 'demographics',
      width: 100,
      render: (record: Patient) => (
        <div>
          <div>{record.gender === 'male' ? '男' : '女'}</div>
          <div style={{ color: '#8c8c8c', fontSize: '12px' }}>{record.age}岁</div>
        </div>
      )
    },
    {
      title: '联系方式',
      key: 'contact',
      width: 200,
      render: (record: Patient) => (
        <div>
          <div><PhoneOutlined /> {record.phone}</div>
          <div style={{ color: '#8c8c8c', fontSize: '12px' }}>
            <EnvironmentOutlined /> {record.address}
          </div>
        </div>
      )
    },
    {
      title: '影像数量',
      dataIndex: 'imageCount',
      key: 'imageCount',
      width: 100,
      render: (count: number) => (
        <div style={{ textAlign: 'center' }}>
          <div style={{ fontSize: '16px', fontWeight: 500, color: '#1890ff' }}>{count}</div>
          <div style={{ fontSize: '12px', color: '#8c8c8c' }}>张</div>
        </div>
      )
    },
    {
      title: '最后就诊',
      dataIndex: 'lastVisit',
      key: 'lastVisit',
      width: 120,
      render: (date: string) => date ? dayjs(date).format('MM-DD HH:mm') : '-'
    },
    {
      title: '状态',
      dataIndex: 'status',
      key: 'status',
      width: 80,
      render: (status: string) => (
        <Tag color={status === 'active' ? 'green' : 'default'}>
          {status === 'active' ? '活跃' : '非活跃'}
        </Tag>
      )
    },
    {
      title: '诊断',
      dataIndex: 'diagnosis',
      key: 'diagnosis',
      width: 120,
      render: (diagnosis: string) => diagnosis || '-'
    },
    {
      title: '操作',
      key: 'actions',
      width: 120,
      render: (record: Patient) => (
        <Space>
          <Tooltip title="查看详情">
            <Button 
              type="text" 
              size="small" 
              icon={<EyeOutlined />} 
              onClick={() => handleViewPatient(record)}
            />
          </Tooltip>
          <Tooltip title="编辑">
            <Button 
              type="text" 
              size="small" 
              icon={<EditOutlined />} 
              onClick={() => handleEditPatient(record)}
            />
          </Tooltip>
          <Dropdown
            menu={{
              items: [
                {
                  key: 'images',
                  label: '查看影像',
                  icon: <FileImageOutlined />
                },
                {
                  key: 'delete',
                  label: '删除',
                  icon: <DeleteOutlined />,
                  danger: true,
                  onClick: () => handleDeletePatient(record)
                }
              ]
            }}
          >
            <Button type="text" size="small" icon={<MoreOutlined />} />
          </Dropdown>
        </Space>
      )
    }
  ]
  
  const paginatedData = filteredPatients.slice(
    (currentPage - 1) * pageSize,
    currentPage * pageSize
  )
  
  return (
    <PatientListContainer>
      <div className="page-header">
        <h1>患者管理</h1>
        <Button type="primary" icon={<PlusOutlined />} onClick={handleAddPatient}>
          添加患者
        </Button>
      </div>
      
      {/* 统计卡片 */}
      <Row gutter={16} className="stats-cards">
        <Col span={6}>
          <Card>
            <div className="stat-card">
              <div className="stat-number">{patients.length}</div>
              <div className="stat-label">总患者数</div>
            </div>
          </Card>
        </Col>
        <Col span={6}>
          <Card>
            <div className="stat-card">
              <div className="stat-number">{patients.filter(p => p.status === 'active').length}</div>
              <div className="stat-label">活跃患者</div>
            </div>
          </Card>
        </Col>
        <Col span={6}>
          <Card>
            <div className="stat-card">
              <div className="stat-number">{patients.reduce((sum, p) => sum + p.imageCount, 0)}</div>
              <div className="stat-label">总影像数</div>
            </div>
          </Card>
        </Col>
        <Col span={6}>
          <Card>
            <div className="stat-card">
              <div className="stat-number">{patients.filter(p => p.lastVisit && dayjs(p.lastVisit).isAfter(dayjs().subtract(7, 'day'))).length}</div>
              <div className="stat-label">本周就诊</div>
            </div>
          </Card>
        </Col>
      </Row>
      
      {/* 筛选栏 */}
      <Card className="filter-bar">
        <div className="filter-row">
          <Search
            placeholder="搜索患者姓名、ID或电话"
            value={searchText}
            onChange={(e) => setSearchText(e.target.value)}
            style={{ width: 250 }}
            allowClear
          />
          
          <Select
            placeholder="性别"
            value={genderFilter}
            onChange={setGenderFilter}
            style={{ width: 120 }}
            allowClear
          >
            <Option value="male">男</Option>
            <Option value="female">女</Option>
          </Select>
          
          <Select
            placeholder="状态"
            value={statusFilter}
            onChange={setStatusFilter}
            style={{ width: 120 }}
            allowClear
          >
            <Option value="active">活跃</Option>
            <Option value="inactive">非活跃</Option>
          </Select>
          
          <RangePicker
            placeholder={['开始日期', '结束日期']}
            value={dateRange}
            onChange={setDateRange}
          />
          
          <Button onClick={() => {
            setSearchText('')
            setGenderFilter('')
            setStatusFilter('')
            setDateRange(null)
          }}>
            重置
          </Button>
        </div>
      </Card>
      
      {/* 患者列表 */}
      <Card>
        <Table
          columns={columns}
          dataSource={paginatedData}
          rowKey="id"
          loading={loading}
          pagination={false}
          scroll={{ x: 1200 }}
        />
        
        <div style={{ marginTop: 16, textAlign: 'right' }}>
          <Pagination
            current={currentPage}
            pageSize={pageSize}
            total={filteredPatients.length}
            onChange={setCurrentPage}
            onShowSizeChange={(current, size) => {
              setCurrentPage(1)
              setPageSize(size)
            }}
            showSizeChanger
            showQuickJumper
            showTotal={(total, range) => `第 ${range[0]}-${range[1]} 条，共 ${total} 条`}
          />
        </div>
      </Card>
      
      {/* 添加患者弹窗 */}
      <Modal
        title="添加患者"
        open={addModalVisible}
        onCancel={() => setAddModalVisible(false)}
        onOk={() => form.submit()}
        width={600}
      >
        <Form
          form={form}
          layout="vertical"
          onFinish={handleSavePatient}
        >
          <Row gutter={16}>
            <Col span={12}>
              <Form.Item
                label="姓名"
                name="name"
                rules={[{ required: true, message: '请输入患者姓名' }]}
              >
                <Input />
              </Form.Item>
            </Col>
            <Col span={12}>
              <Form.Item
                label="性别"
                name="gender"
                rules={[{ required: true, message: '请选择性别' }]}
              >
                <Select>
                  <Option value="male">男</Option>
                  <Option value="female">女</Option>
                </Select>
              </Form.Item>
            </Col>
          </Row>
          
          <Row gutter={16}>
            <Col span={12}>
              <Form.Item
                label="出生日期"
                name="birthDate"
                rules={[{ required: true, message: '请选择出生日期' }]}
              >
                <DatePicker style={{ width: '100%' }} />
              </Form.Item>
            </Col>
            <Col span={12}>
              <Form.Item
                label="年龄"
                name="age"
                rules={[{ required: true, message: '请输入年龄' }]}
              >
                <Input type="number" />
              </Form.Item>
            </Col>
          </Row>
          
          <Row gutter={16}>
            <Col span={12}>
              <Form.Item
                label="联系电话"
                name="phone"
                rules={[{ required: true, message: '请输入联系电话' }]}
              >
                <Input />
              </Form.Item>
            </Col>
            <Col span={12}>
              <Form.Item
                label="地址"
                name="address"
              >
                <Input />
              </Form.Item>
            </Col>
          </Row>
        </Form>
      </Modal>
      
      {/* 编辑患者弹窗 */}
      <Modal
        title="编辑患者信息"
        open={editModalVisible}
        onCancel={() => setEditModalVisible(false)}
        onOk={() => form.submit()}
        width={600}
      >
        <Form
          form={form}
          layout="vertical"
          onFinish={handleSavePatient}
        >
          <Row gutter={16}>
            <Col span={12}>
              <Form.Item
                label="姓名"
                name="name"
                rules={[{ required: true, message: '请输入患者姓名' }]}
              >
                <Input />
              </Form.Item>
            </Col>
            <Col span={12}>
              <Form.Item
                label="性别"
                name="gender"
                rules={[{ required: true, message: '请选择性别' }]}
              >
                <Select>
                  <Option value="male">男</Option>
                  <Option value="female">女</Option>
                </Select>
              </Form.Item>
            </Col>
          </Row>
          
          <Row gutter={16}>
            <Col span={12}>
              <Form.Item
                label="出生日期"
                name="birthDate"
                rules={[{ required: true, message: '请选择出生日期' }]}
              >
                <DatePicker style={{ width: '100%' }} />
              </Form.Item>
            </Col>
            <Col span={12}>
              <Form.Item
                label="年龄"
                name="age"
                rules={[{ required: true, message: '请输入年龄' }]}
              >
                <Input type="number" />
              </Form.Item>
            </Col>
          </Row>
          
          <Row gutter={16}>
            <Col span={12}>
              <Form.Item
                label="联系电话"
                name="phone"
                rules={[{ required: true, message: '请输入联系电话' }]}
              >
                <Input />
              </Form.Item>
            </Col>
            <Col span={12}>
              <Form.Item
                label="地址"
                name="address"
              >
                <Input />
              </Form.Item>
            </Col>
          </Row>
        </Form>
      </Modal>
    </PatientListContainer>
  )
}

export default PatientListPage