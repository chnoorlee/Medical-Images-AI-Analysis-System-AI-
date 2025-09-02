import React, { useState, useEffect } from 'react'
import { useParams, useNavigate } from 'react-router-dom'
import { Card, Row, Col, Descriptions, Avatar, Tag, Button, Table, Timeline, Tabs, Space, Modal, Form, Input, DatePicker, Select, message, Divider, Progress, Image } from 'antd'
import {
  ArrowLeftOutlined,
  EditOutlined,
  FileImageOutlined,
  EyeOutlined,
  DownloadOutlined,
  PrinterOutlined,
  ManOutlined,
  WomanOutlined,
  CalendarOutlined,
  PhoneOutlined,
  EnvironmentOutlined,
  MedicineBoxOutlined,
  HeartOutlined,
  AlertOutlined,
  CheckCircleOutlined,
  ClockCircleOutlined
} from '@ant-design/icons'
import styled from 'styled-components'
import dayjs from 'dayjs'

const { TabPane } = Tabs
const { TextArea } = Input
const { Option } = Select

const PatientDetailContainer = styled.div`
  .page-header {
    display: flex;
    align-items: center;
    margin-bottom: 24px;
    
    .back-button {
      margin-right: 16px;
    }
    
    h1 {
      margin: 0;
      color: #262626;
      font-size: 24px;
      font-weight: 600;
    }
  }
  
  .patient-header {
    .patient-avatar {
      display: flex;
      align-items: center;
      gap: 16px;
      margin-bottom: 16px;
      
      .patient-info {
        .patient-name {
          font-size: 20px;
          font-weight: 600;
          color: #262626;
          margin-bottom: 4px;
        }
        
        .patient-id {
          color: #8c8c8c;
          margin-bottom: 8px;
        }
        
        .patient-tags {
          display: flex;
          gap: 8px;
        }
      }
    }
  }
  
  .image-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
    gap: 16px;
    
    .image-card {
      border: 1px solid #f0f0f0;
      border-radius: 8px;
      overflow: hidden;
      cursor: pointer;
      transition: all 0.3s;
      
      &:hover {
        border-color: #1890ff;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
      }
      
      .image-preview {
        width: 100%;
        height: 150px;
        background: #f5f5f5;
        display: flex;
        align-items: center;
        justify-content: center;
        
        img {
          max-width: 100%;
          max-height: 100%;
          object-fit: contain;
        }
      }
      
      .image-info {
        padding: 12px;
        
        .image-title {
          font-weight: 500;
          margin-bottom: 4px;
        }
        
        .image-meta {
          font-size: 12px;
          color: #8c8c8c;
          display: flex;
          justify-content: space-between;
        }
      }
    }
  }
  
  .timeline-item {
    .timeline-content {
      .timeline-title {
        font-weight: 500;
        margin-bottom: 4px;
      }
      
      .timeline-description {
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
  medicalHistory?: string[]
  allergies?: string[]
  emergencyContact?: {
    name: string
    phone: string
    relationship: string
  }
}

interface PatientImage {
  id: string
  name: string
  type: string
  uploadTime: string
  size: number
  thumbnail: string
  status: 'uploaded' | 'analyzed' | 'reported'
  analysisResults?: {
    findings: string
    confidence: number
    abnormal: boolean
  }
}

interface MedicalRecord {
  id: string
  date: string
  type: 'visit' | 'diagnosis' | 'treatment' | 'image' | 'report'
  title: string
  description: string
  doctor: string
  status?: string
}

const PatientDetailPage: React.FC = () => {
  const { patientId } = useParams<{ patientId: string }>()
  const navigate = useNavigate()
  const [loading, setLoading] = useState(true)
  const [patient, setPatient] = useState<Patient | null>(null)
  const [images, setImages] = useState<PatientImage[]>([])
  const [records, setRecords] = useState<MedicalRecord[]>([])
  const [editModalVisible, setEditModalVisible] = useState(false)
  const [form] = Form.useForm()
  
  // Mock data
  const mockPatient: Patient = {
    id: '1',
    name: '张三',
    patientId: 'P001',
    gender: 'male',
    age: 45,
    phone: '13800138001',
    address: '北京市朝阳区建国路88号',
    birthDate: '1978-05-15',
    createTime: '2024-01-15T10:30:00Z',
    lastVisit: '2024-01-20T14:20:00Z',
    imageCount: 12,
    status: 'active',
    diagnosis: '肺结节',
    doctor: '李医生',
    medicalHistory: ['高血压', '糖尿病'],
    allergies: ['青霉素', '海鲜'],
    emergencyContact: {
      name: '李四',
      phone: '13900139001',
      relationship: '配偶'
    }
  }
  
  const mockImages: PatientImage[] = [
    {
      id: '1',
      name: '胸部CT-001',
      type: 'CT',
      uploadTime: '2024-01-20T14:20:00Z',
      size: 2048000,
      thumbnail: '/api/placeholder/200/150',
      status: 'analyzed',
      analysisResults: {
        findings: '右上肺发现8mm结节',
        confidence: 0.92,
        abnormal: true
      }
    },
    {
      id: '2',
      name: '胸部CT-002',
      type: 'CT',
      uploadTime: '2024-01-19T10:15:00Z',
      size: 1856000,
      thumbnail: '/api/placeholder/200/150',
      status: 'reported'
    },
    {
      id: '3',
      name: '胸部X光-001',
      type: 'X-Ray',
      uploadTime: '2024-01-18T16:30:00Z',
      size: 512000,
      thumbnail: '/api/placeholder/200/150',
      status: 'uploaded'
    }
  ]
  
  const mockRecords: MedicalRecord[] = [
    {
      id: '1',
      date: '2024-01-20T14:20:00Z',
      type: 'image',
      title: '胸部CT检查',
      description: '上传胸部CT影像，AI分析发现右上肺结节',
      doctor: '李医生',
      status: 'completed'
    },
    {
      id: '2',
      date: '2024-01-19T10:15:00Z',
      type: 'report',
      title: '影像报告生成',
      description: '生成详细影像分析报告，建议进一步随访',
      doctor: '李医生',
      status: 'completed'
    },
    {
      id: '3',
      date: '2024-01-18T16:30:00Z',
      type: 'visit',
      title: '门诊就诊',
      description: '患者主诉胸闷气短，建议进行胸部CT检查',
      doctor: '李医生',
      status: 'completed'
    },
    {
      id: '4',
      date: '2024-01-15T10:30:00Z',
      type: 'diagnosis',
      title: '初步诊断',
      description: '疑似肺部疾病，需要进一步检查确诊',
      doctor: '李医生',
      status: 'completed'
    }
  ]
  
  useEffect(() => {
    loadPatientData()
  }, [patientId])
  
  const loadPatientData = async () => {
    setLoading(true)
    // Simulate API calls
    setTimeout(() => {
      setPatient(mockPatient)
      setImages(mockImages)
      setRecords(mockRecords)
      setLoading(false)
    }, 1000)
  }
  
  const handleEditPatient = () => {
    if (!patient) return
    
    form.setFieldsValue({
      ...patient,
      birthDate: dayjs(patient.birthDate),
      emergencyContactName: patient.emergencyContact?.name,
      emergencyContactPhone: patient.emergencyContact?.phone,
      emergencyContactRelationship: patient.emergencyContact?.relationship
    })
    setEditModalVisible(true)
  }
  
  const handleSavePatient = async (values: any) => {
    try {
      // Update patient data
      const updatedPatient = {
        ...patient!,
        ...values,
        birthDate: values.birthDate.format('YYYY-MM-DD'),
        emergencyContact: {
          name: values.emergencyContactName,
          phone: values.emergencyContactPhone,
          relationship: values.emergencyContactRelationship
        }
      }
      setPatient(updatedPatient)
      setEditModalVisible(false)
      message.success('患者信息已更新')
    } catch (error) {
      message.error('更新失败，请重试')
    }
  }
  
  const handleViewImage = (image: PatientImage) => {
    navigate(`/images/${image.id}`)
  }
  
  const getStatusColor = (status: string) => {
    switch (status) {
      case 'uploaded': return 'default'
      case 'analyzed': return 'processing'
      case 'reported': return 'success'
      default: return 'default'
    }
  }
  
  const getStatusText = (status: string) => {
    switch (status) {
      case 'uploaded': return '已上传'
      case 'analyzed': return '已分析'
      case 'reported': return '已报告'
      default: return '未知'
    }
  }
  
  const getTimelineIcon = (type: string) => {
    switch (type) {
      case 'visit': return <CalendarOutlined style={{ color: '#1890ff' }} />
      case 'diagnosis': return <MedicineBoxOutlined style={{ color: '#722ed1' }} />
      case 'treatment': return <HeartOutlined style={{ color: '#52c41a' }} />
      case 'image': return <FileImageOutlined style={{ color: '#fa8c16' }} />
      case 'report': return <CheckCircleOutlined style={{ color: '#13c2c2' }} />
      default: return <ClockCircleOutlined />
    }
  }
  
  if (loading || !patient) {
    return (
      <PatientDetailContainer>
        <div style={{ textAlign: 'center', padding: '100px 0' }}>
          <Progress type="circle" />
          <div style={{ marginTop: 16 }}>加载患者信息...</div>
        </div>
      </PatientDetailContainer>
    )
  }
  
  return (
    <PatientDetailContainer>
      <div className="page-header">
        <Button 
          className="back-button" 
          icon={<ArrowLeftOutlined />} 
          onClick={() => navigate('/patients')}
        >
          返回
        </Button>
        <h1>患者详情</h1>
      </div>
      
      {/* 患者基本信息 */}
      <Card className="patient-header">
        <div className="patient-avatar">
          <Avatar 
            size={80} 
            icon={patient.gender === 'male' ? <ManOutlined /> : <WomanOutlined />}
            style={{ backgroundColor: patient.gender === 'male' ? '#1890ff' : '#eb2f96' }}
          />
          <div className="patient-info">
            <div className="patient-name">{patient.name}</div>
            <div className="patient-id">患者ID: {patient.patientId}</div>
            <div className="patient-tags">
              <Tag color={patient.status === 'active' ? 'green' : 'default'}>
                {patient.status === 'active' ? '活跃' : '非活跃'}
              </Tag>
              {patient.diagnosis && <Tag color="blue">{patient.diagnosis}</Tag>}
              <Tag>{patient.gender === 'male' ? '男' : '女'} • {patient.age}岁</Tag>
            </div>
          </div>
          <div style={{ marginLeft: 'auto' }}>
            <Button type="primary" icon={<EditOutlined />} onClick={handleEditPatient}>
              编辑信息
            </Button>
          </div>
        </div>
        
        <Row gutter={[16, 16]}>
          <Col span={8}>
            <Descriptions title="基本信息" column={1} size="small">
              <Descriptions.Item label="出生日期">
                {dayjs(patient.birthDate).format('YYYY年MM月DD日')}
              </Descriptions.Item>
              <Descriptions.Item label="联系电话">
                <PhoneOutlined /> {patient.phone}
              </Descriptions.Item>
              <Descriptions.Item label="地址">
                <EnvironmentOutlined /> {patient.address}
              </Descriptions.Item>
            </Descriptions>
          </Col>
          <Col span={8}>
            <Descriptions title="医疗信息" column={1} size="small">
              <Descriptions.Item label="主治医生">{patient.doctor}</Descriptions.Item>
              <Descriptions.Item label="病史">
                {patient.medicalHistory?.join(', ') || '无'}
              </Descriptions.Item>
              <Descriptions.Item label="过敏史">
                {patient.allergies?.join(', ') || '无'}
              </Descriptions.Item>
            </Descriptions>
          </Col>
          <Col span={8}>
            <Descriptions title="紧急联系人" column={1} size="small">
              <Descriptions.Item label="姓名">{patient.emergencyContact?.name}</Descriptions.Item>
              <Descriptions.Item label="电话">{patient.emergencyContact?.phone}</Descriptions.Item>
              <Descriptions.Item label="关系">{patient.emergencyContact?.relationship}</Descriptions.Item>
            </Descriptions>
          </Col>
        </Row>
      </Card>
      
      {/* 详细信息标签页 */}
      <Card>
        <Tabs defaultActiveKey="images">
          <TabPane tab={`影像资料 (${images.length})`} key="images">
            <div className="image-grid">
              {images.map(image => (
                <div key={image.id} className="image-card" onClick={() => handleViewImage(image)}>
                  <div className="image-preview">
                    <Image
                      src={image.thumbnail}
                      alt={image.name}
                      preview={false}
                      fallback="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAMIAAADDCAYAAADQvc6UAAABRWlDQ1BJQ0MgUHJvZmlsZQAAKJFjYGASSSwoyGFhYGDIzSspCnJ3UoiIjFJgf8LAwSDCIMogwMCcmFxc4BgQ4ANUwgCjUcG3awyMIPqyLsis7PPOq3QdDFcvjV3jOD1boQVTPQrgSkktTgbSf4A4LbmgqISBgTEFyFYuLykAsTuAbJEioKOA7DkgdjqEvQHEToKwj4DVhAQ5A9k3gGyB5IxEoBmML4BsnSQk8XQkNtReEOBxcfXxUQg1Mjc0dyHgXNJBSWpFCYh2zi+oLMpMzyhRcASGUqqCZ16yno6CkYGRAQMDKMwhqj/fAIcloxgHQqxAjIHBEugw5sUIsSQpBobtQPdLciLEVJYzMPBHMDBsayhILEqEO4DxG0txmrERhM29nYGBddr//5/DGRjYNRkY/l7////39v///y4Dmn+LgeHANwDrkl1AuO+pmgAAADhlWElmTU0AKgAAAAgAAYdpAAQAAAABAAAAGgAAAAAAAqACAAQAAAABAAAAwqADAAQAAAABAAAAwwAAAAD9b/HnAAAHlklEQVR4Ae3dP3Ik1RnG4W+FgYxN"
                    />
                  </div>
                  <div className="image-info">
                    <div className="image-title">{image.name}</div>
                    <div className="image-meta">
                      <span>{image.type}</span>
                      <Tag color={getStatusColor(image.status)} size="small">
                        {getStatusText(image.status)}
                      </Tag>
                    </div>
                    <div className="image-meta">
                      <span>{dayjs(image.uploadTime).format('MM-DD HH:mm')}</span>
                      <span>{(image.size / 1024 / 1024).toFixed(1)}MB</span>
                    </div>
                    {image.analysisResults && (
                      <div style={{ marginTop: 8, fontSize: '12px' }}>
                        <div style={{ color: image.analysisResults.abnormal ? '#ff4d4f' : '#52c41a' }}>
                          {image.analysisResults.findings}
                        </div>
                        <div style={{ color: '#8c8c8c' }}>
                          置信度: {(image.analysisResults.confidence * 100).toFixed(1)}%
                        </div>
                      </div>
                    )}
                  </div>
                </div>
              ))}
            </div>
          </TabPane>
          
          <TabPane tab={`医疗记录 (${records.length})`} key="records">
            <Timeline>
              {records.map(record => (
                <Timeline.Item key={record.id} dot={getTimelineIcon(record.type)}>
                  <div className="timeline-item">
                    <div className="timeline-content">
                      <div className="timeline-title">{record.title}</div>
                      <div className="timeline-description">{record.description}</div>
                      <div style={{ marginTop: 8, fontSize: '12px', color: '#8c8c8c' }}>
                        {dayjs(record.date).format('YYYY年MM月DD日 HH:mm')} • {record.doctor}
                      </div>
                    </div>
                  </div>
                </Timeline.Item>
              ))}
            </Timeline>
          </TabPane>
          
          <TabPane tab="统计分析" key="statistics">
            <Row gutter={[16, 16]}>
              <Col span={8}>
                <Card>
                  <div style={{ textAlign: 'center' }}>
                    <div style={{ fontSize: '24px', fontWeight: 600, color: '#1890ff' }}>
                      {images.length}
                    </div>
                    <div style={{ color: '#8c8c8c' }}>总影像数</div>
                  </div>
                </Card>
              </Col>
              <Col span={8}>
                <Card>
                  <div style={{ textAlign: 'center' }}>
                    <div style={{ fontSize: '24px', fontWeight: 600, color: '#52c41a' }}>
                      {images.filter(img => img.status === 'analyzed').length}
                    </div>
                    <div style={{ color: '#8c8c8c' }}>已分析</div>
                  </div>
                </Card>
              </Col>
              <Col span={8}>
                <Card>
                  <div style={{ textAlign: 'center' }}>
                    <div style={{ fontSize: '24px', fontWeight: 600, color: '#722ed1' }}>
                      {images.filter(img => img.analysisResults?.abnormal).length}
                    </div>
                    <div style={{ color: '#8c8c8c' }}>异常发现</div>
                  </div>
                </Card>
              </Col>
            </Row>
            
            <Divider />
            
            <Row gutter={[16, 16]}>
              <Col span={12}>
                <Card title="检查类型分布">
                  {/* 这里可以添加图表组件 */}
                  <div style={{ textAlign: 'center', padding: '40px 0', color: '#8c8c8c' }}>
                    图表组件待实现
                  </div>
                </Card>
              </Col>
              <Col span={12}>
                <Card title="时间趋势">
                  {/* 这里可以添加图表组件 */}
                  <div style={{ textAlign: 'center', padding: '40px 0', color: '#8c8c8c' }}>
                    图表组件待实现
                  </div>
                </Card>
              </Col>
            </Row>
          </TabPane>
        </Tabs>
      </Card>
      
      {/* 编辑患者信息弹窗 */}
      <Modal
        title="编辑患者信息"
        open={editModalVisible}
        onCancel={() => setEditModalVisible(false)}
        onOk={() => form.submit()}
        width={800}
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
          
          <Form.Item
            label="病史"
            name="medicalHistory"
          >
            <Select mode="tags" placeholder="输入病史信息">
              <Option value="高血压">高血压</Option>
              <Option value="糖尿病">糖尿病</Option>
              <Option value="心脏病">心脏病</Option>
            </Select>
          </Form.Item>
          
          <Form.Item
            label="过敏史"
            name="allergies"
          >
            <Select mode="tags" placeholder="输入过敏信息">
              <Option value="青霉素">青霉素</Option>
              <Option value="海鲜">海鲜</Option>
              <Option value="花粉">花粉</Option>
            </Select>
          </Form.Item>
          
          <Divider>紧急联系人</Divider>
          
          <Row gutter={16}>
            <Col span={8}>
              <Form.Item
                label="姓名"
                name="emergencyContactName"
              >
                <Input />
              </Form.Item>
            </Col>
            <Col span={8}>
              <Form.Item
                label="电话"
                name="emergencyContactPhone"
              >
                <Input />
              </Form.Item>
            </Col>
            <Col span={8}>
              <Form.Item
                label="关系"
                name="emergencyContactRelationship"
              >
                <Select>
                  <Option value="配偶">配偶</Option>
                  <Option value="子女">子女</Option>
                  <Option value="父母">父母</Option>
                  <Option value="兄弟姐妹">兄弟姐妹</Option>
                  <Option value="朋友">朋友</Option>
                </Select>
              </Form.Item>
            </Col>
          </Row>
        </Form>
      </Modal>
    </PatientDetailContainer>
  )
}

export default PatientDetailPage