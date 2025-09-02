import React, { useState, useEffect } from 'react'
import { useParams, useNavigate } from 'react-router-dom'
import { Card, Row, Col, Button, Descriptions, Tag, Space, Modal, Form, Input, Select, message, Divider, Typography, Image, Timeline, Progress, Tooltip, List } from 'antd'
import {
  ArrowLeftOutlined,
  EditOutlined,
  PrinterOutlined,
  DownloadOutlined,
  ShareAltOutlined,
  CheckCircleOutlined,
  ClockCircleOutlined,
  ExclamationCircleOutlined,
  FileImageOutlined,
  UserOutlined,
  CalendarOutlined,
  MedicineBoxOutlined,
  AlertOutlined,
  EyeOutlined,
  SaveOutlined,
  CloseOutlined
} from '@ant-design/icons'
import styled from 'styled-components'
import dayjs from 'dayjs'

const { TextArea } = Input
const { Option } = Select
const { Title, Text, Paragraph } = Typography
const { confirm } = Modal

const ReportDetailContainer = styled.div`
  .page-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: 24px;
    
    .header-left {
      display: flex;
      align-items: center;
      gap: 16px;
      
      h1 {
        margin: 0;
        color: #262626;
        font-size: 24px;
        font-weight: 600;
      }
    }
    
    .header-actions {
      display: flex;
      gap: 8px;
    }
  }
  
  .report-status {
    display: flex;
    align-items: center;
    gap: 12px;
    margin-bottom: 24px;
    padding: 16px;
    background: #fafafa;
    border-radius: 8px;
    
    .status-info {
      flex: 1;
      
      .status-title {
        font-weight: 600;
        margin-bottom: 4px;
      }
      
      .status-description {
        color: #8c8c8c;
        font-size: 14px;
      }
    }
  }
  
  .report-content {
    .content-section {
      margin-bottom: 24px;
      
      .section-title {
        font-size: 16px;
        font-weight: 600;
        margin-bottom: 12px;
        color: #262626;
        display: flex;
        align-items: center;
        gap: 8px;
      }
      
      .section-content {
        background: #fafafa;
        padding: 16px;
        border-radius: 8px;
        border-left: 4px solid #1890ff;
        
        &.editable {
          background: #fff;
          border: 1px solid #d9d9d9;
          border-left: 4px solid #1890ff;
        }
      }
    }
  }
  
  .image-gallery {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(150px, 1fr));
    gap: 12px;
    
    .image-item {
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
        height: 120px;
        background: #f5f5f5;
        display: flex;
        align-items: center;
        justify-content: center;
      }
      
      .image-info {
        padding: 8px;
        text-align: center;
        
        .image-name {
          font-size: 12px;
          font-weight: 500;
          margin-bottom: 2px;
        }
        
        .image-meta {
          font-size: 11px;
          color: #8c8c8c;
        }
      }
    }
  }
  
  .findings-list {
    .finding-item {
      padding: 12px;
      border: 1px solid #f0f0f0;
      border-radius: 8px;
      margin-bottom: 12px;
      
      &:last-child {
        margin-bottom: 0;
      }
      
      .finding-header {
        display: flex;
        align-items: center;
        justify-content: space-between;
        margin-bottom: 8px;
        
        .finding-type {
          font-weight: 600;
          color: #262626;
        }
        
        .finding-confidence {
          font-size: 12px;
          color: #8c8c8c;
        }
      }
      
      .finding-description {
        color: #595959;
        margin-bottom: 8px;
      }
      
      .finding-tags {
        display: flex;
        gap: 4px;
      }
    }
  }
  
  .edit-actions {
    position: fixed;
    bottom: 24px;
    right: 24px;
    display: flex;
    gap: 8px;
    z-index: 1000;
  }
`

interface ReportDetail {
  id: string
  reportNumber: string
  patientId: string
  patientName: string
  patientAge: number
  patientGender: 'male' | 'female'
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
  images: Array<{
    id: string
    name: string
    thumbnail: string
    url: string
  }>
  aiFindings: Array<{
    id: string
    type: string
    description: string
    confidence: number
    severity: 'high' | 'medium' | 'low'
    coordinates?: {
      x: number
      y: number
      width: number
      height: number
    }
  }>
  history: Array<{
    id: string
    action: string
    user: string
    timestamp: string
    description: string
  }>
}

const ReportDetailPage: React.FC = () => {
  const { reportId } = useParams<{ reportId: string }>()
  const navigate = useNavigate()
  const [loading, setLoading] = useState(true)
  const [report, setReport] = useState<ReportDetail | null>(null)
  const [editMode, setEditMode] = useState(false)
  const [form] = Form.useForm()
  
  // Mock data
  const mockReport: ReportDetail = {
    id: '1',
    reportNumber: 'RPT-2024-001',
    patientId: 'P001',
    patientName: '张三',
    patientAge: 45,
    patientGender: 'male',
    studyType: '胸部CT',
    studyDate: '2024-01-20T14:20:00Z',
    reportDate: '2024-01-20T16:30:00Z',
    status: 'completed',
    priority: 'high',
    doctor: '李医生',
    reviewer: '王主任',
    findings: '双肺纹理清晰，右上肺发现一枚约8mm的结节影，边界清晰，密度均匀，CT值约为25HU。左下肺见点状钙化灶。纵隔淋巴结未见明显肿大。胸膜未见异常。',
    impression: '1. 右上肺结节，考虑良性可能性大，建议定期随访观察。\n2. 左下肺钙化灶，陈旧性病变。',
    recommendations: [
      '建议3个月后复查胸部CT',
      '如有咳嗽、胸痛等症状变化请及时就诊',
      '戒烟，保持健康生活方式',
      '定期体检'
    ],
    images: [
      {
        id: '1',
        name: '轴位图像-001',
        thumbnail: '/api/placeholder/150/120',
        url: '/api/placeholder/800/600'
      },
      {
        id: '2',
        name: '轴位图像-002',
        thumbnail: '/api/placeholder/150/120',
        url: '/api/placeholder/800/600'
      },
      {
        id: '3',
        name: '冠状位图像-001',
        thumbnail: '/api/placeholder/150/120',
        url: '/api/placeholder/800/600'
      },
      {
        id: '4',
        name: '矢状位图像-001',
        thumbnail: '/api/placeholder/150/120',
        url: '/api/placeholder/800/600'
      }
    ],
    aiFindings: [
      {
        id: '1',
        type: '肺结节',
        description: '右上肺发现8mm结节，边界清晰，密度均匀，良性可能性较大',
        confidence: 0.92,
        severity: 'medium',
        coordinates: { x: 320, y: 180, width: 40, height: 40 }
      },
      {
        id: '2',
        type: '钙化灶',
        description: '左下肺见点状钙化灶，符合陈旧性病变表现',
        confidence: 0.95,
        severity: 'low'
      }
    ],
    history: [
      {
        id: '1',
        action: '报告审核',
        user: '王主任',
        timestamp: '2024-01-20T17:00:00Z',
        description: '报告已审核通过'
      },
      {
        id: '2',
        action: '报告完成',
        user: '李医生',
        timestamp: '2024-01-20T16:30:00Z',
        description: '完成报告编写'
      },
      {
        id: '3',
        action: 'AI分析',
        user: '系统',
        timestamp: '2024-01-20T15:45:00Z',
        description: 'AI分析完成，发现2处异常'
      },
      {
        id: '4',
        action: '影像上传',
        user: '技师小王',
        timestamp: '2024-01-20T14:30:00Z',
        description: '上传胸部CT影像120张'
      }
    ]
  }
  
  useEffect(() => {
    loadReportDetail()
  }, [reportId])
  
  const loadReportDetail = async () => {
    setLoading(true)
    // Simulate API call
    setTimeout(() => {
      setReport(mockReport)
      form.setFieldsValue({
        findings: mockReport.findings,
        impression: mockReport.impression,
        recommendations: mockReport.recommendations
      })
      setLoading(false)
    }, 1000)
  }
  
  const handleEdit = () => {
    setEditMode(true)
  }
  
  const handleSave = async () => {
    try {
      const values = await form.validateFields()
      // Update report
      setReport(prev => prev ? {
        ...prev,
        findings: values.findings,
        impression: values.impression,
        recommendations: values.recommendations
      } : null)
      setEditMode(false)
      message.success('报告已保存')
    } catch (error) {
      message.error('保存失败，请检查输入内容')
    }
  }
  
  const handleCancel = () => {
    form.setFieldsValue({
      findings: report?.findings,
      impression: report?.impression,
      recommendations: report?.recommendations
    })
    setEditMode(false)
  }
  
  const handlePrint = () => {
    window.print()
  }
  
  const handleDownload = () => {
    message.info('下载功能开发中')
  }
  
  const handleShare = () => {
    message.info('分享功能开发中')
  }
  
  const handleViewImage = (image: any) => {
    navigate(`/images/${image.id}`)
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
      case 'high': return '高优先级'
      case 'medium': return '中优先级'
      case 'low': return '低优先级'
      default: return '未知'
    }
  }
  
  const getSeverityColor = (severity: string) => {
    switch (severity) {
      case 'high': return 'red'
      case 'medium': return 'orange'
      case 'low': return 'blue'
      default: return 'default'
    }
  }
  
  const getSeverityText = (severity: string) => {
    switch (severity) {
      case 'high': return '高风险'
      case 'medium': return '中风险'
      case 'low': return '低风险'
      default: return '未知'
    }
  }
  
  const getTimelineIcon = (action: string) => {
    switch (action) {
      case '报告审核': return <CheckCircleOutlined style={{ color: '#52c41a' }} />
      case '报告完成': return <FileImageOutlined style={{ color: '#1890ff' }} />
      case 'AI分析': return <ExclamationCircleOutlined style={{ color: '#fa8c16' }} />
      case '影像上传': return <FileImageOutlined style={{ color: '#722ed1' }} />
      default: return <ClockCircleOutlined />
    }
  }
  
  if (loading || !report) {
    return (
      <ReportDetailContainer>
        <div style={{ 
          display: 'flex', 
          alignItems: 'center', 
          justifyContent: 'center', 
          height: '50vh',
          flexDirection: 'column',
          gap: 16
        }}>
          <Progress type="circle" />
          <div>加载报告详情...</div>
        </div>
      </ReportDetailContainer>
    )
  }
  
  return (
    <ReportDetailContainer>
      <div className="page-header">
        <div className="header-left">
          <Button 
            icon={<ArrowLeftOutlined />} 
            onClick={() => navigate('/reports')}
          >
            返回
          </Button>
          <h1>{report.reportNumber}</h1>
          <Tag color={getStatusColor(report.status)}>
            {getStatusText(report.status)}
          </Tag>
          <Tag color={getPriorityColor(report.priority)}>
            {getPriorityText(report.priority)}
          </Tag>
        </div>
        <div className="header-actions">
          {!editMode && (
            <>
              <Button icon={<EditOutlined />} onClick={handleEdit}>
                编辑
              </Button>
              <Button icon={<PrinterOutlined />} onClick={handlePrint}>
                打印
              </Button>
              <Button icon={<DownloadOutlined />} onClick={handleDownload}>
                下载
              </Button>
              <Button icon={<ShareAltOutlined />} onClick={handleShare}>
                分享
              </Button>
            </>
          )}
        </div>
      </div>
      
      {/* 报告状态 */}
      <div className="report-status">
        <div className="status-info">
          <div className="status-title">报告状态</div>
          <div className="status-description">
            {report.status === 'reviewed' ? '已审核完成' : 
             report.status === 'completed' ? '已完成，等待审核' :
             report.status === 'pending' ? '正在编写中' : '草稿状态'}
          </div>
        </div>
        <div>
          <Progress 
            type="circle" 
            size={60}
            percent={report.status === 'reviewed' ? 100 : 
                    report.status === 'completed' ? 80 :
                    report.status === 'pending' ? 50 : 25}
            strokeColor={report.status === 'reviewed' ? '#52c41a' : '#1890ff'}
          />
        </div>
      </div>
      
      <Row gutter={[24, 24]}>
        <Col span={16}>
          {/* 基本信息 */}
          <Card title="基本信息" style={{ marginBottom: 24 }}>
            <Row gutter={[16, 16]}>
              <Col span={12}>
                <Descriptions column={1} size="small">
                  <Descriptions.Item label="患者姓名">{report.patientName}</Descriptions.Item>
                  <Descriptions.Item label="患者ID">{report.patientId}</Descriptions.Item>
                  <Descriptions.Item label="性别">{report.patientGender === 'male' ? '男' : '女'}</Descriptions.Item>
                  <Descriptions.Item label="年龄">{report.patientAge}岁</Descriptions.Item>
                </Descriptions>
              </Col>
              <Col span={12}>
                <Descriptions column={1} size="small">
                  <Descriptions.Item label="检查类型">{report.studyType}</Descriptions.Item>
                  <Descriptions.Item label="检查日期">
                    {dayjs(report.studyDate).format('YYYY年MM月DD日 HH:mm')}
                  </Descriptions.Item>
                  <Descriptions.Item label="报告日期">
                    {dayjs(report.reportDate).format('YYYY年MM月DD日 HH:mm')}
                  </Descriptions.Item>
                  <Descriptions.Item label="报告医生">{report.doctor}</Descriptions.Item>
                  {report.reviewer && (
                    <Descriptions.Item label="审核医生">{report.reviewer}</Descriptions.Item>
                  )}
                </Descriptions>
              </Col>
            </Row>
          </Card>
          
          {/* 报告内容 */}
          <div className="report-content">
            <Form form={form} layout="vertical">
              {/* 检查所见 */}
              <div className="content-section">
                <div className="section-title">
                  <MedicineBoxOutlined />
                  检查所见
                </div>
                <div className={`section-content ${editMode ? 'editable' : ''}`}>
                  {editMode ? (
                    <Form.Item name="findings" style={{ margin: 0 }}>
                      <TextArea 
                        rows={6} 
                        placeholder="描述检查发现的异常或正常表现"
                      />
                    </Form.Item>
                  ) : (
                    <Paragraph style={{ margin: 0, whiteSpace: 'pre-wrap' }}>
                      {report.findings}
                    </Paragraph>
                  )}
                </div>
              </div>
              
              {/* 诊断印象 */}
              <div className="content-section">
                <div className="section-title">
                  <CheckCircleOutlined />
                  诊断印象
                </div>
                <div className={`section-content ${editMode ? 'editable' : ''}`}>
                  {editMode ? (
                    <Form.Item name="impression" style={{ margin: 0 }}>
                      <TextArea 
                        rows={4} 
                        placeholder="基于检查所见的诊断结论"
                      />
                    </Form.Item>
                  ) : (
                    <Paragraph style={{ margin: 0, whiteSpace: 'pre-wrap' }}>
                      {report.impression}
                    </Paragraph>
                  )}
                </div>
              </div>
              
              {/* 建议 */}
              <div className="content-section">
                <div className="section-title">
                  <AlertOutlined />
                  建议
                </div>
                <div className={`section-content ${editMode ? 'editable' : ''}`}>
                  {editMode ? (
                    <Form.Item name="recommendations" style={{ margin: 0 }}>
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
                  ) : (
                    <List
                      size="small"
                      dataSource={report.recommendations}
                      renderItem={item => (
                        <List.Item>
                          <Text>• {item}</Text>
                        </List.Item>
                      )}
                    />
                  )}
                </div>
              </div>
            </Form>
          </div>
          
          {/* 影像资料 */}
          <Card title="影像资料" style={{ marginBottom: 24 }}>
            <div className="image-gallery">
              {report.images.map(image => (
                <div 
                  key={image.id} 
                  className="image-item"
                  onClick={() => handleViewImage(image)}
                >
                  <div className="image-preview">
                    <Image
                      src={image.thumbnail}
                      alt={image.name}
                      preview={false}
                      fallback="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAMIAAADDCAYAAADQvc6UAAABRWlDQ1BJQ0MgUHJvZmlsZQAAKJFjYGASSSwoyGFhYGDIzSspCnJ3UoiIjFJgf8LAwSDCIMogwMCcmFxc4BgQ4ANUwgCjUcG3awyMIPqyLsis7PPOq3QdDFcvjV3jOD1boQVTPQrgSkktTgbSf4A4LbmgqISBgTEFyFYuLykAsTuAbJEioKOA7DkgdjqEvQHEToKwj4DVhAQ5A9k3gGyB5IxEoBmML4BsnSQk8XQkNtReEOBxcfXxUQg1Mjc0dyHgXNJBSWpFCYh2zi+oLMpMzyhRcASGUqqCZ16yno6CkYGRAQMDKMwhqj/fAIcloxgHQqxAjIHBEugw5sUIsSQpBobtQPdLciLEVJYzMPBHMDBsayhILEqEO4DxG0txmrERhM29nYGBddr//5/DGRjYNRkY/l7////39v///y4Dmn+LgeHANwDrkl1AuO+pmgAAADhlWElmTU0AKgAAAAgAAYdpAAQAAAABAAAAGgAAAAAAAqACAAQAAAABAAAAwqADAAQAAAABAAAAwwAAAAD9b/HnAAAHlklEQVR4Ae3dP3Ik1RnG4W+FgYxN"
                    />
                  </div>
                  <div className="image-info">
                    <div className="image-name">{image.name}</div>
                    <div className="image-meta">
                      <EyeOutlined /> 查看
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </Card>
        </Col>
        
        <Col span={8}>
          {/* AI分析结果 */}
          <Card title="AI分析结果" style={{ marginBottom: 24 }}>
            <div className="findings-list">
              {report.aiFindings.map(finding => (
                <div key={finding.id} className="finding-item">
                  <div className="finding-header">
                    <span className="finding-type">{finding.type}</span>
                    <span className="finding-confidence">
                      {(finding.confidence * 100).toFixed(1)}%
                    </span>
                  </div>
                  <div className="finding-description">
                    {finding.description}
                  </div>
                  <div className="finding-tags">
                    <Tag color={getSeverityColor(finding.severity)}>
                      {getSeverityText(finding.severity)}
                    </Tag>
                    <Tag>置信度: {(finding.confidence * 100).toFixed(1)}%</Tag>
                  </div>
                </div>
              ))}
            </div>
          </Card>
          
          {/* 操作历史 */}
          <Card title="操作历史">
            <Timeline>
              {report.history.map(item => (
                <Timeline.Item key={item.id} dot={getTimelineIcon(item.action)}>
                  <div>
                    <div style={{ fontWeight: 500 }}>{item.action}</div>
                    <div style={{ fontSize: '12px', color: '#8c8c8c', marginBottom: 4 }}>
                      {dayjs(item.timestamp).format('MM-DD HH:mm')} • {item.user}
                    </div>
                    <div style={{ fontSize: '14px', color: '#595959' }}>
                      {item.description}
                    </div>
                  </div>
                </Timeline.Item>
              ))}
            </Timeline>
          </Card>
        </Col>
      </Row>
      
      {/* 编辑模式操作按钮 */}
      {editMode && (
        <div className="edit-actions">
          <Button 
            type="primary" 
            icon={<SaveOutlined />} 
            size="large"
            onClick={handleSave}
          >
            保存
          </Button>
          <Button 
            icon={<CloseOutlined />} 
            size="large"
            onClick={handleCancel}
          >
            取消
          </Button>
        </div>
      )}
    </ReportDetailContainer>
  )
}

export default ReportDetailPage