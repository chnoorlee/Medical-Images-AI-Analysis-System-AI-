import React, { useState, useEffect, useRef } from 'react'
import { useParams, useNavigate } from 'react-router-dom'
import { Card, Row, Col, Button, Slider, Select, Tabs, List, Tag, Progress, Modal, Form, Input, message, Tooltip, Space, Divider, Typography, Image } from 'antd'
import {
  ArrowLeftOutlined,
  ZoomInOutlined,
  ZoomOutOutlined,
  RotateLeftOutlined,
  RotateRightOutlined,
  ReloadOutlined,
  FullscreenOutlined,
  DownloadOutlined,
  PrinterOutlined,
  SettingOutlined,
  PlayCircleOutlined,
  PauseCircleOutlined,
  EyeOutlined,
  BulbOutlined,
  ContainerOutlined,
  FileImageOutlined,
  AlertOutlined,
  CheckCircleOutlined,
  ClockCircleOutlined,
  InfoCircleOutlined
} from '@ant-design/icons'
import styled from 'styled-components'
import dayjs from 'dayjs'

const { TabPane } = Tabs
const { TextArea } = Input
const { Option } = Select
const { Title, Text } = Typography

const ImageViewerContainer = styled.div`
  height: 100vh;
  display: flex;
  flex-direction: column;
  
  .viewer-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 16px 24px;
    border-bottom: 1px solid #f0f0f0;
    background: #fff;
    
    .header-left {
      display: flex;
      align-items: center;
      gap: 16px;
      
      h1 {
        margin: 0;
        font-size: 18px;
        font-weight: 600;
      }
    }
    
    .header-right {
      display: flex;
      align-items: center;
      gap: 8px;
    }
  }
  
  .viewer-content {
    flex: 1;
    display: flex;
    overflow: hidden;
    
    .viewer-main {
      flex: 1;
      display: flex;
      flex-direction: column;
      
      .viewer-toolbar {
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 12px 16px;
        border-bottom: 1px solid #f0f0f0;
        background: #fafafa;
        
        .toolbar-left {
          display: flex;
          align-items: center;
          gap: 12px;
        }
        
        .toolbar-right {
          display: flex;
          align-items: center;
          gap: 8px;
        }
      }
      
      .viewer-canvas {
        flex: 1;
        position: relative;
        background: #000;
        overflow: hidden;
        
        .image-container {
          width: 100%;
          height: 100%;
          display: flex;
          align-items: center;
          justify-content: center;
          position: relative;
          
          .medical-image {
            max-width: 100%;
            max-height: 100%;
            object-fit: contain;
            transition: transform 0.3s;
          }
          
          .annotations {
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            pointer-events: none;
            
            .annotation {
              position: absolute;
              border: 2px solid #ff4d4f;
              background: rgba(255, 77, 79, 0.1);
              pointer-events: auto;
              cursor: pointer;
              
              &.selected {
                border-color: #1890ff;
                background: rgba(24, 144, 255, 0.1);
              }
              
              .annotation-label {
                position: absolute;
                top: -24px;
                left: 0;
                background: #ff4d4f;
                color: white;
                padding: 2px 6px;
                font-size: 12px;
                border-radius: 2px;
                white-space: nowrap;
              }
            }
          }
          
          .image-info {
            position: absolute;
            top: 16px;
            left: 16px;
            background: rgba(0, 0, 0, 0.7);
            color: white;
            padding: 8px 12px;
            border-radius: 4px;
            font-size: 12px;
            
            .info-item {
              margin-bottom: 4px;
              
              &:last-child {
                margin-bottom: 0;
              }
            }
          }
        }
      }
    }
    
    .viewer-sidebar {
      width: 350px;
      border-left: 1px solid #f0f0f0;
      background: #fff;
      display: flex;
      flex-direction: column;
      
      .sidebar-content {
        flex: 1;
        overflow-y: auto;
      }
    }
  }
  
  .analysis-item {
    padding: 12px;
    border-bottom: 1px solid #f0f0f0;
    
    &:last-child {
      border-bottom: none;
    }
    
    .analysis-header {
      display: flex;
      align-items: center;
      justify-content: space-between;
      margin-bottom: 8px;
      
      .analysis-title {
        font-weight: 500;
        color: #262626;
      }
      
      .confidence-score {
        font-size: 12px;
        color: #8c8c8c;
      }
    }
    
    .analysis-content {
      color: #595959;
      font-size: 14px;
      line-height: 1.5;
    }
    
    .analysis-tags {
      margin-top: 8px;
      display: flex;
      gap: 4px;
      flex-wrap: wrap;
    }
  }
  
  .series-list {
    .series-item {
      display: flex;
      align-items: center;
      padding: 8px 12px;
      cursor: pointer;
      border-bottom: 1px solid #f0f0f0;
      transition: background-color 0.3s;
      
      &:hover {
        background: #f5f5f5;
      }
      
      &.active {
        background: #e6f7ff;
        border-color: #1890ff;
      }
      
      .series-thumbnail {
        width: 40px;
        height: 40px;
        background: #f0f0f0;
        border-radius: 4px;
        margin-right: 12px;
        display: flex;
        align-items: center;
        justify-content: center;
        
        img {
          max-width: 100%;
          max-height: 100%;
          object-fit: cover;
        }
      }
      
      .series-info {
        flex: 1;
        
        .series-name {
          font-weight: 500;
          margin-bottom: 2px;
        }
        
        .series-meta {
          font-size: 12px;
          color: #8c8c8c;
        }
      }
    }
  }
`

interface ImageData {
  id: string
  name: string
  patientId: string
  patientName: string
  studyDate: string
  modality: string
  bodyPart: string
  imageUrl: string
  thumbnail: string
  metadata: {
    width: number
    height: number
    sliceThickness?: number
    pixelSpacing?: number[]
    windowCenter?: number
    windowWidth?: number
  }
  analysisResults?: {
    findings: Array<{
      id: string
      type: string
      description: string
      confidence: number
      coordinates: {
        x: number
        y: number
        width: number
        height: number
      }
      severity: 'low' | 'medium' | 'high'
    }>
    summary: string
    recommendations: string[]
  }
  series?: ImageData[]
}

const ImageViewerPage: React.FC = () => {
  const { imageId } = useParams<{ imageId: string }>()
  const navigate = useNavigate()
  const canvasRef = useRef<HTMLDivElement>(null)
  
  const [loading, setLoading] = useState(true)
  const [imageData, setImageData] = useState<ImageData | null>(null)
  const [currentSeries, setCurrentSeries] = useState(0)
  const [zoom, setZoom] = useState(100)
  const [rotation, setRotation] = useState(0)
  const [brightness, setBrightness] = useState(50)
  const [contrast, setContrast] = useState(50)
  const [selectedAnnotation, setSelectedAnnotation] = useState<string | null>(null)
  const [isPlaying, setIsPlaying] = useState(false)
  const [playSpeed, setPlaySpeed] = useState(1)
  const [windowLevel, setWindowLevel] = useState({ center: 40, width: 400 })
  const [measurementMode, setMeasurementMode] = useState(false)
  const [annotationMode, setAnnotationMode] = useState(false)
  
  // Mock data
  const mockImageData: ImageData = {
    id: '1',
    name: '胸部CT-001',
    patientId: 'P001',
    patientName: '张三',
    studyDate: '2024-01-20T14:20:00Z',
    modality: 'CT',
    bodyPart: '胸部',
    imageUrl: '/api/placeholder/800/600',
    thumbnail: '/api/placeholder/200/150',
    metadata: {
      width: 512,
      height: 512,
      sliceThickness: 1.25,
      pixelSpacing: [0.7, 0.7],
      windowCenter: 40,
      windowWidth: 400
    },
    analysisResults: {
      findings: [
        {
          id: '1',
          type: '结节',
          description: '右上肺发现8mm结节，边界清晰，密度均匀',
          confidence: 0.92,
          coordinates: { x: 320, y: 180, width: 40, height: 40 },
          severity: 'medium'
        },
        {
          id: '2',
          type: '钙化',
          description: '左下肺见点状钙化灶',
          confidence: 0.85,
          coordinates: { x: 180, y: 350, width: 20, height: 20 },
          severity: 'low'
        }
      ],
      summary: '胸部CT显示右上肺结节，建议进一步随访观察',
      recommendations: [
        '建议3个月后复查胸部CT',
        '如有症状变化请及时就诊',
        '戒烟，保持健康生活方式'
      ]
    },
    series: [
      {
        id: '1-1',
        name: '轴位图像 1/120',
        patientId: 'P001',
        patientName: '张三',
        studyDate: '2024-01-20T14:20:00Z',
        modality: 'CT',
        bodyPart: '胸部',
        imageUrl: '/api/placeholder/800/600',
        thumbnail: '/api/placeholder/40/40',
        metadata: { width: 512, height: 512 }
      },
      {
        id: '1-2',
        name: '轴位图像 2/120',
        patientId: 'P001',
        patientName: '张三',
        studyDate: '2024-01-20T14:20:00Z',
        modality: 'CT',
        bodyPart: '胸部',
        imageUrl: '/api/placeholder/800/600',
        thumbnail: '/api/placeholder/40/40',
        metadata: { width: 512, height: 512 }
      }
    ]
  }
  
  useEffect(() => {
    loadImageData()
  }, [imageId])
  
  useEffect(() => {
    let interval: NodeJS.Timeout
    if (isPlaying && imageData?.series) {
      interval = setInterval(() => {
        setCurrentSeries(prev => 
          prev >= (imageData.series!.length - 1) ? 0 : prev + 1
        )
      }, 1000 / playSpeed)
    }
    return () => clearInterval(interval)
  }, [isPlaying, playSpeed, imageData?.series])
  
  const loadImageData = async () => {
    setLoading(true)
    // Simulate API call
    setTimeout(() => {
      setImageData(mockImageData)
      setWindowLevel({
        center: mockImageData.metadata.windowCenter || 40,
        width: mockImageData.metadata.windowWidth || 400
      })
      setLoading(false)
    }, 1000)
  }
  
  const handleZoomIn = () => {
    setZoom(prev => Math.min(prev + 25, 500))
  }
  
  const handleZoomOut = () => {
    setZoom(prev => Math.max(prev - 25, 25))
  }
  
  const handleRotateLeft = () => {
    setRotation(prev => prev - 90)
  }
  
  const handleRotateRight = () => {
    setRotation(prev => prev + 90)
  }
  
  const handleReset = () => {
    setZoom(100)
    setRotation(0)
    setBrightness(50)
    setContrast(50)
    setWindowLevel({
      center: imageData?.metadata.windowCenter || 40,
      width: imageData?.metadata.windowWidth || 400
    })
  }
  
  const handleAnnotationClick = (annotationId: string) => {
    setSelectedAnnotation(annotationId)
  }
  
  const handleSeriesChange = (index: number) => {
    setCurrentSeries(index)
  }
  
  const handlePlayToggle = () => {
    setIsPlaying(!isPlaying)
  }
  
  const getImageStyle = () => {
    return {
      transform: `scale(${zoom / 100}) rotate(${rotation}deg)`,
      filter: `brightness(${brightness}%) contrast(${contrast}%)`
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
  
  if (loading || !imageData) {
    return (
      <ImageViewerContainer>
        <div style={{ 
          display: 'flex', 
          alignItems: 'center', 
          justifyContent: 'center', 
          height: '100vh',
          flexDirection: 'column',
          gap: 16
        }}>
          <Progress type="circle" />
          <div>加载影像数据...</div>
        </div>
      </ImageViewerContainer>
    )
  }
  
  const currentImage = imageData.series?.[currentSeries] || imageData
  
  return (
    <ImageViewerContainer>
      {/* 头部工具栏 */}
      <div className="viewer-header">
        <div className="header-left">
          <Button 
            icon={<ArrowLeftOutlined />} 
            onClick={() => navigate(-1)}
          >
            返回
          </Button>
          <h1>{imageData.name}</h1>
          <Tag color="blue">{imageData.modality}</Tag>
          <span style={{ color: '#8c8c8c' }}>
            {imageData.patientName} • {dayjs(imageData.studyDate).format('YYYY-MM-DD HH:mm')}
          </span>
        </div>
        <div className="header-right">
          <Button icon={<DownloadOutlined />}>下载</Button>
          <Button icon={<PrinterOutlined />}>打印</Button>
          <Button icon={<SettingOutlined />}>设置</Button>
        </div>
      </div>
      
      <div className="viewer-content">
        {/* 主要查看区域 */}
        <div className="viewer-main">
          {/* 工具栏 */}
          <div className="viewer-toolbar">
            <div className="toolbar-left">
              <Space>
                <Button icon={<ZoomInOutlined />} onClick={handleZoomIn} />
                <Button icon={<ZoomOutOutlined />} onClick={handleZoomOut} />
                <span style={{ minWidth: 60, textAlign: 'center' }}>{zoom}%</span>
                <Button icon={<RotateLeftOutlined />} onClick={handleRotateLeft} />
                <Button icon={<RotateRightOutlined />} onClick={handleRotateRight} />
                <Button icon={<ReloadOutlined />} onClick={handleReset} />
              </Space>
              
              <Divider type="vertical" />
              
              <Space>
                <span>亮度:</span>
                <Slider 
                  style={{ width: 100 }} 
                  value={brightness} 
                  onChange={setBrightness}
                  min={0}
                  max={100}
                />
                <span>对比度:</span>
                <Slider 
                  style={{ width: 100 }} 
                  value={contrast} 
                  onChange={setContrast}
                  min={0}
                  max={100}
                />
              </Space>
            </div>
            
            <div className="toolbar-right">
              {imageData.series && (
                <Space>
                  <Button 
                    icon={isPlaying ? <PauseCircleOutlined /> : <PlayCircleOutlined />}
                    onClick={handlePlayToggle}
                  />
                  <Select 
                    value={playSpeed} 
                    onChange={setPlaySpeed}
                    style={{ width: 80 }}
                  >
                    <Option value={0.5}>0.5x</Option>
                    <Option value={1}>1x</Option>
                    <Option value={2}>2x</Option>
                    <Option value={4}>4x</Option>
                  </Select>
                  <span>{currentSeries + 1} / {imageData.series.length}</span>
                </Space>
              )}
            </div>
          </div>
          
          {/* 影像显示区域 */}
          <div className="viewer-canvas" ref={canvasRef}>
            <div className="image-container">
              <Image
                src={currentImage.imageUrl}
                alt={currentImage.name}
                className="medical-image"
                style={getImageStyle()}
                preview={false}
                fallback="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAMIAAADDCAYAAADQvc6UAAABRWlDQ1BJQ0MgUHJvZmlsZQAAKJFjYGASSSwoyGFhYGDIzSspCnJ3UoiIjFJgf8LAwSDCIMogwMCcmFxc4BgQ4ANUwgCjUcG3awyMIPqyLsis7PPOq3QdDFcvjV3jOD1boQVTPQrgSkktTgbSf4A4LbmgqISBgTEFyFYuLykAsTuAbJEioKOA7DkgdjqEvQHEToKwj4DVhAQ5A9k3gGyB5IxEoBmML4BsnSQk8XQkNtReEOBxcfXxUQg1Mjc0dyHgXNJBSWpFCYh2zi+oLMpMzyhRcASGUqqCZ16yno6CkYGRAQMDKMwhqj/fAIcloxgHQqxAjIHBEugw5sUIsSQpBobtQPdLciLEVJYzMPBHMDBsayhILEqEO4DxG0txmrERhM29nYGBddr//5/DGRjYNRkY/l7////39v///y4Dmn+LgeHANwDrkl1AuO+pmgAAADhlWElmTU0AKgAAAAgAAYdpAAQAAAABAAAAGgAAAAAAAqACAAQAAAABAAAAwqADAAQAAAABAAAAwwAAAAD9b/HnAAAHlklEQVR4Ae3dP3Ik1RnG4W+FgYxN"
              />
              
              {/* AI分析标注 */}
              {imageData.analysisResults && (
                <div className="annotations">
                  {imageData.analysisResults.findings.map(finding => (
                    <div
                      key={finding.id}
                      className={`annotation ${selectedAnnotation === finding.id ? 'selected' : ''}`}
                      style={{
                        left: `${(finding.coordinates.x / currentImage.metadata.width) * 100}%`,
                        top: `${(finding.coordinates.y / currentImage.metadata.height) * 100}%`,
                        width: `${(finding.coordinates.width / currentImage.metadata.width) * 100}%`,
                        height: `${(finding.coordinates.height / currentImage.metadata.height) * 100}%`
                      }}
                      onClick={() => handleAnnotationClick(finding.id)}
                    >
                      <div className="annotation-label">
                        {finding.type} ({(finding.confidence * 100).toFixed(0)}%)
                      </div>
                    </div>
                  ))}
                </div>
              )}
              
              {/* 影像信息 */}
              <div className="image-info">
                <div className="info-item">患者: {imageData.patientName}</div>
                <div className="info-item">检查: {imageData.modality} - {imageData.bodyPart}</div>
                <div className="info-item">尺寸: {currentImage.metadata.width} × {currentImage.metadata.height}</div>
                {currentImage.metadata.sliceThickness && (
                  <div className="info-item">层厚: {currentImage.metadata.sliceThickness}mm</div>
                )}
                <div className="info-item">窗位/窗宽: {windowLevel.center}/{windowLevel.width}</div>
              </div>
            </div>
          </div>
        </div>
        
        {/* 侧边栏 */}
        <div className="viewer-sidebar">
          <div className="sidebar-content">
            <Tabs defaultActiveKey="analysis">
              <TabPane tab="AI分析" key="analysis">
                {imageData.analysisResults ? (
                  <div>
                    {/* 分析摘要 */}
                    <Card size="small" style={{ margin: '0 12px 12px' }}>
                      <Title level={5}>分析摘要</Title>
                      <Text>{imageData.analysisResults.summary}</Text>
                    </Card>
                    
                    {/* 发现列表 */}
                    <div>
                      {imageData.analysisResults.findings.map(finding => (
                        <div 
                          key={finding.id} 
                          className={`analysis-item ${selectedAnnotation === finding.id ? 'selected' : ''}`}
                          onClick={() => handleAnnotationClick(finding.id)}
                          style={{ cursor: 'pointer' }}
                        >
                          <div className="analysis-header">
                            <span className="analysis-title">{finding.type}</span>
                            <span className="confidence-score">
                              {(finding.confidence * 100).toFixed(1)}%
                            </span>
                          </div>
                          <div className="analysis-content">
                            {finding.description}
                          </div>
                          <div className="analysis-tags">
                            <Tag color={getSeverityColor(finding.severity)}>
                              {getSeverityText(finding.severity)}
                            </Tag>
                            <Tag>置信度: {(finding.confidence * 100).toFixed(1)}%</Tag>
                          </div>
                        </div>
                      ))}
                    </div>
                    
                    {/* 建议 */}
                    <Card size="small" style={{ margin: 12 }}>
                      <Title level={5}>建议</Title>
                      <List
                        size="small"
                        dataSource={imageData.analysisResults.recommendations}
                        renderItem={item => (
                          <List.Item>
                            <Text>• {item}</Text>
                          </List.Item>
                        )}
                      />
                    </Card>
                  </div>
                ) : (
                  <div style={{ textAlign: 'center', padding: '40px 20px', color: '#8c8c8c' }}>
                    <AlertOutlined style={{ fontSize: 24, marginBottom: 8 }} />
                    <div>暂无AI分析结果</div>
                  </div>
                )}
              </TabPane>
              
              <TabPane tab="序列" key="series">
                {imageData.series ? (
                  <div className="series-list">
                    {imageData.series.map((series, index) => (
                      <div 
                        key={series.id}
                        className={`series-item ${index === currentSeries ? 'active' : ''}`}
                        onClick={() => handleSeriesChange(index)}
                      >
                        <div className="series-thumbnail">
                          <Image
                            src={series.thumbnail}
                            alt={series.name}
                            preview={false}
                            fallback="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAMIAAADDCAYAAADQvc6UAAABRWlDQ1BJQ0MgUHJvZmlsZQAAKJFjYGASSSwoyGFhYGDIzSspCnJ3UoiIjFJgf8LAwSDCIMogwMCcmFxc4BgQ4ANUwgCjUcG3awyMIPqyLsis7PPOq3QdDFcvjV3jOD1boQVTPQrgSkktTgbSf4A4LbmgqISBgTEFyFYuLykAsTuAbJEioKOA7DkgdjqEvQHEToKwj4DVhAQ5A9k3gGyB5IxEoBmML4BsnSQk8XQkNtReEOBxcfXxUQg1Mjc0dyHgXNJBSWpFCYh2zi+oLMpMzyhRcASGUqqCZ16yno6CkYGRAQMDKMwhqj/fAIcloxgHQqxAjIHBEugw5sUIsSQpBobtQPdLciLEVJYzMPBHMDBsayhILEqEO4DxG0txmrERhM29nYGBddr//5/DGRjYNRkY/l7////39v///y4Dmn+LgeHANwDrkl1AuO+pmgAAADhlWElmTU0AKgAAAAgAAYdpAAQAAAABAAAAGgAAAAAAAqACAAQAAAABAAAAwqADAAQAAAABAAAAwwAAAAD9b/HnAAAHlklEQVR4Ae3dP3Ik1RnG4W+FgYxN"
                          />
                        </div>
                        <div className="series-info">
                          <div className="series-name">{series.name}</div>
                          <div className="series-meta">
                            {series.metadata.width} × {series.metadata.height}
                          </div>
                        </div>
                      </div>
                    ))}
                  </div>
                ) : (
                  <div style={{ textAlign: 'center', padding: '40px 20px', color: '#8c8c8c' }}>
                    <FileImageOutlined style={{ fontSize: 24, marginBottom: 8 }} />
                    <div>单张影像</div>
                  </div>
                )}
              </TabPane>
              
              <TabPane tab="测量" key="measurement">
                <div style={{ padding: 12 }}>
                  <Button 
                    type={measurementMode ? 'primary' : 'default'}
                    block
                    style={{ marginBottom: 12 }}
                    onClick={() => setMeasurementMode(!measurementMode)}
                  >
                    {measurementMode ? '退出测量' : '开始测量'}
                  </Button>
                  
                  <div style={{ textAlign: 'center', padding: '40px 20px', color: '#8c8c8c' }}>
                    <InfoCircleOutlined style={{ fontSize: 24, marginBottom: 8 }} />
                    <div>测量功能开发中</div>
                  </div>
                </div>
              </TabPane>
              
              <TabPane tab="标注" key="annotation">
                <div style={{ padding: 12 }}>
                  <Button 
                    type={annotationMode ? 'primary' : 'default'}
                    block
                    style={{ marginBottom: 12 }}
                    onClick={() => setAnnotationMode(!annotationMode)}
                  >
                    {annotationMode ? '退出标注' : '开始标注'}
                  </Button>
                  
                  <div style={{ textAlign: 'center', padding: '40px 20px', color: '#8c8c8c' }}>
                    <InfoCircleOutlined style={{ fontSize: 24, marginBottom: 8 }} />
                    <div>标注功能开发中</div>
                  </div>
                </div>
              </TabPane>
            </Tabs>
          </div>
        </div>
      </div>
    </ImageViewerContainer>
  )
}

export default ImageViewerPage