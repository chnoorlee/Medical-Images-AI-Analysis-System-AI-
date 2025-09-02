import React, { useState, useEffect, useRef } from 'react'
import { Row, Col, Card, Button, Upload, Select, Tabs, Table, Tag, Modal, Form, Input, Space, Divider, Progress, Alert, Tooltip, Drawer } from 'antd'
import {
  UploadOutlined,
  PlayCircleOutlined,
  PauseCircleOutlined,
  EyeOutlined,
  DownloadOutlined,
  SettingOutlined,
  FullscreenOutlined,
  ZoomInOutlined,
  ZoomOutOutlined,
  RotateLeftOutlined,
  RotateRightOutlined,
  ReloadOutlined,
  SaveOutlined,
  PrinterOutlined,
  ShareAltOutlined
} from '@ant-design/icons'
import { UploadFile } from 'antd/es/upload/interface'
import styled from 'styled-components'
import { useSelector } from 'react-redux'
import { RootState } from '../../store'

const { Option } = Select
const { TabPane } = Tabs
const { TextArea } = Input

const WorkstationContainer = styled.div`
  height: calc(100vh - 112px);
  display: flex;
  flex-direction: column;
  
  .workstation-header {
    margin-bottom: 16px;
    
    h1 {
      margin: 0;
      color: #262626;
      font-size: 24px;
      font-weight: 600;
    }
  }
  
  .workstation-content {
    flex: 1;
    display: flex;
    gap: 16px;
    min-height: 0;
  }
  
  .left-panel {
    width: 300px;
    display: flex;
    flex-direction: column;
    gap: 16px;
  }
  
  .center-panel {
    flex: 1;
    display: flex;
    flex-direction: column;
    min-width: 0;
  }
  
  .right-panel {
    width: 350px;
    display: flex;
    flex-direction: column;
    gap: 16px;
  }
  
  .image-viewer {
    flex: 1;
    background: #000;
    border-radius: 8px;
    position: relative;
    display: flex;
    align-items: center;
    justify-content: center;
    min-height: 400px;
    
    .viewer-controls {
      position: absolute;
      top: 16px;
      right: 16px;
      z-index: 10;
    }
    
    .viewer-info {
      position: absolute;
      bottom: 16px;
      left: 16px;
      color: white;
      background: rgba(0, 0, 0, 0.6);
      padding: 8px 12px;
      border-radius: 4px;
      font-size: 12px;
    }
    
    .no-image {
      color: #8c8c8c;
      text-align: center;
      
      .upload-hint {
        margin-top: 16px;
        font-size: 14px;
      }
    }
  }
  
  .analysis-panel {
    .ant-card-body {
      padding: 16px;
    }
  }
  
  .result-item {
    padding: 12px;
    border: 1px solid #f0f0f0;
    border-radius: 6px;
    margin-bottom: 8px;
    
    &.abnormal {
      border-color: #ff4d4f;
      background: #fff2f0;
    }
    
    &.normal {
      border-color: #52c41a;
      background: #f6ffed;
    }
    
    .result-header {
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-bottom: 8px;
      
      .result-title {
        font-weight: 500;
      }
      
      .confidence {
        font-size: 12px;
        color: #8c8c8c;
      }
    }
    
    .result-description {
      font-size: 14px;
      color: #595959;
      line-height: 1.4;
    }
  }
`

interface ImageFile {
  id: string
  name: string
  url: string
  size: number
  uploadTime: string
  patientId?: string
  studyType: string
}

interface AnalysisResult {
  id: string
  type: string
  title: string
  description: string
  confidence: number
  severity: 'normal' | 'abnormal' | 'critical'
  coordinates?: { x: number; y: number; width: number; height: number }
}

interface AnalysisTask {
  id: string
  imageId: string
  modelId: string
  status: 'pending' | 'running' | 'completed' | 'failed'
  progress: number
  results?: AnalysisResult[]
  startTime: string
  endTime?: string
}

const WorkstationPage: React.FC = () => {
  const { user } = useSelector((state: RootState) => state.auth)
  const [selectedImage, setSelectedImage] = useState<ImageFile | null>(null)
  const [imageList, setImageList] = useState<ImageFile[]>([])
  const [analysisTask, setAnalysisTask] = useState<AnalysisTask | null>(null)
  const [selectedModel, setSelectedModel] = useState<string>('lung-nodule-detection')
  const [viewerSettings, setViewerSettings] = useState({
    zoom: 100,
    rotation: 0,
    brightness: 50,
    contrast: 50
  })
  const [reportForm] = Form.useForm()
  const [reportVisible, setReportVisible] = useState(false)
  const [settingsVisible, setSettingsVisible] = useState(false)
  
  const viewerRef = useRef<HTMLDivElement>(null)
  
  // Mock data
  const availableModels = [
    { id: 'lung-nodule-detection', name: '肺结节检测', description: '检测肺部结节和肿块' },
    { id: 'brain-tumor-detection', name: '脑肿瘤检测', description: '检测脑部肿瘤和异常' },
    { id: 'bone-fracture-detection', name: '骨折检测', description: '检测骨折和骨损伤' },
    { id: 'cardiac-analysis', name: '心脏分析', description: '心脏功能和结构分析' }
  ]
  
  const mockResults: AnalysisResult[] = [
    {
      id: '1',
      type: 'nodule',
      title: '肺结节',
      description: '右上肺发现直径约8mm的结节，边界清晰，密度均匀。建议进一步随访观察。',
      confidence: 0.92,
      severity: 'abnormal',
      coordinates: { x: 120, y: 80, width: 20, height: 20 }
    },
    {
      id: '2',
      type: 'nodule',
      title: '肺结节',
      description: '左下肺发现直径约5mm的小结节，边界模糊，需要密切关注。',
      confidence: 0.78,
      severity: 'abnormal',
      coordinates: { x: 200, y: 150, width: 15, height: 15 }
    },
    {
      id: '3',
      type: 'normal',
      title: '正常结构',
      description: '心脏轮廓正常，纵隔无异常，胸膜无积液。',
      confidence: 0.95,
      severity: 'normal'
    }
  ]
  
  const handleImageUpload = (file: UploadFile) => {
    const newImage: ImageFile = {
      id: Date.now().toString(),
      name: file.name,
      url: URL.createObjectURL(file as any),
      size: file.size || 0,
      uploadTime: new Date().toISOString(),
      studyType: 'CT'
    }
    
    setImageList(prev => [newImage, ...prev])
    setSelectedImage(newImage)
    return false // Prevent default upload
  }
  
  const handleStartAnalysis = () => {
    if (!selectedImage) return
    
    const task: AnalysisTask = {
      id: Date.now().toString(),
      imageId: selectedImage.id,
      modelId: selectedModel,
      status: 'running',
      progress: 0,
      startTime: new Date().toISOString()
    }
    
    setAnalysisTask(task)
    
    // Simulate analysis progress
    const interval = setInterval(() => {
      setAnalysisTask(prev => {
        if (!prev) return null
        
        const newProgress = Math.min(prev.progress + 10, 100)
        
        if (newProgress === 100) {
          clearInterval(interval)
          return {
            ...prev,
            status: 'completed',
            progress: 100,
            results: mockResults,
            endTime: new Date().toISOString()
          }
        }
        
        return {
          ...prev,
          progress: newProgress
        }
      })
    }, 500)
  }
  
  const handleViewerControl = (action: string) => {
    switch (action) {
      case 'zoomIn':
        setViewerSettings(prev => ({ ...prev, zoom: Math.min(prev.zoom + 25, 400) }))
        break
      case 'zoomOut':
        setViewerSettings(prev => ({ ...prev, zoom: Math.max(prev.zoom - 25, 25) }))
        break
      case 'rotateLeft':
        setViewerSettings(prev => ({ ...prev, rotation: prev.rotation - 90 }))
        break
      case 'rotateRight':
        setViewerSettings(prev => ({ ...prev, rotation: prev.rotation + 90 }))
        break
      case 'reset':
        setViewerSettings({ zoom: 100, rotation: 0, brightness: 50, contrast: 50 })
        break
    }
  }
  
  const handleGenerateReport = () => {
    setReportVisible(true)
    reportForm.setFieldsValue({
      patientName: '张三',
      studyDate: new Date().toISOString().split('T')[0],
      studyType: 'CT胸部平扫',
      findings: analysisTask?.results?.map(r => r.description).join('\n\n') || '',
      impression: '右上肺结节，建议进一步随访观察。',
      recommendation: '建议3个月后复查CT，必要时行增强扫描。'
    })
  }
  
  return (
    <WorkstationContainer>
      <div className="workstation-header">
        <h1>医生工作台</h1>
      </div>
      
      <div className="workstation-content">
        {/* 左侧面板 - 图像列表和上传 */}
        <div className="left-panel">
          <Card title="影像上传" size="small">
            <Upload
              accept=".dcm,.jpg,.jpeg,.png"
              beforeUpload={handleImageUpload}
              showUploadList={false}
            >
              <Button icon={<UploadOutlined />} block>
                选择影像文件
              </Button>
            </Upload>
            <div style={{ marginTop: 8, fontSize: '12px', color: '#8c8c8c' }}>
              支持 DICOM、JPG、PNG 格式
            </div>
          </Card>
          
          <Card title="影像列表" size="small" style={{ flex: 1 }}>
            <div style={{ maxHeight: '300px', overflowY: 'auto' }}>
              {imageList.map(image => (
                <div
                  key={image.id}
                  style={{
                    padding: '8px',
                    border: selectedImage?.id === image.id ? '2px solid #1890ff' : '1px solid #f0f0f0',
                    borderRadius: '4px',
                    marginBottom: '8px',
                    cursor: 'pointer'
                  }}
                  onClick={() => setSelectedImage(image)}
                >
                  <div style={{ fontWeight: 500, fontSize: '14px' }}>{image.name}</div>
                  <div style={{ fontSize: '12px', color: '#8c8c8c' }}>
                    {image.studyType} • {(image.size / 1024 / 1024).toFixed(1)}MB
                  </div>
                </div>
              ))}
            </div>
          </Card>
          
          <Card title="AI分析" size="small">
            <Space direction="vertical" style={{ width: '100%' }}>
              <div>
                <div style={{ marginBottom: 8, fontSize: '14px', fontWeight: 500 }}>选择模型</div>
                <Select
                  value={selectedModel}
                  onChange={setSelectedModel}
                  style={{ width: '100%' }}
                  size="small"
                >
                  {availableModels.map(model => (
                    <Option key={model.id} value={model.id}>
                      {model.name}
                    </Option>
                  ))}
                </Select>
              </div>
              
              <Button
                type="primary"
                icon={analysisTask?.status === 'running' ? <PauseCircleOutlined /> : <PlayCircleOutlined />}
                onClick={handleStartAnalysis}
                disabled={!selectedImage || analysisTask?.status === 'running'}
                block
              >
                {analysisTask?.status === 'running' ? '分析中...' : '开始分析'}
              </Button>
              
              {analysisTask?.status === 'running' && (
                <Progress percent={analysisTask.progress} size="small" />
              )}
            </Space>
          </Card>
        </div>
        
        {/* 中间面板 - 图像查看器 */}
        <div className="center-panel">
          <Card title="影像查看器" size="small" style={{ flex: 1 }}>
            <div className="image-viewer" ref={viewerRef}>
              {selectedImage ? (
                <>
                  <img
                    src={selectedImage.url}
                    alt={selectedImage.name}
                    style={{
                      maxWidth: '100%',
                      maxHeight: '100%',
                      transform: `scale(${viewerSettings.zoom / 100}) rotate(${viewerSettings.rotation}deg)`,
                      filter: `brightness(${viewerSettings.brightness}%) contrast(${viewerSettings.contrast}%)`
                    }}
                  />
                  
                  {/* 分析结果标注 */}
                  {analysisTask?.results?.map(result => (
                    result.coordinates && (
                      <div
                        key={result.id}
                        style={{
                          position: 'absolute',
                          left: result.coordinates.x,
                          top: result.coordinates.y,
                          width: result.coordinates.width,
                          height: result.coordinates.height,
                          border: `2px solid ${result.severity === 'abnormal' ? '#ff4d4f' : '#52c41a'}`,
                          borderRadius: '4px',
                          backgroundColor: `${result.severity === 'abnormal' ? '#ff4d4f' : '#52c41a'}20`
                        }}
                      />
                    )
                  ))}
                  
                  <div className="viewer-controls">
                    <Space>
                      <Tooltip title="放大">
                        <Button size="small" icon={<ZoomInOutlined />} onClick={() => handleViewerControl('zoomIn')} />
                      </Tooltip>
                      <Tooltip title="缩小">
                        <Button size="small" icon={<ZoomOutOutlined />} onClick={() => handleViewerControl('zoomOut')} />
                      </Tooltip>
                      <Tooltip title="左转">
                        <Button size="small" icon={<RotateLeftOutlined />} onClick={() => handleViewerControl('rotateLeft')} />
                      </Tooltip>
                      <Tooltip title="右转">
                        <Button size="small" icon={<RotateRightOutlined />} onClick={() => handleViewerControl('rotateRight')} />
                      </Tooltip>
                      <Tooltip title="重置">
                        <Button size="small" icon={<ReloadOutlined />} onClick={() => handleViewerControl('reset')} />
                      </Tooltip>
                      <Tooltip title="全屏">
                        <Button size="small" icon={<FullscreenOutlined />} />
                      </Tooltip>
                      <Tooltip title="设置">
                        <Button size="small" icon={<SettingOutlined />} onClick={() => setSettingsVisible(true)} />
                      </Tooltip>
                    </Space>
                  </div>
                  
                  <div className="viewer-info">
                    缩放: {viewerSettings.zoom}% | 旋转: {viewerSettings.rotation}°
                  </div>
                </>
              ) : (
                <div className="no-image">
                  <EyeOutlined style={{ fontSize: '48px' }} />
                  <div className="upload-hint">请选择或上传影像文件</div>
                </div>
              )}
            </div>
          </Card>
        </div>
        
        {/* 右侧面板 - 分析结果和报告 */}
        <div className="right-panel">
          <Card title="分析结果" size="small" className="analysis-panel" style={{ flex: 1 }}>
            {analysisTask?.status === 'completed' && analysisTask.results ? (
              <div>
                {analysisTask.results.map(result => (
                  <div key={result.id} className={`result-item ${result.severity}`}>
                    <div className="result-header">
                      <span className="result-title">{result.title}</span>
                      <span className="confidence">置信度: {(result.confidence * 100).toFixed(1)}%</span>
                    </div>
                    <div className="result-description">{result.description}</div>
                  </div>
                ))}
                
                <Divider />
                
                <Space style={{ width: '100%' }} direction="vertical">
                  <Button type="primary" icon={<SaveOutlined />} onClick={handleGenerateReport} block>
                    生成报告
                  </Button>
                  <Button icon={<DownloadOutlined />} block>
                    导出结果
                  </Button>
                  <Button icon={<ShareAltOutlined />} block>
                    分享结果
                  </Button>
                </Space>
              </div>
            ) : analysisTask?.status === 'running' ? (
              <div style={{ textAlign: 'center', padding: '40px 0' }}>
                <Progress type="circle" percent={analysisTask.progress} />
                <div style={{ marginTop: 16, color: '#8c8c8c' }}>AI正在分析影像...</div>
              </div>
            ) : (
              <div style={{ textAlign: 'center', padding: '40px 0', color: '#8c8c8c' }}>
                <ExperimentOutlined style={{ fontSize: '48px', marginBottom: 16 }} />
                <div>请上传影像并开始AI分析</div>
              </div>
            )}
          </Card>
        </div>
      </div>
      
      {/* 报告生成弹窗 */}
      <Modal
        title="生成影像报告"
        open={reportVisible}
        onCancel={() => setReportVisible(false)}
        width={800}
        footer={[
          <Button key="cancel" onClick={() => setReportVisible(false)}>
            取消
          </Button>,
          <Button key="save" type="default" icon={<SaveOutlined />}>
            保存草稿
          </Button>,
          <Button key="print" type="default" icon={<PrinterOutlined />}>
            打印
          </Button>,
          <Button key="submit" type="primary">
            提交报告
          </Button>
        ]}
      >
        <Form form={reportForm} layout="vertical">
          <Row gutter={16}>
            <Col span={12}>
              <Form.Item label="患者姓名" name="patientName">
                <Input />
              </Form.Item>
            </Col>
            <Col span={12}>
              <Form.Item label="检查日期" name="studyDate">
                <Input type="date" />
              </Form.Item>
            </Col>
          </Row>
          
          <Form.Item label="检查类型" name="studyType">
            <Input />
          </Form.Item>
          
          <Form.Item label="影像所见" name="findings">
            <TextArea rows={6} placeholder="描述影像中观察到的异常和正常结构..." />
          </Form.Item>
          
          <Form.Item label="影像印象" name="impression">
            <TextArea rows={3} placeholder="总结主要发现和诊断印象..." />
          </Form.Item>
          
          <Form.Item label="建议" name="recommendation">
            <TextArea rows={3} placeholder="提供进一步检查或治疗建议..." />
          </Form.Item>
        </Form>
      </Modal>
      
      {/* 查看器设置抽屉 */}
      <Drawer
        title="查看器设置"
        placement="right"
        onClose={() => setSettingsVisible(false)}
        open={settingsVisible}
        width={300}
      >
        <Space direction="vertical" style={{ width: '100%' }}>
          <div>
            <div style={{ marginBottom: 8 }}>亮度: {viewerSettings.brightness}%</div>
            <input
              type="range"
              min="0"
              max="200"
              value={viewerSettings.brightness}
              onChange={(e) => setViewerSettings(prev => ({ ...prev, brightness: Number(e.target.value) }))}
              style={{ width: '100%' }}
            />
          </div>
          
          <div>
            <div style={{ marginBottom: 8 }}>对比度: {viewerSettings.contrast}%</div>
            <input
              type="range"
              min="0"
              max="200"
              value={viewerSettings.contrast}
              onChange={(e) => setViewerSettings(prev => ({ ...prev, contrast: Number(e.target.value) }))}
              style={{ width: '100%' }}
            />
          </div>
          
          <Button onClick={() => handleViewerControl('reset')} block>
            重置所有设置
          </Button>
        </Space>
      </Drawer>
    </WorkstationContainer>
  )
}

export default WorkstationPage