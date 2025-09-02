import React, { useState, useEffect } from 'react';
import {
  Card,
  Table,
  Button,
  Space,
  Tag,
  Modal,
  Form,
  Input,
  Select,
  Upload,
  Progress,
  Statistic,
  Row,
  Col,
  Tabs,
  message,
  Tooltip,
  Divider,
  Switch,
  Tree,
  Checkbox,
  Radio,
  DatePicker,
  Alert,
  Badge,
  Dropdown,
  Menu
} from 'antd';
import {
  UploadOutlined,
  DownloadOutlined,
  EyeOutlined,
  EditOutlined,
  DeleteOutlined,
  PlusOutlined,
  SearchOutlined,
  FolderOutlined,
  FileImageOutlined,
  TagOutlined,
  ExportOutlined,
  ImportOutlined,
  SyncOutlined,
  SettingOutlined,
  BarChartOutlined,
  CheckCircleOutlined,
  CloseCircleOutlined,
  ExclamationCircleOutlined,
  MoreOutlined
} from '@ant-design/icons';
import styled from 'styled-components';
import { useDispatch, useSelector } from 'react-redux';
import type { RootState } from '../../store';
import dayjs from 'dayjs';

const { TabPane } = Tabs;
const { Option } = Select;
const { Dragger } = Upload;
const { TextArea } = Input;
const { TreeNode } = Tree;

// 样式组件
const PageContainer = styled.div`
  padding: 24px;
  background: #f5f5f5;
  min-height: 100vh;
`;

const StatsCard = styled(Card)`
  .ant-card-body {
    padding: 20px;
  }
  
  .ant-statistic-title {
    color: #666;
    font-size: 14px;
  }
  
  .ant-statistic-content {
    color: #1890ff;
  }
`;

const DatasetCard = styled(Card)`
  margin-bottom: 16px;
  
  .dataset-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 12px;
  }
  
  .dataset-stats {
    display: flex;
    gap: 24px;
    margin-top: 12px;
    
    .stat-item {
      text-align: center;
      
      .stat-value {
        font-size: 18px;
        font-weight: bold;
        color: #1890ff;
      }
      
      .stat-label {
        font-size: 12px;
        color: #666;
        margin-top: 4px;
      }
    }
  }
`;

const AnnotationCard = styled(Card)`
  .annotation-item {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 8px 0;
    border-bottom: 1px solid #f0f0f0;
    
    &:last-child {
      border-bottom: none;
    }
    
    .annotation-info {
      flex: 1;
      
      .annotation-name {
        font-weight: 500;
        margin-bottom: 4px;
      }
      
      .annotation-meta {
        font-size: 12px;
        color: #666;
      }
    }
    
    .annotation-actions {
      display: flex;
      gap: 8px;
    }
  }
`;

// 模拟数据
const mockDataStats = {
  totalDatasets: 15,
  totalImages: 12580,
  annotatedImages: 9876,
  totalStorage: 2.5, // TB
  todayUploads: 156
};

const mockDatasets = [
  {
    id: '1',
    name: 'CT胸部数据集',
    description: '用于肺部疾病诊断的CT图像数据集',
    type: 'CT',
    category: '胸部',
    imageCount: 2580,
    annotatedCount: 2450,
    size: '450GB',
    status: 'active',
    quality: 'high',
    createdAt: '2024-01-10 09:00:00',
    updatedAt: '2024-01-15 14:30:00',
    creator: '张医生',
    tags: ['肺部', '疾病诊断', 'CT']
  },
  {
    id: '2',
    name: 'MRI脑部数据集',
    description: '脑部疾病诊断MRI图像数据集',
    type: 'MRI',
    category: '脑部',
    imageCount: 1890,
    annotatedCount: 1650,
    size: '320GB',
    status: 'processing',
    quality: 'medium',
    createdAt: '2024-01-08 15:20:00',
    updatedAt: '2024-01-14 10:45:00',
    creator: '李医生',
    tags: ['脑部', 'MRI', '神经科']
  }
];

const mockAnnotations = [
  {
    id: '1',
    imageId: 'IMG_001',
    annotationType: 'segmentation',
    annotationName: '肺结节分割',
    annotator: '张医生',
    status: 'completed',
    confidence: 0.95,
    createdAt: '2024-01-15 10:30:00',
    reviewedAt: '2024-01-15 14:20:00',
    reviewer: '王医生'
  },
  {
    id: '2',
    imageId: 'IMG_002',
    annotationType: 'classification',
    annotationName: '病变分类',
    annotator: '李医生',
    status: 'pending_review',
    confidence: 0.88,
    createdAt: '2024-01-15 11:15:00',
    reviewedAt: null,
    reviewer: null
  }
];

const mockDataTree = [
  {
    title: '医学影像数据',
    key: 'root',
    children: [
      {
        title: 'CT数据',
        key: 'ct',
        children: [
          { title: '胸部CT (2580张)', key: 'ct-chest' },
          { title: '腹部CT (1890张)', key: 'ct-abdomen' },
          { title: '头部CT (1245张)', key: 'ct-head' }
        ]
      },
      {
        title: 'MRI数据',
        key: 'mri',
        children: [
          { title: '脑部MRI (1890张)', key: 'mri-brain' },
          { title: '脊柱MRI (980张)', key: 'mri-spine' },
          { title: '关节MRI (756张)', key: 'mri-joint' }
        ]
      },
      {
        title: 'X光数据',
        key: 'xray',
        children: [
          { title: '胸部X光 (3450张)', key: 'xray-chest' },
          { title: '骨骼X光 (2100张)', key: 'xray-bone' }
        ]
      }
    ]
  }
];

const DataManagePage: React.FC = () => {
  const [activeTab, setActiveTab] = useState('datasets');
  const [datasetModalVisible, setDatasetModalVisible] = useState(false);
  const [uploadModalVisible, setUploadModalVisible] = useState(false);
  const [annotationModalVisible, setAnnotationModalVisible] = useState(false);
  const [selectedDataset, setSelectedDataset] = useState<any>(null);
  const [selectedAnnotation, setSelectedAnnotation] = useState<any>(null);
  const [datasetForm] = Form.useForm();
  const [uploadForm] = Form.useForm();
  const [annotationForm] = Form.useForm();
  const [loading, setLoading] = useState(false);
  const [searchText, setSearchText] = useState('');
  const [filterType, setFilterType] = useState<string | undefined>();
  const [filterStatus, setFilterStatus] = useState<string | undefined>();
  const [uploadProgress, setUploadProgress] = useState(0);
  const [selectedTreeKeys, setSelectedTreeKeys] = useState<string[]>([]);

  // 获取状态标签
  const getStatusTag = (status: string) => {
    const statusConfig = {
      active: { color: 'green', text: '活跃' },
      processing: { color: 'blue', text: '处理中' },
      completed: { color: 'green', text: '已完成' },
      pending_review: { color: 'orange', text: '待审核' },
      inactive: { color: 'default', text: '未激活' },
      error: { color: 'red', text: '错误' }
    };
    const config = statusConfig[status as keyof typeof statusConfig] || { color: 'default', text: status };
    return <Tag color={config.color}>{config.text}</Tag>;
  };

  // 获取质量标签
  const getQualityTag = (quality: string) => {
    const qualityConfig = {
      high: { color: 'green', text: '高' },
      medium: { color: 'orange', text: '中' },
      low: { color: 'red', text: '低' }
    };
    const config = qualityConfig[quality as keyof typeof qualityConfig] || { color: 'default', text: quality };
    return <Tag color={config.color}>{config.text}</Tag>;
  };

  // 数据集表格列
  const datasetColumns = [
    {
      title: '数据集名称',
      dataIndex: 'name',
      key: 'name',
      width: 200,
      render: (name: string, record: any) => (
        <div>
          <div style={{ fontWeight: 500 }}>{name}</div>
          <div style={{ fontSize: 12, color: '#666' }}>{record.description}</div>
        </div>
      )
    },
    {
      title: '类型',
      dataIndex: 'type',
      key: 'type',
      width: 80,
      render: (type: string) => <Tag>{type}</Tag>
    },
    {
      title: '分类',
      dataIndex: 'category',
      key: 'category',
      width: 100
    },
    {
      title: '图像数量',
      dataIndex: 'imageCount',
      key: 'imageCount',
      width: 100,
      render: (count: number) => count.toLocaleString()
    },
    {
      title: '标注进度',
      key: 'annotationProgress',
      width: 150,
      render: (_, record) => {
        const progress = Math.round((record.annotatedCount / record.imageCount) * 100);
        return (
          <div>
            <Progress percent={progress} size="small" />
            <div style={{ fontSize: 12, color: '#666', marginTop: 4 }}>
              {record.annotatedCount}/{record.imageCount}
            </div>
          </div>
        );
      }
    },
    {
      title: '存储大小',
      dataIndex: 'size',
      key: 'size',
      width: 100
    },
    {
      title: '质量',
      dataIndex: 'quality',
      key: 'quality',
      width: 80,
      render: (quality: string) => getQualityTag(quality)
    },
    {
      title: '状态',
      dataIndex: 'status',
      key: 'status',
      width: 100,
      render: (status: string) => getStatusTag(status)
    },
    {
      title: '创建者',
      dataIndex: 'creator',
      key: 'creator',
      width: 100
    },
    {
      title: '更新时间',
      dataIndex: 'updatedAt',
      key: 'updatedAt',
      width: 150,
      render: (date: string) => dayjs(date).format('YYYY-MM-DD HH:mm')
    },
    {
      title: '操作',
      key: 'action',
      width: 150,
      render: (_, record) => (
        <Space size="small">
          <Tooltip title="查看详情">
            <Button
              type="text"
              icon={<EyeOutlined />}
              onClick={() => {
                setSelectedDataset(record);
                setDatasetModalVisible(true);
              }}
            />
          </Tooltip>
          <Tooltip title="编辑">
            <Button
              type="text"
              icon={<EditOutlined />}
              onClick={() => {
                setSelectedDataset(record);
                datasetForm.setFieldsValue(record);
                setDatasetModalVisible(true);
              }}
            />
          </Tooltip>
          <Dropdown
            overlay={
              <Menu>
                <Menu.Item key="export" icon={<ExportOutlined />}>
                  导出数据集
                </Menu.Item>
                <Menu.Item key="sync" icon={<SyncOutlined />}>
                  同步数据
                </Menu.Item>
                <Menu.Divider />
                <Menu.Item key="delete" icon={<DeleteOutlined />} danger>
                  删除数据集
                </Menu.Item>
              </Menu>
            }
            trigger={['click']}
          >
            <Button type="text" icon={<MoreOutlined />} />
          </Dropdown>
        </Space>
      )
    }
  ];

  // 标注表格列
  const annotationColumns = [
    {
      title: '图像ID',
      dataIndex: 'imageId',
      key: 'imageId',
      width: 120
    },
    {
      title: '标注类型',
      dataIndex: 'annotationType',
      key: 'annotationType',
      width: 120,
      render: (type: string) => {
        const typeConfig = {
          segmentation: { color: 'blue', text: '分割' },
          classification: { color: 'green', text: '分类' },
          detection: { color: 'orange', text: '检测' },
          keypoint: { color: 'purple', text: '关键点' }
        };
        const config = typeConfig[type as keyof typeof typeConfig] || { color: 'default', text: type };
        return <Tag color={config.color}>{config.text}</Tag>;
      }
    },
    {
      title: '标注名称',
      dataIndex: 'annotationName',
      key: 'annotationName',
      width: 150
    },
    {
      title: '标注者',
      dataIndex: 'annotator',
      key: 'annotator',
      width: 100
    },
    {
      title: '置信度',
      dataIndex: 'confidence',
      key: 'confidence',
      width: 100,
      render: (confidence: number) => (
        <Progress
          percent={Math.round(confidence * 100)}
          size="small"
          format={(percent) => `${percent}%`}
        />
      )
    },
    {
      title: '状态',
      dataIndex: 'status',
      key: 'status',
      width: 120,
      render: (status: string) => getStatusTag(status)
    },
    {
      title: '审核者',
      dataIndex: 'reviewer',
      key: 'reviewer',
      width: 100,
      render: (reviewer: string) => reviewer || '-'
    },
    {
      title: '创建时间',
      dataIndex: 'createdAt',
      key: 'createdAt',
      width: 150,
      render: (date: string) => dayjs(date).format('YYYY-MM-DD HH:mm')
    },
    {
      title: '操作',
      key: 'action',
      width: 120,
      render: (_, record) => (
        <Space size="small">
          <Tooltip title="查看">
            <Button
              type="text"
              icon={<EyeOutlined />}
              onClick={() => {
                setSelectedAnnotation(record);
                setAnnotationModalVisible(true);
              }}
            />
          </Tooltip>
          <Tooltip title="编辑">
            <Button
              type="text"
              icon={<EditOutlined />}
              onClick={() => {
                setSelectedAnnotation(record);
                annotationForm.setFieldsValue(record);
                setAnnotationModalVisible(true);
              }}
            />
          </Tooltip>
          {record.status === 'pending_review' && (
            <Tooltip title="审核">
              <Button
                type="text"
                icon={<CheckCircleOutlined />}
                onClick={() => handleReviewAnnotation(record.id)}
              />
            </Tooltip>
          )}
        </Space>
      )
    }
  ];

  // 处理数据上传
  const handleUpload = {
    name: 'file',
    multiple: true,
    accept: '.dcm,.jpg,.png,.nii,.nii.gz',
    beforeUpload: () => false, // 阻止自动上传
    onChange: (info: any) => {
      const { status } = info.file;
      if (status === 'done') {
        message.success(`${info.file.name} 上传成功`);
      } else if (status === 'error') {
        message.error(`${info.file.name} 上传失败`);
      }
    }
  };

  // 处理批量导入
  const handleBatchImport = async () => {
    setLoading(true);
    try {
      // 模拟批量导入
      for (let i = 0; i <= 100; i += 10) {
        setUploadProgress(i);
        await new Promise(resolve => setTimeout(resolve, 200));
      }
      message.success('批量导入完成');
      setUploadModalVisible(false);
      setUploadProgress(0);
    } catch (error) {
      message.error('批量导入失败');
    } finally {
      setLoading(false);
    }
  };

  // 处理标注审核
  const handleReviewAnnotation = (annotationId: string) => {
    Modal.confirm({
      title: '审核标注',
      content: '确认通过这个标注吗？',
      onOk: () => {
        message.success('标注审核通过');
      }
    });
  };

  // 处理保存数据集
  const handleSaveDataset = async () => {
    try {
      const values = await datasetForm.validateFields();
      setLoading(true);
      
      // 模拟API调用
      await new Promise(resolve => setTimeout(resolve, 1000));
      
      message.success(selectedDataset ? '数据集更新成功' : '数据集创建成功');
      setDatasetModalVisible(false);
      setSelectedDataset(null);
      datasetForm.resetFields();
    } catch (error) {
      message.error('保存失败');
    } finally {
      setLoading(false);
    }
  };

  // 处理保存标注
  const handleSaveAnnotation = async () => {
    try {
      const values = await annotationForm.validateFields();
      setLoading(true);
      
      // 模拟API调用
      await new Promise(resolve => setTimeout(resolve, 1000));
      
      message.success(selectedAnnotation ? '标注更新成功' : '标注创建成功');
      setAnnotationModalVisible(false);
      setSelectedAnnotation(null);
      annotationForm.resetFields();
    } catch (error) {
      message.error('保存失败');
    } finally {
      setLoading(false);
    }
  };

  return (
    <PageContainer>
      {/* 统计卡片 */}
      <Row gutter={[16, 16]} style={{ marginBottom: 24 }}>
        <Col xs={24} sm={12} md={6}>
          <StatsCard>
            <Statistic
              title="数据集总数"
              value={mockDataStats.totalDatasets}
              prefix={<FolderOutlined />}
            />
          </StatsCard>
        </Col>
        <Col xs={24} sm={12} md={6}>
          <StatsCard>
            <Statistic
              title="图像总数"
              value={mockDataStats.totalImages}
              prefix={<FileImageOutlined />}
            />
          </StatsCard>
        </Col>
        <Col xs={24} sm={12} md={6}>
          <StatsCard>
            <Statistic
              title="已标注图像"
              value={mockDataStats.annotatedImages}
              prefix={<TagOutlined />}
              valueStyle={{ color: '#3f8600' }}
            />
          </StatsCard>
        </Col>
        <Col xs={24} sm={12} md={6}>
          <StatsCard>
            <Statistic
              title="存储空间"
              value={mockDataStats.totalStorage}
              precision={1}
              suffix="TB"
              prefix={<BarChartOutlined />}
              valueStyle={{ color: '#1890ff' }}
            />
          </StatsCard>
        </Col>
      </Row>

      {/* 主要内容 */}
      <Card>
        <Tabs activeKey={activeTab} onChange={setActiveTab}>
          {/* 数据集管理 */}
          <TabPane tab="数据集管理" key="datasets">
            <div style={{ marginBottom: 16 }}>
              <Row gutter={[16, 16]}>
                <Col xs={24} sm={12} md={6}>
                  <Input
                    placeholder="搜索数据集名称"
                    prefix={<SearchOutlined />}
                    value={searchText}
                    onChange={(e) => setSearchText(e.target.value)}
                  />
                </Col>
                <Col xs={24} sm={12} md={4}>
                  <Select
                    placeholder="数据类型"
                    value={filterType}
                    onChange={setFilterType}
                    allowClear
                    style={{ width: '100%' }}
                  >
                    <Option value="CT">CT</Option>
                    <Option value="MRI">MRI</Option>
                    <Option value="X-Ray">X光</Option>
                    <Option value="Ultrasound">超声</Option>
                  </Select>
                </Col>
                <Col xs={24} sm={12} md={4}>
                  <Select
                    placeholder="状态"
                    value={filterStatus}
                    onChange={setFilterStatus}
                    allowClear
                    style={{ width: '100%' }}
                  >
                    <Option value="active">活跃</Option>
                    <Option value="processing">处理中</Option>
                    <Option value="inactive">未激活</Option>
                  </Select>
                </Col>
                <Col xs={24} sm={12} md={10}>
                  <Space>
                    <Button
                      type="primary"
                      icon={<PlusOutlined />}
                      onClick={() => {
                        setSelectedDataset(null);
                        datasetForm.resetFields();
                        setDatasetModalVisible(true);
                      }}
                    >
                      新建数据集
                    </Button>
                    <Button
                      icon={<UploadOutlined />}
                      onClick={() => setUploadModalVisible(true)}
                    >
                      批量导入
                    </Button>
                    <Button icon={<ExportOutlined />}>
                      导出数据
                    </Button>
                    <Button icon={<SyncOutlined />}>
                      同步数据
                    </Button>
                  </Space>
                </Col>
              </Row>
            </div>

            <Table
              columns={datasetColumns}
              dataSource={mockDatasets}
              rowKey="id"
              pagination={{
                total: mockDatasets.length,
                pageSize: 10,
                showSizeChanger: true,
                showQuickJumper: true,
                showTotal: (total, range) => `第 ${range[0]}-${range[1]} 条/共 ${total} 条`
              }}
              scroll={{ x: 1400 }}
            />
          </TabPane>

          {/* 数据浏览 */}
          <TabPane tab="数据浏览" key="browse">
            <Row gutter={16}>
              <Col xs={24} lg={8}>
                <Card title="数据目录" size="small">
                  <Tree
                    showIcon
                    defaultExpandAll
                    selectedKeys={selectedTreeKeys}
                    onSelect={setSelectedTreeKeys}
                    treeData={mockDataTree}
                  />
                </Card>
              </Col>
              <Col xs={24} lg={16}>
                <Card title="数据详情" size="small">
                  {selectedTreeKeys.length > 0 ? (
                    <div>
                      <Alert
                        message="数据集信息"
                        description={`已选择: ${selectedTreeKeys[0]}`}
                        type="info"
                        showIcon
                        style={{ marginBottom: 16 }}
                      />
                      <Row gutter={[16, 16]}>
                        <Col span={8}>
                          <Statistic title="图像数量" value={2580} />
                        </Col>
                        <Col span={8}>
                          <Statistic title="已标注" value={2450} />
                        </Col>
                        <Col span={8}>
                          <Statistic title="存储大小" value="450GB" />
                        </Col>
                      </Row>
                      <Divider />
                      <div style={{ textAlign: 'center', padding: '40px 0' }}>
                        <FileImageOutlined style={{ fontSize: 48, color: '#d9d9d9' }} />
                        <p style={{ color: '#999', marginTop: 16 }}>图像预览区域</p>
                      </div>
                    </div>
                  ) : (
                    <div style={{ textAlign: 'center', padding: '60px 0' }}>
                      <FolderOutlined style={{ fontSize: 48, color: '#d9d9d9' }} />
                      <p style={{ color: '#999', marginTop: 16 }}>请选择数据目录</p>
                    </div>
                  )}
                </Card>
              </Col>
            </Row>
          </TabPane>

          {/* 标注管理 */}
          <TabPane tab="标注管理" key="annotations">
            <div style={{ marginBottom: 16 }}>
              <Row gutter={[16, 16]}>
                <Col xs={24} sm={12} md={6}>
                  <Input
                    placeholder="搜索图像ID"
                    prefix={<SearchOutlined />}
                    value={searchText}
                    onChange={(e) => setSearchText(e.target.value)}
                  />
                </Col>
                <Col xs={24} sm={12} md={4}>
                  <Select
                    placeholder="标注类型"
                    allowClear
                    style={{ width: '100%' }}
                  >
                    <Option value="segmentation">分割</Option>
                    <Option value="classification">分类</Option>
                    <Option value="detection">检测</Option>
                    <Option value="keypoint">关键点</Option>
                  </Select>
                </Col>
                <Col xs={24} sm={12} md={4}>
                  <Select
                    placeholder="状态"
                    allowClear
                    style={{ width: '100%' }}
                  >
                    <Option value="completed">已完成</Option>
                    <Option value="pending_review">待审核</Option>
                    <Option value="in_progress">进行中</Option>
                  </Select>
                </Col>
                <Col xs={24} sm={12} md={10}>
                  <Space>
                    <Button
                      type="primary"
                      icon={<PlusOutlined />}
                      onClick={() => {
                        setSelectedAnnotation(null);
                        annotationForm.resetFields();
                        setAnnotationModalVisible(true);
                      }}
                    >
                      新建标注
                    </Button>
                    <Button icon={<CheckCircleOutlined />}>
                      批量审核
                    </Button>
                    <Button icon={<ExportOutlined />}>
                      导出标注
                    </Button>
                  </Space>
                </Col>
              </Row>
            </div>

            <Table
              columns={annotationColumns}
              dataSource={mockAnnotations}
              rowKey="id"
              pagination={{
                total: mockAnnotations.length,
                pageSize: 10,
                showSizeChanger: true,
                showQuickJumper: true,
                showTotal: (total, range) => `第 ${range[0]}-${range[1]} 条/共 ${total} 条`
              }}
              scroll={{ x: 1200 }}
            />
          </TabPane>

          {/* 数据统计 */}
          <TabPane tab="数据统计" key="statistics">
            <Row gutter={[16, 16]}>
              <Col xs={24} lg={12}>
                <Card title="数据分布" extra={<Button icon={<SyncOutlined />} />}>
                  <div style={{ height: 300, display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                    <div style={{ textAlign: 'center' }}>
                      <BarChartOutlined style={{ fontSize: 48, color: '#d9d9d9' }} />
                      <p style={{ color: '#999', marginTop: 16 }}>数据分布图表</p>
                    </div>
                  </div>
                </Card>
              </Col>
              <Col xs={24} lg={12}>
                <Card title="标注进度" extra={<Button icon={<SyncOutlined />} />}>
                  <div style={{ height: 300, display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                    <div style={{ textAlign: 'center' }}>
                      <BarChartOutlined style={{ fontSize: 48, color: '#d9d9d9' }} />
                      <p style={{ color: '#999', marginTop: 16 }}>标注进度图表</p>
                    </div>
                  </div>
                </Card>
              </Col>
              <Col xs={24}>
                <Card title="存储使用情况">
                  <Row gutter={[16, 16]}>
                    <Col xs={24} sm={8}>
                      <div style={{ textAlign: 'center' }}>
                        <Progress
                          type="circle"
                          percent={65}
                          format={(percent) => `${percent}%`}
                          strokeColor="#1890ff"
                        />
                        <p style={{ marginTop: 8 }}>总存储使用率</p>
                      </div>
                    </Col>
                    <Col xs={24} sm={16}>
                      <div>
                        <div style={{ marginBottom: 16 }}>
                          <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 4 }}>
                            <span>CT数据</span>
                            <span>1.2TB / 2.0TB</span>
                          </div>
                          <Progress percent={60} strokeColor="#52c41a" />
                        </div>
                        <div style={{ marginBottom: 16 }}>
                          <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 4 }}>
                            <span>MRI数据</span>
                            <span>0.8TB / 1.5TB</span>
                          </div>
                          <Progress percent={53} strokeColor="#1890ff" />
                        </div>
                        <div>
                          <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 4 }}>
                            <span>X光数据</span>
                            <span>0.5TB / 1.0TB</span>
                          </div>
                          <Progress percent={50} strokeColor="#faad14" />
                        </div>
                      </div>
                    </Col>
                  </Row>
                </Card>
              </Col>
            </Row>
          </TabPane>
        </Tabs>
      </Card>

      {/* 数据集模态框 */}
      <Modal
        title={selectedDataset ? "编辑数据集" : "新建数据集"}
        open={datasetModalVisible}
        onOk={handleSaveDataset}
        onCancel={() => {
          setDatasetModalVisible(false);
          setSelectedDataset(null);
          datasetForm.resetFields();
        }}
        confirmLoading={loading}
        width={600}
      >
        <Form form={datasetForm} layout="vertical">
          <Row gutter={16}>
            <Col span={12}>
              <Form.Item
                name="name"
                label="数据集名称"
                rules={[{ required: true, message: '请输入数据集名称' }]}
              >
                <Input placeholder="请输入数据集名称" />
              </Form.Item>
            </Col>
            <Col span={12}>
              <Form.Item
                name="type"
                label="数据类型"
                rules={[{ required: true, message: '请选择数据类型' }]}
              >
                <Select placeholder="请选择数据类型">
                  <Option value="CT">CT</Option>
                  <Option value="MRI">MRI</Option>
                  <Option value="X-Ray">X光</Option>
                  <Option value="Ultrasound">超声</Option>
                </Select>
              </Form.Item>
            </Col>
          </Row>
          <Row gutter={16}>
            <Col span={12}>
              <Form.Item
                name="category"
                label="分类"
                rules={[{ required: true, message: '请输入分类' }]}
              >
                <Input placeholder="请输入分类" />
              </Form.Item>
            </Col>
            <Col span={12}>
              <Form.Item
                name="quality"
                label="质量等级"
                rules={[{ required: true, message: '请选择质量等级' }]}
              >
                <Select placeholder="请选择质量等级">
                  <Option value="high">高</Option>
                  <Option value="medium">中</Option>
                  <Option value="low">低</Option>
                </Select>
              </Form.Item>
            </Col>
          </Row>
          <Form.Item
            name="description"
            label="描述"
            rules={[{ required: true, message: '请输入描述' }]}
          >
            <TextArea rows={3} placeholder="请输入数据集描述" />
          </Form.Item>
          <Form.Item name="tags" label="标签">
            <Select
              mode="tags"
              placeholder="请输入标签"
              style={{ width: '100%' }}
            >
              <Option value="肺部">肺部</Option>
              <Option value="脑部">脑部</Option>
              <Option value="骨骼">骨骼</Option>
              <Option value="心脏">心脏</Option>
            </Select>
          </Form.Item>
        </Form>
      </Modal>

      {/* 批量上传模态框 */}
      <Modal
        title="批量数据导入"
        open={uploadModalVisible}
        onOk={handleBatchImport}
        onCancel={() => {
          setUploadModalVisible(false);
          setUploadProgress(0);
        }}
        confirmLoading={loading}
        width={600}
      >
        <Form form={uploadForm} layout="vertical">
          <Form.Item name="dataset" label="目标数据集" rules={[{ required: true }]}>
            <Select placeholder="请选择目标数据集">
              {mockDatasets.map(dataset => (
                <Option key={dataset.id} value={dataset.id}>{dataset.name}</Option>
              ))}
            </Select>
          </Form.Item>
          
          <Form.Item label="上传文件">
            <Dragger {...handleUpload}>
              <p className="ant-upload-drag-icon">
                <UploadOutlined />
              </p>
              <p className="ant-upload-text">点击或拖拽文件到此区域上传</p>
              <p className="ant-upload-hint">
                支持 DICOM (.dcm)、JPEG (.jpg)、PNG (.png)、NIfTI (.nii, .nii.gz) 格式
              </p>
            </Dragger>
          </Form.Item>
          
          {uploadProgress > 0 && (
            <Form.Item label="上传进度">
              <Progress percent={uploadProgress} />
            </Form.Item>
          )}
        </Form>
      </Modal>

      {/* 标注模态框 */}
      <Modal
        title={selectedAnnotation ? "编辑标注" : "新建标注"}
        open={annotationModalVisible}
        onOk={handleSaveAnnotation}
        onCancel={() => {
          setAnnotationModalVisible(false);
          setSelectedAnnotation(null);
          annotationForm.resetFields();
        }}
        confirmLoading={loading}
        width={600}
      >
        <Form form={annotationForm} layout="vertical">
          <Row gutter={16}>
            <Col span={12}>
              <Form.Item
                name="imageId"
                label="图像ID"
                rules={[{ required: true, message: '请输入图像ID' }]}
              >
                <Input placeholder="请输入图像ID" />
              </Form.Item>
            </Col>
            <Col span={12}>
              <Form.Item
                name="annotationType"
                label="标注类型"
                rules={[{ required: true, message: '请选择标注类型' }]}
              >
                <Select placeholder="请选择标注类型">
                  <Option value="segmentation">分割</Option>
                  <Option value="classification">分类</Option>
                  <Option value="detection">检测</Option>
                  <Option value="keypoint">关键点</Option>
                </Select>
              </Form.Item>
            </Col>
          </Row>
          <Form.Item
            name="annotationName"
            label="标注名称"
            rules={[{ required: true, message: '请输入标注名称' }]}
          >
            <Input placeholder="请输入标注名称" />
          </Form.Item>
          <Form.Item name="description" label="标注描述">
            <TextArea rows={3} placeholder="请输入标注描述" />
          </Form.Item>
        </Form>
      </Modal>
    </PageContainer>
  );
};

export default DataManagePage;