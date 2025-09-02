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
  DatePicker,
  Progress,
  Statistic,
  Row,
  Col,
  Tabs,
  Upload,
  message,
  Tooltip,
  Divider,
  Switch,
  InputNumber,
  Alert,
  Badge
} from 'antd';
import {
  UploadOutlined,
  DownloadOutlined,
  EyeOutlined,
  EditOutlined,
  DeleteOutlined,
  PlusOutlined,
  SearchOutlined,
  ReloadOutlined,
  SettingOutlined,
  BarChartOutlined,
  ExclamationCircleOutlined,
  CheckCircleOutlined,
  CloseCircleOutlined,
  WarningOutlined
} from '@ant-design/icons';
import styled from 'styled-components';
import { useDispatch, useSelector } from 'react-redux';
import type { RootState } from '../../store';
import dayjs from 'dayjs';

const { TabPane } = Tabs;
const { Option } = Select;
const { RangePicker } = DatePicker;
const { TextArea } = Input;

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

const QualityCard = styled(Card)`
  margin-bottom: 16px;
  
  .quality-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 16px;
  }
  
  .quality-score {
    text-align: center;
    padding: 20px;
    
    .score-circle {
      margin-bottom: 8px;
    }
    
    .score-label {
      color: #666;
      font-size: 12px;
    }
  }
`;

const RuleCard = styled(Card)`
  margin-bottom: 16px;
  
  .rule-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
  }
  
  .rule-content {
    margin-top: 12px;
    
    .rule-condition {
      background: #f8f9fa;
      padding: 8px 12px;
      border-radius: 4px;
      margin: 4px 0;
      font-family: 'Courier New', monospace;
      font-size: 12px;
    }
  }
`;

// 模拟数据
const mockQualityStats = {
  totalAssessments: 1248,
  passedAssessments: 1156,
  failedAssessments: 92,
  averageScore: 87.5,
  todayAssessments: 45,
  activeRules: 28
};

const mockQualityAssessments = [
  {
    id: '1',
    imageId: 'IMG_001',
    patientName: '张三',
    studyType: 'CT胸部',
    overallScore: 92.5,
    grade: 'excellent',
    technicalScore: 94.0,
    clinicalScore: 91.0,
    issues: [
      { type: 'warning', message: '图像对比度略低' }
    ],
    assessmentDate: '2024-01-15 14:30:00',
    assessor: 'AI系统',
    status: 'completed'
  },
  {
    id: '2',
    imageId: 'IMG_002',
    patientName: '李四',
    studyType: 'MRI脑部',
    overallScore: 78.3,
    grade: 'good',
    technicalScore: 82.0,
    clinicalScore: 74.6,
    issues: [
      { type: 'error', message: '运动伪影明显' },
      { type: 'warning', message: '信噪比偏低' }
    ],
    assessmentDate: '2024-01-15 13:45:00',
    assessor: '王医生',
    status: 'completed'
  }
];

const mockQualityRules = [
  {
    id: '1',
    name: '图像对比度检查',
    description: '检查医学图像的对比度是否符合诊断要求',
    type: 'technical',
    category: '图像质量',
    severity: 'medium',
    conditions: {
      contrast_ratio: { min: 0.3, max: 1.0 },
      brightness: { min: 50, max: 200 }
    },
    thresholds: {
      excellent: 0.8,
      good: 0.6,
      acceptable: 0.4
    },
    isActive: true,
    createdAt: '2024-01-10 09:00:00',
    updatedAt: '2024-01-14 16:30:00'
  },
  {
    id: '2',
    name: '解剖结构完整性',
    description: '检查关键解剖结构是否完整可见',
    type: 'clinical',
    category: '临床质量',
    severity: 'high',
    conditions: {
      structure_visibility: { min: 0.9 },
      coverage_completeness: { min: 0.95 }
    },
    thresholds: {
      excellent: 0.95,
      good: 0.85,
      acceptable: 0.75
    },
    isActive: true,
    createdAt: '2024-01-08 14:20:00',
    updatedAt: '2024-01-12 11:15:00'
  }
];

const QualityControlPage: React.FC = () => {
  const [activeTab, setActiveTab] = useState('assessments');
  const [assessmentModalVisible, setAssessmentModalVisible] = useState(false);
  const [ruleModalVisible, setRuleModalVisible] = useState(false);
  const [selectedAssessment, setSelectedAssessment] = useState<any>(null);
  const [selectedRule, setSelectedRule] = useState<any>(null);
  const [assessmentForm] = Form.useForm();
  const [ruleForm] = Form.useForm();
  const [loading, setLoading] = useState(false);
  const [searchText, setSearchText] = useState('');
  const [filterGrade, setFilterGrade] = useState<string | undefined>();
  const [filterType, setFilterType] = useState<string | undefined>();
  const [dateRange, setDateRange] = useState<any[]>([]);

  // 获取质量等级标签
  const getGradeTag = (grade: string) => {
    const gradeConfig = {
      excellent: { color: 'green', text: '优秀' },
      good: { color: 'blue', text: '良好' },
      acceptable: { color: 'orange', text: '可接受' },
      poor: { color: 'red', text: '差' }
    };
    const config = gradeConfig[grade as keyof typeof gradeConfig] || { color: 'default', text: grade };
    return <Tag color={config.color}>{config.text}</Tag>;
  };

  // 获取严重程度标签
  const getSeverityTag = (severity: string) => {
    const severityConfig = {
      low: { color: 'green', text: '低' },
      medium: { color: 'orange', text: '中' },
      high: { color: 'red', text: '高' },
      critical: { color: 'magenta', text: '严重' }
    };
    const config = severityConfig[severity as keyof typeof severityConfig] || { color: 'default', text: severity };
    return <Tag color={config.color}>{config.text}</Tag>;
  };

  // 获取问题图标
  const getIssueIcon = (type: string) => {
    switch (type) {
      case 'error':
        return <CloseCircleOutlined style={{ color: '#ff4d4f' }} />;
      case 'warning':
        return <WarningOutlined style={{ color: '#faad14' }} />;
      case 'info':
        return <ExclamationCircleOutlined style={{ color: '#1890ff' }} />;
      default:
        return <CheckCircleOutlined style={{ color: '#52c41a' }} />;
    }
  };

  // 质量评估表格列
  const assessmentColumns = [
    {
      title: '图像ID',
      dataIndex: 'imageId',
      key: 'imageId',
      width: 120
    },
    {
      title: '患者姓名',
      dataIndex: 'patientName',
      key: 'patientName',
      width: 100
    },
    {
      title: '检查类型',
      dataIndex: 'studyType',
      key: 'studyType',
      width: 120
    },
    {
      title: '综合评分',
      dataIndex: 'overallScore',
      key: 'overallScore',
      width: 100,
      render: (score: number) => (
        <div style={{ textAlign: 'center' }}>
          <Progress
            type="circle"
            size={40}
            percent={score}
            format={(percent) => `${percent}`}
            strokeColor={score >= 90 ? '#52c41a' : score >= 80 ? '#1890ff' : score >= 70 ? '#faad14' : '#ff4d4f'}
          />
        </div>
      )
    },
    {
      title: '质量等级',
      dataIndex: 'grade',
      key: 'grade',
      width: 100,
      render: (grade: string) => getGradeTag(grade)
    },
    {
      title: '技术评分',
      dataIndex: 'technicalScore',
      key: 'technicalScore',
      width: 100,
      render: (score: number) => `${score.toFixed(1)}`
    },
    {
      title: '临床评分',
      dataIndex: 'clinicalScore',
      key: 'clinicalScore',
      width: 100,
      render: (score: number) => `${score.toFixed(1)}`
    },
    {
      title: '问题数量',
      dataIndex: 'issues',
      key: 'issues',
      width: 100,
      render: (issues: any[]) => (
        <Badge count={issues.length} showZero style={{ backgroundColor: issues.length > 0 ? '#ff4d4f' : '#52c41a' }} />
      )
    },
    {
      title: '评估时间',
      dataIndex: 'assessmentDate',
      key: 'assessmentDate',
      width: 150,
      render: (date: string) => dayjs(date).format('YYYY-MM-DD HH:mm')
    },
    {
      title: '评估者',
      dataIndex: 'assessor',
      key: 'assessor',
      width: 100
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
                setSelectedAssessment(record);
                setAssessmentModalVisible(true);
              }}
            />
          </Tooltip>
          <Tooltip title="重新评估">
            <Button
              type="text"
              icon={<ReloadOutlined />}
              onClick={() => handleReassess(record.id)}
            />
          </Tooltip>
          <Tooltip title="下载报告">
            <Button
              type="text"
              icon={<DownloadOutlined />}
              onClick={() => handleDownloadReport(record.id)}
            />
          </Tooltip>
        </Space>
      )
    }
  ];

  // 质量规则表格列
  const ruleColumns = [
    {
      title: '规则名称',
      dataIndex: 'name',
      key: 'name',
      width: 150
    },
    {
      title: '描述',
      dataIndex: 'description',
      key: 'description',
      ellipsis: true
    },
    {
      title: '类型',
      dataIndex: 'type',
      key: 'type',
      width: 100,
      render: (type: string) => (
        <Tag color={type === 'technical' ? 'blue' : 'green'}>
          {type === 'technical' ? '技术' : '临床'}
        </Tag>
      )
    },
    {
      title: '分类',
      dataIndex: 'category',
      key: 'category',
      width: 100
    },
    {
      title: '严重程度',
      dataIndex: 'severity',
      key: 'severity',
      width: 100,
      render: (severity: string) => getSeverityTag(severity)
    },
    {
      title: '状态',
      dataIndex: 'isActive',
      key: 'isActive',
      width: 80,
      render: (isActive: boolean) => (
        <Switch
          checked={isActive}
          size="small"
          onChange={(checked) => handleToggleRule(checked)}
        />
      )
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
      width: 120,
      render: (_, record) => (
        <Space size="small">
          <Tooltip title="编辑">
            <Button
              type="text"
              icon={<EditOutlined />}
              onClick={() => {
                setSelectedRule(record);
                ruleForm.setFieldsValue(record);
                setRuleModalVisible(true);
              }}
            />
          </Tooltip>
          <Tooltip title="删除">
            <Button
              type="text"
              danger
              icon={<DeleteOutlined />}
              onClick={() => handleDeleteRule(record.id)}
            />
          </Tooltip>
        </Space>
      )
    }
  ];

  // 处理重新评估
  const handleReassess = async (assessmentId: string) => {
    setLoading(true);
    try {
      // 模拟API调用
      await new Promise(resolve => setTimeout(resolve, 2000));
      message.success('重新评估完成');
    } catch (error) {
      message.error('重新评估失败');
    } finally {
      setLoading(false);
    }
  };

  // 处理下载报告
  const handleDownloadReport = (assessmentId: string) => {
    message.info('正在生成报告...');
    // 模拟下载
    setTimeout(() => {
      message.success('报告下载完成');
    }, 1000);
  };

  // 处理规则开关
  const handleToggleRule = (checked: boolean) => {
    message.success(`规则已${checked ? '启用' : '禁用'}`);
  };

  // 处理删除规则
  const handleDeleteRule = (ruleId: string) => {
    Modal.confirm({
      title: '确认删除',
      content: '确定要删除这个质量规则吗？',
      onOk: () => {
        message.success('规则删除成功');
      }
    });
  };

  // 处理批量评估
  const handleBatchAssessment = () => {
    Modal.confirm({
      title: '批量质量评估',
      content: '确定要对所有未评估的图像进行质量评估吗？',
      onOk: async () => {
        setLoading(true);
        try {
          // 模拟批量评估
          await new Promise(resolve => setTimeout(resolve, 3000));
          message.success('批量评估完成');
        } catch (error) {
          message.error('批量评估失败');
        } finally {
          setLoading(false);
        }
      }
    });
  };

  // 处理保存规则
  const handleSaveRule = async () => {
    try {
      const values = await ruleForm.validateFields();
      setLoading(true);
      
      // 模拟API调用
      await new Promise(resolve => setTimeout(resolve, 1000));
      
      message.success(selectedRule ? '规则更新成功' : '规则创建成功');
      setRuleModalVisible(false);
      setSelectedRule(null);
      ruleForm.resetFields();
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
              title="总评估数"
              value={mockQualityStats.totalAssessments}
              prefix={<BarChartOutlined />}
            />
          </StatsCard>
        </Col>
        <Col xs={24} sm={12} md={6}>
          <StatsCard>
            <Statistic
              title="通过评估"
              value={mockQualityStats.passedAssessments}
              prefix={<CheckCircleOutlined />}
              valueStyle={{ color: '#3f8600' }}
            />
          </StatsCard>
        </Col>
        <Col xs={24} sm={12} md={6}>
          <StatsCard>
            <Statistic
              title="未通过评估"
              value={mockQualityStats.failedAssessments}
              prefix={<CloseCircleOutlined />}
              valueStyle={{ color: '#cf1322' }}
            />
          </StatsCard>
        </Col>
        <Col xs={24} sm={12} md={6}>
          <StatsCard>
            <Statistic
              title="平均评分"
              value={mockQualityStats.averageScore}
              precision={1}
              suffix="分"
              prefix={<BarChartOutlined />}
              valueStyle={{ color: '#1890ff' }}
            />
          </StatsCard>
        </Col>
      </Row>

      {/* 主要内容 */}
      <Card>
        <Tabs activeKey={activeTab} onChange={setActiveTab}>
          {/* 质量评估 */}
          <TabPane tab="质量评估" key="assessments">
            <div style={{ marginBottom: 16 }}>
              <Row gutter={[16, 16]}>
                <Col xs={24} sm={12} md={6}>
                  <Input
                    placeholder="搜索图像ID或患者姓名"
                    prefix={<SearchOutlined />}
                    value={searchText}
                    onChange={(e) => setSearchText(e.target.value)}
                  />
                </Col>
                <Col xs={24} sm={12} md={4}>
                  <Select
                    placeholder="质量等级"
                    value={filterGrade}
                    onChange={setFilterGrade}
                    allowClear
                    style={{ width: '100%' }}
                  >
                    <Option value="excellent">优秀</Option>
                    <Option value="good">良好</Option>
                    <Option value="acceptable">可接受</Option>
                    <Option value="poor">差</Option>
                  </Select>
                </Col>
                <Col xs={24} sm={12} md={6}>
                  <RangePicker
                    value={dateRange}
                    onChange={setDateRange}
                    style={{ width: '100%' }}
                  />
                </Col>
                <Col xs={24} sm={12} md={8}>
                  <Space>
                    <Button
                      type="primary"
                      icon={<PlusOutlined />}
                      onClick={() => setAssessmentModalVisible(true)}
                    >
                      新建评估
                    </Button>
                    <Button
                      icon={<ReloadOutlined />}
                      onClick={handleBatchAssessment}
                      loading={loading}
                    >
                      批量评估
                    </Button>
                    <Button icon={<DownloadOutlined />}>
                      导出报告
                    </Button>
                  </Space>
                </Col>
              </Row>
            </div>

            <Table
              columns={assessmentColumns}
              dataSource={mockQualityAssessments}
              rowKey="id"
              pagination={{
                total: mockQualityAssessments.length,
                pageSize: 10,
                showSizeChanger: true,
                showQuickJumper: true,
                showTotal: (total, range) => `第 ${range[0]}-${range[1]} 条/共 ${total} 条`
              }}
              scroll={{ x: 1200 }}
            />
          </TabPane>

          {/* 质量规则 */}
          <TabPane tab="质量规则" key="rules">
            <div style={{ marginBottom: 16 }}>
              <Row gutter={[16, 16]}>
                <Col xs={24} sm={12} md={6}>
                  <Input
                    placeholder="搜索规则名称"
                    prefix={<SearchOutlined />}
                    value={searchText}
                    onChange={(e) => setSearchText(e.target.value)}
                  />
                </Col>
                <Col xs={24} sm={12} md={4}>
                  <Select
                    placeholder="规则类型"
                    value={filterType}
                    onChange={setFilterType}
                    allowClear
                    style={{ width: '100%' }}
                  >
                    <Option value="technical">技术</Option>
                    <Option value="clinical">临床</Option>
                  </Select>
                </Col>
                <Col xs={24} sm={12} md={14}>
                  <Space>
                    <Button
                      type="primary"
                      icon={<PlusOutlined />}
                      onClick={() => {
                        setSelectedRule(null);
                        ruleForm.resetFields();
                        setRuleModalVisible(true);
                      }}
                    >
                      新建规则
                    </Button>
                    <Button icon={<SettingOutlined />}>
                      规则配置
                    </Button>
                    <Button icon={<DownloadOutlined />}>
                      导出规则
                    </Button>
                  </Space>
                </Col>
              </Row>
            </div>

            <Table
              columns={ruleColumns}
              dataSource={mockQualityRules}
              rowKey="id"
              pagination={{
                total: mockQualityRules.length,
                pageSize: 10,
                showSizeChanger: true,
                showQuickJumper: true,
                showTotal: (total, range) => `第 ${range[0]}-${range[1]} 条/共 ${total} 条`
              }}
              scroll={{ x: 1000 }}
            />
          </TabPane>

          {/* 数据质量监控 */}
          <TabPane tab="数据质量监控" key="monitoring">
            <Row gutter={[16, 16]}>
              <Col xs={24} lg={12}>
                <Card title="数据质量趋势" extra={<Button icon={<ReloadOutlined />} />}>
                  <div style={{ height: 300, display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                    <div style={{ textAlign: 'center' }}>
                      <BarChartOutlined style={{ fontSize: 48, color: '#d9d9d9' }} />
                      <p style={{ color: '#999', marginTop: 16 }}>质量趋势图表</p>
                    </div>
                  </div>
                </Card>
              </Col>
              <Col xs={24} lg={12}>
                <Card title="质量分布" extra={<Button icon={<ReloadOutlined />} />}>
                  <div style={{ height: 300, display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                    <div style={{ textAlign: 'center' }}>
                      <BarChartOutlined style={{ fontSize: 48, color: '#d9d9d9' }} />
                      <p style={{ color: '#999', marginTop: 16 }}>质量分布图表</p>
                    </div>
                  </div>
                </Card>
              </Col>
              <Col xs={24}>
                <Card title="质量告警">
                  <Alert
                    message="数据质量告警"
                    description="检测到3个图像的对比度低于阈值，建议进行人工复核。"
                    type="warning"
                    showIcon
                    style={{ marginBottom: 16 }}
                  />
                  <Alert
                    message="规则执行异常"
                    description="解剖结构完整性检查规则执行失败，请检查规则配置。"
                    type="error"
                    showIcon
                  />
                </Card>
              </Col>
            </Row>
          </TabPane>
        </Tabs>
      </Card>

      {/* 质量评估详情模态框 */}
      <Modal
        title={selectedAssessment ? "质量评估详情" : "新建质量评估"}
        open={assessmentModalVisible}
        onCancel={() => {
          setAssessmentModalVisible(false);
          setSelectedAssessment(null);
        }}
        footer={[
          <Button key="cancel" onClick={() => setAssessmentModalVisible(false)}>
            关闭
          </Button>,
          !selectedAssessment && (
            <Button key="submit" type="primary" loading={loading}>
              开始评估
            </Button>
          )
        ]}
        width={800}
      >
        {selectedAssessment ? (
          <div>
            <Row gutter={[16, 16]}>
              <Col span={12}>
                <QualityCard>
                  <div className="quality-score">
                    <Progress
                      type="circle"
                      percent={selectedAssessment.overallScore}
                      format={(percent) => `${percent}分`}
                      strokeColor={selectedAssessment.overallScore >= 90 ? '#52c41a' : selectedAssessment.overallScore >= 80 ? '#1890ff' : '#faad14'}
                      className="score-circle"
                    />
                    <div className="score-label">综合评分</div>
                  </div>
                </QualityCard>
              </Col>
              <Col span={12}>
                <Card title="评分详情">
                  <div style={{ marginBottom: 8 }}>
                    <span>技术评分：</span>
                    <Progress
                      percent={selectedAssessment.technicalScore}
                      size="small"
                      format={(percent) => `${percent}分`}
                    />
                  </div>
                  <div>
                    <span>临床评分：</span>
                    <Progress
                      percent={selectedAssessment.clinicalScore}
                      size="small"
                      format={(percent) => `${percent}分`}
                    />
                  </div>
                </Card>
              </Col>
            </Row>
            
            <Divider />
            
            <Card title="发现的问题" size="small">
              {selectedAssessment.issues.map((issue: any, index: number) => (
                <div key={index} style={{ marginBottom: 8, display: 'flex', alignItems: 'center' }}>
                  {getIssueIcon(issue.type)}
                  <span style={{ marginLeft: 8 }}>{issue.message}</span>
                </div>
              ))}
            </Card>
            
            <Card title="基本信息" size="small" style={{ marginTop: 16 }}>
              <Row gutter={[16, 8]}>
                <Col span={12}>
                  <strong>图像ID：</strong>{selectedAssessment.imageId}
                </Col>
                <Col span={12}>
                  <strong>患者姓名：</strong>{selectedAssessment.patientName}
                </Col>
                <Col span={12}>
                  <strong>检查类型：</strong>{selectedAssessment.studyType}
                </Col>
                <Col span={12}>
                  <strong>评估者：</strong>{selectedAssessment.assessor}
                </Col>
                <Col span={24}>
                  <strong>评估时间：</strong>{selectedAssessment.assessmentDate}
                </Col>
              </Row>
            </Card>
          </div>
        ) : (
          <Form form={assessmentForm} layout="vertical">
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
                  name="assessmentType"
                  label="评估类型"
                  rules={[{ required: true, message: '请选择评估类型' }]}
                >
                  <Select placeholder="请选择评估类型">
                    <Option value="technical">技术质量</Option>
                    <Option value="clinical">临床质量</Option>
                    <Option value="comprehensive">综合评估</Option>
                  </Select>
                </Form.Item>
              </Col>
            </Row>
            <Form.Item name="description" label="评估说明">
              <TextArea rows={3} placeholder="请输入评估说明" />
            </Form.Item>
          </Form>
        )}
      </Modal>

      {/* 质量规则模态框 */}
      <Modal
        title={selectedRule ? "编辑质量规则" : "新建质量规则"}
        open={ruleModalVisible}
        onOk={handleSaveRule}
        onCancel={() => {
          setRuleModalVisible(false);
          setSelectedRule(null);
          ruleForm.resetFields();
        }}
        confirmLoading={loading}
        width={800}
      >
        <Form form={ruleForm} layout="vertical">
          <Row gutter={16}>
            <Col span={12}>
              <Form.Item
                name="name"
                label="规则名称"
                rules={[{ required: true, message: '请输入规则名称' }]}
              >
                <Input placeholder="请输入规则名称" />
              </Form.Item>
            </Col>
            <Col span={12}>
              <Form.Item
                name="type"
                label="规则类型"
                rules={[{ required: true, message: '请选择规则类型' }]}
              >
                <Select placeholder="请选择规则类型">
                  <Option value="technical">技术</Option>
                  <Option value="clinical">临床</Option>
                </Select>
              </Form.Item>
            </Col>
          </Row>
          <Row gutter={16}>
            <Col span={12}>
              <Form.Item
                name="category"
                label="规则分类"
                rules={[{ required: true, message: '请输入规则分类' }]}
              >
                <Input placeholder="请输入规则分类" />
              </Form.Item>
            </Col>
            <Col span={12}>
              <Form.Item
                name="severity"
                label="严重程度"
                rules={[{ required: true, message: '请选择严重程度' }]}
              >
                <Select placeholder="请选择严重程度">
                  <Option value="low">低</Option>
                  <Option value="medium">中</Option>
                  <Option value="high">高</Option>
                  <Option value="critical">严重</Option>
                </Select>
              </Form.Item>
            </Col>
          </Row>
          <Form.Item
            name="description"
            label="规则描述"
            rules={[{ required: true, message: '请输入规则描述' }]}
          >
            <TextArea rows={3} placeholder="请输入规则描述" />
          </Form.Item>
          <Row gutter={16}>
            <Col span={8}>
              <Form.Item
                name="excellentThreshold"
                label="优秀阈值"
                rules={[{ required: true, message: '请输入优秀阈值' }]}
              >
                <InputNumber
                  min={0}
                  max={1}
                  step={0.01}
                  placeholder="0.90"
                  style={{ width: '100%' }}
                />
              </Form.Item>
            </Col>
            <Col span={8}>
              <Form.Item
                name="goodThreshold"
                label="良好阈值"
                rules={[{ required: true, message: '请输入良好阈值' }]}
              >
                <InputNumber
                  min={0}
                  max={1}
                  step={0.01}
                  placeholder="0.80"
                  style={{ width: '100%' }}
                />
              </Form.Item>
            </Col>
            <Col span={8}>
              <Form.Item
                name="acceptableThreshold"
                label="可接受阈值"
                rules={[{ required: true, message: '请输入可接受阈值' }]}
              >
                <InputNumber
                  min={0}
                  max={1}
                  step={0.01}
                  placeholder="0.70"
                  style={{ width: '100%' }}
                />
              </Form.Item>
            </Col>
          </Row>
          <Form.Item name="isActive" label="启用状态" valuePropName="checked">
            <Switch />
          </Form.Item>
        </Form>
      </Modal>
    </PageContainer>
  );
};

export default QualityControlPage;