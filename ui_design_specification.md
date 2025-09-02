# 医学图像AI分析系统用户界面设计规范

## 1. 概述 (Overview)

### 1.1 设计目标

本文档旨在为医学图像AI分析系统设计一套完整的用户界面解决方案，确保系统具有优秀的用户体验、高效的工作流程和直观的操作界面。设计遵循医疗行业标准和用户习惯，同时考虑不同用户群体的特定需求。

### 1.2 设计原则

1. **以用户为中心**: 深入理解医生、技师、管理员等不同用户的工作流程和需求
2. **简洁高效**: 界面简洁明了，操作流程高效，减少用户认知负担
3. **安全可靠**: 确保患者数据安全，操作可追溯，符合医疗行业规范
4. **响应式设计**: 支持多种设备和屏幕尺寸，提供一致的用户体验
5. **可访问性**: 遵循无障碍设计标准，确保所有用户都能正常使用
6. **可扩展性**: 界面设计具有良好的扩展性，便于后续功能迭代

### 1.3 目标用户群体

- **放射科医生**: 主要用户，负责图像诊断和报告生成
- **影像技师**: 负责图像采集、预处理和质量控制
- **临床医生**: 查看AI分析结果，辅助临床决策
- **系统管理员**: 负责系统配置、用户管理和数据维护
- **质控人员**: 负责质量监控和合规性检查

## 2. 整体设计架构 (Overall Design Architecture)

### 2.1 系统架构概览

```
医学图像AI分析系统UI架构
├── Web端应用 (主要平台)
│   ├── 医生工作台 (Physician Workstation)
│   ├── 管理后台 (Admin Dashboard)
│   └── 质控中心 (Quality Control Center)
├── 移动端应用 (Mobile Application)
│   ├── iOS应用
│   └── Android应用
└── 集成接口 (Integration Interface)
    ├── PACS集成界面
    ├── HIS集成界面
    └── 第三方系统接口
```

### 2.2 技术栈选择

#### 2.2.1 前端技术栈
```javascript
// 主要技术栈
const techStack = {
    framework: 'React 18.x',
    stateManagement: 'Redux Toolkit + RTK Query',
    uiLibrary: 'Ant Design + Custom Components',
    styling: 'Styled-components + CSS Modules',
    imageViewer: 'Cornerstone.js + OHIF Viewer',
    charts: 'D3.js + Recharts',
    testing: 'Jest + React Testing Library',
    bundler: 'Vite',
    typeScript: 'TypeScript 5.x'
};

// 移动端技术栈
const mobileStack = {
    framework: 'React Native',
    navigation: 'React Navigation 6.x',
    stateManagement: 'Redux Toolkit',
    uiLibrary: 'NativeBase + Custom Components',
    imageHandling: 'React Native Image Viewer'
};
```

### 2.3 设计系统 (Design System)

#### 2.3.1 色彩系统
```css
/* 主色调 */
:root {
    /* 主色 - 医疗蓝 */
    --primary-50: #e3f2fd;
    --primary-100: #bbdefb;
    --primary-200: #90caf9;
    --primary-300: #64b5f6;
    --primary-400: #42a5f5;
    --primary-500: #2196f3; /* 主色 */
    --primary-600: #1e88e5;
    --primary-700: #1976d2;
    --primary-800: #1565c0;
    --primary-900: #0d47a1;
    
    /* 辅助色 */
    --secondary-500: #ff9800; /* 警告橙 */
    --success-500: #4caf50;   /* 成功绿 */
    --error-500: #f44336;     /* 错误红 */
    --warning-500: #ff9800;   /* 警告橙 */
    --info-500: #2196f3;      /* 信息蓝 */
    
    /* 中性色 */
    --gray-50: #fafafa;
    --gray-100: #f5f5f5;
    --gray-200: #eeeeee;
    --gray-300: #e0e0e0;
    --gray-400: #bdbdbd;
    --gray-500: #9e9e9e;
    --gray-600: #757575;
    --gray-700: #616161;
    --gray-800: #424242;
    --gray-900: #212121;
    
    /* 医疗专用色 */
    --medical-critical: #d32f2f;  /* 危急值红 */
    --medical-abnormal: #f57c00;  /* 异常橙 */
    --medical-normal: #388e3c;    /* 正常绿 */
    --medical-pending: #1976d2;   /* 待处理蓝 */
}
```

#### 2.3.2 字体系统
```css
/* 字体定义 */
:root {
    /* 字体族 */
    --font-family-primary: 'Inter', 'PingFang SC', 'Microsoft YaHei', sans-serif;
    --font-family-mono: 'JetBrains Mono', 'Consolas', monospace;
    
    /* 字体大小 */
    --font-size-xs: 0.75rem;   /* 12px */
    --font-size-sm: 0.875rem;  /* 14px */
    --font-size-base: 1rem;    /* 16px */
    --font-size-lg: 1.125rem;  /* 18px */
    --font-size-xl: 1.25rem;   /* 20px */
    --font-size-2xl: 1.5rem;   /* 24px */
    --font-size-3xl: 1.875rem; /* 30px */
    --font-size-4xl: 2.25rem;  /* 36px */
    
    /* 行高 */
    --line-height-tight: 1.25;
    --line-height-normal: 1.5;
    --line-height-relaxed: 1.75;
    
    /* 字重 */
    --font-weight-normal: 400;
    --font-weight-medium: 500;
    --font-weight-semibold: 600;
    --font-weight-bold: 700;
}
```

#### 2.3.3 间距系统
```css
/* 间距系统 */
:root {
    --spacing-0: 0;
    --spacing-1: 0.25rem;  /* 4px */
    --spacing-2: 0.5rem;   /* 8px */
    --spacing-3: 0.75rem;  /* 12px */
    --spacing-4: 1rem;     /* 16px */
    --spacing-5: 1.25rem;  /* 20px */
    --spacing-6: 1.5rem;   /* 24px */
    --spacing-8: 2rem;     /* 32px */
    --spacing-10: 2.5rem;  /* 40px */
    --spacing-12: 3rem;    /* 48px */
    --spacing-16: 4rem;    /* 64px */
    --spacing-20: 5rem;    /* 80px */
    --spacing-24: 6rem;    /* 96px */
}
```

## 3. 医生工作台设计 (Physician Workstation)

### 3.1 整体布局设计

#### 3.1.1 主界面布局
```jsx
// 医生工作台主界面组件
import React from 'react';
import { Layout, Menu, Avatar, Badge, Dropdown } from 'antd';
import { 
    DashboardOutlined, 
    FileImageOutlined, 
    HistoryOutlined,
    SettingOutlined,
    UserOutlined,
    BellOutlined
} from '@ant-design/icons';

const { Header, Sider, Content } = Layout;

const PhysicianWorkstation = () => {
    return (
        <Layout className="physician-workstation" style={{ minHeight: '100vh' }}>
            {/* 顶部导航栏 */}
            <Header className="workstation-header">
                <div className="header-left">
                    <div className="logo">
                        <img src="/logo.svg" alt="Medical AI" />
                        <span>医学影像AI分析系统</span>
                    </div>
                </div>
                
                <div className="header-center">
                    <div className="current-patient-info">
                        <span className="patient-name">张三</span>
                        <span className="patient-id">ID: 20240101001</span>
                        <span className="study-date">2024-01-15</span>
                    </div>
                </div>
                
                <div className="header-right">
                    <Badge count={5} className="notification-badge">
                        <BellOutlined className="notification-icon" />
                    </Badge>
                    
                    <Dropdown menu={{
                        items: [
                            { key: 'profile', label: '个人资料' },
                            { key: 'settings', label: '系统设置' },
                            { key: 'logout', label: '退出登录' }
                        ]
                    }}>
                        <div className="user-info">
                            <Avatar icon={<UserOutlined />} />
                            <span className="username">李医生</span>
                        </div>
                    </Dropdown>
                </div>
            </Header>
            
            <Layout>
                {/* 左侧导航菜单 */}
                <Sider width={240} className="workstation-sider">
                    <Menu
                        mode="inline"
                        defaultSelectedKeys={['dashboard']}
                        items={[
                            {
                                key: 'dashboard',
                                icon: <DashboardOutlined />,
                                label: '工作台概览'
                            },
                            {
                                key: 'worklist',
                                icon: <FileImageOutlined />,
                                label: '工作列表',
                                children: [
                                    { key: 'pending', label: '待处理' },
                                    { key: 'in-progress', label: '处理中' },
                                    { key: 'completed', label: '已完成' }
                                ]
                            },
                            {
                                key: 'history',
                                icon: <HistoryOutlined />,
                                label: '历史记录'
                            },
                            {
                                key: 'settings',
                                icon: <SettingOutlined />,
                                label: '个人设置'
                            }
                        ]}
                    />
                </Sider>
                
                {/* 主内容区域 */}
                <Content className="workstation-content">
                    {/* 内容区域将根据选中的菜单项动态渲染 */}
                </Content>
            </Layout>
        </Layout>
    );
};

export default PhysicianWorkstation;
```

### 3.2 工作列表界面

#### 3.2.1 工作列表组件
```jsx
// 工作列表组件
import React, { useState, useEffect } from 'react';
import { 
    Table, 
    Tag, 
    Button, 
    Input, 
    Select, 
    DatePicker, 
    Space,
    Tooltip,
    Progress,
    Badge
} from 'antd';
import { 
    SearchOutlined, 
    FilterOutlined, 
    EyeOutlined,
    ClockCircleOutlined,
    CheckCircleOutlined,
    ExclamationCircleOutlined
} from '@ant-design/icons';

const { Search } = Input;
const { Option } = Select;
const { RangePicker } = DatePicker;

const WorkList = () => {
    const [worklistData, setWorklistData] = useState([]);
    const [loading, setLoading] = useState(false);
    const [filters, setFilters] = useState({
        status: 'all',
        modality: 'all',
        priority: 'all',
        dateRange: null
    });
    
    // 表格列定义
    const columns = [
        {
            title: '患者信息',
            key: 'patient',
            width: 200,
            render: (_, record) => (
                <div className="patient-info">
                    <div className="patient-name">{record.patientName}</div>
                    <div className="patient-details">
                        <span className="patient-id">ID: {record.patientId}</span>
                        <span className="patient-age">{record.age}岁</span>
                        <span className="patient-gender">{record.gender}</span>
                    </div>
                </div>
            )
        },
        {
            title: '检查信息',
            key: 'study',
            width: 250,
            render: (_, record) => (
                <div className="study-info">
                    <div className="study-description">{record.studyDescription}</div>
                    <div className="study-details">
                        <Tag color="blue">{record.modality}</Tag>
                        <span className="study-date">{record.studyDate}</span>
                        <span className="series-count">{record.seriesCount}序列</span>
                    </div>
                </div>
            )
        },
        {
            title: '优先级',
            dataIndex: 'priority',
            key: 'priority',
            width: 100,
            render: (priority) => {
                const priorityConfig = {
                    urgent: { color: 'red', text: '紧急' },
                    high: { color: 'orange', text: '高' },
                    normal: { color: 'green', text: '普通' },
                    low: { color: 'gray', text: '低' }
                };
                const config = priorityConfig[priority];
                return <Tag color={config.color}>{config.text}</Tag>;
            }
        },
        {
            title: 'AI分析状态',
            key: 'aiStatus',
            width: 150,
            render: (_, record) => {
                const statusConfig = {
                    pending: { 
                        icon: <ClockCircleOutlined />, 
                        color: 'default', 
                        text: '等待中' 
                    },
                    processing: { 
                        icon: <Progress type="circle" size="small" percent={record.progress} />, 
                        color: 'processing', 
                        text: '分析中' 
                    },
                    completed: { 
                        icon: <CheckCircleOutlined />, 
                        color: 'success', 
                        text: '已完成' 
                    },
                    failed: { 
                        icon: <ExclamationCircleOutlined />, 
                        color: 'error', 
                        text: '失败' 
                    }
                };
                const config = statusConfig[record.aiStatus];
                return (
                    <div className="ai-status">
                        <Badge status={config.color} />
                        {config.icon}
                        <span>{config.text}</span>
                    </div>
                );
            }
        },
        {
            title: '分配医生',
            dataIndex: 'assignedDoctor',
            key: 'assignedDoctor',
            width: 120
        },
        {
            title: '创建时间',
            dataIndex: 'createdAt',
            key: 'createdAt',
            width: 150,
            sorter: true
        },
        {
            title: '操作',
            key: 'actions',
            width: 150,
            render: (_, record) => (
                <Space>
                    <Tooltip title="查看详情">
                        <Button 
                            type="primary" 
                            icon={<EyeOutlined />} 
                            size="small"
                            onClick={() => handleViewStudy(record)}
                        >
                            查看
                        </Button>
                    </Tooltip>
                    {record.aiStatus === 'completed' && (
                        <Button 
                            type="default" 
                            size="small"
                            onClick={() => handleGenerateReport(record)}
                        >
                            生成报告
                        </Button>
                    )}
                </Space>
            )
        }
    ];
    
    const handleViewStudy = (record) => {
        // 跳转到图像查看器
        console.log('查看检查:', record);
    };
    
    const handleGenerateReport = (record) => {
        // 生成诊断报告
        console.log('生成报告:', record);
    };
    
    return (
        <div className="worklist-container">
            {/* 筛选工具栏 */}
            <div className="worklist-toolbar">
                <div className="toolbar-left">
                    <Search
                        placeholder="搜索患者姓名或ID"
                        allowClear
                        style={{ width: 250 }}
                        onSearch={(value) => console.log('搜索:', value)}
                    />
                </div>
                
                <div className="toolbar-right">
                    <Space>
                        <Select
                            value={filters.status}
                            style={{ width: 120 }}
                            onChange={(value) => setFilters({...filters, status: value})}
                        >
                            <Option value="all">全部状态</Option>
                            <Option value="pending">待处理</Option>
                            <Option value="processing">处理中</Option>
                            <Option value="completed">已完成</Option>
                        </Select>
                        
                        <Select
                            value={filters.modality}
                            style={{ width: 120 }}
                            onChange={(value) => setFilters({...filters, modality: value})}
                        >
                            <Option value="all">全部模态</Option>
                            <Option value="CT">CT</Option>
                            <Option value="MRI">MRI</Option>
                            <Option value="X-Ray">X-Ray</Option>
                            <Option value="US">超声</Option>
                        </Select>
                        
                        <RangePicker
                            onChange={(dates) => setFilters({...filters, dateRange: dates})}
                        />
                        
                        <Button icon={<FilterOutlined />}>高级筛选</Button>
                    </Space>
                </div>
            </div>
            
            {/* 工作列表表格 */}
            <Table
                columns={columns}
                dataSource={worklistData}
                loading={loading}
                rowKey="id"
                pagination={{
                    total: 1000,
                    pageSize: 20,
                    showSizeChanger: true,
                    showQuickJumper: true,
                    showTotal: (total, range) => 
                        `第 ${range[0]}-${range[1]} 条，共 ${total} 条记录`
                }}
                scroll={{ y: 600 }}
                className="worklist-table"
            />
        </div>
    );
};

export default WorkList;
```

### 3.3 图像查看器界面

#### 3.3.1 图像查看器组件
```jsx
// 图像查看器组件
import React, { useState, useEffect, useRef } from 'react';
import { 
    Layout, 
    Tabs, 
    Button, 
    Slider, 
    Select, 
    Tooltip, 
    Drawer,
    Card,
    List,
    Tag,
    Progress,
    Space
} from 'antd';
import {
    ZoomInOutlined,
    ZoomOutOutlined,
    RotateLeftOutlined,
    RotateRightOutlined,
    FullscreenOutlined,
    SettingOutlined,
    EyeOutlined,
    FileTextOutlined,
    HistoryOutlined
} from '@ant-design/icons';

const { Sider, Content } = Layout;
const { TabPane } = Tabs;
const { Option } = Select;

const ImageViewer = ({ studyData }) => {
    const viewerRef = useRef(null);
    const [currentSeries, setCurrentSeries] = useState(0);
    const [currentImage, setCurrentImage] = useState(0);
    const [viewerSettings, setViewerSettings] = useState({
        windowWidth: 400,
        windowCenter: 40,
        zoom: 1.0,
        rotation: 0,
        invert: false
    });
    const [aiResults, setAiResults] = useState(null);
    const [showAiOverlay, setShowAiOverlay] = useState(true);
    const [sidebarVisible, setSidebarVisible] = useState(true);
    
    // AI分析结果组件
    const AIResultsPanel = () => (
        <Card title="AI分析结果" size="small" className="ai-results-panel">
            {aiResults ? (
                <div className="ai-results-content">
                    <div className="confidence-score">
                        <span>置信度: </span>
                        <Progress 
                            percent={aiResults.confidence * 100} 
                            size="small" 
                            status={aiResults.confidence > 0.8 ? 'success' : 'normal'}
                        />
                    </div>
                    
                    <div className="findings-list">
                        <h4>发现:</h4>
                        <List
                            size="small"
                            dataSource={aiResults.findings}
                            renderItem={(finding) => (
                                <List.Item>
                                    <div className="finding-item">
                                        <Tag color={finding.severity === 'high' ? 'red' : 
                                                   finding.severity === 'medium' ? 'orange' : 'green'}>
                                            {finding.category}
                                        </Tag>
                                        <span className="finding-description">{finding.description}</span>
                                        <span className="finding-confidence">({(finding.confidence * 100).toFixed(1)}%)</span>
                                    </div>
                                </List.Item>
                            )}
                        />
                    </div>
                    
                    <div className="recommendations">
                        <h4>建议:</h4>
                        <ul>
                            {aiResults.recommendations.map((rec, index) => (
                                <li key={index}>{rec}</li>
                            ))}
                        </ul>
                    </div>
                </div>
            ) : (
                <div className="ai-loading">
                    <Progress type="circle" percent={75} />
                    <p>AI分析中...</p>
                </div>
            )}
        </Card>
    );
    
    // 图像工具栏组件
    const ImageToolbar = () => (
        <div className="image-toolbar">
            <div className="toolbar-group">
                <Tooltip title="放大">
                    <Button 
                        icon={<ZoomInOutlined />} 
                        onClick={() => handleZoom(1.2)}
                    />
                </Tooltip>
                <Tooltip title="缩小">
                    <Button 
                        icon={<ZoomOutOutlined />} 
                        onClick={() => handleZoom(0.8)}
                    />
                </Tooltip>
                <Tooltip title="重置">
                    <Button onClick={handleReset}>重置</Button>
                </Tooltip>
            </div>
            
            <div className="toolbar-group">
                <Tooltip title="左旋转">
                    <Button 
                        icon={<RotateLeftOutlined />} 
                        onClick={() => handleRotate(-90)}
                    />
                </Tooltip>
                <Tooltip title="右旋转">
                    <Button 
                        icon={<RotateRightOutlined />} 
                        onClick={() => handleRotate(90)}
                    />
                </Tooltip>
            </div>
            
            <div className="toolbar-group">
                <span>窗宽: </span>
                <Slider
                    min={1}
                    max={2000}
                    value={viewerSettings.windowWidth}
                    onChange={(value) => handleWindowChange('width', value)}
                    style={{ width: 100 }}
                />
            </div>
            
            <div className="toolbar-group">
                <span>窗位: </span>
                <Slider
                    min={-1000}
                    max={1000}
                    value={viewerSettings.windowCenter}
                    onChange={(value) => handleWindowChange('center', value)}
                    style={{ width: 100 }}
                />
            </div>
            
            <div className="toolbar-group">
                <Button 
                    type={showAiOverlay ? 'primary' : 'default'}
                    icon={<EyeOutlined />}
                    onClick={() => setShowAiOverlay(!showAiOverlay)}
                >
                    AI标注
                </Button>
                
                <Tooltip title="全屏">
                    <Button 
                        icon={<FullscreenOutlined />} 
                        onClick={handleFullscreen}
                    />
                </Tooltip>
                
                <Tooltip title="设置">
                    <Button 
                        icon={<SettingOutlined />} 
                        onClick={() => setSidebarVisible(!sidebarVisible)}
                    />
                </Tooltip>
            </div>
        </div>
    );
    
    const handleZoom = (factor) => {
        setViewerSettings(prev => ({
            ...prev,
            zoom: prev.zoom * factor
        }));
    };
    
    const handleRotate = (angle) => {
        setViewerSettings(prev => ({
            ...prev,
            rotation: (prev.rotation + angle) % 360
        }));
    };
    
    const handleReset = () => {
        setViewerSettings({
            windowWidth: 400,
            windowCenter: 40,
            zoom: 1.0,
            rotation: 0,
            invert: false
        });
    };
    
    const handleWindowChange = (type, value) => {
        setViewerSettings(prev => ({
            ...prev,
            [type === 'width' ? 'windowWidth' : 'windowCenter']: value
        }));
    };
    
    const handleFullscreen = () => {
        if (viewerRef.current) {
            viewerRef.current.requestFullscreen();
        }
    };
    
    return (
        <Layout className="image-viewer-layout">
            <Content className="viewer-content">
                {/* 图像工具栏 */}
                <ImageToolbar />
                
                {/* 主图像显示区域 */}
                <div className="image-display-area" ref={viewerRef}>
                    <div className="image-container">
                        {/* 这里集成Cornerstone.js或OHIF Viewer */}
                        <div className="cornerstone-viewport" id="dicomImage">
                            {/* DICOM图像将在这里渲染 */}
                        </div>
                        
                        {/* AI标注覆盖层 */}
                        {showAiOverlay && aiResults && (
                            <div className="ai-overlay">
                                {/* AI检测结果的可视化标注 */}
                            </div>
                        )}
                    </div>
                    
                    {/* 序列缩略图 */}
                    <div className="series-thumbnails">
                        {studyData?.series?.map((series, index) => (
                            <div 
                                key={series.id}
                                className={`thumbnail ${index === currentSeries ? 'active' : ''}`}
                                onClick={() => setCurrentSeries(index)}
                            >
                                <img src={series.thumbnail} alt={`Series ${index + 1}`} />
                                <span className="series-info">{series.description}</span>
                            </div>
                        ))}
                    </div>
                </div>
            </Content>
            
            {/* 右侧信息面板 */}
            {sidebarVisible && (
                <Sider width={350} className="viewer-sidebar">
                    <Tabs defaultActiveKey="ai-results">
                        <TabPane tab="AI分析" key="ai-results">
                            <AIResultsPanel />
                        </TabPane>
                        
                        <TabPane tab="患者信息" key="patient-info">
                            <Card title="患者信息" size="small">
                                <div className="patient-details">
                                    <p><strong>姓名:</strong> {studyData?.patient?.name}</p>
                                    <p><strong>ID:</strong> {studyData?.patient?.id}</p>
                                    <p><strong>性别:</strong> {studyData?.patient?.gender}</p>
                                    <p><strong>年龄:</strong> {studyData?.patient?.age}</p>
                                    <p><strong>检查日期:</strong> {studyData?.studyDate}</p>
                                    <p><strong>检查部位:</strong> {studyData?.bodyPart}</p>
                                </div>
                            </Card>
                        </TabPane>
                        
                        <TabPane tab="检查信息" key="study-info">
                            <Card title="检查信息" size="small">
                                <div className="study-details">
                                    <p><strong>检查描述:</strong> {studyData?.description}</p>
                                    <p><strong>设备型号:</strong> {studyData?.equipment}</p>
                                    <p><strong>扫描参数:</strong> {studyData?.parameters}</p>
                                    <p><strong>序列数量:</strong> {studyData?.series?.length}</p>
                                </div>
                            </Card>
                        </TabPane>
                        
                        <TabPane tab="历史记录" key="history">
                            <Card title="历史检查" size="small">
                                <List
                                    size="small"
                                    dataSource={studyData?.history || []}
                                    renderItem={(item) => (
                                        <List.Item>
                                            <div className="history-item">
                                                <div className="history-date">{item.date}</div>
                                                <div className="history-description">{item.description}</div>
                                            </div>
                                        </List.Item>
                                    )}
                                />
                            </Card>
                        </TabPane>
                    </Tabs>
                </Sider>
            )}
        </Layout>
    );
};

export default ImageViewer;
```

### 3.4 报告生成界面

#### 3.4.1 智能报告编辑器
```jsx
// 智能报告编辑器组件
import React, { useState, useEffect } from 'react';
import { 
    Layout, 
    Card, 
    Button, 
    Input, 
    Select, 
    Tabs, 
    Tag,
    Space,
    Modal,
    message,
    Divider,
    Tooltip,
    Progress
} from 'antd';
import {
    SaveOutlined,
    PrinterOutlined,
    SendOutlined,
    RobotOutlined,
    EditOutlined,
    HistoryOutlined,
    CheckCircleOutlined
} from '@ant-design/icons';

const { TextArea } = Input;
const { Option } = Select;
const { TabPane } = Tabs;

const ReportEditor = ({ studyData, aiResults }) => {
    const [reportData, setReportData] = useState({
        findings: '',
        impression: '',
        recommendations: '',
        status: 'draft'
    });
    const [aiSuggestions, setAiSuggestions] = useState(null);
    const [isGeneratingReport, setIsGeneratingReport] = useState(false);
    const [templateModalVisible, setTemplateModalVisible] = useState(false);
    const [reportTemplates, setReportTemplates] = useState([]);
    
    // AI报告生成
    const generateAIReport = async () => {
        setIsGeneratingReport(true);
        try {
            // 调用AI报告生成API
            const response = await fetch('/api/ai/generate-report', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    studyId: studyData.id,
                    aiResults: aiResults,
                    template: 'standard'
                })
            });
            
            const aiReport = await response.json();
            setReportData({
                findings: aiReport.findings,
                impression: aiReport.impression,
                recommendations: aiReport.recommendations,
                status: 'ai_generated'
            });
            
            message.success('AI报告生成成功');
        } catch (error) {
            message.error('AI报告生成失败');
        } finally {
            setIsGeneratingReport(false);
        }
    };
    
    // 报告模板选择
    const ReportTemplateModal = () => (
        <Modal
            title="选择报告模板"
            visible={templateModalVisible}
            onCancel={() => setTemplateModalVisible(false)}
            footer={null}
            width={800}
        >
            <div className="template-grid">
                {reportTemplates.map(template => (
                    <Card
                        key={template.id}
                        hoverable
                        className="template-card"
                        onClick={() => applyTemplate(template)}
                    >
                        <div className="template-header">
                            <h4>{template.name}</h4>
                            <Tag color={template.category === 'CT' ? 'blue' : 'green'}>
                                {template.category}
                            </Tag>
                        </div>
                        <p className="template-description">{template.description}</p>
                        <div className="template-usage">
                            <span>使用次数: {template.usageCount}</span>
                        </div>
                    </Card>
                ))}
            </div>
        </Modal>
    );
    
    const applyTemplate = (template) => {
        setReportData({
            findings: template.findingsTemplate,
            impression: template.impressionTemplate,
            recommendations: template.recommendationsTemplate,
            status: 'template_applied'
        });
        setTemplateModalVisible(false);
        message.success('模板应用成功');
    };
    
    // 保存报告
    const saveReport = async (status = 'draft') => {
        try {
            await fetch('/api/reports/save', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    studyId: studyData.id,
                    ...reportData,
                    status
                })
            });
            
            setReportData(prev => ({ ...prev, status }));
            message.success('报告保存成功');
        } catch (error) {
            message.error('报告保存失败');
        }
    };
    
    // 提交审核
    const submitForReview = () => {
        Modal.confirm({
            title: '确认提交审核',
            content: '提交后报告将无法修改，确认提交吗？',
            onOk: () => saveReport('submitted')
        });
    };
    
    return (
        <Layout className="report-editor-layout">
            <div className="report-header">
                <div className="header-left">
                    <h2>诊断报告</h2>
                    <Tag color={reportData.status === 'draft' ? 'orange' : 
                              reportData.status === 'submitted' ? 'blue' : 'green'}>
                        {reportData.status === 'draft' ? '草稿' :
                         reportData.status === 'submitted' ? '已提交' : '已完成'}
                    </Tag>
                </div>
                
                <div className="header-right">
                    <Space>
                        <Button 
                            icon={<RobotOutlined />}
                            loading={isGeneratingReport}
                            onClick={generateAIReport}
                        >
                            AI生成报告
                        </Button>
                        
                        <Button 
                            icon={<EditOutlined />}
                            onClick={() => setTemplateModalVisible(true)}
                        >
                            使用模板
                        </Button>
                        
                        <Button 
                            icon={<SaveOutlined />}
                            onClick={() => saveReport('draft')}
                        >
                            保存草稿
                        </Button>
                        
                        <Button 
                            type="primary"
                            icon={<CheckCircleOutlined />}
                            onClick={submitForReview}
                            disabled={reportData.status !== 'draft'}
                        >
                            提交审核
                        </Button>
                        
                        <Button 
                            icon={<PrinterOutlined />}
                            onClick={() => window.print()}
                        >
                            打印
                        </Button>
                    </Space>
                </div>
            </div>
            
            <div className="report-content">
                <div className="report-main">
                    <Card title="患者基本信息" className="patient-info-card">
                        <div className="patient-info-grid">
                            <div className="info-item">
                                <label>患者姓名:</label>
                                <span>{studyData.patient.name}</span>
                            </div>
                            <div className="info-item">
                                <label>患者ID:</label>
                                <span>{studyData.patient.id}</span>
                            </div>
                            <div className="info-item">
                                <label>性别:</label>
                                <span>{studyData.patient.gender}</span>
                            </div>
                            <div className="info-item">
                                <label>年龄:</label>
                                <span>{studyData.patient.age}</span>
                            </div>
                            <div className="info-item">
                                <label>检查日期:</label>
                                <span>{studyData.studyDate}</span>
                            </div>
                            <div className="info-item">
                                <label>检查部位:</label>
                                <span>{studyData.bodyPart}</span>
                            </div>
                        </div>
                    </Card>
                    
                    <Card title="影像所见" className="findings-card">
                        <TextArea
                            value={reportData.findings}
                            onChange={(e) => setReportData(prev => ({
                                ...prev,
                                findings: e.target.value
                            }))}
                            placeholder="请描述影像学发现..."
                            rows={8}
                            className="report-textarea"
                        />
                        
                        {aiSuggestions?.findings && (
                            <div className="ai-suggestions">
                                <Divider orientation="left">AI建议</Divider>
                                <div className="suggestion-content">
                                    {aiSuggestions.findings}
                                </div>
                                <Button 
                                    size="small" 
                                    type="link"
                                    onClick={() => setReportData(prev => ({
                                        ...prev,
                                        findings: aiSuggestions.findings
                                    }))}
                                >
                                    采用建议
                                </Button>
                            </div>
                        )}
                    </Card>
                    
                    <Card title="诊断意见" className="impression-card">
                        <TextArea
                            value={reportData.impression}
                            onChange={(e) => setReportData(prev => ({
                                ...prev,
                                impression: e.target.value
                            }))}
                            placeholder="请输入诊断意见..."
                            rows={4}
                            className="report-textarea"
                        />
                    </Card>
                    
                    <Card title="建议" className="recommendations-card">
                        <TextArea
                            value={reportData.recommendations}
                            onChange={(e) => setReportData(prev => ({
                                ...prev,
                                recommendations: e.target.value
                            }))}
                            placeholder="请输入建议..."
                            rows={3}
                            className="report-textarea"
                        />
                    </Card>
                </div>
                
                <div className="report-sidebar">
                    <Card title="AI分析结果" size="small">
                        {aiResults ? (
                            <div className="ai-results-summary">
                                <div className="confidence-display">
                                    <span>整体置信度:</span>
                                    <Progress 
                                        type="circle" 
                                        percent={aiResults.overallConfidence * 100}
                                        width={60}
                                    />
                                </div>
                                
                                <div className="key-findings">
                                    <h4>关键发现:</h4>
                                    {aiResults.keyFindings.map((finding, index) => (
                                        <Tag 
                                            key={index}
                                            color={finding.severity === 'high' ? 'red' : 'orange'}
                                            className="finding-tag"
                                        >
                                            {finding.name}
                                        </Tag>
                                    ))}
                                </div>
                                
                                <Button 
                                    type="link" 
                                    size="small"
                                    onClick={() => setAiSuggestions(aiResults.suggestions)}
                                >
                                    查看AI建议
                                </Button>
                            </div>
                        ) : (
                            <div className="no-ai-results">
                                <p>暂无AI分析结果</p>
                            </div>
                        )}
                    </Card>
                    
                    <Card title="报告历史" size="small">
                        <div className="report-history">
                            <div className="history-item">
                                <span className="history-time">2024-01-15 10:30</span>
                                <span className="history-action">创建草稿</span>
                            </div>
                            <div className="history-item">
                                <span className="history-time">2024-01-15 11:15</span>
                                <span className="history-action">AI生成内容</span>
                            </div>
                            <div className="history-item">
                                <span className="history-time">2024-01-15 11:45</span>
                                <span className="history-action">医生修改</span>
                            </div>
                        </div>
                    </Card>
                </div>
            </div>
            
            <ReportTemplateModal />
        </Layout>
    );
};

export default ReportEditor;
```

## 4. 管理后台设计 (Admin Dashboard)

### 4.1 管理后台主界面

#### 4.1.1 仪表板概览
```jsx
// 管理后台仪表板组件
import React, { useState, useEffect } from 'react';
import { 
    Layout, 
    Card, 
    Row, 
    Col, 
    Statistic, 
    Table, 
    Progress,
    List,
    Tag,
    Button,
    DatePicker,
    Select,
    Space
} from 'antd';
import {
    UserOutlined,
    FileImageOutlined,
    ClockCircleOutlined,
    CheckCircleOutlined,
    ExclamationCircleOutlined,
    TrendingUpOutlined,
    DownloadOutlined
} from '@ant-design/icons';
import { Line, Column, Pie } from '@ant-design/plots';

const { RangePicker } = DatePicker;
const { Option } = Select;

const AdminDashboard = () => {
    const [dashboardData, setDashboardData] = useState(null);
    const [dateRange, setDateRange] = useState(null);
    const [loading, setLoading] = useState(true);
    
    // 系统统计数据
    const systemStats = {
        totalUsers: 156,
        activeUsers: 89,
        totalStudies: 12450,
        todayStudies: 234,
        aiProcessed: 11890,
        aiAccuracy: 94.5,
        systemUptime: 99.8,
        avgProcessingTime: 45
    };
    
    // 处理量趋势数据
    const processingTrendData = [
        { date: '2024-01-01', studies: 180, aiProcessed: 165 },
        { date: '2024-01-02', studies: 220, aiProcessed: 198 },
        { date: '2024-01-03', studies: 195, aiProcessed: 185 },
        { date: '2024-01-04', studies: 240, aiProcessed: 225 },
        { date: '2024-01-05', studies: 210, aiProcessed: 195 },
        { date: '2024-01-06', studies: 185, aiProcessed: 170 },
        { date: '2024-01-07', studies: 260, aiProcessed: 245 }
    ];
    
    // AI性能数据
    const aiPerformanceData = [
        { category: 'CT胸部', accuracy: 96.2, processed: 3450 },
        { category: 'MRI脑部', accuracy: 94.8, processed: 2890 },
        { category: 'X光胸片', accuracy: 92.5, processed: 4120 },
        { category: '超声心脏', accuracy: 89.3, processed: 1980 }
    ];
    
    // 用户活跃度数据
    const userActivityData = [
        { type: '放射科医生', count: 45, percentage: 52 },
        { type: '影像技师', count: 28, percentage: 32 },
        { type: '临床医生', count: 12, percentage: 14 },
        { type: '管理员', count: 4, percentage: 2 }
    ];
    
    // 处理量趋势图配置
    const trendChartConfig = {
        data: processingTrendData,
        xField: 'date',
        yField: 'studies',
        seriesField: 'type',
        smooth: true,
        animation: {
            appear: {
                animation: 'path-in',
                duration: 1000
            }
        }
    };
    
    // AI性能柱状图配置
    const performanceChartConfig = {
        data: aiPerformanceData,
        xField: 'category',
        yField: 'accuracy',
        columnStyle: {
            radius: [4, 4, 0, 0]
        },
        meta: {
            accuracy: {
                alias: '准确率(%)',
                min: 80,
                max: 100
            }
        }
    };
    
    // 用户分布饼图配置
    const userDistributionConfig = {
        data: userActivityData,
        angleField: 'count',
        colorField: 'type',
        radius: 0.8,
        label: {
            type: 'outer',
            content: '{name} {percentage}%'
        },
        interactions: [
            {
                type: 'element-active'
            }
        ]
    };
    
    return (
        <div className="admin-dashboard">
            {/* 页面头部 */}
            <div className="dashboard-header">
                <div className="header-left">
                    <h1>系统概览</h1>
                    <p>医学图像AI分析系统管理后台</p>
                </div>
                
                <div className="header-right">
                    <Space>
                        <RangePicker 
                            onChange={setDateRange}
                            placeholder={['开始日期', '结束日期']}
                        />
                        <Select defaultValue="today" style={{ width: 120 }}>
                            <Option value="today">今日</Option>
                            <Option value="week">本周</Option>
                            <Option value="month">本月</Option>
                            <Option value="quarter">本季度</Option>
                        </Select>
                        <Button icon={<DownloadOutlined />}>导出报告</Button>
                    </Space>
                </div>
            </div>
            
            {/* 关键指标卡片 */}
            <Row gutter={[16, 16]} className="stats-cards">
                <Col xs={24} sm={12} lg={6}>
                    <Card>
                        <Statistic
                            title="总用户数"
                            value={systemStats.totalUsers}
                            prefix={<UserOutlined />}
                            suffix="人"
                            valueStyle={{ color: '#1890ff' }}
                        />
                        <div className="stat-extra">
                            <span className="stat-trend positive">
                                <TrendingUpOutlined /> +12%
                            </span>
                            <span className="stat-period">较上月</span>
                        </div>
                    </Card>
                </Col>
                
                <Col xs={24} sm={12} lg={6}>
                    <Card>
                        <Statistic
                            title="今日检查"
                            value={systemStats.todayStudies}
                            prefix={<FileImageOutlined />}
                            suffix="例"
                            valueStyle={{ color: '#52c41a' }}
                        />
                        <div className="stat-extra">
                            <span className="stat-trend positive">
                                <TrendingUpOutlined /> +8%
                            </span>
                            <span className="stat-period">较昨日</span>
                        </div>
                    </Card>
                </Col>
                
                <Col xs={24} sm={12} lg={6}>
                    <Card>
                        <Statistic
                            title="AI准确率"
                            value={systemStats.aiAccuracy}
                            suffix="%"
                            precision={1}
                            valueStyle={{ color: '#722ed1' }}
                        />
                        <Progress 
                            percent={systemStats.aiAccuracy} 
                            size="small" 
                            showInfo={false}
                            strokeColor="#722ed1"
                        />
                    </Card>
                </Col>
                
                <Col xs={24} sm={12} lg={6}>
                    <Card>
                        <Statistic
                            title="系统可用性"
                            value={systemStats.systemUptime}
                            suffix="%"
                            precision={1}
                            valueStyle={{ color: '#fa8c16' }}
                        />
                        <div className="stat-extra">
                            <span className="uptime-status online">● 运行正常</span>
                        </div>
                    </Card>
                </Col>
            </Row>
            
            {/* 图表区域 */}
            <Row gutter={[16, 16]} className="charts-section">
                <Col xs={24} lg={16}>
                    <Card title="处理量趋势" className="chart-card">
                        <Line {...trendChartConfig} height={300} />
                    </Card>
                </Col>
                
                <Col xs={24} lg={8}>
                    <Card title="用户分布" className="chart-card">
                        <Pie {...userDistributionConfig} height={300} />
                    </Card>
                </Col>
            </Row>
            
            <Row gutter={[16, 16]}>
                <Col xs={24} lg={12}>
                    <Card title="AI性能统计" className="chart-card">
                        <Column {...performanceChartConfig} height={250} />
                    </Card>
                </Col>
                
                <Col xs={24} lg={12}>
                    <Card title="实时活动" className="activity-card">
                        <List
                            size="small"
                            dataSource={[
                                {
                                    id: 1,
                                    user: '张医生',
                                    action: '完成CT胸部检查报告',
                                    time: '2分钟前',
                                    type: 'success'
                                },
                                {
                                    id: 2,
                                    user: 'AI系统',
                                    action: '处理MRI脑部扫描',
                                    time: '5分钟前',
                                    type: 'processing'
                                },
                                {
                                    id: 3,
                                    user: '李技师',
                                    action: '上传X光胸片',
                                    time: '8分钟前',
                                    type: 'info'
                                },
                                {
                                    id: 4,
                                    user: '系统',
                                    action: '数据备份完成',
                                    time: '15分钟前',
                                    type: 'success'
                                },
                                {
                                    id: 5,
                                    user: '王医生',
                                    action: '审核超声心脏报告',
                                    time: '20分钟前',
                                    type: 'warning'
                                }
                            ]}
                            renderItem={(item) => (
                                <List.Item>
                                    <div className="activity-item">
                                        <div className="activity-content">
                                            <span className="activity-user">{item.user}</span>
                                            <span className="activity-action">{item.action}</span>
                                        </div>
                                        <div className="activity-meta">
                                            <Tag color={item.type === 'success' ? 'green' :
                                                      item.type === 'warning' ? 'orange' :
                                                      item.type === 'processing' ? 'blue' : 'default'}>
                                                {item.type === 'success' ? '完成' :
                                                 item.type === 'warning' ? '审核' :
                                                 item.type === 'processing' ? '处理中' : '信息'}
                                            </Tag>
                                            <span className="activity-time">{item.time}</span>
                                        </div>
                                    </div>
                                </List.Item>
                            )}
                        />
                    </Card>
                </Col>
            </Row>
            
            {/* 快速操作区域 */}
            <Row gutter={[16, 16]}>
                <Col xs={24}>
                    <Card title="快速操作" className="quick-actions-card">
                        <Space size="large">
                            <Button type="primary" icon={<UserOutlined />}>
                                用户管理
                            </Button>
                            <Button icon={<FileImageOutlined />}>
                                数据管理
                            </Button>
                            <Button icon={<SettingOutlined />}>
                                系统配置
                            </Button>
                            <Button icon={<ExclamationCircleOutlined />}>
                                告警管理
                            </Button>
                            <Button icon={<DownloadOutlined />}>
                                导出报告
                            </Button>
                        </Space>
                    </Card>
                </Col>
            </Row>
        </div>
    );
};

export default AdminDashboard;
```

### 4.2 用户管理界面

#### 4.2.1 用户管理组件
```jsx
// 用户管理组件
import React, { useState, useEffect } from 'react';
import { 
    Table, 
    Button, 
    Modal, 
    Form, 
    Input, 
    Select, 
    Switch,
    Tag,
    Space,
    message,
    Popconfirm,
    Avatar,
    Badge
} from 'antd';
import {
    PlusOutlined,
    EditOutlined,
    DeleteOutlined,
    UserOutlined,
    LockOutlined,
    UnlockOutlined
} from '@ant-design/icons';

const { Option } = Select;

const UserManagement = () => {
    const [users, setUsers] = useState([]);
    const [loading, setLoading] = useState(false);
    const [modalVisible, setModalVisible] = useState(false);
    const [editingUser, setEditingUser] = useState(null);
    const [form] = Form.useForm();
    
    // 用户角色配置
    const userRoles = {
        admin: { label: '系统管理员', color: 'red' },
        doctor: { label: '医生', color: 'blue' },
        technician: { label: '技师', color: 'green' },
        viewer: { label: '查看者', color: 'gray' }
    };
    
    // 表格列定义
    const columns = [
        {
            title: '用户信息',
            key: 'userInfo',
            render: (_, record) => (
                <div className="user-info">
                    <Avatar 
                        src={record.avatar} 
                        icon={<UserOutlined />}
                        size="large"
                    />
                    <div className="user-details">
                        <div className="user-name">{record.name}</div>
                        <div className="user-email">{record.email}</div>
                    </div>
                </div>
            )
        },
        {
            title: '角色',
            dataIndex: 'role',
            key: 'role',
            render: (role) => (
                <Tag color={userRoles[role]?.color}>
                    {userRoles[role]?.label}
                </Tag>
            )
        },
        {
            title: '部门',
            dataIndex: 'department',
            key: 'department'
        },
        {
            title: '状态',
            key: 'status',
            render: (_, record) => (
                <div className="user-status">
                    <Badge 
                        status={record.isActive ? 'success' : 'default'}
                        text={record.isActive ? '活跃' : '禁用'}
                    />
                    {record.isOnline && (
                        <Tag color="green" size="small">在线</Tag>
                    )}
                </div>
            )
        },
        {
            title: '最后登录',
            dataIndex: 'lastLogin',
            key: 'lastLogin'
        },
        {
            title: '操作',
            key: 'actions',
            render: (_, record) => (
                <Space>
                    <Button 
                        icon={<EditOutlined />}
                        size="small"
                        onClick={() => handleEdit(record)}
                    >
                        编辑
                    </Button>
                    
                    <Button 
                        icon={record.isActive ? <LockOutlined /> : <UnlockOutlined />}
                        size="small"
                        onClick={() => handleToggleStatus(record)}
                    >
                        {record.isActive ? '禁用' : '启用'}
                    </Button>
                    
                    <Popconfirm
                        title="确定删除此用户吗？"
                        onConfirm={() => handleDelete(record.id)}
                    >
                        <Button 
                            icon={<DeleteOutlined />}
                            size="small"
                            danger
                        >
                            删除
                        </Button>
                    </Popconfirm>
                </Space>
            )
        }
    ];
    
    const handleEdit = (user) => {
        setEditingUser(user);
        form.setFieldsValue(user);
        setModalVisible(true);
    };
    
    const handleAdd = () => {
        setEditingUser(null);
        form.resetFields();
        setModalVisible(true);
    };
    
    const handleSave = async (values) => {
        try {
            if (editingUser) {
                // 更新用户
                await updateUser(editingUser.id, values);
                message.success('用户更新成功');
            } else {
                // 创建用户
                await createUser(values);
                message.success('用户创建成功');
            }
            setModalVisible(false);
            loadUsers();
        } catch (error) {
            message.error('操作失败');
        }
    };
    
    return (
        <div className="user-management">
            <div className="page-header">
                <h2>用户管理</h2>
                <Button 
                    type="primary" 
                    icon={<PlusOutlined />}
                    onClick={handleAdd}
                >
                    添加用户
                </Button>
            </div>
            
            <Table
                columns={columns}
                dataSource={users}
                loading={loading}
                rowKey="id"
                pagination={{
                    showSizeChanger: true,
                    showQuickJumper: true
                }}
            />
            
            {/* 用户编辑模态框 */}
            <Modal
                title={editingUser ? '编辑用户' : '添加用户'}
                visible={modalVisible}
                onCancel={() => setModalVisible(false)}
                footer={null}
            >
                <Form
                    form={form}
                    layout="vertical"
                    onFinish={handleSave}
                >
                    <Form.Item
                        name="name"
                        label="姓名"
                        rules={[{ required: true, message: '请输入姓名' }]}
                    >
                        <Input />
                    </Form.Item>
                    
                    <Form.Item
                        name="email"
                        label="邮箱"
                        rules={[
                            { required: true, message: '请输入邮箱' },
                            { type: 'email', message: '邮箱格式不正确' }
                        ]}
                    >
                        <Input />
                    </Form.Item>
                    
                    <Form.Item
                        name="role"
                        label="角色"
                        rules={[{ required: true, message: '请选择角色' }]}
                    >
                        <Select>
                            {Object.entries(userRoles).map(([key, value]) => (
                                <Option key={key} value={key}>
                                    {value.label}
                                </Option>
                            ))}
                        </Select>
                    </Form.Item>
                    
                    <Form.Item
                        name="department"
                        label="部门"
                        rules={[{ required: true, message: '请输入部门' }]}
                    >
                        <Input />
                    </Form.Item>
                    
                    <Form.Item
                        name="isActive"
                        label="状态"
                        valuePropName="checked"
                    >
                        <Switch checkedChildren="启用" unCheckedChildren="禁用" />
                    </Form.Item>
                    
                    <Form.Item>
                        <Space>
                            <Button type="primary" htmlType="submit">
                                保存
                            </Button>
                            <Button onClick={() => setModalVisible(false)}>
                                取消
                            </Button>
                        </Space>
                    </Form.Item>
                </Form>
            </Modal>
        </div>
    );
};

export default UserManagement;
```

## 5. 移动端应用设计 (Mobile Application)

### 5.1 移动端整体架构

#### 5.1.1 移动端主导航
```jsx
// 移动端主应用组件
import React from 'react';
import { NavigationContainer } from '@react-navigation/native';
import { createBottomTabNavigator } from '@react-navigation/bottom-tabs';
import { createStackNavigator } from '@react-navigation/stack';
import Icon from 'react-native-vector-icons/MaterialIcons';

// 导入页面组件
import HomeScreen from './screens/HomeScreen';
import WorklistScreen from './screens/WorklistScreen';
import StudyViewerScreen from './screens/StudyViewerScreen';
import ProfileScreen from './screens/ProfileScreen';
import NotificationsScreen from './screens/NotificationsScreen';

const Tab = createBottomTabNavigator();
const Stack = createStackNavigator();

// 工作列表堆栈导航
const WorklistStack = () => (
    <Stack.Navigator>
        <Stack.Screen 
            name="WorklistMain" 
            component={WorklistScreen}
            options={{ title: '工作列表' }}
        />
        <Stack.Screen 
            name="StudyViewer" 
            component={StudyViewerScreen}
            options={{ title: '图像查看' }}
        />
    </Stack.Navigator>
);

// 主标签导航
const MobileApp = () => {
    return (
        <NavigationContainer>
            <Tab.Navigator
                screenOptions={({ route }) => ({
                    tabBarIcon: ({ focused, color, size }) => {
                        let iconName;
                        
                        switch (route.name) {
                            case 'Home':
                                iconName = 'dashboard';
                                break;
                            case 'Worklist':
                                iconName = 'list';
                                break;
                            case 'Notifications':
                                iconName = 'notifications';
                                break;
                            case 'Profile':
                                iconName = 'person';
                                break;
                            default:
                                iconName = 'help';
                        }
                        
                        return <Icon name={iconName} size={size} color={color} />;
                    },
                    tabBarActiveTintColor: '#2196F3',
                    tabBarInactiveTintColor: 'gray',
                    headerShown: false
                })}
            >
                <Tab.Screen 
                    name="Home" 
                    component={HomeScreen}
                    options={{ title: '首页' }}
                />
                <Tab.Screen 
                    name="Worklist" 
                    component={WorklistStack}
                    options={{ title: '工作列表' }}
                />
                <Tab.Screen 
                    name="Notifications" 
                    component={NotificationsScreen}
                    options={{ title: '通知' }}
                />
                <Tab.Screen 
                    name="Profile" 
                    component={ProfileScreen}
                    options={{ title: '我的' }}
                />
            </Tab.Navigator>
        </NavigationContainer>
    );
};

export default MobileApp;
```

### 5.2 移动端首页

#### 5.2.1 首页组件
```jsx
// 移动端首页组件
import React, { useState, useEffect } from 'react';
import {
    View,
    Text,
    ScrollView,
    StyleSheet,
    TouchableOpacity,
    RefreshControl,
    Dimensions
} from 'react-native';
import {
    Card,
    Avatar,
    Badge,
    Progress,
    Button
} from 'react-native-elements';
import Icon from 'react-native-vector-icons/MaterialIcons';

const { width } = Dimensions.get('window');

const HomeScreen = ({ navigation }) => {
    const [refreshing, setRefreshing] = useState(false);
    const [dashboardData, setDashboardData] = useState({
        pendingStudies: 12,
        completedToday: 8,
        aiProcessing: 3,
        notifications: 5
    });
    
    const onRefresh = async () => {
        setRefreshing(true);
        // 刷新数据逻辑
        setTimeout(() => setRefreshing(false), 1000);
    };
    
    // 快速操作按钮
    const QuickActions = () => (
        <View style={styles.quickActions}>
            <TouchableOpacity 
                style={styles.actionButton}
                onPress={() => navigation.navigate('Worklist')}
            >
                <Icon name="list" size={24} color="#2196F3" />
                <Text style={styles.actionText}>工作列表</Text>
            </TouchableOpacity>
            
            <TouchableOpacity 
                style={styles.actionButton}
                onPress={() => navigation.navigate('Notifications')}
            >
                <Icon name="notifications" size={24} color="#FF9800" />
                <Text style={styles.actionText}>通知</Text>
                {dashboardData.notifications > 0 && (
                    <Badge 
                        value={dashboardData.notifications}
                        status="error"
                        containerStyle={styles.badge}
                    />
                )}
            </TouchableOpacity>
            
            <TouchableOpacity style={styles.actionButton}>
                <Icon name="search" size={24} color="#4CAF50" />
                <Text style={styles.actionText}>搜索</Text>
            </TouchableOpacity>
            
            <TouchableOpacity style={styles.actionButton}>
                <Icon name="settings" size={24} color="#9E9E9E" />
                <Text style={styles.actionText}>设置</Text>
            </TouchableOpacity>
        </View>
    );
    
    // 统计卡片
    const StatsCards = () => (
        <View style={styles.statsContainer}>
            <View style={styles.statsRow}>
                <Card containerStyle={[styles.statCard, { backgroundColor: '#E3F2FD' }]}>
                    <View style={styles.statContent}>
                        <Icon name="pending" size={32} color="#2196F3" />
                        <Text style={styles.statNumber}>{dashboardData.pendingStudies}</Text>
                        <Text style={styles.statLabel}>待处理</Text>
                    </View>
                </Card>
                
                <Card containerStyle={[styles.statCard, { backgroundColor: '#E8F5E8' }]}>
                    <View style={styles.statContent}>
                        <Icon name="check-circle" size={32} color="#4CAF50" />
                        <Text style={styles.statNumber}>{dashboardData.completedToday}</Text>
                        <Text style={styles.statLabel}>今日完成</Text>
                    </View>
                </Card>
            </View>
            
            <View style={styles.statsRow}>
                <Card containerStyle={[styles.statCard, { backgroundColor: '#FFF3E0' }]}>
                    <View style={styles.statContent}>
                        <Icon name="memory" size={32} color="#FF9800" />
                        <Text style={styles.statNumber}>{dashboardData.aiProcessing}</Text>
                        <Text style={styles.statLabel}>AI处理中</Text>
                    </View>
                </Card>
                
                <Card containerStyle={[styles.statCard, { backgroundColor: '#FCE4EC' }]}>
                    <View style={styles.statContent}>
                        <Icon name="notifications" size={32} color="#E91E63" />
                        <Text style={styles.statNumber}>{dashboardData.notifications}</Text>
                        <Text style={styles.statLabel}>新通知</Text>
                    </View>
                </Card>
            </View>
        </View>
    );
    
    // 最近活动
    const RecentActivity = () => (
        <Card title="最近活动" containerStyle={styles.activityCard}>
            <View style={styles.activityList}>
                <View style={styles.activityItem}>
                    <Icon name="image" size={20} color="#2196F3" />
                    <View style={styles.activityContent}>
                        <Text style={styles.activityTitle}>CT胸部检查完成</Text>
                        <Text style={styles.activityTime}>5分钟前</Text>
                    </View>
                </View>
                
                <View style={styles.activityItem}>
                    <Icon name="memory" size={20} color="#FF9800" />
                    <View style={styles.activityContent}>
                        <Text style={styles.activityTitle}>AI分析结果已生成</Text>
                        <Text style={styles.activityTime}>10分钟前</Text>
                    </View>
                </View>
                
                <View style={styles.activityItem}>
                    <Icon name="assignment" size={20} color="#4CAF50" />
                    <View style={styles.activityContent}>
                        <Text style={styles.activityTitle}>报告已提交审核</Text>
                        <Text style={styles.activityTime}>15分钟前</Text>
                    </View>
                </View>
            </View>
        </Card>
    );
    
    return (
        <ScrollView 
            style={styles.container}
            refreshControl={
                <RefreshControl refreshing={refreshing} onRefresh={onRefresh} />
            }
        >
            {/* 用户信息头部 */}
            <View style={styles.header}>
                <View style={styles.userInfo}>
                    <Avatar
                        rounded
                        source={{ uri: 'https://example.com/avatar.jpg' }}
                        size="medium"
                    />
                    <View style={styles.userDetails}>
                        <Text style={styles.userName}>李医生</Text>
                        <Text style={styles.userRole}>放射科医生</Text>
                    </View>
                </View>
                
                <TouchableOpacity style={styles.notificationIcon}>
                    <Icon name="notifications" size={24} color="#666" />
                    <Badge 
                        value={dashboardData.notifications}
                        status="error"
                        containerStyle={styles.headerBadge}
                    />
                </TouchableOpacity>
            </View>
            
            {/* 快速操作 */}
            <QuickActions />
            
            {/* 统计卡片 */}
            <StatsCards />
            
            {/* 最近活动 */}
            <RecentActivity />
        </ScrollView>
    );
};

const styles = StyleSheet.create({
    container: {
        flex: 1,
        backgroundColor: '#f5f5f5'
    },
    header: {
        flexDirection: 'row',
        justifyContent: 'space-between',
        alignItems: 'center',
        padding: 16,
        backgroundColor: '#fff',
        marginBottom: 8
    },
    userInfo: {
        flexDirection: 'row',
        alignItems: 'center'
    },
    userDetails: {
        marginLeft: 12
    },
    userName: {
        fontSize: 18,
        fontWeight: 'bold',
        color: '#333'
    },
    userRole: {
        fontSize: 14,
        color: '#666',
        marginTop: 2
    },
    notificationIcon: {
        position: 'relative'
    },
    headerBadge: {
        position: 'absolute',
        top: -8,
        right: -8
    },
    quickActions: {
        flexDirection: 'row',
        justifyContent: 'space-around',
        backgroundColor: '#fff',
        paddingVertical: 16,
        marginBottom: 8
    },
    actionButton: {
        alignItems: 'center',
        position: 'relative'
    },
    actionText: {
        fontSize: 12,
        color: '#666',
        marginTop: 4
    },
    badge: {
        position: 'absolute',
        top: -5,
        right: -5
    },
    statsContainer: {
        paddingHorizontal: 8,
        marginBottom: 8
    },
    statsRow: {
        flexDirection: 'row',
        justifyContent: 'space-between',
        marginBottom: 8
    },
    statCard: {
        flex: 1,
        marginHorizontal: 4,
        borderRadius: 8,
        elevation: 2
    },
    statContent: {
        alignItems: 'center',
        paddingVertical: 8
    },
    statNumber: {
        fontSize: 24,
        fontWeight: 'bold',
        color: '#333',
        marginTop: 4
    },
    statLabel: {
        fontSize: 12,
        color: '#666',
        marginTop: 2
    },
    activityCard: {
        margin: 8,
        borderRadius: 8
    },
    activityList: {
        marginTop: 8
    },
    activityItem: {
        flexDirection: 'row',
        alignItems: 'center',
        paddingVertical: 8,
        borderBottomWidth: 1,
        borderBottomColor: '#f0f0f0'
    },
    activityContent: {
        marginLeft: 12,
        flex: 1
    },
    activityTitle: {
        fontSize: 14,
        color: '#333'
    },
    activityTime: {
        fontSize: 12,
        color: '#999',
        marginTop: 2
    }
});

export default HomeScreen;
```

## 6. 响应式设计与适配 (Responsive Design)

### 6.1 断点系统

```css
/* 响应式断点定义 */
:root {
    /* 断点定义 */
    --breakpoint-xs: 480px;
    --breakpoint-sm: 768px;
    --breakpoint-md: 1024px;
    --breakpoint-lg: 1280px;
    --breakpoint-xl: 1920px;
}

/* 媒体查询混合器 */
@media (max-width: 480px) {
    /* 超小屏幕 - 手机竖屏 */
    .physician-workstation {
        .workstation-header {
            padding: 8px 12px;
            
            .header-center {
                display: none; /* 隐藏患者信息 */
            }
            
            .logo span {
                display: none; /* 隐藏文字 */
            }
        }
        
        .workstation-sider {
            width: 60px !important;
            
            .ant-menu-item-title {
                display: none; /* 只显示图标 */
            }
        }
    }
    
    .worklist-toolbar {
        flex-direction: column;
        gap: 8px;
        
        .toolbar-right {
            width: 100%;
            justify-content: space-between;
        }
    }
    
    .image-viewer-layout {
        .viewer-sidebar {
            position: fixed;
            top: 0;
            right: -350px;
            height: 100vh;
            z-index: 1000;
            transition: right 0.3s ease;
            
            &.visible {
                right: 0;
            }
        }
    }
}

@media (min-width: 481px) and (max-width: 768px) {
    /* 小屏幕 - 手机横屏/平板竖屏 */
    .physician-workstation {
        .workstation-sider {
            width: 180px !important;
        }
    }
    
    .worklist-table {
        .ant-table-tbody > tr > td {
            padding: 8px 4px;
            font-size: 12px;
        }
    }
    
    .image-viewer-layout {
        .image-toolbar {
            flex-wrap: wrap;
            gap: 4px;
            
            .toolbar-group {
                margin-right: 8px;
            }
        }
    }
}

@media (min-width: 769px) and (max-width: 1024px) {
    /* 中等屏幕 - 平板横屏 */
    .physician-workstation {
        .workstation-sider {
            width: 200px !important;
        }
    }
    
    .image-viewer-layout {
        .viewer-sidebar {
            width: 280px !important;
        }
    }
}

@media (min-width: 1025px) {
    /* 大屏幕 - 桌面 */
    .physician-workstation {
        .workstation-sider {
            width: 240px !important;
        }
    }
    
    .image-viewer-layout {
        .viewer-sidebar {
            width: 350px !important;
        }
    }
}
```

### 6.2 触摸优化

```css
/* 触摸设备优化 */
@media (hover: none) and (pointer: coarse) {
    /* 触摸设备样式 */
    .ant-btn {
        min-height: 44px; /* 符合触摸标准 */
        padding: 8px 16px;
    }
    
    .ant-table-tbody > tr > td {
        padding: 12px 8px; /* 增加触摸区域 */
    }
    
    .image-toolbar {
        .toolbar-group {
            gap: 12px; /* 增加按钮间距 */
        }
        
        .ant-btn {
            min-width: 48px;
            min-height: 48px;
        }
    }
    
    /* 滑动手势支持 */
    .image-container {
        touch-action: pan-x pan-y pinch-zoom;
        user-select: none;
    }
    
    .series-thumbnails {
        overflow-x: auto;
        -webkit-overflow-scrolling: touch;
        
        .thumbnail {
            min-width: 80px;
            min-height: 80px;
        }
    }
}
```

## 7. 可访问性设计 (Accessibility)

### 7.1 ARIA标签和语义化

```jsx
// 可访问性增强组件示例
import React from 'react';

const AccessibleImageViewer = () => {
    return (
        <div 
            className="image-viewer"
            role="application"
            aria-label="医学图像查看器"
        >
            <div 
                className="image-container"
                role="img"
                aria-label="DICOM医学图像"
                tabIndex={0}
            >
                {/* 图像内容 */}
            </div>
            
            <div 
                className="ai-overlay"
                role="region"
                aria-label="AI分析结果标注"
                aria-live="polite"
            >
                {/* AI标注内容 */}
            </div>
            
            <nav 
                className="image-toolbar"
                role="toolbar"
                aria-label="图像操作工具"
            >
                <button 
                    aria-label="放大图像"
                    title="放大图像 (快捷键: +)"
                >
                    <ZoomInIcon />
                </button>
                
                <button 
                    aria-label="缩小图像"
                    title="缩小图像 (快捷键: -)"
                >
                    <ZoomOutIcon />
                </button>
            </nav>
        </div>
    );
};
```

### 7.2 键盘导航支持

```css
/* 键盘焦点样式 */
.keyboard-focus {
    outline: 2px solid #2196F3;
    outline-offset: 2px;
    border-radius: 4px;
}

/* 跳过链接 */
.skip-link {
    position: absolute;
    top: -40px;
    left: 6px;
    background: #2196F3;
    color: white;
    padding: 8px;
    text-decoration: none;
    border-radius: 4px;
    z-index: 9999;
    
    &:focus {
        top: 6px;
    }
}

/* 高对比度模式支持 */
@media (prefers-contrast: high) {
    :root {
        --primary-500: #0000FF;
        --success-500: #008000;
        --error-500: #FF0000;
        --gray-900: #000000;
        --gray-100: #FFFFFF;
    }
}

/* 减少动画模式 */
@media (prefers-reduced-motion: reduce) {
    * {
        animation-duration: 0.01ms !important;
        animation-iteration-count: 1 !important;
        transition-duration: 0.01ms !important;
    }
}
```

## 8. 性能优化策略 (Performance Optimization)

### 8.1 代码分割和懒加载

```jsx
// 路由级别的代码分割
import { lazy, Suspense } from 'react';
import { Spin } from 'antd';

// 懒加载组件
const PhysicianWorkstation = lazy(() => import('./components/PhysicianWorkstation'));
const AdminDashboard = lazy(() => import('./components/AdminDashboard'));
const ImageViewer = lazy(() => import('./components/ImageViewer'));

// 加载中组件
const LoadingSpinner = () => (
    <div style={{ 
        display: 'flex', 
        justifyContent: 'center', 
        alignItems: 'center', 
        height: '100vh' 
    }}>
        <Spin size="large" tip="加载中..." />
    </div>
);

// 应用路由
const AppRouter = () => {
    return (
        <Router>
            <Suspense fallback={<LoadingSpinner />}>
                <Routes>
                    <Route 
                        path="/workstation" 
                        element={<PhysicianWorkstation />} 
                    />
                    <Route 
                        path="/admin" 
                        element={<AdminDashboard />} 
                    />
                    <Route 
                        path="/viewer/:studyId" 
                        element={<ImageViewer />} 
                    />
                </Routes>
            </Suspense>
        </Router>
    );
};
```

### 8.2 图像优化

```jsx
// 图像懒加载和优化
import { useState, useEffect, useRef } from 'react';

const OptimizedImageViewer = ({ imageUrl, thumbnail }) => {
    const [isLoaded, setIsLoaded] = useState(false);
    const [isInView, setIsInView] = useState(false);
    const imgRef = useRef(null);
    
    // 交叉观察器用于懒加载
    useEffect(() => {
        const observer = new IntersectionObserver(
            ([entry]) => {
                if (entry.isIntersecting) {
                    setIsInView(true);
                    observer.disconnect();
                }
            },
            { threshold: 0.1 }
        );
        
        if (imgRef.current) {
            observer.observe(imgRef.current);
        }
        
        return () => observer.disconnect();
    }, []);
    
    return (
        <div ref={imgRef} className="optimized-image-container">
            {/* 缩略图预览 */}
            {!isLoaded && thumbnail && (
                <img 
                    src={thumbnail}
                    alt="缩略图"
                    className="thumbnail-preview"
                    style={{ filter: 'blur(5px)' }}
                />
            )}
            
            {/* 主图像 */}
            {isInView && (
                <img
                    src={imageUrl}
                    alt="医学图像"
                    onLoad={() => setIsLoaded(true)}
                    style={{ 
                        opacity: isLoaded ? 1 : 0,
                        transition: 'opacity 0.3s ease'
                    }}
                />
            )}
        </div>
    );
};
```

## 9. 总结与实施建议 (Summary and Implementation)

### 9.1 设计系统核心优势

1. **统一性**: 建立了完整的设计系统，确保各平台界面一致性
2. **可扩展性**: 模块化设计支持功能快速迭代和扩展
3. **用户体验**: 针对医疗场景优化的工作流程和交互设计
4. **技术先进性**: 采用现代前端技术栈，支持高性能渲染
5. **可访问性**: 全面的无障碍设计，符合医疗行业标准

### 9.2 实施路线图

#### 第一阶段 (1-2个月): 基础框架
- 搭建设计系统和组件库
- 实现医生工作台核心功能
- 完成图像查看器基础版本
- 建立CI/CD流水线

#### 第二阶段 (2-3个月): 功能完善
- 完成管理后台开发
- 实现AI结果可视化
- 开发报告生成系统
- 移动端应用开发

#### 第三阶段 (1-2个月): 优化与测试
- 性能优化和用户体验改进
- 全面测试和bug修复
- 用户培训和文档编写
- 部署和上线准备

### 9.3 关键成功因素

1. **用户参与**: 持续收集医生和技师的反馈
2. **迭代开发**: 采用敏捷开发方法，快速响应需求变化
3. **质量保证**: 建立完善的测试体系和质量控制流程
4. **性能监控**: 实时监控系统性能和用户体验指标
5. **安全合规**: 确保符合医疗数据安全和隐私保护要求

### 9.4 风险缓解策略

1. **技术风险**: 建立技术预研和原型验证机制
2. **用户接受度**: 提供充分的用户培训和支持
3. **性能风险**: 进行压力测试和性能优化
4. **合规风险**: 与法规专家密切合作，确保合规性
5. **维护风险**: 建立完善的文档和知识传承机制

通过以上全面的UI设计规范，医学图像AI分析系统将能够为用户提供优秀的使用体验，同时满足医疗行业的专业需求和合规要求。