import { createSlice, createAsyncThunk, PayloadAction } from '@reduxjs/toolkit'
import type { RootState } from '../index'

// 报告相关接口定义
export interface Report {
  id: string
  patientId: string
  patientName: string
  studyId: string
  studyDate: string
  modality: string
  bodyPart: string
  title: string
  description: string
  findings: string
  impression: string
  recommendations: string
  status: 'draft' | 'pending_review' | 'reviewed' | 'approved' | 'signed' | 'cancelled'
  priority: 'low' | 'normal' | 'high' | 'urgent'
  type: 'diagnostic' | 'screening' | 'follow_up' | 'emergency' | 'research'
  template: {
    id: string
    name: string
    version: string
  }
  sections: Array<{
    id: string
    name: string
    content: string
    type: 'text' | 'structured' | 'measurement' | 'image'
    required: boolean
    order: number
  }>
  measurements: Array<{
    id: string
    name: string
    value: number
    unit: string
    normalRange: {
      min: number
      max: number
    }
    isAbnormal: boolean
    category: string
  }>
  images: Array<{
    id: string
    imageId: string
    seriesId: string
    instanceId: string
    thumbnailUrl: string
    annotations: Array<{
      id: string
      type: 'arrow' | 'circle' | 'rectangle' | 'text' | 'measurement'
      coordinates: number[]
      text?: string
      color: string
    }>
    keyImage: boolean
  }>
  aiAnalysis: {
    modelId: string
    modelName: string
    confidence: number
    findings: Array<{
      category: string
      description: string
      confidence: number
      severity: 'normal' | 'mild' | 'moderate' | 'severe'
      location?: {
        x: number
        y: number
        width: number
        height: number
      }
    }>
    suggestions: string[]
    processingTime: number
    version: string
  } | null
  workflow: {
    currentStep: string
    steps: Array<{
      id: string
      name: string
      status: 'pending' | 'in_progress' | 'completed' | 'skipped'
      assignedTo?: string
      assignedToName?: string
      startedAt?: string
      completedAt?: string
      comments?: string
    }>
    history: Array<{
      id: string
      action: string
      performedBy: string
      performedByName: string
      timestamp: string
      comments?: string
      previousStatus?: string
      newStatus?: string
    }>
  }
  quality: {
    score: number
    metrics: {
      completeness: number
      accuracy: number
      consistency: number
      timeliness: number
    }
    issues: Array<{
      type: 'missing_field' | 'inconsistent_data' | 'quality_concern' | 'validation_error'
      severity: 'low' | 'medium' | 'high'
      description: string
      field?: string
      suggestion?: string
    }>
    lastChecked: string
  }
  collaboration: {
    comments: Array<{
      id: string
      userId: string
      userName: string
      content: string
      timestamp: string
      type: 'general' | 'question' | 'suggestion' | 'approval'
      resolved: boolean
      replies: Array<{
        id: string
        userId: string
        userName: string
        content: string
        timestamp: string
      }>
    }>
    reviewers: Array<{
      userId: string
      userName: string
      role: string
      status: 'pending' | 'reviewing' | 'approved' | 'rejected'
      assignedAt: string
      reviewedAt?: string
      comments?: string
    }>
    sharedWith: Array<{
      userId: string
      userName: string
      permission: 'read' | 'comment' | 'edit'
      sharedAt: string
    }>
  }
  metadata: {
    createdBy: string
    createdByName: string
    createdAt: string
    updatedBy: string
    updatedByName: string
    updatedAt: string
    signedBy?: string
    signedByName?: string
    signedAt?: string
    version: number
    tags: string[]
    department: string
    institution: string
    referringPhysician?: string
    urgentNotification: boolean
    estimatedCompletionTime?: string
    actualCompletionTime?: string
  }
  attachments: Array<{
    id: string
    name: string
    type: string
    size: number
    url: string
    uploadedBy: string
    uploadedAt: string
  }>
}

export interface ReportTemplate {
  id: string
  name: string
  description: string
  category: string
  modality: string
  bodyPart: string
  isActive: boolean
  isDefault: boolean
  version: string
  structure: {
    sections: Array<{
      id: string
      name: string
      type: 'text' | 'structured' | 'measurement' | 'image'
      required: boolean
      order: number
      defaultContent?: string
      options?: string[]
      validation?: {
        minLength?: number
        maxLength?: number
        pattern?: string
        required?: boolean
      }
    }>
    measurements: Array<{
      id: string
      name: string
      unit: string
      normalRange: {
        min: number
        max: number
      }
      category: string
      required: boolean
    }>
    workflow: Array<{
      id: string
      name: string
      role: string
      required: boolean
      order: number
      autoAssign?: boolean
    }>
  }
  settings: {
    autoSave: boolean
    requireReview: boolean
    requireSignature: boolean
    allowCollaboration: boolean
    qualityChecks: boolean
    aiAnalysis: boolean
    notifications: {
      onCreate: boolean
      onStatusChange: boolean
      onReview: boolean
      onSign: boolean
    }
  }
  usage: {
    count: number
    lastUsed: string
    averageCompletionTime: number
  }
  createdBy: string
  createdAt: string
  updatedAt: string
}

export interface ReportFilter {
  search?: string
  patientId?: string
  studyId?: string
  status?: string[]
  priority?: string[]
  type?: string[]
  modality?: string[]
  bodyPart?: string[]
  createdBy?: string[]
  assignedTo?: string[]
  dateRange?: {
    start: string
    end: string
    type: 'created' | 'updated' | 'signed' | 'study'
  }
  tags?: string[]
  department?: string[]
  hasAiAnalysis?: boolean
  qualityScore?: {
    min: number
    max: number
  }
  urgent?: boolean
}

export interface ReportStatistics {
  overview: {
    total: number
    draft: number
    pending: number
    completed: number
    overdue: number
    averageCompletionTime: number
    qualityScore: number
  }
  byStatus: Record<string, number>
  byPriority: Record<string, number>
  byType: Record<string, number>
  byModality: Record<string, number>
  byDepartment: Record<string, number>
  trends: {
    daily: Array<{
      date: string
      created: number
      completed: number
      signed: number
    }>
    monthly: Array<{
      month: string
      created: number
      completed: number
      averageTime: number
    }>
  }
  performance: {
    topPerformers: Array<{
      userId: string
      userName: string
      completed: number
      averageTime: number
      qualityScore: number
    }>
    bottlenecks: Array<{
      step: string
      averageTime: number
      count: number
    }>
  }
  quality: {
    averageScore: number
    distribution: Record<string, number>
    commonIssues: Array<{
      type: string
      count: number
      percentage: number
    }>
  }
}

// 状态接口
export interface ReportState {
  // 报告列表
  reports: Report[]
  currentReport: Report | null
  selectedReports: string[]
  
  // 报告模板
  templates: ReportTemplate[]
  currentTemplate: ReportTemplate | null
  
  // 过滤和搜索
  filters: ReportFilter
  searchResults: Report[]
  
  // 分页
  pagination: {
    page: number
    pageSize: number
    total: number
    totalPages: number
  }
  
  // 排序
  sorting: {
    field: string
    order: 'asc' | 'desc'
  }
  
  // 统计数据
  statistics: ReportStatistics | null
  
  // 工作流状态
  workflow: {
    currentStep: string
    availableActions: string[]
    pendingTasks: Array<{
      reportId: string
      taskType: string
      assignedTo: string
      dueDate: string
    }>
  }
  
  // 质量控制
  qualityControl: {
    enabled: boolean
    rules: Array<{
      id: string
      name: string
      type: string
      condition: string
      action: string
      enabled: boolean
    }>
    issues: Array<{
      reportId: string
      type: string
      severity: string
      description: string
      resolved: boolean
    }>
  }
  
  // 协作状态
  collaboration: {
    activeReviewers: Array<{
      reportId: string
      userId: string
      userName: string
      status: string
    }>
    recentComments: Array<{
      reportId: string
      commentId: string
      content: string
      author: string
      timestamp: string
    }>
  }
  
  // UI状态
  ui: {
    viewMode: 'list' | 'grid' | 'timeline'
    sidebarOpen: boolean
    filterPanelOpen: boolean
    selectedTab: 'all' | 'my_reports' | 'pending_review' | 'drafts'
    previewMode: boolean
    fullscreen: boolean
  }
  
  // 加载状态
  loading: {
    reports: boolean
    currentReport: boolean
    templates: boolean
    statistics: boolean
    saving: boolean
    deleting: boolean
    exporting: boolean
  }
  
  // 错误状态
  error: {
    reports: string | null
    currentReport: string | null
    templates: string | null
    statistics: string | null
    saving: string | null
    validation: Record<string, string>
  }
  
  // 缓存
  cache: {
    lastFetch: string | null
    reportDetails: Record<string, Report>
    searchCache: Record<string, Report[]>
  }
}

// 初始状态
const initialState: ReportState = {
  reports: [],
  currentReport: null,
  selectedReports: [],
  
  templates: [],
  currentTemplate: null,
  
  filters: {},
  searchResults: [],
  
  pagination: {
    page: 1,
    pageSize: 20,
    total: 0,
    totalPages: 0,
  },
  
  sorting: {
    field: 'updatedAt',
    order: 'desc',
  },
  
  statistics: null,
  
  workflow: {
    currentStep: '',
    availableActions: [],
    pendingTasks: [],
  },
  
  qualityControl: {
    enabled: true,
    rules: [],
    issues: [],
  },
  
  collaboration: {
    activeReviewers: [],
    recentComments: [],
  },
  
  ui: {
    viewMode: 'list',
    sidebarOpen: true,
    filterPanelOpen: false,
    selectedTab: 'all',
    previewMode: false,
    fullscreen: false,
  },
  
  loading: {
    reports: false,
    currentReport: false,
    templates: false,
    statistics: false,
    saving: false,
    deleting: false,
    exporting: false,
  },
  
  error: {
    reports: null,
    currentReport: null,
    templates: null,
    statistics: null,
    saving: null,
    validation: {},
  },
  
  cache: {
    lastFetch: null,
    reportDetails: {},
    searchCache: {},
  },
}

// 异步操作
export const loadReports = createAsyncThunk(
  'report/loadReports',
  async (params: {
    filters?: ReportFilter
    page?: number
    pageSize?: number
    sorting?: { field: string; order: 'asc' | 'desc' }
  }) => {
    const response = await fetch('/api/reports', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(params),
    })
    if (!response.ok) throw new Error('Failed to load reports')
    return response.json()
  }
)

export const fetchReports = createAsyncThunk(
  'report/fetchReports',
  async (params: {
    page?: number
    pageSize?: number
    filters?: ReportFilter
    sorting?: { field: string; order: 'asc' | 'desc' }
    searchQuery?: string
  }) => {
    const response = await fetch('/api/reports', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(params),
    })
    if (!response.ok) throw new Error('Failed to fetch reports')
    return response.json()
  }
)

export const fetchReportById = createAsyncThunk(
  'report/fetchReportById',
  async (reportId: string) => {
    const response = await fetch(`/api/reports/${reportId}`)
    if (!response.ok) throw new Error('Failed to fetch report')
    return response.json()
  }
)

export const createReport = createAsyncThunk(
  'report/createReport',
  async (reportData: Partial<Report>) => {
    const response = await fetch('/api/reports', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(reportData),
    })
    if (!response.ok) throw new Error('Failed to create report')
    return response.json()
  }
)

export const updateReport = createAsyncThunk(
  'report/updateReport',
  async (params: { reportId: string; updates: Partial<Report> }) => {
    const { reportId, updates } = params
    
    const response = await fetch(`/api/reports/${reportId}`, {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(updates),
    })
    if (!response.ok) throw new Error('Failed to update report')
    return response.json()
  }
)

export const deleteReport = createAsyncThunk(
  'report/deleteReport',
  async (reportId: string) => {
    const response = await fetch(`/api/reports/${reportId}`, {
      method: 'DELETE',
    })
    if (!response.ok) throw new Error('Failed to delete report')
    return reportId
  }
)

export const fetchReportStatistics = createAsyncThunk(
  'report/fetchReportStatistics',
  async (params?: {
    dateRange?: { start: string; end: string }
    department?: string
    filters?: ReportFilter
  }) => {
    const response = await fetch('/api/reports/statistics', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(params || {}),
    })
    if (!response.ok) throw new Error('Failed to fetch statistics')
    return response.json()
  }
)

export const fetchReportTemplates = createAsyncThunk(
  'report/fetchReportTemplates',
  async (params?: {
    category?: string
    modality?: string
    bodyPart?: string
    isActive?: boolean
  }) => {
    const queryParams = new URLSearchParams()
    if (params) {
      Object.entries(params).forEach(([key, value]) => {
        if (value !== undefined) {
          queryParams.append(key, String(value))
        }
      })
    }
    
    const response = await fetch(`/api/reports/templates?${queryParams}`)
    if (!response.ok) throw new Error('Failed to fetch templates')
    return response.json()
  }
)

export const searchReports = createAsyncThunk(
  'report/searchReports',
  async (params: {
    query: string
    filters?: ReportFilter
    page?: number
    pageSize?: number
    sorting?: { field: string; order: 'asc' | 'desc' }
  }) => {
    const response = await fetch('/api/reports/search', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(params),
    })
    if (!response.ok) throw new Error('Failed to search reports')
    return response.json()
  }
)

export const validateReport = createAsyncThunk(
  'report/validateReport',
  async (report: Partial<Report>) => {
    const response = await fetch('/api/reports/validate', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(report),
    })
    if (!response.ok) throw new Error('Failed to validate report')
    return response.json()
  }
)

export const duplicateReport = createAsyncThunk(
  'report/duplicateReport',
  async (reportId: string) => {
    const response = await fetch(`/api/reports/${reportId}/duplicate`, {
      method: 'POST',
    })
    if (!response.ok) throw new Error('Failed to duplicate report')
    return response.json()
  }
)

export const exportReport = createAsyncThunk(
  'report/exportReport',
  async (params: { reportId: string; format: 'pdf' | 'docx' | 'html' }) => {
    const response = await fetch(`/api/reports/${params.reportId}/export`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ format: params.format }),
    })
    if (!response.ok) throw new Error('Failed to export report')
    return response.blob()
  }
)

const reportSlice = createSlice({
  name: 'report',
  initialState,
  reducers: {
    setCurrentReport: (state, action: PayloadAction<Report | null>) => {
      state.currentReport = action.payload
    },
    setSearchQuery: (state, action: PayloadAction<string>) => {
      state.searchQuery = action.payload
    },
    setFilters: (state, action: PayloadAction<ReportState['filters']>) => {
      state.filters = action.payload
    },
    setPagination: (state, action: PayloadAction<Partial<ReportState['pagination']>>) => {
      state.pagination = { ...state.pagination, ...action.payload }
    },
    setSortBy: (state, action: PayloadAction<ReportState['sortBy']>) => {
      state.sortBy = action.payload
    },
    setIsEditing: (state, action: PayloadAction<boolean>) => {
      state.isEditing = action.payload
    },
    setEditingFields: (state, action: PayloadAction<string[]>) => {
      state.editingFields = action.payload
    },
    addEditingField: (state, action: PayloadAction<string>) => {
      if (!state.editingFields.includes(action.payload)) {
        state.editingFields.push(action.payload)
      }
    },
    removeEditingField: (state, action: PayloadAction<string>) => {
      state.editingFields = state.editingFields.filter(field => field !== action.payload)
    },
    updateCurrentReportField: (state, action: PayloadAction<{ field: string; value: any }>) => {
      if (state.currentReport) {
        (state.currentReport as any)[action.payload.field] = action.payload.value
        state.currentReport.updatedAt = new Date().toISOString()
      }
    },
    addReportHistoryEntry: (state, action: PayloadAction<{
      action: string
      description: string
      performedBy: string
    }>) => {
      if (state.currentReport) {
        const historyEntry = {
          id: Date.now().toString(),
          ...action.payload,
          timestamp: new Date().toISOString()
        }
        state.currentReport.history.push(historyEntry)
      }
    },
    clearError: (state) => {
      state.error = {
        reports: null,
        currentReport: null,
        saving: null,
        deleting: null,
        searching: null,
        validating: null,
        duplicating: null,
        exporting: null,
        statistics: null,
        templates: null
      }
    },
    setSelectedReports: (state, action: PayloadAction<string[]>) => {
      state.selectedReports = action.payload
    },
    setCurrentTemplate: (state, action: PayloadAction<ReportTemplate | null>) => {
      state.currentTemplate = action.payload
    },
    setWorkflowStatus: (state, action: PayloadAction<Partial<ReportState['workflow']>>) => {
      state.workflow = { ...state.workflow, ...action.payload }
    },
    setQualityControlStatus: (state, action: PayloadAction<Partial<ReportState['qualityControl']>>) => {
      state.qualityControl = { ...state.qualityControl, ...action.payload }
    },
    setCollaborationMode: (state, action: PayloadAction<Partial<ReportState['collaboration']>>) => {
      state.collaboration = { ...state.collaboration, ...action.payload }
    },
    setUIState: (state, action: PayloadAction<Partial<ReportState['ui']>>) => {
      state.ui = { ...state.ui, ...action.payload }
    },
    clearCache: (state) => {
      state.cache = {
        reportDetails: {},
        lastFetch: null,
        invalidatedKeys: []
      }
    },
    resetFilters: (state) => {
      state.filters = {}
    },
    resetPagination: (state) => {
      state.pagination = {
        page: 1,
        pageSize: 20,
        total: 0
      }
    }
  },
  extraReducers: (builder) => {
    builder
      // Load reports
      .addCase(loadReports.pending, (state) => {
        state.loading.reports = true
        state.error.reports = null
      })
      .addCase(loadReports.fulfilled, (state, action) => {
        state.loading.reports = false
        state.reports = action.payload.reports
        state.pagination.total = action.payload.total
        state.cache.lastFetch = Date.now()
      })
      .addCase(loadReports.rejected, (state, action) => {
        state.loading.reports = false
        state.error.reports = action.error.message || '获取报告列表失败'
      })
      
      // Load report
      .addCase(fetchReportById.pending, (state) => {
        state.loading.currentReport = true
        state.error.currentReport = null
      })
      .addCase(fetchReportById.fulfilled, (state, action) => {
        state.loading.currentReport = false
        state.currentReport = action.payload
        state.cache.reportDetails[action.payload.id] = {
          data: action.payload,
          timestamp: Date.now()
        }
      })
      .addCase(fetchReportById.rejected, (state, action) => {
        state.loading.currentReport = false
        state.error.currentReport = action.error.message || '获取报告详情失败'
      })
      
      // Save report
      .addCase(updateReport.pending, (state) => {
        state.loading.saving = true
        state.error.saving = null
      })
      .addCase(updateReport.fulfilled, (state, action) => {
        state.loading.saving = false
        const savedReport = action.payload
        
        // Update or add to reports list
        const existingIndex = state.reports.findIndex(r => r.id === savedReport.id)
        if (existingIndex !== -1) {
          state.reports[existingIndex] = savedReport
        } else {
          state.reports.unshift(savedReport)
          state.pagination.total += 1
        }
        
        // Update current report if it's the same
        if (state.currentReport?.id === savedReport.id) {
          state.currentReport = savedReport
        }
        
        // Update cache
        state.cache.reportDetails[savedReport.id] = {
          data: savedReport,
          timestamp: Date.now()
        }
      })
      .addCase(updateReport.rejected, (state, action) => {
        state.loading.saving = false
        state.error.saving = action.error.message || '保存报告失败'
      })
      
      // Delete report
      .addCase(deleteReport.pending, (state) => {
        state.loading.deleting = true
        state.error.deleting = null
      })
      .addCase(deleteReport.fulfilled, (state, action) => {
        state.loading.deleting = false
        const reportId = action.payload
        state.reports = state.reports.filter(report => report.id !== reportId)
        if (state.currentReport?.id === reportId) {
          state.currentReport = null
        }
        state.pagination.total -= 1
        delete state.cache.reportDetails[reportId]
      })
      .addCase(deleteReport.rejected, (state, action) => {
        state.loading.deleting = false
        state.error.deleting = action.error.message || '删除报告失败'
      })
      
      // Search reports
      .addCase(searchReports.pending, (state) => {
        state.loading.searching = true
        state.error.searching = null
      })
      .addCase(searchReports.fulfilled, (state, action) => {
        state.loading.searching = false
        state.searchResults = action.payload.reports
        state.ui.searchResultsCount = action.payload.total
      })
      .addCase(searchReports.rejected, (state, action) => {
        state.loading.searching = false
        state.error.searching = action.error.message || '搜索报告失败'
      })
      
      // Validate report
      .addCase(validateReport.pending, (state) => {
        state.loading.validating = true
        state.error.validating = null
      })
      .addCase(validateReport.fulfilled, (state, action) => {
        state.loading.validating = false
        state.qualityControl.validationResults = action.payload
      })
      .addCase(validateReport.rejected, (state, action) => {
        state.loading.validating = false
        state.error.validating = action.error.message || '验证报告失败'
      })
      
      // Duplicate report
      .addCase(duplicateReport.pending, (state) => {
        state.loading.duplicating = true
        state.error.duplicating = null
      })
      .addCase(duplicateReport.fulfilled, (state, action) => {
        state.loading.duplicating = false
        state.reports.unshift(action.payload)
        state.pagination.total += 1
      })
      .addCase(duplicateReport.rejected, (state, action) => {
        state.loading.duplicating = false
        state.error.duplicating = action.error.message || '复制报告失败'
      })
      
      // Export report
      .addCase(exportReport.pending, (state) => {
        state.loading.exporting = true
        state.error.exporting = null
      })
      .addCase(exportReport.fulfilled, (state) => {
        state.loading.exporting = false
      })
      .addCase(exportReport.rejected, (state, action) => {
        state.loading.exporting = false
        state.error.exporting = action.error.message || '导出报告失败'
      })
      
      // Fetch statistics
      .addCase(fetchReportStatistics.pending, (state) => {
        state.loading.statistics = true
        state.error.statistics = null
      })
      .addCase(fetchReportStatistics.fulfilled, (state, action) => {
        state.loading.statistics = false
        state.statistics = action.payload
      })
      .addCase(fetchReportStatistics.rejected, (state, action) => {
        state.loading.statistics = false
        state.error.statistics = action.error.message || '获取统计数据失败'
      })
      
      // Fetch templates
      .addCase(fetchReportTemplates.pending, (state) => {
        state.loading.templates = true
        state.error.templates = null
      })
      .addCase(fetchReportTemplates.fulfilled, (state, action) => {
        state.loading.templates = false
        state.templates = action.payload
      })
      .addCase(fetchReportTemplates.rejected, (state, action) => {
        state.loading.templates = false
        state.error.templates = action.error.message || '获取模板失败'
      })
  }
})

export const {
  setCurrentReport,
  setSearchQuery,
  setFilters,
  setPagination,
  setSortBy,
  setIsEditing,
  setEditingFields,
  addEditingField,
  removeEditingField,
  updateCurrentReportField,
  addReportHistoryEntry,
  clearError,
  setSelectedReports,
  setCurrentTemplate,
  setWorkflowStatus,
  setQualityControlStatus,
  setCollaborationMode,
  setUIState,
  clearCache,
  resetFilters,
  resetPagination
} = reportSlice.actions

export default reportSlice.reducer