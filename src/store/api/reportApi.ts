import { createApi, fetchBaseQuery } from '@reduxjs/toolkit/query/react'
import type { RootState } from '../index'
import type { Report, ReportTemplate, ReportStatistics } from '../slices/reportSlice'

export interface ReportListParams {
  page?: number
  pageSize?: number
  search?: string
  patientId?: string
  doctorId?: string
  status?: 'draft' | 'pending' | 'reviewed' | 'approved' | 'rejected'
  priority?: 'low' | 'medium' | 'high' | 'urgent'
  studyType?: string
  dateRange?: [string, string]
  hasAbnormalFindings?: boolean
  sortBy?: 'reportNumber' | 'patientName' | 'studyDate' | 'createdAt' | 'priority'
  sortOrder?: 'asc' | 'desc'
  assignedToMe?: boolean
}

export interface ReportListResponse {
  reports: Report[]
  total: number
  page: number
  pageSize: number
  totalPages: number
}

export interface CreateReportRequest {
  patientId: string
  patientName: string
  studyType: string
  studyDate: string
  priority: 'low' | 'medium' | 'high' | 'urgent'
  assignedDoctor: string
  templateId?: string
  imageIds?: string[]
  clinicalHistory?: string
  indication?: string
  technique?: string
  findings?: string
  impression?: string
  recommendations?: string
  urgentFindings?: boolean
  followUpRequired?: boolean
  followUpDate?: string
}

export interface UpdateReportRequest extends Partial<CreateReportRequest> {
  id: string
  status?: 'draft' | 'pending' | 'reviewed' | 'approved' | 'rejected'
  reviewNotes?: string
  rejectionReason?: string
}

export interface ReportReviewRequest {
  reportId: string
  action: 'approve' | 'reject' | 'request_changes'
  notes?: string
  changes?: Array<{
    section: string
    originalText: string
    suggestedText: string
    reason: string
  }>
}

export interface ReportSignRequest {
  reportId: string
  signature: string
  timestamp: string
  location?: string
}

export interface ReportShareRequest {
  reportId: string
  recipients: Array<{
    type: 'email' | 'internal' | 'external'
    address: string
    name?: string
    permissions: ('view' | 'download' | 'print')[]
  }>
  message?: string
  expirationDate?: string
  requireAuthentication?: boolean
}

export interface ReportExportRequest {
  reportIds: string[]
  format: 'pdf' | 'docx' | 'html' | 'dicom_sr'
  includeImages?: boolean
  includeAnnotations?: boolean
  watermark?: string
  template?: string
}

export interface ReportSearchRequest {
  query: string
  searchIn: ('findings' | 'impression' | 'recommendations' | 'patientName' | 'reportNumber')[]
  filters?: ReportListParams
  highlightMatches?: boolean
}

export interface ReportSearchResponse {
  reports: Array<Report & {
    highlights?: Record<string, string[]>
    relevanceScore?: number
  }>
  total: number
  searchTime: number
  suggestions?: string[]
}

export interface ReportAnalyticsRequest {
  dateRange: [string, string]
  groupBy?: 'day' | 'week' | 'month' | 'quarter'
  metrics?: ('count' | 'turnaround_time' | 'quality_score' | 'revision_rate')[]
  filters?: {
    studyType?: string[]
    department?: string[]
    doctor?: string[]
    priority?: string[]
  }
}

export interface ReportAnalyticsResponse {
  summary: {
    totalReports: number
    averageTurnaroundTime: number
    averageQualityScore: number
    revisionRate: number
    abnormalFindingsRate: number
  }
  trends: Array<{
    date: string
    count: number
    turnaroundTime: number
    qualityScore: number
    revisionRate: number
  }>
  distributions: {
    byStudyType: Record<string, number>
    byPriority: Record<string, number>
    byStatus: Record<string, number>
    byDoctor: Record<string, number>
  }
  performance: {
    topPerformers: Array<{
      doctorId: string
      doctorName: string
      reportCount: number
      averageTurnaroundTime: number
      qualityScore: number
    }>
    bottlenecks: Array<{
      stage: string
      averageTime: number
      reportCount: number
    }>
  }
}

export interface ReportQualityMetrics {
  reportId: string
  overallScore: number
  metrics: {
    completeness: number
    accuracy: number
    clarity: number
    consistency: number
    timeliness: number
  }
  issues: Array<{
    type: 'missing_section' | 'unclear_language' | 'inconsistent_terminology' | 'delayed_completion'
    severity: 'low' | 'medium' | 'high'
    description: string
    suggestions: string[]
  }>
  recommendations: string[]
}

export const reportApi = createApi({
  reducerPath: 'reportApi',
  baseQuery: fetchBaseQuery({
    baseUrl: '/api/reports',
    prepareHeaders: (headers, { getState }) => {
      const token = (getState() as RootState).auth.token
      if (token) {
        headers.set('authorization', `Bearer ${token}`)
      }
      return headers
    },
  }),
  tagTypes: ['Report', 'ReportTemplate', 'ReportStats', 'ReportAnalytics'],
  endpoints: (builder) => ({
    // 获取报告列表
    getReports: builder.query<ReportListResponse, ReportListParams>({
      query: (params) => ({
        url: '',
        params,
      }),
      providesTags: (result) =>
        result
          ? [
              ...result.reports.map(({ id }) => ({ type: 'Report' as const, id })),
              { type: 'Report', id: 'LIST' },
            ]
          : [{ type: 'Report', id: 'LIST' }],
    }),

    // 获取单个报告
    getReport: builder.query<Report, string>({
      query: (id) => `/${id}`,
      providesTags: (result, error, id) => [{ type: 'Report', id }],
    }),

    // 创建报告
    createReport: builder.mutation<Report, CreateReportRequest>({
      query: (data) => ({
        url: '',
        method: 'POST',
        body: data,
      }),
      invalidatesTags: [{ type: 'Report', id: 'LIST' }, 'ReportStats'],
    }),

    // 更新报告
    updateReport: builder.mutation<Report, UpdateReportRequest>({
      query: ({ id, ...data }) => ({
        url: `/${id}`,
        method: 'PUT',
        body: data,
      }),
      invalidatesTags: (result, error, { id }) => [
        { type: 'Report', id },
        { type: 'Report', id: 'LIST' },
      ],
    }),

    // 删除报告
    deleteReport: builder.mutation<void, string>({
      query: (id) => ({
        url: `/${id}`,
        method: 'DELETE',
      }),
      invalidatesTags: (result, error, id) => [
        { type: 'Report', id },
        { type: 'Report', id: 'LIST' },
        'ReportStats',
      ],
    }),

    // 复制报告
    duplicateReport: builder.mutation<Report, string>({
      query: (id) => ({
        url: `/${id}/duplicate`,
        method: 'POST',
      }),
      invalidatesTags: [{ type: 'Report', id: 'LIST' }],
    }),

    // 审核报告
    reviewReport: builder.mutation<Report, ReportReviewRequest>({
      query: ({ reportId, ...data }) => ({
        url: `/${reportId}/review`,
        method: 'POST',
        body: data,
      }),
      invalidatesTags: (result, error, { reportId }) => [
        { type: 'Report', id: reportId },
        { type: 'Report', id: 'LIST' },
      ],
    }),

    // 签署报告
    signReport: builder.mutation<Report, ReportSignRequest>({
      query: ({ reportId, ...data }) => ({
        url: `/${reportId}/sign`,
        method: 'POST',
        body: data,
      }),
      invalidatesTags: (result, error, { reportId }) => [
        { type: 'Report', id: reportId },
        { type: 'Report', id: 'LIST' },
      ],
    }),

    // 分享报告
    shareReport: builder.mutation<{ shareId: string; shareUrl: string }, ReportShareRequest>({
      query: ({ reportId, ...data }) => ({
        url: `/${reportId}/share`,
        method: 'POST',
        body: data,
      }),
    }),

    // 导出报告
    exportReports: builder.mutation<{ downloadUrl: string }, ReportExportRequest>({
      query: (data) => ({
        url: '/export',
        method: 'POST',
        body: data,
      }),
    }),

    // 搜索报告
    searchReports: builder.query<ReportSearchResponse, ReportSearchRequest>({
      query: (data) => ({
        url: '/search',
        method: 'POST',
        body: data,
      }),
    }),

    // 获取报告模板列表
    getReportTemplates: builder.query<ReportTemplate[], {
      studyType?: string
      category?: string
      active?: boolean
    }>({
      query: (params) => ({
        url: '/templates',
        params,
      }),
      providesTags: ['ReportTemplate'],
    }),

    // 获取单个报告模板
    getReportTemplate: builder.query<ReportTemplate, string>({
      query: (id) => `/templates/${id}`,
      providesTags: (result, error, id) => [{ type: 'ReportTemplate', id }],
    }),

    // 创建报告模板
    createReportTemplate: builder.mutation<ReportTemplate, Omit<ReportTemplate, 'id' | 'createdAt' | 'updatedAt'>>({
      query: (data) => ({
        url: '/templates',
        method: 'POST',
        body: data,
      }),
      invalidatesTags: ['ReportTemplate'],
    }),

    // 更新报告模板
    updateReportTemplate: builder.mutation<ReportTemplate, {
      id: string
      data: Partial<ReportTemplate>
    }>({
      query: ({ id, data }) => ({
        url: `/templates/${id}`,
        method: 'PUT',
        body: data,
      }),
      invalidatesTags: (result, error, { id }) => [
        { type: 'ReportTemplate', id },
        'ReportTemplate',
      ],
    }),

    // 删除报告模板
    deleteReportTemplate: builder.mutation<void, string>({
      query: (id) => ({
        url: `/templates/${id}`,
        method: 'DELETE',
      }),
      invalidatesTags: ['ReportTemplate'],
    }),

    // 获取报告统计信息
    getReportStatistics: builder.query<ReportStatistics, {
      dateRange?: [string, string]
      doctorId?: string
      department?: string
    }>({
      query: (params) => ({
        url: '/statistics',
        params,
      }),
      providesTags: ['ReportStats'],
    }),

    // 获取报告分析数据
    getReportAnalytics: builder.query<ReportAnalyticsResponse, ReportAnalyticsRequest>({
      query: (params) => ({
        url: '/analytics',
        params,
      }),
      providesTags: ['ReportAnalytics'],
    }),

    // 获取报告质量评估
    getReportQualityMetrics: builder.query<ReportQualityMetrics, string>({
      query: (reportId) => `/${reportId}/quality-metrics`,
    }),

    // 批量操作报告
    batchUpdateReports: builder.mutation<void, {
      reportIds: string[]
      action: 'approve' | 'reject' | 'assign' | 'priority' | 'status'
      data: Record<string, any>
    }>({
      query: (data) => ({
        url: '/batch-update',
        method: 'POST',
        body: data,
      }),
      invalidatesTags: [{ type: 'Report', id: 'LIST' }],
    }),

    // 获取报告历史版本
    getReportHistory: builder.query<Array<{
      version: number
      timestamp: string
      author: string
      changes: Array<{
        field: string
        oldValue: any
        newValue: any
      }>
      comment?: string
    }>, string>({
      query: (reportId) => `/${reportId}/history`,
    }),

    // 恢复报告版本
    restoreReportVersion: builder.mutation<Report, {
      reportId: string
      version: number
    }>({
      query: ({ reportId, version }) => ({
        url: `/${reportId}/restore/${version}`,
        method: 'POST',
      }),
      invalidatesTags: (result, error, { reportId }) => [
        { type: 'Report', id: reportId },
      ],
    }),

    // 获取报告评论
    getReportComments: builder.query<Array<{
      id: string
      author: string
      content: string
      timestamp: string
      section?: string
      resolved: boolean
    }>, string>({
      query: (reportId) => `/${reportId}/comments`,
    }),

    // 添加报告评论
    addReportComment: builder.mutation<void, {
      reportId: string
      content: string
      section?: string
    }>({
      query: ({ reportId, ...data }) => ({
        url: `/${reportId}/comments`,
        method: 'POST',
        body: data,
      }),
      invalidatesTags: (result, error, { reportId }) => [
        { type: 'Report', id: reportId },
      ],
    }),

    // 解决报告评论
    resolveReportComment: builder.mutation<void, {
      reportId: string
      commentId: string
    }>({
      query: ({ reportId, commentId }) => ({
        url: `/${reportId}/comments/${commentId}/resolve`,
        method: 'POST',
      }),
      invalidatesTags: (result, error, { reportId }) => [
        { type: 'Report', id: reportId },
      ],
    }),

    // 获取我的待办报告
    getMyPendingReports: builder.query<Report[], {
      priority?: 'high' | 'urgent'
      limit?: number
    }>({
      query: (params) => ({
        url: '/my-pending',
        params,
      }),
      providesTags: ['Report'],
    }),

    // 获取报告提醒
    getReportReminders: builder.query<Array<{
      reportId: string
      type: 'overdue' | 'due_soon' | 'follow_up'
      message: string
      dueDate: string
      priority: 'low' | 'medium' | 'high'
    }>, void>({
      query: () => '/reminders',
    }),

    // 验证报告数据
    validateReport: builder.mutation<{
      valid: boolean
      errors: Array<{
        field: string
        message: string
        severity: 'error' | 'warning'
      }>
    }, Partial<Report>>({
      query: (data) => ({
        url: '/validate',
        method: 'POST',
        body: data,
      }),
    }),
  }),
})

export const {
  useGetReportsQuery,
  useGetReportQuery,
  useCreateReportMutation,
  useUpdateReportMutation,
  useDeleteReportMutation,
  useDuplicateReportMutation,
  useReviewReportMutation,
  useSignReportMutation,
  useShareReportMutation,
  useExportReportsMutation,
  useSearchReportsQuery,
  useGetReportTemplatesQuery,
  useGetReportTemplateQuery,
  useCreateReportTemplateMutation,
  useUpdateReportTemplateMutation,
  useDeleteReportTemplateMutation,
  useGetReportStatisticsQuery,
  useGetReportAnalyticsQuery,
  useGetReportQualityMetricsQuery,
  useBatchUpdateReportsMutation,
  useGetReportHistoryQuery,
  useRestoreReportVersionMutation,
  useGetReportCommentsQuery,
  useAddReportCommentMutation,
  useResolveReportCommentMutation,
  useGetMyPendingReportsQuery,
  useGetReportRemindersQuery,
  useValidateReportMutation,
} = reportApi

export default reportApi