import { createApi, fetchBaseQuery } from '@reduxjs/toolkit/query/react'
import type { RootState } from '../index'

export interface WorklistItem {
  id: string
  type: 'study_review' | 'report_approval' | 'ai_analysis' | 'quality_check' | 'follow_up'
  title: string
  description: string
  priority: 'low' | 'medium' | 'high' | 'urgent'
  status: 'pending' | 'in_progress' | 'completed' | 'cancelled' | 'overdue'
  assignedTo: string
  assignedBy: string
  createdAt: string
  updatedAt: string
  dueDate?: string
  estimatedDuration?: number
  actualDuration?: number
  tags: string[]
  metadata: {
    patientId?: string
    patientName?: string
    studyId?: string
    studyType?: string
    imageId?: string
    reportId?: string
    modelId?: string
    department?: string
    location?: string
    [key: string]: any
  }
  dependencies?: string[]
  attachments?: Array<{
    id: string
    name: string
    type: string
    url: string
    size: number
  }>
  comments?: Array<{
    id: string
    author: string
    content: string
    timestamp: string
  }>
  history: Array<{
    timestamp: string
    action: string
    user: string
    details: string
  }>
}

export interface WorklistParams {
  page?: number
  pageSize?: number
  search?: string
  type?: string
  priority?: string
  status?: string
  assignedTo?: string
  assignedBy?: string
  department?: string
  dateRange?: [string, string]
  dueDateRange?: [string, string]
  tags?: string[]
  sortBy?: 'priority' | 'dueDate' | 'createdAt' | 'updatedAt' | 'title'
  sortOrder?: 'asc' | 'desc'
  overdue?: boolean
  myTasks?: boolean
}

export interface WorklistResponse {
  items: WorklistItem[]
  total: number
  page: number
  pageSize: number
  totalPages: number
  summary: {
    pending: number
    inProgress: number
    completed: number
    overdue: number
    byPriority: Record<string, number>
    byType: Record<string, number>
  }
}

export interface CreateWorklistItemRequest {
  type: 'study_review' | 'report_approval' | 'ai_analysis' | 'quality_check' | 'follow_up'
  title: string
  description: string
  priority: 'low' | 'medium' | 'high' | 'urgent'
  assignedTo: string
  dueDate?: string
  estimatedDuration?: number
  tags?: string[]
  metadata?: Record<string, any>
  dependencies?: string[]
  attachments?: Array<{
    name: string
    type: string
    content: string | File
  }>
}

export interface UpdateWorklistItemRequest extends Partial<CreateWorklistItemRequest> {
  id: string
  status?: 'pending' | 'in_progress' | 'completed' | 'cancelled'
  actualDuration?: number
  completionNotes?: string
}

export interface WorklistTemplate {
  id: string
  name: string
  description: string
  type: string
  category: string
  isActive: boolean
  template: {
    title: string
    description: string
    priority: 'low' | 'medium' | 'high' | 'urgent'
    estimatedDuration: number
    tags: string[]
    steps: Array<{
      id: string
      title: string
      description: string
      required: boolean
      type: 'text' | 'checkbox' | 'file' | 'selection'
      options?: string[]
    }>
    dependencies: string[]
    notifications: {
      onCreate: boolean
      onAssign: boolean
      onDue: boolean
      onComplete: boolean
    }
  }
  createdBy: string
  createdAt: string
  updatedAt: string
  usageCount: number
}

export interface WorklistRule {
  id: string
  name: string
  description: string
  isActive: boolean
  trigger: {
    event: 'patient_created' | 'image_uploaded' | 'report_created' | 'analysis_completed' | 'scheduled'
    conditions: Array<{
      field: string
      operator: 'eq' | 'ne' | 'gt' | 'lt' | 'gte' | 'lte' | 'in' | 'contains' | 'regex'
      value: any
    }>
    schedule?: {
      type: 'once' | 'recurring'
      cron?: string
      timezone?: string
    }
  }
  actions: Array<{
    type: 'create_task' | 'assign_task' | 'send_notification' | 'update_status' | 'run_analysis'
    config: Record<string, any>
  }>
  priority: number
  createdBy: string
  createdAt: string
  updatedAt: string
  executionCount: number
  lastExecuted?: string
}

export interface WorklistStatistics {
  overview: {
    totalTasks: number
    pendingTasks: number
    inProgressTasks: number
    completedTasks: number
    overdueTasks: number
    averageCompletionTime: number
    completionRate: number
  }
  byUser: Array<{
    userId: string
    userName: string
    assignedTasks: number
    completedTasks: number
    averageCompletionTime: number
    completionRate: number
    overdueCount: number
  }>
  byType: Array<{
    type: string
    count: number
    averageCompletionTime: number
    completionRate: number
  }>
  byPriority: Record<string, {
    count: number
    averageCompletionTime: number
    completionRate: number
  }>
  trends: {
    daily: Array<{
      date: string
      created: number
      completed: number
      overdue: number
    }>
    weekly: Array<{
      week: string
      created: number
      completed: number
      overdue: number
    }>
    monthly: Array<{
      month: string
      created: number
      completed: number
      overdue: number
    }>
  }
  performance: {
    bottlenecks: Array<{
      type: string
      averageTime: number
      taskCount: number
      suggestions: string[]
    }>
    topPerformers: Array<{
      userId: string
      userName: string
      completionRate: number
      averageTime: number
    }>
  }
}

export interface BulkActionRequest {
  itemIds: string[]
  action: 'assign' | 'update_priority' | 'update_status' | 'add_tags' | 'remove_tags' | 'delete'
  data: Record<string, any>
}

export interface WorklistCalendar {
  date: string
  tasks: Array<{
    id: string
    title: string
    type: string
    priority: string
    status: string
    assignedTo: string
    dueTime?: string
    estimatedDuration?: number
  }>
  summary: {
    totalTasks: number
    byPriority: Record<string, number>
    byStatus: Record<string, number>
    totalDuration: number
  }
}

export const worklistApi = createApi({
  reducerPath: 'worklistApi',
  baseQuery: fetchBaseQuery({
    baseUrl: '/api/worklist',
    prepareHeaders: (headers, { getState }) => {
      const token = (getState() as RootState).auth.token
      if (token) {
        headers.set('authorization', `Bearer ${token}`)
      }
      return headers
    },
  }),
  tagTypes: ['WorklistItem', 'WorklistTemplate', 'WorklistRule', 'WorklistStats'],
  endpoints: (builder) => ({
    // 工作列表项管理
    getWorklistItems: builder.query<WorklistResponse, WorklistParams>({
      query: (params) => ({
        url: '/items',
        params,
      }),
      providesTags: (result) =>
        result
          ? [
              ...result.items.map(({ id }) => ({ type: 'WorklistItem' as const, id })),
              { type: 'WorklistItem', id: 'LIST' },
            ]
          : [{ type: 'WorklistItem', id: 'LIST' }],
    }),

    getWorklistItem: builder.query<WorklistItem, string>({
      query: (id) => `/items/${id}`,
      providesTags: (result, error, id) => [{ type: 'WorklistItem', id }],
    }),

    createWorklistItem: builder.mutation<WorklistItem, CreateWorklistItemRequest>({
      query: (data) => ({
        url: '/items',
        method: 'POST',
        body: data,
      }),
      invalidatesTags: [{ type: 'WorklistItem', id: 'LIST' }, 'WorklistStats'],
    }),

    updateWorklistItem: builder.mutation<WorklistItem, UpdateWorklistItemRequest>({
      query: ({ id, ...data }) => ({
        url: `/items/${id}`,
        method: 'PUT',
        body: data,
      }),
      invalidatesTags: (result, error, { id }) => [
        { type: 'WorklistItem', id },
        { type: 'WorklistItem', id: 'LIST' },
        'WorklistStats',
      ],
    }),

    deleteWorklistItem: builder.mutation<void, string>({
      query: (id) => ({
        url: `/items/${id}`,
        method: 'DELETE',
      }),
      invalidatesTags: (result, error, id) => [
        { type: 'WorklistItem', id },
        { type: 'WorklistItem', id: 'LIST' },
        'WorklistStats',
      ],
    }),

    // 批量操作
    bulkUpdateWorklistItems: builder.mutation<void, BulkActionRequest>({
      query: (data) => ({
        url: '/items/bulk',
        method: 'POST',
        body: data,
      }),
      invalidatesTags: [{ type: 'WorklistItem', id: 'LIST' }, 'WorklistStats'],
    }),

    // 任务分配
    assignTask: builder.mutation<WorklistItem, {
      itemId: string
      assignedTo: string
      notes?: string
    }>({
      query: ({ itemId, ...data }) => ({
        url: `/items/${itemId}/assign`,
        method: 'POST',
        body: data,
      }),
      invalidatesTags: (result, error, { itemId }) => [
        { type: 'WorklistItem', id: itemId },
        { type: 'WorklistItem', id: 'LIST' },
      ],
    }),

    // 任务状态更新
    updateTaskStatus: builder.mutation<WorklistItem, {
      itemId: string
      status: 'pending' | 'in_progress' | 'completed' | 'cancelled'
      notes?: string
      actualDuration?: number
    }>({
      query: ({ itemId, ...data }) => ({
        url: `/items/${itemId}/status`,
        method: 'POST',
        body: data,
      }),
      invalidatesTags: (result, error, { itemId }) => [
        { type: 'WorklistItem', id: itemId },
        { type: 'WorklistItem', id: 'LIST' },
        'WorklistStats',
      ],
    }),

    // 添加评论
    addComment: builder.mutation<void, {
      itemId: string
      content: string
    }>({
      query: ({ itemId, content }) => ({
        url: `/items/${itemId}/comments`,
        method: 'POST',
        body: { content },
      }),
      invalidatesTags: (result, error, { itemId }) => [
        { type: 'WorklistItem', id: itemId },
      ],
    }),

    // 工作列表模板
    getWorklistTemplates: builder.query<WorklistTemplate[], {
      type?: string
      category?: string
      active?: boolean
    }>({
      query: (params) => ({
        url: '/templates',
        params,
      }),
      providesTags: ['WorklistTemplate'],
    }),

    getWorklistTemplate: builder.query<WorklistTemplate, string>({
      query: (id) => `/templates/${id}`,
      providesTags: (result, error, id) => [{ type: 'WorklistTemplate', id }],
    }),

    createWorklistTemplate: builder.mutation<WorklistTemplate, Omit<WorklistTemplate, 'id' | 'createdAt' | 'updatedAt' | 'usageCount'>>({
      query: (data) => ({
        url: '/templates',
        method: 'POST',
        body: data,
      }),
      invalidatesTags: ['WorklistTemplate'],
    }),

    updateWorklistTemplate: builder.mutation<WorklistTemplate, {
      id: string
      data: Partial<WorklistTemplate>
    }>({
      query: ({ id, data }) => ({
        url: `/templates/${id}`,
        method: 'PUT',
        body: data,
      }),
      invalidatesTags: (result, error, { id }) => [
        { type: 'WorklistTemplate', id },
        'WorklistTemplate',
      ],
    }),

    deleteWorklistTemplate: builder.mutation<void, string>({
      query: (id) => ({
        url: `/templates/${id}`,
        method: 'DELETE',
      }),
      invalidatesTags: ['WorklistTemplate'],
    }),

    // 从模板创建任务
    createFromTemplate: builder.mutation<WorklistItem, {
      templateId: string
      data: Record<string, any>
    }>({
      query: ({ templateId, data }) => ({
        url: `/templates/${templateId}/create`,
        method: 'POST',
        body: data,
      }),
      invalidatesTags: [{ type: 'WorklistItem', id: 'LIST' }],
    }),

    // 工作列表规则
    getWorklistRules: builder.query<WorklistRule[], {
      active?: boolean
      event?: string
    }>({
      query: (params) => ({
        url: '/rules',
        params,
      }),
      providesTags: ['WorklistRule'],
    }),

    getWorklistRule: builder.query<WorklistRule, string>({
      query: (id) => `/rules/${id}`,
      providesTags: (result, error, id) => [{ type: 'WorklistRule', id }],
    }),

    createWorklistRule: builder.mutation<WorklistRule, Omit<WorklistRule, 'id' | 'createdAt' | 'updatedAt' | 'executionCount' | 'lastExecuted'>>({
      query: (data) => ({
        url: '/rules',
        method: 'POST',
        body: data,
      }),
      invalidatesTags: ['WorklistRule'],
    }),

    updateWorklistRule: builder.mutation<WorklistRule, {
      id: string
      data: Partial<WorklistRule>
    }>({
      query: ({ id, data }) => ({
        url: `/rules/${id}`,
        method: 'PUT',
        body: data,
      }),
      invalidatesTags: (result, error, { id }) => [
        { type: 'WorklistRule', id },
        'WorklistRule',
      ],
    }),

    deleteWorklistRule: builder.mutation<void, string>({
      query: (id) => ({
        url: `/rules/${id}`,
        method: 'DELETE',
      }),
      invalidatesTags: ['WorklistRule'],
    }),

    // 测试规则
    testWorklistRule: builder.mutation<{
      success: boolean
      matches: number
      preview: WorklistItem[]
    }, {
      ruleId: string
      testData?: Record<string, any>
    }>({
      query: ({ ruleId, testData }) => ({
        url: `/rules/${ruleId}/test`,
        method: 'POST',
        body: { testData },
      }),
    }),

    // 统计信息
    getWorklistStatistics: builder.query<WorklistStatistics, {
      dateRange?: [string, string]
      userId?: string
      department?: string
      type?: string
    }>({
      query: (params) => ({
        url: '/statistics',
        params,
      }),
      providesTags: ['WorklistStats'],
    }),

    // 日历视图
    getWorklistCalendar: builder.query<WorklistCalendar[], {
      startDate: string
      endDate: string
      userId?: string
      type?: string
    }>({
      query: (params) => ({
        url: '/calendar',
        params,
      }),
    }),

    // 我的任务
    getMyTasks: builder.query<WorklistResponse, {
      status?: string
      priority?: string
      dueToday?: boolean
      overdue?: boolean
      limit?: number
    }>({
      query: (params) => ({
        url: '/my-tasks',
        params,
      }),
      providesTags: ['WorklistItem'],
    }),

    // 任务提醒
    getTaskReminders: builder.query<Array<{
      itemId: string
      title: string
      type: 'due_soon' | 'overdue' | 'dependency_ready'
      message: string
      dueDate: string
      priority: string
    }>, void>({
      query: () => '/reminders',
    }),

    // 导出工作列表
    exportWorklist: builder.mutation<{ downloadUrl: string }, {
      format: 'csv' | 'excel' | 'pdf'
      filters: WorklistParams
      includeComments?: boolean
      includeHistory?: boolean
    }>({
      query: (data) => ({
        url: '/export',
        method: 'POST',
        body: data,
      }),
    }),

    // 工作流集成
    triggerWorkflow: builder.mutation<{ workflowId: string }, {
      itemId: string
      workflowType: string
      parameters?: Record<string, any>
    }>({
      query: (data) => ({
        url: '/workflow/trigger',
        method: 'POST',
        body: data,
      }),
    }),

    // 任务依赖管理
    updateDependencies: builder.mutation<WorklistItem, {
      itemId: string
      dependencies: string[]
    }>({
      query: ({ itemId, dependencies }) => ({
        url: `/items/${itemId}/dependencies`,
        method: 'PUT',
        body: { dependencies },
      }),
      invalidatesTags: (result, error, { itemId }) => [
        { type: 'WorklistItem', id: itemId },
      ],
    }),

    // 获取可用的依赖项
    getAvailableDependencies: builder.query<Array<{
      id: string
      title: string
      type: string
      status: string
    }>, {
      excludeId?: string
      type?: string
    }>({
      query: (params) => ({
        url: '/dependencies',
        params,
      }),
    }),
  }),
})

export const {
  useGetWorklistItemsQuery,
  useGetWorklistItemQuery,
  useCreateWorklistItemMutation,
  useUpdateWorklistItemMutation,
  useDeleteWorklistItemMutation,
  useBulkUpdateWorklistItemsMutation,
  useAssignTaskMutation,
  useUpdateTaskStatusMutation,
  useAddCommentMutation,
  useGetWorklistTemplatesQuery,
  useGetWorklistTemplateQuery,
  useCreateWorklistTemplateMutation,
  useUpdateWorklistTemplateMutation,
  useDeleteWorklistTemplateMutation,
  useCreateFromTemplateMutation,
  useGetWorklistRulesQuery,
  useGetWorklistRuleQuery,
  useCreateWorklistRuleMutation,
  useUpdateWorklistRuleMutation,
  useDeleteWorklistRuleMutation,
  useTestWorklistRuleMutation,
  useGetWorklistStatisticsQuery,
  useGetWorklistCalendarQuery,
  useGetMyTasksQuery,
  useGetTaskRemindersQuery,
  useExportWorklistMutation,
  useTriggerWorkflowMutation,
  useUpdateDependenciesMutation,
  useGetAvailableDependenciesQuery,
} = worklistApi

export default worklistApi