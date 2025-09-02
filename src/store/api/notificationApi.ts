import { createApi, fetchBaseQuery } from '@reduxjs/toolkit/query/react'
import type { RootState } from '../index'

export interface Notification {
  id: string
  type: 'info' | 'success' | 'warning' | 'error' | 'system' | 'task' | 'report' | 'ai_analysis'
  title: string
  message: string
  priority: 'low' | 'medium' | 'high' | 'urgent'
  status: 'unread' | 'read' | 'archived'
  category: 'system' | 'workflow' | 'security' | 'maintenance' | 'user_action' | 'ai_result'
  recipientId: string
  senderId?: string
  senderName?: string
  createdAt: string
  readAt?: string
  expiresAt?: string
  metadata: {
    patientId?: string
    patientName?: string
    studyId?: string
    reportId?: string
    taskId?: string
    imageId?: string
    modelId?: string
    url?: string
    actionRequired?: boolean
    actionUrl?: string
    actionText?: string
    [key: string]: any
  }
  channels: ('in_app' | 'email' | 'sms' | 'push' | 'webhook')[]
  deliveryStatus: {
    in_app?: 'pending' | 'delivered' | 'failed'
    email?: 'pending' | 'sent' | 'delivered' | 'bounced' | 'failed'
    sms?: 'pending' | 'sent' | 'delivered' | 'failed'
    push?: 'pending' | 'sent' | 'delivered' | 'failed'
    webhook?: 'pending' | 'sent' | 'delivered' | 'failed'
  }
  attachments?: Array<{
    id: string
    name: string
    type: string
    url: string
    size: number
  }>
  actions?: Array<{
    id: string
    label: string
    type: 'button' | 'link'
    url?: string
    method?: 'GET' | 'POST' | 'PUT' | 'DELETE'
    data?: Record<string, any>
    style?: 'primary' | 'secondary' | 'success' | 'warning' | 'danger'
  }>
}

export interface NotificationParams {
  page?: number
  pageSize?: number
  search?: string
  type?: string
  priority?: string
  status?: 'unread' | 'read' | 'archived'
  category?: string
  dateRange?: [string, string]
  senderId?: string
  sortBy?: 'createdAt' | 'priority' | 'title' | 'readAt'
  sortOrder?: 'asc' | 'desc'
  unreadOnly?: boolean
}

export interface NotificationResponse {
  notifications: Notification[]
  total: number
  page: number
  pageSize: number
  totalPages: number
  unreadCount: number
  summary: {
    byType: Record<string, number>
    byPriority: Record<string, number>
    byCategory: Record<string, number>
    byStatus: Record<string, number>
  }
}

export interface CreateNotificationRequest {
  type: 'info' | 'success' | 'warning' | 'error' | 'system' | 'task' | 'report' | 'ai_analysis'
  title: string
  message: string
  priority?: 'low' | 'medium' | 'high' | 'urgent'
  category?: 'system' | 'workflow' | 'security' | 'maintenance' | 'user_action' | 'ai_result'
  recipients: Array<{
    type: 'user' | 'role' | 'department' | 'group'
    id: string
  }>
  channels?: ('in_app' | 'email' | 'sms' | 'push' | 'webhook')[]
  metadata?: Record<string, any>
  expiresAt?: string
  scheduledAt?: string
  template?: {
    id: string
    variables: Record<string, any>
  }
  attachments?: Array<{
    name: string
    type: string
    content: string | File
  }>
  actions?: Array<{
    label: string
    type: 'button' | 'link'
    url?: string
    method?: 'GET' | 'POST' | 'PUT' | 'DELETE'
    data?: Record<string, any>
    style?: 'primary' | 'secondary' | 'success' | 'warning' | 'danger'
  }>
}

export interface NotificationTemplate {
  id: string
  name: string
  description: string
  type: string
  category: string
  isActive: boolean
  template: {
    title: string
    message: string
    priority: 'low' | 'medium' | 'high' | 'urgent'
    channels: ('in_app' | 'email' | 'sms' | 'push' | 'webhook')[]
    variables: Array<{
      name: string
      type: 'string' | 'number' | 'boolean' | 'date' | 'url'
      required: boolean
      defaultValue?: any
      description?: string
    }>
    conditions?: Array<{
      field: string
      operator: 'eq' | 'ne' | 'gt' | 'lt' | 'gte' | 'lte' | 'in' | 'contains'
      value: any
    }>
  }
  createdBy: string
  createdAt: string
  updatedAt: string
  usageCount: number
}

export interface NotificationRule {
  id: string
  name: string
  description: string
  isActive: boolean
  trigger: {
    event: string
    conditions: Array<{
      field: string
      operator: 'eq' | 'ne' | 'gt' | 'lt' | 'gte' | 'lte' | 'in' | 'contains' | 'regex'
      value: any
    }>
    schedule?: {
      type: 'immediate' | 'delayed' | 'scheduled'
      delay?: number
      cron?: string
      timezone?: string
    }
  }
  notification: {
    templateId?: string
    title: string
    message: string
    type: string
    priority: string
    category: string
    channels: string[]
    recipients: Array<{
      type: 'user' | 'role' | 'department' | 'group'
      id: string
    }>
  }
  priority: number
  createdBy: string
  createdAt: string
  updatedAt: string
  executionCount: number
  lastExecuted?: string
}

export interface NotificationSettings {
  userId: string
  preferences: {
    channels: {
      in_app: boolean
      email: boolean
      sms: boolean
      push: boolean
    }
    categories: Record<string, {
      enabled: boolean
      channels: string[]
      priority: 'low' | 'medium' | 'high' | 'urgent'
    }>
    schedule: {
      quietHours: {
        enabled: boolean
        start: string
        end: string
        timezone: string
      }
      weekends: {
        enabled: boolean
        channels: string[]
      }
      digest: {
        enabled: boolean
        frequency: 'daily' | 'weekly'
        time: string
        channels: string[]
      }
    }
    filters: {
      keywords: string[]
      senders: string[]
      priorities: string[]
    }
  }
  devices: Array<{
    id: string
    type: 'web' | 'mobile' | 'desktop'
    token: string
    platform: string
    isActive: boolean
    lastUsed: string
  }>
  updatedAt: string
}

export interface NotificationStatistics {
  overview: {
    totalSent: number
    totalDelivered: number
    totalRead: number
    deliveryRate: number
    readRate: number
    averageReadTime: number
  }
  byChannel: Record<string, {
    sent: number
    delivered: number
    failed: number
    deliveryRate: number
  }>
  byType: Record<string, {
    sent: number
    read: number
    readRate: number
  }>
  byPriority: Record<string, {
    sent: number
    read: number
    averageReadTime: number
  }>
  trends: {
    daily: Array<{
      date: string
      sent: number
      delivered: number
      read: number
    }>
    hourly: Array<{
      hour: number
      sent: number
      delivered: number
      read: number
    }>
  }
  performance: {
    topCategories: Array<{
      category: string
      sent: number
      readRate: number
    }>
    engagementMetrics: {
      clickThroughRate: number
      actionCompletionRate: number
      unsubscribeRate: number
    }
  }
}

export interface BulkNotificationRequest {
  notificationIds: string[]
  action: 'mark_read' | 'mark_unread' | 'archive' | 'delete'
}

export const notificationApi = createApi({
  reducerPath: 'notificationApi',
  baseQuery: fetchBaseQuery({
    baseUrl: '/api/notifications',
    prepareHeaders: (headers, { getState }) => {
      const token = (getState() as RootState).auth.token
      if (token) {
        headers.set('authorization', `Bearer ${token}`)
      }
      return headers
    },
  }),
  tagTypes: ['Notification', 'NotificationTemplate', 'NotificationRule', 'NotificationSettings', 'NotificationStats'],
  endpoints: (builder) => ({
    // 通知管理
    getNotifications: builder.query<NotificationResponse, NotificationParams>({
      query: (params) => ({
        url: '',
        params,
      }),
      providesTags: (result) =>
        result
          ? [
              ...result.notifications.map(({ id }) => ({ type: 'Notification' as const, id })),
              { type: 'Notification', id: 'LIST' },
            ]
          : [{ type: 'Notification', id: 'LIST' }],
    }),

    getNotification: builder.query<Notification, string>({
      query: (id) => `/${id}`,
      providesTags: (result, error, id) => [{ type: 'Notification', id }],
    }),

    createNotification: builder.mutation<Notification, CreateNotificationRequest>({
      query: (data) => ({
        url: '',
        method: 'POST',
        body: data,
      }),
      invalidatesTags: [{ type: 'Notification', id: 'LIST' }, 'NotificationStats'],
    }),

    markAsRead: builder.mutation<void, string>({
      query: (id) => ({
        url: `/${id}/read`,
        method: 'POST',
      }),
      invalidatesTags: (result, error, id) => [
        { type: 'Notification', id },
        { type: 'Notification', id: 'LIST' },
      ],
    }),

    markAsUnread: builder.mutation<void, string>({
      query: (id) => ({
        url: `/${id}/unread`,
        method: 'POST',
      }),
      invalidatesTags: (result, error, id) => [
        { type: 'Notification', id },
        { type: 'Notification', id: 'LIST' },
      ],
    }),

    archiveNotification: builder.mutation<void, string>({
      query: (id) => ({
        url: `/${id}/archive`,
        method: 'POST',
      }),
      invalidatesTags: (result, error, id) => [
        { type: 'Notification', id },
        { type: 'Notification', id: 'LIST' },
      ],
    }),

    deleteNotification: builder.mutation<void, string>({
      query: (id) => ({
        url: `/${id}`,
        method: 'DELETE',
      }),
      invalidatesTags: (result, error, id) => [
        { type: 'Notification', id },
        { type: 'Notification', id: 'LIST' },
      ],
    }),

    // 批量操作
    bulkUpdateNotifications: builder.mutation<void, BulkNotificationRequest>({
      query: (data) => ({
        url: '/bulk',
        method: 'POST',
        body: data,
      }),
      invalidatesTags: [{ type: 'Notification', id: 'LIST' }],
    }),

    markAllAsRead: builder.mutation<void, void>({
      query: () => ({
        url: '/mark-all-read',
        method: 'POST',
      }),
      invalidatesTags: [{ type: 'Notification', id: 'LIST' }],
    }),

    // 获取未读通知数量
    getUnreadCount: builder.query<{ count: number }, void>({
      query: () => '/unread-count',
      providesTags: ['Notification'],
    }),

    // 通知模板
    getNotificationTemplates: builder.query<NotificationTemplate[], {
      type?: string
      category?: string
      active?: boolean
    }>({
      query: (params) => ({
        url: '/templates',
        params,
      }),
      providesTags: ['NotificationTemplate'],
    }),

    getNotificationTemplate: builder.query<NotificationTemplate, string>({
      query: (id) => `/templates/${id}`,
      providesTags: (result, error, id) => [{ type: 'NotificationTemplate', id }],
    }),

    createNotificationTemplate: builder.mutation<NotificationTemplate, Omit<NotificationTemplate, 'id' | 'createdAt' | 'updatedAt' | 'usageCount'>>({
      query: (data) => ({
        url: '/templates',
        method: 'POST',
        body: data,
      }),
      invalidatesTags: ['NotificationTemplate'],
    }),

    updateNotificationTemplate: builder.mutation<NotificationTemplate, {
      id: string
      data: Partial<NotificationTemplate>
    }>({
      query: ({ id, data }) => ({
        url: `/templates/${id}`,
        method: 'PUT',
        body: data,
      }),
      invalidatesTags: (result, error, { id }) => [
        { type: 'NotificationTemplate', id },
        'NotificationTemplate',
      ],
    }),

    deleteNotificationTemplate: builder.mutation<void, string>({
      query: (id) => ({
        url: `/templates/${id}`,
        method: 'DELETE',
      }),
      invalidatesTags: ['NotificationTemplate'],
    }),

    // 通知规则
    getNotificationRules: builder.query<NotificationRule[], {
      active?: boolean
      event?: string
    }>({
      query: (params) => ({
        url: '/rules',
        params,
      }),
      providesTags: ['NotificationRule'],
    }),

    getNotificationRule: builder.query<NotificationRule, string>({
      query: (id) => `/rules/${id}`,
      providesTags: (result, error, id) => [{ type: 'NotificationRule', id }],
    }),

    createNotificationRule: builder.mutation<NotificationRule, Omit<NotificationRule, 'id' | 'createdAt' | 'updatedAt' | 'executionCount' | 'lastExecuted'>>({
      query: (data) => ({
        url: '/rules',
        method: 'POST',
        body: data,
      }),
      invalidatesTags: ['NotificationRule'],
    }),

    updateNotificationRule: builder.mutation<NotificationRule, {
      id: string
      data: Partial<NotificationRule>
    }>({
      query: ({ id, data }) => ({
        url: `/rules/${id}`,
        method: 'PUT',
        body: data,
      }),
      invalidatesTags: (result, error, { id }) => [
        { type: 'NotificationRule', id },
        'NotificationRule',
      ],
    }),

    deleteNotificationRule: builder.mutation<void, string>({
      query: (id) => ({
        url: `/rules/${id}`,
        method: 'DELETE',
      }),
      invalidatesTags: ['NotificationRule'],
    }),

    // 测试通知规则
    testNotificationRule: builder.mutation<{
      success: boolean
      matches: number
      preview: Notification[]
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

    // 用户通知设置
    getNotificationSettings: builder.query<NotificationSettings, string | void>({
      query: (userId) => userId ? `/settings/${userId}` : '/settings',
      providesTags: ['NotificationSettings'],
    }),

    updateNotificationSettings: builder.mutation<NotificationSettings, {
      userId?: string
      settings: Partial<NotificationSettings['preferences']>
    }>({
      query: ({ userId, settings }) => ({
        url: userId ? `/settings/${userId}` : '/settings',
        method: 'PUT',
        body: settings,
      }),
      invalidatesTags: ['NotificationSettings'],
    }),

    // 设备管理
    registerDevice: builder.mutation<void, {
      type: 'web' | 'mobile' | 'desktop'
      token: string
      platform: string
    }>({
      query: (data) => ({
        url: '/devices',
        method: 'POST',
        body: data,
      }),
      invalidatesTags: ['NotificationSettings'],
    }),

    unregisterDevice: builder.mutation<void, string>({
      query: (deviceId) => ({
        url: `/devices/${deviceId}`,
        method: 'DELETE',
      }),
      invalidatesTags: ['NotificationSettings'],
    }),

    // 通知统计
    getNotificationStatistics: builder.query<NotificationStatistics, {
      dateRange?: [string, string]
      userId?: string
      type?: string
      category?: string
    }>({
      query: (params) => ({
        url: '/statistics',
        params,
      }),
      providesTags: ['NotificationStats'],
    }),

    // 发送测试通知
    sendTestNotification: builder.mutation<{ success: boolean; message: string }, {
      type: string
      title: string
      message: string
      channels: string[]
      recipients: string[]
    }>({
      query: (data) => ({
        url: '/test',
        method: 'POST',
        body: data,
      }),
    }),

    // 通知历史
    getNotificationHistory: builder.query<{
      notifications: Notification[]
      total: number
    }, {
      userId?: string
      dateRange?: [string, string]
      page?: number
      pageSize?: number
    }>({
      query: (params) => ({
        url: '/history',
        params,
      }),
    }),

    // 导出通知
    exportNotifications: builder.mutation<{ downloadUrl: string }, {
      format: 'csv' | 'excel' | 'json'
      filters: NotificationParams
    }>({
      query: (data) => ({
        url: '/export',
        method: 'POST',
        body: data,
      }),
    }),

    // 通知摘要
    getNotificationDigest: builder.query<{
      period: 'daily' | 'weekly'
      summary: {
        totalNotifications: number
        unreadCount: number
        importantCount: number
        categories: Record<string, number>
      }
      highlights: Notification[]
      trends: Array<{
        date: string
        count: number
      }>
    }, {
      period: 'daily' | 'weekly'
      date?: string
    }>({
      query: (params) => ({
        url: '/digest',
        params,
      }),
    }),

    // 通知搜索
    searchNotifications: builder.query<{
      notifications: Notification[]
      total: number
      searchTime: number
    }, {
      query: string
      filters?: NotificationParams
      highlightMatches?: boolean
    }>({
      query: (params) => ({
        url: '/search',
        params,
      }),
    }),

    // 通知操作
    executeNotificationAction: builder.mutation<{ success: boolean; result?: any }, {
      notificationId: string
      actionId: string
      data?: Record<string, any>
    }>({
      query: ({ notificationId, actionId, data }) => ({
        url: `/${notificationId}/actions/${actionId}`,
        method: 'POST',
        body: data,
      }),
      invalidatesTags: (result, error, { notificationId }) => [
        { type: 'Notification', id: notificationId },
      ],
    }),

    // 通知订阅管理
    subscribeToNotifications: builder.mutation<void, {
      topics: string[]
      channels: string[]
    }>({
      query: (data) => ({
        url: '/subscribe',
        method: 'POST',
        body: data,
      }),
      invalidatesTags: ['NotificationSettings'],
    }),

    unsubscribeFromNotifications: builder.mutation<void, {
      topics: string[]
      channels?: string[]
    }>({
      query: (data) => ({
        url: '/unsubscribe',
        method: 'POST',
        body: data,
      }),
      invalidatesTags: ['NotificationSettings'],
    }),
  }),
})

export const {
  useGetNotificationsQuery,
  useGetNotificationQuery,
  useCreateNotificationMutation,
  useMarkAsReadMutation,
  useMarkAsUnreadMutation,
  useArchiveNotificationMutation,
  useDeleteNotificationMutation,
  useBulkUpdateNotificationsMutation,
  useMarkAllAsReadMutation,
  useGetUnreadCountQuery,
  useGetNotificationTemplatesQuery,
  useGetNotificationTemplateQuery,
  useCreateNotificationTemplateMutation,
  useUpdateNotificationTemplateMutation,
  useDeleteNotificationTemplateMutation,
  useGetNotificationRulesQuery,
  useGetNotificationRuleQuery,
  useCreateNotificationRuleMutation,
  useUpdateNotificationRuleMutation,
  useDeleteNotificationRuleMutation,
  useTestNotificationRuleMutation,
  useGetNotificationSettingsQuery,
  useUpdateNotificationSettingsMutation,
  useRegisterDeviceMutation,
  useUnregisterDeviceMutation,
  useGetNotificationStatisticsQuery,
  useSendTestNotificationMutation,
  useGetNotificationHistoryQuery,
  useExportNotificationsMutation,
  useGetNotificationDigestQuery,
  useSearchNotificationsQuery,
  useExecuteNotificationActionMutation,
  useSubscribeToNotificationsMutation,
  useUnsubscribeFromNotificationsMutation,
} = notificationApi

export default notificationApi