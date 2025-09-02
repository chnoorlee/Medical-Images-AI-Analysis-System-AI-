import { createApi, fetchBaseQuery } from '@reduxjs/toolkit/query/react'
import type { RootState } from '../index'
import type { User } from '../slices/authSlice'

export interface SystemUser extends User {
  lastLoginAt?: string
  loginCount: number
  isActive: boolean
  permissions: string[]
  groups: string[]
  createdBy: string
  updatedBy: string
}

export interface UserListParams {
  page?: number
  pageSize?: number
  search?: string
  role?: 'admin' | 'doctor' | 'technician' | 'nurse' | 'researcher'
  department?: string
  status?: 'active' | 'inactive' | 'suspended' | 'pending'
  sortBy?: 'name' | 'email' | 'role' | 'lastLoginAt' | 'createdAt'
  sortOrder?: 'asc' | 'desc'
  dateRange?: [string, string]
}

export interface UserListResponse {
  users: SystemUser[]
  total: number
  page: number
  pageSize: number
  totalPages: number
}

export interface CreateUserRequest {
  email: string
  firstName: string
  lastName: string
  role: 'admin' | 'doctor' | 'technician' | 'nurse' | 'researcher'
  department: string
  phone?: string
  licenseNumber?: string
  specialization?: string
  permissions: string[]
  groups?: string[]
  temporaryPassword?: boolean
  sendWelcomeEmail?: boolean
  expirationDate?: string
}

export interface UpdateUserRequest extends Partial<CreateUserRequest> {
  id: string
  isActive?: boolean
  forcePasswordReset?: boolean
}

export interface SystemSettings {
  general: {
    siteName: string
    siteDescription: string
    logoUrl?: string
    faviconUrl?: string
    timezone: string
    language: string
    dateFormat: string
    timeFormat: string
  }
  security: {
    passwordPolicy: {
      minLength: number
      requireUppercase: boolean
      requireLowercase: boolean
      requireNumbers: boolean
      requireSpecialChars: boolean
      maxAge: number
      preventReuse: number
    }
    sessionTimeout: number
    maxLoginAttempts: number
    lockoutDuration: number
    twoFactorRequired: boolean
    ipWhitelist: string[]
    allowedFileTypes: string[]
    maxFileSize: number
  }
  email: {
    smtpHost: string
    smtpPort: number
    smtpUser: string
    smtpPassword: string
    smtpSecure: boolean
    fromAddress: string
    fromName: string
    templates: Record<string, {
      subject: string
      body: string
      variables: string[]
    }>
  }
  storage: {
    provider: 'local' | 'aws' | 'azure' | 'gcp'
    config: Record<string, any>
    maxStorageSize: number
    retentionPeriod: number
    compressionEnabled: boolean
    encryptionEnabled: boolean
  }
  ai: {
    defaultModel: string
    maxConcurrentTasks: number
    analysisTimeout: number
    confidenceThreshold: number
    autoAnalysisEnabled: boolean
    modelUpdateInterval: number
    resourceLimits: {
      cpu: number
      memory: number
      gpu: number
    }
  }
  integration: {
    dicom: {
      enabled: boolean
      aeTitle: string
      port: number
      storageClass: string[]
      transferSyntax: string[]
    }
    hl7: {
      enabled: boolean
      version: string
      endpoint: string
      messageTypes: string[]
    }
    fhir: {
      enabled: boolean
      version: string
      baseUrl: string
      authentication: {
        type: 'none' | 'basic' | 'oauth2' | 'bearer'
        config: Record<string, any>
      }
    }
  }
}

export interface AuditLog {
  id: string
  timestamp: string
  userId: string
  userName: string
  action: string
  resource: string
  resourceId?: string
  details: Record<string, any>
  ipAddress: string
  userAgent: string
  success: boolean
  errorMessage?: string
}

export interface AuditLogParams {
  page?: number
  pageSize?: number
  userId?: string
  action?: string
  resource?: string
  dateRange?: [string, string]
  success?: boolean
  ipAddress?: string
}

export interface SystemMetrics {
  overview: {
    totalUsers: number
    activeUsers: number
    totalPatients: number
    totalImages: number
    totalReports: number
    storageUsed: number
    storageLimit: number
  }
  performance: {
    averageResponseTime: number
    requestsPerMinute: number
    errorRate: number
    uptime: number
    cpuUsage: number
    memoryUsage: number
    diskUsage: number
  }
  usage: {
    dailyActiveUsers: Array<{ date: string; count: number }>
    imageUploads: Array<{ date: string; count: number; size: number }>
    reportGeneration: Array<{ date: string; count: number }>
    aiAnalysis: Array<{ date: string; count: number; processingTime: number }>
  }
  errors: {
    recent: Array<{
      timestamp: string
      level: 'error' | 'warning'
      message: string
      source: string
      count: number
    }>
    byCategory: Record<string, number>
  }
}

export interface BackupConfig {
  enabled: boolean
  schedule: string
  retention: {
    daily: number
    weekly: number
    monthly: number
  }
  destination: {
    type: 'local' | 'aws' | 'azure' | 'gcp'
    config: Record<string, any>
  }
  encryption: {
    enabled: boolean
    algorithm: string
    keyId?: string
  }
  compression: boolean
  includeFiles: boolean
  excludePatterns: string[]
}

export interface BackupJob {
  id: string
  status: 'running' | 'completed' | 'failed' | 'cancelled'
  startTime: string
  endTime?: string
  size?: number
  fileCount?: number
  progress: number
  errorMessage?: string
  downloadUrl?: string
}

export interface MaintenanceMode {
  enabled: boolean
  message: string
  allowedIPs: string[]
  allowedUsers: string[]
  startTime?: string
  endTime?: string
}

export interface SystemHealth {
  status: 'healthy' | 'warning' | 'critical'
  checks: Array<{
    name: string
    status: 'pass' | 'fail' | 'warn'
    message: string
    responseTime?: number
    lastChecked: string
  }>
  dependencies: Array<{
    name: string
    type: 'database' | 'storage' | 'external_api' | 'service'
    status: 'connected' | 'disconnected' | 'degraded'
    responseTime?: number
    errorMessage?: string
  }>
}

export interface LicenseInfo {
  type: 'trial' | 'basic' | 'professional' | 'enterprise'
  status: 'active' | 'expired' | 'suspended'
  expirationDate: string
  features: Record<string, boolean>
  limits: {
    maxUsers: number
    maxPatients: number
    maxStorage: number
    maxAIAnalyses: number
  }
  usage: {
    currentUsers: number
    currentPatients: number
    currentStorage: number
    currentAIAnalyses: number
  }
}

export const adminApi = createApi({
  reducerPath: 'adminApi',
  baseQuery: fetchBaseQuery({
    baseUrl: '/api/admin',
    prepareHeaders: (headers, { getState }) => {
      const token = (getState() as RootState).auth.token
      if (token) {
        headers.set('authorization', `Bearer ${token}`)
      }
      return headers
    },
  }),
  tagTypes: ['User', 'Settings', 'AuditLog', 'Metrics', 'Backup', 'Health', 'License'],
  endpoints: (builder) => ({
    // 用户管理
    getUsers: builder.query<UserListResponse, UserListParams>({
      query: (params) => ({
        url: '/users',
        params,
      }),
      providesTags: (result) =>
        result
          ? [
              ...result.users.map(({ id }) => ({ type: 'User' as const, id })),
              { type: 'User', id: 'LIST' },
            ]
          : [{ type: 'User', id: 'LIST' }],
    }),

    getUser: builder.query<SystemUser, string>({
      query: (id) => `/users/${id}`,
      providesTags: (result, error, id) => [{ type: 'User', id }],
    }),

    createUser: builder.mutation<SystemUser, CreateUserRequest>({
      query: (data) => ({
        url: '/users',
        method: 'POST',
        body: data,
      }),
      invalidatesTags: [{ type: 'User', id: 'LIST' }],
    }),

    updateUser: builder.mutation<SystemUser, UpdateUserRequest>({
      query: ({ id, ...data }) => ({
        url: `/users/${id}`,
        method: 'PUT',
        body: data,
      }),
      invalidatesTags: (result, error, { id }) => [
        { type: 'User', id },
        { type: 'User', id: 'LIST' },
      ],
    }),

    deleteUser: builder.mutation<void, string>({
      query: (id) => ({
        url: `/users/${id}`,
        method: 'DELETE',
      }),
      invalidatesTags: (result, error, id) => [
        { type: 'User', id },
        { type: 'User', id: 'LIST' },
      ],
    }),

    suspendUser: builder.mutation<void, { id: string; reason: string; duration?: number }>({
      query: ({ id, ...data }) => ({
        url: `/users/${id}/suspend`,
        method: 'POST',
        body: data,
      }),
      invalidatesTags: (result, error, { id }) => [
        { type: 'User', id },
        { type: 'User', id: 'LIST' },
      ],
    }),

    activateUser: builder.mutation<void, string>({
      query: (id) => ({
        url: `/users/${id}/activate`,
        method: 'POST',
      }),
      invalidatesTags: (result, error, id) => [
        { type: 'User', id },
        { type: 'User', id: 'LIST' },
      ],
    }),

    resetUserPassword: builder.mutation<{ temporaryPassword: string }, string>({
      query: (id) => ({
        url: `/users/${id}/reset-password`,
        method: 'POST',
      }),
    }),

    // 系统设置
    getSystemSettings: builder.query<SystemSettings, void>({
      query: () => '/settings',
      providesTags: ['Settings'],
    }),

    updateSystemSettings: builder.mutation<SystemSettings, Partial<SystemSettings>>({
      query: (data) => ({
        url: '/settings',
        method: 'PUT',
        body: data,
      }),
      invalidatesTags: ['Settings'],
    }),

    testEmailSettings: builder.mutation<{ success: boolean; message: string }, {
      to: string
      subject: string
      body: string
    }>({
      query: (data) => ({
        url: '/settings/test-email',
        method: 'POST',
        body: data,
      }),
    }),

    testDatabaseConnection: builder.mutation<{ success: boolean; message: string }, Record<string, any>>({
      query: (config) => ({
        url: '/settings/test-database',
        method: 'POST',
        body: config,
      }),
    }),

    // 审计日志
    getAuditLogs: builder.query<{
      logs: AuditLog[]
      total: number
      page: number
      pageSize: number
    }, AuditLogParams>({
      query: (params) => ({
        url: '/audit-logs',
        params,
      }),
      providesTags: ['AuditLog'],
    }),

    exportAuditLogs: builder.mutation<{ downloadUrl: string }, {
      format: 'csv' | 'json' | 'pdf'
      filters: AuditLogParams
    }>({
      query: (data) => ({
        url: '/audit-logs/export',
        method: 'POST',
        body: data,
      }),
    }),

    // 系统指标
    getSystemMetrics: builder.query<SystemMetrics, {
      period: 'hour' | 'day' | 'week' | 'month'
      startDate?: string
      endDate?: string
    }>({
      query: (params) => ({
        url: '/metrics',
        params,
      }),
      providesTags: ['Metrics'],
    }),

    // 备份管理
    getBackupConfig: builder.query<BackupConfig, void>({
      query: () => '/backup/config',
      providesTags: ['Backup'],
    }),

    updateBackupConfig: builder.mutation<BackupConfig, Partial<BackupConfig>>({
      query: (data) => ({
        url: '/backup/config',
        method: 'PUT',
        body: data,
      }),
      invalidatesTags: ['Backup'],
    }),

    startBackup: builder.mutation<BackupJob, {
      type: 'full' | 'incremental'
      description?: string
    }>({
      query: (data) => ({
        url: '/backup/start',
        method: 'POST',
        body: data,
      }),
      invalidatesTags: ['Backup'],
    }),

    getBackupJobs: builder.query<{
      jobs: BackupJob[]
      total: number
    }, {
      page?: number
      pageSize?: number
      status?: string
    }>({
      query: (params) => ({
        url: '/backup/jobs',
        params,
      }),
      providesTags: ['Backup'],
    }),

    getBackupJob: builder.query<BackupJob, string>({
      query: (id) => `/backup/jobs/${id}`,
    }),

    cancelBackupJob: builder.mutation<void, string>({
      query: (id) => ({
        url: `/backup/jobs/${id}/cancel`,
        method: 'POST',
      }),
      invalidatesTags: ['Backup'],
    }),

    restoreBackup: builder.mutation<{ jobId: string }, {
      backupId: string
      options: {
        overwriteExisting: boolean
        restoreFiles: boolean
        restoreDatabase: boolean
      }
    }>({
      query: (data) => ({
        url: '/backup/restore',
        method: 'POST',
        body: data,
      }),
    }),

    // 维护模式
    getMaintenanceMode: builder.query<MaintenanceMode, void>({
      query: () => '/maintenance',
    }),

    setMaintenanceMode: builder.mutation<MaintenanceMode, Partial<MaintenanceMode>>({
      query: (data) => ({
        url: '/maintenance',
        method: 'PUT',
        body: data,
      }),
    }),

    // 系统健康检查
    getSystemHealth: builder.query<SystemHealth, void>({
      query: () => '/health',
      providesTags: ['Health'],
    }),

    runHealthCheck: builder.mutation<SystemHealth, { checks?: string[] }>({
      query: (data) => ({
        url: '/health/check',
        method: 'POST',
        body: data,
      }),
      invalidatesTags: ['Health'],
    }),

    // 许可证管理
    getLicenseInfo: builder.query<LicenseInfo, void>({
      query: () => '/license',
      providesTags: ['License'],
    }),

    updateLicense: builder.mutation<LicenseInfo, { licenseKey: string }>({
      query: (data) => ({
        url: '/license',
        method: 'PUT',
        body: data,
      }),
      invalidatesTags: ['License'],
    }),

    // 系统日志
    getSystemLogs: builder.query<{
      logs: Array<{
        timestamp: string
        level: 'debug' | 'info' | 'warn' | 'error'
        message: string
        source: string
        metadata?: Record<string, any>
      }>
      total: number
    }, {
      page?: number
      pageSize?: number
      level?: string
      source?: string
      dateRange?: [string, string]
      search?: string
    }>({
      query: (params) => ({
        url: '/logs',
        params,
      }),
    }),

    downloadSystemLogs: builder.mutation<{ downloadUrl: string }, {
      format: 'txt' | 'json'
      filters: Record<string, any>
    }>({
      query: (data) => ({
        url: '/logs/download',
        method: 'POST',
        body: data,
      }),
    }),

    // 系统配置导入导出
    exportSystemConfig: builder.mutation<{ downloadUrl: string }, {
      includeSecrets: boolean
      sections: string[]
    }>({
      query: (data) => ({
        url: '/config/export',
        method: 'POST',
        body: data,
      }),
    }),

    importSystemConfig: builder.mutation<{ success: boolean; message: string }, {
      configFile: File
      overwriteExisting: boolean
      validateOnly: boolean
    }>({
      query: (data) => {
        const formData = new FormData()
        formData.append('configFile', data.configFile)
        formData.append('overwriteExisting', String(data.overwriteExisting))
        formData.append('validateOnly', String(data.validateOnly))
        return {
          url: '/config/import',
          method: 'POST',
          body: formData,
        }
      },
    }),

    // 系统更新
    checkForUpdates: builder.query<{
      available: boolean
      currentVersion: string
      latestVersion?: string
      releaseNotes?: string
      downloadUrl?: string
    }, void>({
      query: () => '/updates/check',
    }),

    downloadUpdate: builder.mutation<{ downloadId: string }, string>({
      query: (version) => ({
        url: `/updates/download/${version}`,
        method: 'POST',
      }),
    }),

    installUpdate: builder.mutation<{ success: boolean; message: string }, {
      downloadId: string
      scheduledTime?: string
    }>({
      query: (data) => ({
        url: '/updates/install',
        method: 'POST',
        body: data,
      }),
    }),

    // 性能优化
    optimizeDatabase: builder.mutation<{
      success: boolean
      message: string
      statistics: Record<string, any>
    }, {
      operations: ('vacuum' | 'reindex' | 'analyze' | 'cleanup')[]
    }>({
      query: (data) => ({
        url: '/optimize/database',
        method: 'POST',
        body: data,
      }),
    }),

    clearCache: builder.mutation<{ success: boolean; message: string }, {
      cacheTypes: ('redis' | 'memory' | 'file' | 'database')[]
    }>({
      query: (data) => ({
        url: '/optimize/cache',
        method: 'POST',
        body: data,
      }),
    }),

    // 安全扫描
    runSecurityScan: builder.mutation<{
      scanId: string
      status: 'running' | 'completed' | 'failed'
      results?: {
        vulnerabilities: Array<{
          severity: 'low' | 'medium' | 'high' | 'critical'
          type: string
          description: string
          recommendation: string
        }>
        score: number
        summary: string
      }
    }, {
      scanType: 'full' | 'quick' | 'custom'
      options?: Record<string, any>
    }>({
      query: (data) => ({
        url: '/security/scan',
        method: 'POST',
        body: data,
      }),
    }),

    getSecurityScans: builder.query<Array<{
      scanId: string
      scanType: string
      status: string
      startTime: string
      endTime?: string
      score?: number
      vulnerabilityCount: number
    }>, {
      page?: number
      pageSize?: number
    }>({
      query: (params) => ({
        url: '/security/scans',
        params,
      }),
    }),
  }),
})

export const {
  useGetUsersQuery,
  useGetUserQuery,
  useCreateUserMutation,
  useUpdateUserMutation,
  useDeleteUserMutation,
  useSuspendUserMutation,
  useActivateUserMutation,
  useResetUserPasswordMutation,
  useGetSystemSettingsQuery,
  useUpdateSystemSettingsMutation,
  useTestEmailSettingsMutation,
  useTestDatabaseConnectionMutation,
  useGetAuditLogsQuery,
  useExportAuditLogsMutation,
  useGetSystemMetricsQuery,
  useGetBackupConfigQuery,
  useUpdateBackupConfigMutation,
  useStartBackupMutation,
  useGetBackupJobsQuery,
  useGetBackupJobQuery,
  useCancelBackupJobMutation,
  useRestoreBackupMutation,
  useGetMaintenanceModeQuery,
  useSetMaintenanceModeMutation,
  useGetSystemHealthQuery,
  useRunHealthCheckMutation,
  useGetLicenseInfoQuery,
  useUpdateLicenseMutation,
  useGetSystemLogsQuery,
  useDownloadSystemLogsMutation,
  useExportSystemConfigMutation,
  useImportSystemConfigMutation,
  useCheckForUpdatesQuery,
  useDownloadUpdateMutation,
  useInstallUpdateMutation,
  useOptimizeDatabaseMutation,
  useClearCacheMutation,
  useRunSecurityScanMutation,
  useGetSecurityScansQuery,
} = adminApi

export default adminApi