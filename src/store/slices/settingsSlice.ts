import { createSlice, createAsyncThunk, PayloadAction } from '@reduxjs/toolkit'

export interface UserPreferences {
  theme: 'light' | 'dark' | 'auto'
  language: 'zh-CN' | 'en-US'
  timezone: string
  dateFormat: string
  timeFormat: '12h' | '24h'
  pageSize: number
  autoSave: boolean
  notifications: {
    email: boolean
    push: boolean
    sound: boolean
    desktop: boolean
  }
  shortcuts: Record<string, string>
}

export interface SystemSettings {
  siteName: string
  siteDescription: string
  logo: string
  favicon: string
  maintenanceMode: boolean
  registrationEnabled: boolean
  emailVerificationRequired: boolean
  maxFileSize: number
  allowedFileTypes: string[]
  sessionTimeout: number
  passwordPolicy: {
    minLength: number
    requireUppercase: boolean
    requireLowercase: boolean
    requireNumbers: boolean
    requireSpecialChars: boolean
    expirationDays: number
  }
  backup: {
    enabled: boolean
    frequency: 'daily' | 'weekly' | 'monthly'
    retention: number
    location: string
  }
  security: {
    twoFactorAuth: boolean
    ipWhitelist: string[]
    maxLoginAttempts: number
    lockoutDuration: number
  }
}

export interface AISettings {
  defaultModel: string
  analysisTimeout: number
  maxConcurrentAnalysis: number
  confidenceThreshold: number
  autoAnalysis: boolean
  saveAnalysisHistory: boolean
  modelUpdateCheck: boolean
  gpuAcceleration: boolean
  batchProcessing: {
    enabled: boolean
    batchSize: number
    priority: 'low' | 'medium' | 'high'
  }
  qualityControl: {
    enabled: boolean
    minImageQuality: number
    autoReject: boolean
  }
}

export interface NotificationSettings {
  email: {
    enabled: boolean
    smtp: {
      host: string
      port: number
      secure: boolean
      username: string
      password: string
    }
    templates: Record<string, {
      subject: string
      body: string
    }>
  }
  push: {
    enabled: boolean
    vapidKeys: {
      publicKey: string
      privateKey: string
    }
  }
  webhook: {
    enabled: boolean
    url: string
    secret: string
    events: string[]
  }
}

export interface StorageSettings {
  provider: 'local' | 'aws' | 'azure' | 'gcp'
  local: {
    path: string
    maxSize: number
  }
  aws: {
    accessKeyId: string
    secretAccessKey: string
    region: string
    bucket: string
  }
  azure: {
    connectionString: string
    containerName: string
  }
  gcp: {
    projectId: string
    keyFile: string
    bucket: string
  }
  cleanup: {
    enabled: boolean
    retentionDays: number
    schedule: string
  }
}

export interface DatabaseSettings {
  type: 'postgresql' | 'mysql' | 'sqlite'
  host: string
  port: number
  database: string
  username: string
  password: string
  ssl: boolean
  poolSize: number
  backup: {
    enabled: boolean
    schedule: string
    retention: number
  }
  monitoring: {
    enabled: boolean
    slowQueryThreshold: number
    logQueries: boolean
  }
}

interface SettingsState {
  userPreferences: UserPreferences
  systemSettings: SystemSettings
  aiSettings: AISettings
  notificationSettings: NotificationSettings
  storageSettings: StorageSettings
  databaseSettings: DatabaseSettings
  loading: boolean
  error: string | null
  hasUnsavedChanges: boolean
  lastSaved: string | null
}

const initialState: SettingsState = {
  userPreferences: {
    theme: 'light',
    language: 'zh-CN',
    timezone: 'Asia/Shanghai',
    dateFormat: 'YYYY-MM-DD',
    timeFormat: '24h',
    pageSize: 10,
    autoSave: true,
    notifications: {
      email: true,
      push: true,
      sound: true,
      desktop: false
    },
    shortcuts: {
      'save': 'Ctrl+S',
      'search': 'Ctrl+F',
      'new': 'Ctrl+N',
      'refresh': 'F5'
    }
  },
  systemSettings: {
    siteName: '医疗AI影像分析系统',
    siteDescription: '基于人工智能的医疗影像分析平台',
    logo: '/logo.png',
    favicon: '/favicon.ico',
    maintenanceMode: false,
    registrationEnabled: true,
    emailVerificationRequired: false,
    maxFileSize: 100 * 1024 * 1024, // 100MB
    allowedFileTypes: ['dcm', 'jpg', 'jpeg', 'png', 'tiff'],
    sessionTimeout: 3600, // 1 hour
    passwordPolicy: {
      minLength: 8,
      requireUppercase: true,
      requireLowercase: true,
      requireNumbers: true,
      requireSpecialChars: false,
      expirationDays: 90
    },
    backup: {
      enabled: true,
      frequency: 'daily',
      retention: 30,
      location: '/backups'
    },
    security: {
      twoFactorAuth: false,
      ipWhitelist: [],
      maxLoginAttempts: 5,
      lockoutDuration: 900 // 15 minutes
    }
  },
  aiSettings: {
    defaultModel: 'lung-nodule-v2',
    analysisTimeout: 300, // 5 minutes
    maxConcurrentAnalysis: 3,
    confidenceThreshold: 0.7,
    autoAnalysis: true,
    saveAnalysisHistory: true,
    modelUpdateCheck: true,
    gpuAcceleration: true,
    batchProcessing: {
      enabled: true,
      batchSize: 10,
      priority: 'medium'
    },
    qualityControl: {
      enabled: true,
      minImageQuality: 0.8,
      autoReject: false
    }
  },
  notificationSettings: {
    email: {
      enabled: false,
      smtp: {
        host: '',
        port: 587,
        secure: false,
        username: '',
        password: ''
      },
      templates: {
        'analysis_complete': {
          subject: 'AI分析完成通知',
          body: '您的影像分析已完成，请查看结果。'
        },
        'report_ready': {
          subject: '报告生成完成',
          body: '您的医疗报告已生成完成，请及时查看。'
        }
      }
    },
    push: {
      enabled: false,
      vapidKeys: {
        publicKey: '',
        privateKey: ''
      }
    },
    webhook: {
      enabled: false,
      url: '',
      secret: '',
      events: []
    }
  },
  storageSettings: {
    provider: 'local',
    local: {
      path: '/uploads',
      maxSize: 1024 * 1024 * 1024 // 1GB
    },
    aws: {
      accessKeyId: '',
      secretAccessKey: '',
      region: 'us-east-1',
      bucket: ''
    },
    azure: {
      connectionString: '',
      containerName: ''
    },
    gcp: {
      projectId: '',
      keyFile: '',
      bucket: ''
    },
    cleanup: {
      enabled: true,
      retentionDays: 365,
      schedule: '0 2 * * *' // Daily at 2 AM
    }
  },
  databaseSettings: {
    type: 'postgresql',
    host: 'localhost',
    port: 5432,
    database: 'medical_ai',
    username: 'postgres',
    password: '',
    ssl: false,
    poolSize: 10,
    backup: {
      enabled: true,
      schedule: '0 3 * * *', // Daily at 3 AM
      retention: 30
    },
    monitoring: {
      enabled: true,
      slowQueryThreshold: 1000, // 1 second
      logQueries: false
    }
  },
  loading: false,
  error: null,
  hasUnsavedChanges: false,
  lastSaved: null
}

// Async thunks
export const loadSettings = createAsyncThunk(
  'settings/loadSettings',
  async () => {
    // Simulate API call
    await new Promise(resolve => setTimeout(resolve, 1000))
    
    // In a real app, this would fetch from the server
    // For now, return the initial state
    return {
      userPreferences: initialState.userPreferences,
      systemSettings: initialState.systemSettings,
      aiSettings: initialState.aiSettings,
      notificationSettings: initialState.notificationSettings,
      storageSettings: initialState.storageSettings,
      databaseSettings: initialState.databaseSettings
    }
  }
)

export const saveSettings = createAsyncThunk(
  'settings/saveSettings',
  async (settings: Partial<SettingsState>) => {
    // Simulate API call
    await new Promise(resolve => setTimeout(resolve, 1500))
    
    // In a real app, this would save to the server
    return {
      ...settings,
      lastSaved: new Date().toISOString()
    }
  }
)

export const resetSettings = createAsyncThunk(
  'settings/resetSettings',
  async (category: keyof SettingsState) => {
    // Simulate API call
    await new Promise(resolve => setTimeout(resolve, 800))
    
    return {
      category,
      settings: (initialState as any)[category]
    }
  }
)

export const testEmailSettings = createAsyncThunk(
  'settings/testEmailSettings',
  async (emailSettings: NotificationSettings['email']) => {
    // Simulate API call to test email settings
    await new Promise(resolve => setTimeout(resolve, 2000))
    
    // Mock test result
    const success = Math.random() > 0.3 // 70% success rate for demo
    
    if (!success) {
      throw new Error('邮件服务器连接失败，请检查配置')
    }
    
    return { success: true, message: '邮件配置测试成功' }
  }
)

export const testDatabaseConnection = createAsyncThunk(
  'settings/testDatabaseConnection',
  async (dbSettings: DatabaseSettings) => {
    // Simulate API call to test database connection
    await new Promise(resolve => setTimeout(resolve, 1500))
    
    // Mock test result
    const success = Math.random() > 0.2 // 80% success rate for demo
    
    if (!success) {
      throw new Error('数据库连接失败，请检查配置')
    }
    
    return { success: true, message: '数据库连接测试成功' }
  }
)

const settingsSlice = createSlice({
  name: 'settings',
  initialState,
  reducers: {
    updateUserPreferences: (state, action: PayloadAction<Partial<UserPreferences>>) => {
      state.userPreferences = { ...state.userPreferences, ...action.payload }
      state.hasUnsavedChanges = true
    },
    updateSystemSettings: (state, action: PayloadAction<Partial<SystemSettings>>) => {
      state.systemSettings = { ...state.systemSettings, ...action.payload }
      state.hasUnsavedChanges = true
    },
    updateAISettings: (state, action: PayloadAction<Partial<AISettings>>) => {
      state.aiSettings = { ...state.aiSettings, ...action.payload }
      state.hasUnsavedChanges = true
    },
    updateNotificationSettings: (state, action: PayloadAction<Partial<NotificationSettings>>) => {
      state.notificationSettings = { ...state.notificationSettings, ...action.payload }
      state.hasUnsavedChanges = true
    },
    updateStorageSettings: (state, action: PayloadAction<Partial<StorageSettings>>) => {
      state.storageSettings = { ...state.storageSettings, ...action.payload }
      state.hasUnsavedChanges = true
    },
    updateDatabaseSettings: (state, action: PayloadAction<Partial<DatabaseSettings>>) => {
      state.databaseSettings = { ...state.databaseSettings, ...action.payload }
      state.hasUnsavedChanges = true
    },
    updateUserNotificationPreference: (state, action: PayloadAction<{
      type: keyof UserPreferences['notifications']
      enabled: boolean
    }>) => {
      const { type, enabled } = action.payload
      state.userPreferences.notifications[type] = enabled
      state.hasUnsavedChanges = true
    },
    updateShortcut: (state, action: PayloadAction<{ action: string; shortcut: string }>) => {
      const { action: actionName, shortcut } = action.payload
      state.userPreferences.shortcuts[actionName] = shortcut
      state.hasUnsavedChanges = true
    },
    addIPToWhitelist: (state, action: PayloadAction<string>) => {
      if (!state.systemSettings.security.ipWhitelist.includes(action.payload)) {
        state.systemSettings.security.ipWhitelist.push(action.payload)
        state.hasUnsavedChanges = true
      }
    },
    removeIPFromWhitelist: (state, action: PayloadAction<string>) => {
      state.systemSettings.security.ipWhitelist = state.systemSettings.security.ipWhitelist.filter(
        ip => ip !== action.payload
      )
      state.hasUnsavedChanges = true
    },
    updateEmailTemplate: (state, action: PayloadAction<{
      templateId: string
      template: { subject: string; body: string }
    }>) => {
      const { templateId, template } = action.payload
      state.notificationSettings.email.templates[templateId] = template
      state.hasUnsavedChanges = true
    },
    addWebhookEvent: (state, action: PayloadAction<string>) => {
      if (!state.notificationSettings.webhook.events.includes(action.payload)) {
        state.notificationSettings.webhook.events.push(action.payload)
        state.hasUnsavedChanges = true
      }
    },
    removeWebhookEvent: (state, action: PayloadAction<string>) => {
      state.notificationSettings.webhook.events = state.notificationSettings.webhook.events.filter(
        event => event !== action.payload
      )
      state.hasUnsavedChanges = true
    },
    markAsSaved: (state) => {
      state.hasUnsavedChanges = false
      state.lastSaved = new Date().toISOString()
    },
    clearError: (state) => {
      state.error = null
    }
  },
  extraReducers: (builder) => {
    builder
      // Load settings
      .addCase(loadSettings.pending, (state) => {
        state.loading = true
        state.error = null
      })
      .addCase(loadSettings.fulfilled, (state, action) => {
        state.loading = false
        state.userPreferences = action.payload.userPreferences
        state.systemSettings = action.payload.systemSettings
        state.aiSettings = action.payload.aiSettings
        state.notificationSettings = action.payload.notificationSettings
        state.storageSettings = action.payload.storageSettings
        state.databaseSettings = action.payload.databaseSettings
        state.hasUnsavedChanges = false
      })
      .addCase(loadSettings.rejected, (state, action) => {
        state.loading = false
        state.error = action.error.message || '加载设置失败'
      })
      
      // Save settings
      .addCase(saveSettings.pending, (state) => {
        state.loading = true
        state.error = null
      })
      .addCase(saveSettings.fulfilled, (state, action) => {
        state.loading = false
        state.hasUnsavedChanges = false
        state.lastSaved = action.payload.lastSaved
      })
      .addCase(saveSettings.rejected, (state, action) => {
        state.loading = false
        state.error = action.error.message || '保存设置失败'
      })
      
      // Reset settings
      .addCase(resetSettings.fulfilled, (state, action) => {
        const { category, settings } = action.payload
        ;(state as any)[category] = settings
        state.hasUnsavedChanges = true
      })
      
      // Test email settings
      .addCase(testEmailSettings.rejected, (state, action) => {
        state.error = action.error.message || '邮件配置测试失败'
      })
      
      // Test database connection
      .addCase(testDatabaseConnection.rejected, (state, action) => {
        state.error = action.error.message || '数据库连接测试失败'
      })
  }
})

export const {
  updateUserPreferences,
  updateSystemSettings,
  updateAISettings,
  updateNotificationSettings,
  updateStorageSettings,
  updateDatabaseSettings,
  updateUserNotificationPreference,
  updateShortcut,
  addIPToWhitelist,
  removeIPFromWhitelist,
  updateEmailTemplate,
  addWebhookEvent,
  removeWebhookEvent,
  markAsSaved,
  clearError
} = settingsSlice.actions

export default settingsSlice.reducer