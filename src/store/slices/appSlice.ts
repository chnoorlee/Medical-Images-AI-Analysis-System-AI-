import { createSlice, createAsyncThunk, PayloadAction } from '@reduxjs/toolkit'

// Types
interface AppState {
  isInitialized: boolean
  isLoading: boolean
  theme: 'light' | 'dark'
  language: 'zh-CN' | 'en-US'
  sidebarCollapsed: boolean
  notifications: Notification[]
  systemInfo: SystemInfo | null
  error: string | null
}

interface Notification {
  id: string
  type: 'success' | 'error' | 'warning' | 'info'
  title: string
  message: string
  timestamp: number
  read: boolean
}

interface SystemInfo {
  version: string
  buildTime: string
  environment: string
  features: string[]
}

// Initial state
const initialState: AppState = {
  isInitialized: false,
  isLoading: false,
  theme: 'light',
  language: 'zh-CN',
  sidebarCollapsed: false,
  notifications: [],
  systemInfo: null,
  error: null,
}

// Async thunks
export const initializeApp = createAsyncThunk(
  'app/initialize',
  async (_, { rejectWithValue }) => {
    try {
      // Load system configuration
      const response = await fetch('/api/system/info')
      if (!response.ok) {
        throw new Error('Failed to load system info')
      }
      const systemInfo = await response.json()

      // Load user preferences from localStorage
      const savedTheme = localStorage.getItem('theme') as 'light' | 'dark' | null
      const savedLanguage = localStorage.getItem('language') as 'zh-CN' | 'en-US' | null
      const savedSidebarState = localStorage.getItem('sidebarCollapsed')

      return {
        systemInfo,
        theme: savedTheme || 'light',
        language: savedLanguage || 'zh-CN',
        sidebarCollapsed: savedSidebarState === 'true',
      }
    } catch (error) {
      return rejectWithValue(error instanceof Error ? error.message : 'Unknown error')
    }
  }
)

export const loadNotifications = createAsyncThunk(
  'app/loadNotifications',
  async (_, { rejectWithValue }) => {
    try {
      const response = await fetch('/api/notifications')
      if (!response.ok) {
        throw new Error('Failed to load notifications')
      }
      return await response.json()
    } catch (error) {
      return rejectWithValue(error instanceof Error ? error.message : 'Unknown error')
    }
  }
)

// Slice
const appSlice = createSlice({
  name: 'app',
  initialState,
  reducers: {
    setLoading: (state, action: PayloadAction<boolean>) => {
      state.isLoading = action.payload
    },
    setTheme: (state, action: PayloadAction<'light' | 'dark'>) => {
      state.theme = action.payload
      localStorage.setItem('theme', action.payload)
    },
    setLanguage: (state, action: PayloadAction<'zh-CN' | 'en-US'>) => {
      state.language = action.payload
      localStorage.setItem('language', action.payload)
    },
    toggleSidebar: (state) => {
      state.sidebarCollapsed = !state.sidebarCollapsed
      localStorage.setItem('sidebarCollapsed', state.sidebarCollapsed.toString())
    },
    setSidebarCollapsed: (state, action: PayloadAction<boolean>) => {
      state.sidebarCollapsed = action.payload
      localStorage.setItem('sidebarCollapsed', action.payload.toString())
    },
    addNotification: (state, action: PayloadAction<Omit<Notification, 'id' | 'timestamp' | 'read'>>) => {
      const notification: Notification = {
        ...action.payload,
        id: Date.now().toString(),
        timestamp: Date.now(),
        read: false,
      }
      state.notifications.unshift(notification)
      
      // Keep only last 50 notifications
      if (state.notifications.length > 50) {
        state.notifications = state.notifications.slice(0, 50)
      }
    },
    markNotificationAsRead: (state, action: PayloadAction<string>) => {
      const notification = state.notifications.find(n => n.id === action.payload)
      if (notification) {
        notification.read = true
      }
    },
    markAllNotificationsAsRead: (state) => {
      state.notifications.forEach(notification => {
        notification.read = true
      })
    },
    removeNotification: (state, action: PayloadAction<string>) => {
      state.notifications = state.notifications.filter(n => n.id !== action.payload)
    },
    clearNotifications: (state) => {
      state.notifications = []
    },
    clearError: (state) => {
      state.error = null
    },
  },
  extraReducers: (builder) => {
    builder
      // Initialize app
      .addCase(initializeApp.pending, (state) => {
        state.isLoading = true
        state.error = null
      })
      .addCase(initializeApp.fulfilled, (state, action) => {
        state.isLoading = false
        state.isInitialized = true
        state.systemInfo = action.payload.systemInfo
        state.theme = action.payload.theme
        state.language = action.payload.language
        state.sidebarCollapsed = action.payload.sidebarCollapsed
      })
      .addCase(initializeApp.rejected, (state, action) => {
        state.isLoading = false
        state.isInitialized = true // Still mark as initialized even if failed
        state.error = action.payload as string
      })
      // Load notifications
      .addCase(loadNotifications.fulfilled, (state, action) => {
        state.notifications = action.payload
      })
  },
})

export const {
  setLoading,
  setTheme,
  setLanguage,
  toggleSidebar,
  setSidebarCollapsed,
  addNotification,
  markNotificationAsRead,
  markAllNotificationsAsRead,
  removeNotification,
  clearNotifications,
  clearError,
} = appSlice.actions

export default appSlice.reducer