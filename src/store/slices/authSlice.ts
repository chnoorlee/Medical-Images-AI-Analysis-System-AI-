import { createSlice, createAsyncThunk, PayloadAction } from '@reduxjs/toolkit'

// Types
interface User {
  id: string
  username: string
  email: string
  fullName: string
  role: 'admin' | 'doctor' | 'technician' | 'viewer'
  department: string
  avatar?: string
  permissions: string[]
  lastLoginAt?: string
  createdAt: string
  updatedAt: string
}

interface AuthState {
  isAuthenticated: boolean
  user: User | null
  token: string | null
  refreshToken: string | null
  isLoading: boolean
  error: string | null
  loginAttempts: number
  lastLoginAttempt: number | null
}

interface LoginCredentials {
  username: string
  password: string
  rememberMe?: boolean
}

interface LoginResponse {
  user: User
  token: string
  refreshToken: string
}

// Initial state
const initialState: AuthState = {
  isAuthenticated: false,
  user: null,
  token: localStorage.getItem('token'),
  refreshToken: localStorage.getItem('refreshToken'),
  isLoading: false,
  error: null,
  loginAttempts: 0,
  lastLoginAttempt: null,
}

// Async thunks
export const login = createAsyncThunk(
  'auth/login',
  async (credentials: LoginCredentials, { rejectWithValue }) => {
    try {
      const response = await fetch('/api/auth/login', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(credentials),
      })

      if (!response.ok) {
        const error = await response.json()
        throw new Error(error.message || 'Login failed')
      }

      const data: LoginResponse = await response.json()
      
      // Store tokens
      localStorage.setItem('token', data.token)
      localStorage.setItem('refreshToken', data.refreshToken)
      
      if (credentials.rememberMe) {
        localStorage.setItem('rememberMe', 'true')
      }

      return data
    } catch (error) {
      return rejectWithValue(error instanceof Error ? error.message : 'Unknown error')
    }
  }
)

export const logout = createAsyncThunk(
  'auth/logout',
  async (_, { getState, rejectWithValue }) => {
    try {
      const state = getState() as { auth: AuthState }
      const token = state.auth.token

      if (token) {
        await fetch('/api/auth/logout', {
          method: 'POST',
          headers: {
            'Authorization': `Bearer ${token}`,
          },
        })
      }

      // Clear local storage
      localStorage.removeItem('token')
      localStorage.removeItem('refreshToken')
      localStorage.removeItem('rememberMe')

      return null
    } catch (error) {
      // Even if logout fails on server, clear local data
      localStorage.removeItem('token')
      localStorage.removeItem('refreshToken')
      localStorage.removeItem('rememberMe')
      
      return rejectWithValue(error instanceof Error ? error.message : 'Unknown error')
    }
  }
)

export const refreshAccessToken = createAsyncThunk(
  'auth/refreshToken',
  async (_, { getState, rejectWithValue }) => {
    try {
      const state = getState() as { auth: AuthState }
      const refreshToken = state.auth.refreshToken

      if (!refreshToken) {
        throw new Error('No refresh token available')
      }

      const response = await fetch('/api/auth/refresh', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ refreshToken }),
      })

      if (!response.ok) {
        throw new Error('Token refresh failed')
      }

      const data = await response.json()
      
      // Update stored token
      localStorage.setItem('token', data.token)
      
      return data
    } catch (error) {
      // If refresh fails, clear all auth data
      localStorage.removeItem('token')
      localStorage.removeItem('refreshToken')
      localStorage.removeItem('rememberMe')
      
      return rejectWithValue(error instanceof Error ? error.message : 'Unknown error')
    }
  }
)

export const getCurrentUser = createAsyncThunk(
  'auth/getCurrentUser',
  async (_, { getState, rejectWithValue }) => {
    try {
      const state = getState() as { auth: AuthState }
      const token = state.auth.token

      if (!token) {
        throw new Error('No token available')
      }

      const response = await fetch('/api/auth/me', {
        headers: {
          'Authorization': `Bearer ${token}`,
        },
      })

      if (!response.ok) {
        throw new Error('Failed to get user info')
      }

      return await response.json()
    } catch (error) {
      return rejectWithValue(error instanceof Error ? error.message : 'Unknown error')
    }
  }
)

export const updateProfile = createAsyncThunk(
  'auth/updateProfile',
  async (profileData: Partial<User>, { getState, rejectWithValue }) => {
    try {
      const state = getState() as { auth: AuthState }
      const token = state.auth.token

      const response = await fetch('/api/auth/profile', {
        method: 'PUT',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${token}`,
        },
        body: JSON.stringify(profileData),
      })

      if (!response.ok) {
        const error = await response.json()
        throw new Error(error.message || 'Profile update failed')
      }

      return await response.json()
    } catch (error) {
      return rejectWithValue(error instanceof Error ? error.message : 'Unknown error')
    }
  }
)

// Slice
const authSlice = createSlice({
  name: 'auth',
  initialState,
  reducers: {
    clearError: (state) => {
      state.error = null
    },
    resetLoginAttempts: (state) => {
      state.loginAttempts = 0
      state.lastLoginAttempt = null
    },
    setToken: (state, action: PayloadAction<string>) => {
      state.token = action.payload
      localStorage.setItem('token', action.payload)
    },
    clearAuth: (state) => {
      state.isAuthenticated = false
      state.user = null
      state.token = null
      state.refreshToken = null
      state.error = null
      localStorage.removeItem('token')
      localStorage.removeItem('refreshToken')
      localStorage.removeItem('rememberMe')
    },
  },
  extraReducers: (builder) => {
    builder
      // Login
      .addCase(login.pending, (state) => {
        state.isLoading = true
        state.error = null
      })
      .addCase(login.fulfilled, (state, action) => {
        state.isLoading = false
        state.isAuthenticated = true
        state.user = action.payload.user
        state.token = action.payload.token
        state.refreshToken = action.payload.refreshToken
        state.loginAttempts = 0
        state.lastLoginAttempt = null
      })
      .addCase(login.rejected, (state, action) => {
        state.isLoading = false
        state.error = action.payload as string
        state.loginAttempts += 1
        state.lastLoginAttempt = Date.now()
      })
      // Logout
      .addCase(logout.fulfilled, (state) => {
        state.isAuthenticated = false
        state.user = null
        state.token = null
        state.refreshToken = null
        state.error = null
      })
      // Refresh token
      .addCase(refreshAccessToken.fulfilled, (state, action) => {
        state.token = action.payload.token
        if (action.payload.user) {
          state.user = action.payload.user
        }
      })
      .addCase(refreshAccessToken.rejected, (state) => {
        state.isAuthenticated = false
        state.user = null
        state.token = null
        state.refreshToken = null
      })
      // Get current user
      .addCase(getCurrentUser.fulfilled, (state, action) => {
        state.isAuthenticated = true
        state.user = action.payload
      })
      .addCase(getCurrentUser.rejected, (state) => {
        state.isAuthenticated = false
        state.user = null
        state.token = null
        state.refreshToken = null
      })
      // Update profile
      .addCase(updateProfile.fulfilled, (state, action) => {
        if (state.user) {
          state.user = { ...state.user, ...action.payload }
        }
      })
  },
})

export const {
  clearError,
  resetLoginAttempts,
  setToken,
  clearAuth,
} = authSlice.actions

export default authSlice.reducer