import { createApi, fetchBaseQuery } from '@reduxjs/toolkit/query/react'
import type { RootState } from '../index'

export interface LoginRequest {
  username: string
  password: string
  rememberMe?: boolean
}

export interface LoginResponse {
  user: {
    id: string
    username: string
    email: string
    firstName: string
    lastName: string
    role: 'admin' | 'doctor' | 'technician' | 'viewer'
    department: string
    avatar?: string
    permissions: string[]
    lastLogin?: string
    isActive: boolean
  }
  token: string
  refreshToken: string
  expiresIn: number
}

export interface RefreshTokenRequest {
  refreshToken: string
}

export interface ChangePasswordRequest {
  currentPassword: string
  newPassword: string
}

export interface ResetPasswordRequest {
  email: string
}

export interface UpdateProfileRequest {
  firstName?: string
  lastName?: string
  email?: string
  department?: string
  avatar?: string
}

export interface TwoFactorSetupResponse {
  qrCode: string
  secret: string
  backupCodes: string[]
}

export interface TwoFactorVerifyRequest {
  token: string
  code: string
}

export const authApi = createApi({
  reducerPath: 'authApi',
  baseQuery: fetchBaseQuery({
    baseUrl: '/api/auth',
    prepareHeaders: (headers, { getState }) => {
      const token = (getState() as RootState).auth.token
      if (token) {
        headers.set('authorization', `Bearer ${token}`)
      }
      return headers
    },
  }),
  tagTypes: ['User', 'Session'],
  endpoints: (builder) => ({
    // 登录
    login: builder.mutation<LoginResponse, LoginRequest>({
      query: (credentials) => ({
        url: '/login',
        method: 'POST',
        body: credentials,
      }),
      invalidatesTags: ['User', 'Session'],
    }),

    // 登出
    logout: builder.mutation<void, void>({
      query: () => ({
        url: '/logout',
        method: 'POST',
      }),
      invalidatesTags: ['User', 'Session'],
    }),

    // 刷新令牌
    refreshToken: builder.mutation<LoginResponse, RefreshTokenRequest>({
      query: (data) => ({
        url: '/refresh',
        method: 'POST',
        body: data,
      }),
      invalidatesTags: ['Session'],
    }),

    // 获取当前用户信息
    getCurrentUser: builder.query<LoginResponse['user'], void>({
      query: () => '/me',
      providesTags: ['User'],
    }),

    // 更新用户资料
    updateProfile: builder.mutation<LoginResponse['user'], UpdateProfileRequest>({
      query: (data) => ({
        url: '/profile',
        method: 'PUT',
        body: data,
      }),
      invalidatesTags: ['User'],
    }),

    // 修改密码
    changePassword: builder.mutation<void, ChangePasswordRequest>({
      query: (data) => ({
        url: '/change-password',
        method: 'POST',
        body: data,
      }),
    }),

    // 重置密码
    resetPassword: builder.mutation<void, ResetPasswordRequest>({
      query: (data) => ({
        url: '/reset-password',
        method: 'POST',
        body: data,
      }),
    }),

    // 验证重置密码令牌
    verifyResetToken: builder.query<{ valid: boolean }, string>({
      query: (token) => `/reset-password/verify/${token}`,
    }),

    // 设置新密码
    setNewPassword: builder.mutation<void, { token: string; password: string }>({
      query: (data) => ({
        url: '/reset-password/confirm',
        method: 'POST',
        body: data,
      }),
    }),

    // 设置双因素认证
    setupTwoFactor: builder.mutation<TwoFactorSetupResponse, void>({
      query: () => ({
        url: '/2fa/setup',
        method: 'POST',
      }),
    }),

    // 验证双因素认证
    verifyTwoFactor: builder.mutation<void, TwoFactorVerifyRequest>({
      query: (data) => ({
        url: '/2fa/verify',
        method: 'POST',
        body: data,
      }),
      invalidatesTags: ['User'],
    }),

    // 禁用双因素认证
    disableTwoFactor: builder.mutation<void, { password: string }>({
      query: (data) => ({
        url: '/2fa/disable',
        method: 'POST',
        body: data,
      }),
      invalidatesTags: ['User'],
    }),

    // 获取会话信息
    getSessionInfo: builder.query<{
      sessions: Array<{
        id: string
        device: string
        browser: string
        ip: string
        location: string
        lastActive: string
        current: boolean
      }>
    }, void>({
      query: () => '/sessions',
      providesTags: ['Session'],
    }),

    // 终止会话
    terminateSession: builder.mutation<void, string>({
      query: (sessionId) => ({
        url: `/sessions/${sessionId}`,
        method: 'DELETE',
      }),
      invalidatesTags: ['Session'],
    }),

    // 终止所有其他会话
    terminateAllOtherSessions: builder.mutation<void, void>({
      query: () => ({
        url: '/sessions/terminate-others',
        method: 'POST',
      }),
      invalidatesTags: ['Session'],
    }),

    // 检查用户名是否可用
    checkUsernameAvailability: builder.query<{ available: boolean }, string>({
      query: (username) => `/check-username/${username}`,
    }),

    // 检查邮箱是否可用
    checkEmailAvailability: builder.query<{ available: boolean }, string>({
      query: (email) => `/check-email/${email}`,
    }),

    // 获取用户权限
    getUserPermissions: builder.query<string[], void>({
      query: () => '/permissions',
      providesTags: ['User'],
    }),

    // 验证令牌
    validateToken: builder.query<{ valid: boolean; user?: LoginResponse['user'] }, void>({
      query: () => '/validate',
      providesTags: ['Session'],
    }),
  }),
})

export const {
  useLoginMutation,
  useLogoutMutation,
  useRefreshTokenMutation,
  useGetCurrentUserQuery,
  useUpdateProfileMutation,
  useChangePasswordMutation,
  useResetPasswordMutation,
  useVerifyResetTokenQuery,
  useSetNewPasswordMutation,
  useSetupTwoFactorMutation,
  useVerifyTwoFactorMutation,
  useDisableTwoFactorMutation,
  useGetSessionInfoQuery,
  useTerminateSessionMutation,
  useTerminateAllOtherSessionsMutation,
  useCheckUsernameAvailabilityQuery,
  useCheckEmailAvailabilityQuery,
  useGetUserPermissionsQuery,
  useValidateTokenQuery,
} = authApi

export default authApi