// 错误边界工具函数

// 全局错误处理
window.addEventListener('error', (event) => {
  console.error('Global error:', event.error)
  
  // 发送错误报告到监控服务
  if (import.meta.env.PROD) {
    reportError({
      type: 'javascript',
      message: event.error?.message || 'Unknown error',
      stack: event.error?.stack,
      filename: event.filename,
      lineno: event.lineno,
      colno: event.colno,
      timestamp: new Date().toISOString(),
      userAgent: navigator.userAgent,
      url: window.location.href
    })
  }
})

// 未处理的Promise拒绝
window.addEventListener('unhandledrejection', (event) => {
  console.error('Unhandled promise rejection:', event.reason)
  
  // 发送错误报告到监控服务
  if (import.meta.env.PROD) {
    reportError({
      type: 'promise',
      message: event.reason?.message || 'Unhandled promise rejection',
      stack: event.reason?.stack,
      timestamp: new Date().toISOString(),
      userAgent: navigator.userAgent,
      url: window.location.href
    })
  }
  
  // 阻止默认的控制台错误输出
  event.preventDefault()
})

// 错误报告接口
interface ErrorReport {
  type: 'javascript' | 'promise' | 'react' | 'network' | 'custom'
  message: string
  stack?: string
  filename?: string
  lineno?: number
  colno?: number
  timestamp: string
  userAgent: string
  url: string
  userId?: string
  sessionId?: string
  buildVersion?: string
  extra?: Record<string, any>
}

// 错误报告函数
export const reportError = async (error: ErrorReport) => {
  try {
    // 添加额外的上下文信息
    const enhancedError = {
      ...error,
      buildVersion: import.meta.env.VITE_BUILD_VERSION || 'unknown',
      sessionId: getSessionId(),
      userId: getCurrentUserId(),
      extra: {
        ...error.extra,
        memoryUsage: getMemoryUsage(),
        connectionType: getConnectionType(),
        viewport: {
          width: window.innerWidth,
          height: window.innerHeight
        }
      }
    }

    // 发送到错误监控服务
    if (import.meta.env.VITE_ERROR_REPORTING_URL) {
      await fetch(import.meta.env.VITE_ERROR_REPORTING_URL, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify(enhancedError)
      })
    }

    // 本地存储错误日志（用于调试）
    if (import.meta.env.DEV) {
      const errorLogs = getLocalErrorLogs()
      errorLogs.push(enhancedError)
      
      // 只保留最近100条错误日志
      if (errorLogs.length > 100) {
        errorLogs.splice(0, errorLogs.length - 100)
      }
      
      localStorage.setItem('medical-ai-error-logs', JSON.stringify(errorLogs))
    }
  } catch (reportingError) {
    console.error('Failed to report error:', reportingError)
  }
}

// 获取会话ID
const getSessionId = (): string => {
  let sessionId = sessionStorage.getItem('medical-ai-session-id')
  if (!sessionId) {
    sessionId = generateUUID()
    sessionStorage.setItem('medical-ai-session-id', sessionId)
  }
  return sessionId
}

// 获取当前用户ID
const getCurrentUserId = (): string | undefined => {
  try {
    const authState = localStorage.getItem('medical-ai-auth')
    if (authState) {
      const parsed = JSON.parse(authState)
      return parsed.user?.id
    }
  } catch (error) {
    console.warn('Failed to get user ID:', error)
  }
  return undefined
}

// 获取内存使用情况
const getMemoryUsage = () => {
  if ('memory' in performance) {
    const memory = (performance as any).memory
    return {
      usedJSHeapSize: memory.usedJSHeapSize,
      totalJSHeapSize: memory.totalJSHeapSize,
      jsHeapSizeLimit: memory.jsHeapSizeLimit
    }
  }
  return null
}

// 获取网络连接类型
const getConnectionType = () => {
  if ('connection' in navigator) {
    const connection = (navigator as any).connection
    return {
      effectiveType: connection.effectiveType,
      downlink: connection.downlink,
      rtt: connection.rtt
    }
  }
  return null
}

// 获取本地错误日志
const getLocalErrorLogs = (): ErrorReport[] => {
  try {
    const logs = localStorage.getItem('medical-ai-error-logs')
    return logs ? JSON.parse(logs) : []
  } catch (error) {
    console.warn('Failed to get local error logs:', error)
    return []
  }
}

// 生成UUID
const generateUUID = (): string => {
  return 'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, (c) => {
    const r = Math.random() * 16 | 0
    const v = c === 'x' ? r : (r & 0x3 | 0x8)
    return v.toString(16)
  })
}

// React错误边界错误报告
export const reportReactError = (error: Error, errorInfo: any) => {
  reportError({
    type: 'react',
    message: error.message,
    stack: error.stack,
    timestamp: new Date().toISOString(),
    userAgent: navigator.userAgent,
    url: window.location.href,
    extra: {
      componentStack: errorInfo.componentStack,
      errorBoundary: true
    }
  })
}

// 网络错误报告
export const reportNetworkError = (url: string, status: number, statusText: string, responseText?: string) => {
  reportError({
    type: 'network',
    message: `Network error: ${status} ${statusText}`,
    timestamp: new Date().toISOString(),
    userAgent: navigator.userAgent,
    url: window.location.href,
    extra: {
      requestUrl: url,
      status,
      statusText,
      responseText: responseText?.substring(0, 1000) // 限制响应文本长度
    }
  })
}

// 自定义错误报告
export const reportCustomError = (message: string, extra?: Record<string, any>) => {
  reportError({
    type: 'custom',
    message,
    timestamp: new Date().toISOString(),
    userAgent: navigator.userAgent,
    url: window.location.href,
    extra
  })
}

// 清理错误日志
export const clearErrorLogs = () => {
  localStorage.removeItem('medical-ai-error-logs')
}

// 获取错误统计
export const getErrorStats = () => {
  const logs = getLocalErrorLogs()
  const stats = {
    total: logs.length,
    byType: {} as Record<string, number>,
    recent: logs.filter(log => {
      const logTime = new Date(log.timestamp).getTime()
      const now = Date.now()
      return now - logTime < 24 * 60 * 60 * 1000 // 最近24小时
    }).length
  }

  logs.forEach(log => {
    stats.byType[log.type] = (stats.byType[log.type] || 0) + 1
  })

  return stats
}

export default {
  reportError,
  reportReactError,
  reportNetworkError,
  reportCustomError,
  clearErrorLogs,
  getErrorStats
}