// 性能监控工具函数

// 性能指标接口
interface PerformanceMetric {
  name: string
  value: number
  timestamp: number
  url: string
  userAgent: string
  userId?: string
  sessionId?: string
  extra?: Record<string, any>
}

// 页面加载性能监控
export const initPagePerformanceMonitoring = () => {
  // 等待页面完全加载
  window.addEventListener('load', () => {
    // 使用 setTimeout 确保所有资源都已加载
    setTimeout(() => {
      collectPageLoadMetrics()
    }, 0)
  })

  // 监听页面可见性变化
  document.addEventListener('visibilitychange', () => {
    if (document.visibilityState === 'hidden') {
      collectPageVisibilityMetrics()
    }
  })

  // 监听页面卸载
  window.addEventListener('beforeunload', () => {
    collectPageUnloadMetrics()
  })
}

// 收集页面加载性能指标
const collectPageLoadMetrics = () => {
  try {
    const navigation = performance.getEntriesByType('navigation')[0] as PerformanceNavigationTiming
    const paint = performance.getEntriesByType('paint')
    
    if (navigation) {
      // DNS查询时间
      reportMetric({
        name: 'dns_lookup_time',
        value: navigation.domainLookupEnd - navigation.domainLookupStart
      })

      // TCP连接时间
      reportMetric({
        name: 'tcp_connect_time',
        value: navigation.connectEnd - navigation.connectStart
      })

      // SSL握手时间
      if (navigation.secureConnectionStart > 0) {
        reportMetric({
          name: 'ssl_handshake_time',
          value: navigation.connectEnd - navigation.secureConnectionStart
        })
      }

      // 请求响应时间
      reportMetric({
        name: 'request_response_time',
        value: navigation.responseEnd - navigation.requestStart
      })

      // DOM解析时间
      reportMetric({
        name: 'dom_parse_time',
        value: navigation.domContentLoadedEventEnd - navigation.domContentLoadedEventStart
      })

      // 页面完全加载时间
      reportMetric({
        name: 'page_load_time',
        value: navigation.loadEventEnd - navigation.loadEventStart
      })

      // 总的页面加载时间
      reportMetric({
        name: 'total_load_time',
        value: navigation.loadEventEnd - navigation.navigationStart
      })

      // 首次内容绘制时间
      const fcp = paint.find(entry => entry.name === 'first-contentful-paint')
      if (fcp) {
        reportMetric({
          name: 'first_contentful_paint',
          value: fcp.startTime
        })
      }

      // 首次绘制时间
      const fp = paint.find(entry => entry.name === 'first-paint')
      if (fp) {
        reportMetric({
          name: 'first_paint',
          value: fp.startTime
        })
      }
    }

    // 收集资源加载性能
    collectResourceMetrics()
    
    // 收集内存使用情况
    collectMemoryMetrics()
    
  } catch (error) {
    console.warn('Failed to collect page load metrics:', error)
  }
}

// 收集资源加载性能指标
const collectResourceMetrics = () => {
  try {
    const resources = performance.getEntriesByType('resource') as PerformanceResourceTiming[]
    
    const resourceStats = {
      total: resources.length,
      totalSize: 0,
      totalDuration: 0,
      byType: {} as Record<string, { count: number; size: number; duration: number }>
    }

    resources.forEach(resource => {
      const duration = resource.responseEnd - resource.startTime
      const size = resource.transferSize || 0
      
      resourceStats.totalSize += size
      resourceStats.totalDuration += duration
      
      // 根据资源类型分类
      const type = getResourceType(resource.name)
      if (!resourceStats.byType[type]) {
        resourceStats.byType[type] = { count: 0, size: 0, duration: 0 }
      }
      
      resourceStats.byType[type].count++
      resourceStats.byType[type].size += size
      resourceStats.byType[type].duration += duration
    })

    // 报告资源统计
    reportMetric({
      name: 'resource_count',
      value: resourceStats.total
    })

    reportMetric({
      name: 'resource_total_size',
      value: resourceStats.totalSize
    })

    reportMetric({
      name: 'resource_avg_duration',
      value: resourceStats.total > 0 ? resourceStats.totalDuration / resourceStats.total : 0
    })

    // 报告各类型资源统计
    Object.entries(resourceStats.byType).forEach(([type, stats]) => {
      reportMetric({
        name: `resource_${type}_count`,
        value: stats.count
      })
      
      reportMetric({
        name: `resource_${type}_size`,
        value: stats.size
      })
      
      reportMetric({
        name: `resource_${type}_avg_duration`,
        value: stats.count > 0 ? stats.duration / stats.count : 0
      })
    })
    
  } catch (error) {
    console.warn('Failed to collect resource metrics:', error)
  }
}

// 获取资源类型
const getResourceType = (url: string): string => {
  const extension = url.split('.').pop()?.toLowerCase()
  
  if (['js', 'mjs'].includes(extension || '')) return 'script'
  if (['css'].includes(extension || '')) return 'stylesheet'
  if (['png', 'jpg', 'jpeg', 'gif', 'svg', 'webp'].includes(extension || '')) return 'image'
  if (['woff', 'woff2', 'ttf', 'otf'].includes(extension || '')) return 'font'
  if (['json', 'xml'].includes(extension || '')) return 'xhr'
  
  return 'other'
}

// 扩展Performance接口以包含memory属性
interface PerformanceWithMemory extends Performance {
  memory?: {
    usedJSHeapSize: number
    totalJSHeapSize: number
    jsHeapSizeLimit: number
  }
}

// 收集内存使用指标
const collectMemoryMetrics = () => {
  try {
    const performanceWithMemory = performance as PerformanceWithMemory
    if (performanceWithMemory.memory) {
      const memory = performanceWithMemory.memory
      
      reportMetric({
        name: 'memory_used_js_heap_size',
        value: memory.usedJSHeapSize
      })
      
      reportMetric({
        name: 'memory_total_js_heap_size',
        value: memory.totalJSHeapSize
      })
      
      reportMetric({
        name: 'memory_js_heap_size_limit',
        value: memory.jsHeapSizeLimit
      })
      
      reportMetric({
        name: 'memory_usage_ratio',
        value: memory.usedJSHeapSize / memory.jsHeapSizeLimit
      })
    }
  } catch (error) {
    console.warn('Failed to collect memory metrics:', error)
  }
}

// 收集页面可见性指标
const collectPageVisibilityMetrics = () => {
  try {
    const sessionStart = getSessionStartTime()
    const now = Date.now()
    
    reportMetric({
      name: 'session_duration',
      value: now - sessionStart
    })
    
  } catch (error) {
    console.warn('Failed to collect page visibility metrics:', error)
  }
}

// 收集页面卸载指标
const collectPageUnloadMetrics = () => {
  try {
    const sessionStart = getSessionStartTime()
    const now = Date.now()
    
    reportMetric({
      name: 'session_total_duration',
      value: now - sessionStart
    })
    
  } catch (error) {
    console.warn('Failed to collect page unload metrics:', error)
  }
}

// 获取会话开始时间
const getSessionStartTime = (): number => {
  let startTime = sessionStorage.getItem('medical-ai-session-start')
  if (!startTime) {
    startTime = Date.now().toString()
    sessionStorage.setItem('medical-ai-session-start', startTime)
  }
  return parseInt(startTime, 10)
}

// 报告性能指标
const reportMetric = (metric: Omit<PerformanceMetric, 'timestamp' | 'url' | 'userAgent' | 'userId' | 'sessionId'>) => {
  try {
    const fullMetric: PerformanceMetric = {
      ...metric,
      timestamp: Date.now(),
      url: window.location.href,
      userAgent: navigator.userAgent,
      userId: getCurrentUserId(),
      sessionId: getSessionId()
    }

    // 发送到性能监控服务
    if (import.meta.env.VITE_PERFORMANCE_MONITORING_URL) {
      sendMetricToService(fullMetric)
    }

    // 本地存储（用于调试）
    if (import.meta.env.DEV) {
      storeMetricLocally(fullMetric)
    }
    
  } catch (error) {
    console.warn('Failed to report metric:', error)
  }
}

// 发送指标到监控服务
const sendMetricToService = async (metric: PerformanceMetric) => {
  try {
    await fetch(import.meta.env.VITE_PERFORMANCE_MONITORING_URL, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify(metric)
    })
  } catch (error) {
    console.warn('Failed to send metric to service:', error)
  }
}

// 本地存储指标
const storeMetricLocally = (metric: PerformanceMetric) => {
  try {
    const metrics = getLocalMetrics()
    metrics.push(metric)
    
    // 只保留最近1000条指标
    if (metrics.length > 1000) {
      metrics.splice(0, metrics.length - 1000)
    }
    
    localStorage.setItem('medical-ai-performance-metrics', JSON.stringify(metrics))
  } catch (error) {
    console.warn('Failed to store metric locally:', error)
  }
}

// 获取本地指标
const getLocalMetrics = (): PerformanceMetric[] => {
  try {
    const metrics = localStorage.getItem('medical-ai-performance-metrics')
    return metrics ? JSON.parse(metrics) : []
  } catch (error) {
    console.warn('Failed to get local metrics:', error)
    return []
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

// 生成UUID
const generateUUID = (): string => {
  return 'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, (c) => {
    const r = Math.random() * 16 | 0
    const v = c === 'x' ? r : (r & 0x3 | 0x8)
    return v.toString(16)
  })
}

// 自定义性能标记
export const markPerformance = (name: string) => {
  try {
    performance.mark(name)
  } catch (error) {
    console.warn('Failed to mark performance:', error)
  }
}

// 测量性能
export const measurePerformance = (name: string, startMark: string, endMark?: string) => {
  try {
    const endMarkName = endMark || `${startMark}-end`
    if (endMark) {
      performance.mark(endMarkName)
    }
    
    const measure = performance.measure(name, startMark, endMarkName)
    
    reportMetric({
      name: `custom_${name}`,
      value: measure.duration
    })
    
    return measure.duration
  } catch (error) {
    console.warn('Failed to measure performance:', error)
    return 0
  }
}

// 监控函数执行时间
export const monitorFunction = <T extends (...args: any[]) => any>(
  fn: T,
  name: string
): T => {
  return ((...args: any[]) => {
    const startTime = performance.now()
    
    try {
      const result = fn(...args)
      
      // 如果是Promise，监控异步执行时间
      if (result && typeof result.then === 'function') {
        return result.finally(() => {
          const endTime = performance.now()
          reportMetric({
            name: `function_${name}_duration`,
            value: endTime - startTime
          })
        })
      } else {
        const endTime = performance.now()
        reportMetric({
          name: `function_${name}_duration`,
          value: endTime - startTime
        })
        return result
      }
    } catch (error) {
      const endTime = performance.now()
      reportMetric({
        name: `function_${name}_duration`,
        value: endTime - startTime,
        extra: { error: true }
      })
      throw error
    }
  }) as T
}

// 监控React组件渲染时间
export const monitorComponentRender = (componentName: string) => {
  return {
    start: () => markPerformance(`${componentName}-render-start`),
    end: () => {
      markPerformance(`${componentName}-render-end`)
      return measurePerformance(
        `${componentName}-render`,
        `${componentName}-render-start`,
        `${componentName}-render-end`
      )
    }
  }
}

// 清理性能数据
export const clearPerformanceData = () => {
  localStorage.removeItem('medical-ai-performance-metrics')
  sessionStorage.removeItem('medical-ai-session-start')
}

// 获取性能统计
export const getPerformanceStats = () => {
  const metrics = getLocalMetrics()
  const stats = {
    total: metrics.length,
    byName: {} as Record<string, { count: number; avg: number; min: number; max: number }>,
    recent: metrics.filter(metric => {
      const metricTime = metric.timestamp
      const now = Date.now()
      return now - metricTime < 60 * 60 * 1000 // 最近1小时
    }).length
  }

  metrics.forEach(metric => {
    if (!stats.byName[metric.name]) {
      stats.byName[metric.name] = {
        count: 0,
        avg: 0,
        min: Infinity,
        max: -Infinity
      }
    }
    
    const stat = stats.byName[metric.name]
    stat.count++
    stat.min = Math.min(stat.min, metric.value)
    stat.max = Math.max(stat.max, metric.value)
    stat.avg = (stat.avg * (stat.count - 1) + metric.value) / stat.count
  })

  return stats
}

export default {
  initPagePerformanceMonitoring,
  markPerformance,
  measurePerformance,
  monitorFunction,
  monitorComponentRender,
  clearPerformanceData,
  getPerformanceStats
}