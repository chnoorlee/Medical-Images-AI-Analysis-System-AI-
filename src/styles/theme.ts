// 主题配置文件
export const theme = {
  // 颜色系统
  colors: {
    // 主色调
    primary: {
      50: '#e6f7ff',
      100: '#bae7ff',
      200: '#91d5ff',
      300: '#69c0ff',
      400: '#40a9ff',
      500: '#1890ff', // 主色
      600: '#096dd9',
      700: '#0050b3',
      800: '#003a8c',
      900: '#002766'
    },
    
    // 成功色
    success: {
      50: '#f6ffed',
      100: '#d9f7be',
      200: '#b7eb8f',
      300: '#95de64',
      400: '#73d13d',
      500: '#52c41a', // 成功色
      600: '#389e0d',
      700: '#237804',
      800: '#135200',
      900: '#092b00'
    },
    
    // 警告色
    warning: {
      50: '#fffbe6',
      100: '#fff1b8',
      200: '#ffe58f',
      300: '#ffd666',
      400: '#ffc53d',
      500: '#faad14', // 警告色
      600: '#d48806',
      700: '#ad6800',
      800: '#874d00',
      900: '#613400'
    },
    
    // 错误色
    error: {
      50: '#fff2f0',
      100: '#ffccc7',
      200: '#ffa39e',
      300: '#ff7875',
      400: '#ff4d4f',
      500: '#f5222d', // 错误色
      600: '#cf1322',
      700: '#a8071a',
      800: '#820014',
      900: '#5c0011'
    },
    
    // 信息色
    info: {
      50: '#e6f7ff',
      100: '#bae7ff',
      200: '#91d5ff',
      300: '#69c0ff',
      400: '#40a9ff',
      500: '#1890ff', // 信息色
      600: '#096dd9',
      700: '#0050b3',
      800: '#003a8c',
      900: '#002766'
    },
    
    // 灰色系
    gray: {
      50: '#fafafa',
      100: '#f5f5f5',
      200: '#f0f0f0',
      300: '#d9d9d9',
      400: '#bfbfbf',
      500: '#8c8c8c',
      600: '#595959',
      700: '#434343',
      800: '#262626',
      900: '#1f1f1f'
    },
    
    // 医学专用颜色
    medical: {
      // 骨骼
      bone: '#f5f5dc',
      // 软组织
      tissue: '#dda0dd',
      // 血管
      vessel: '#ff6347',
      // 病灶
      lesion: '#ff4500',
      // 正常
      normal: '#90ee90',
      // 异常
      abnormal: '#ff69b4'
    }
  },
  
  // 字体系统
  fonts: {
    primary: '-apple-system, BlinkMacSystemFont, "Segoe UI", "PingFang SC", "Hiragino Sans GB", "Microsoft YaHei", "Helvetica Neue", Helvetica, Arial, sans-serif',
    mono: '"SFMono-Regular", Consolas, "Liberation Mono", Menlo, Courier, monospace'
  },
  
  // 字体大小
  fontSizes: {
    xs: '12px',
    sm: '14px',
    md: '16px',
    lg: '18px',
    xl: '20px',
    '2xl': '24px',
    '3xl': '30px',
    '4xl': '36px',
    '5xl': '48px'
  },
  
  // 字体粗细
  fontWeights: {
    normal: 400,
    medium: 500,
    semibold: 600,
    bold: 700
  },
  
  // 行高
  lineHeights: {
    tight: 1.25,
    normal: 1.5,
    relaxed: 1.75
  },
  
  // 间距系统
  spacing: {
    0: '0',
    1: '4px',
    2: '8px',
    3: '12px',
    4: '16px',
    5: '20px',
    6: '24px',
    8: '32px',
    10: '40px',
    12: '48px',
    16: '64px',
    20: '80px',
    24: '96px'
  },
  
  // 圆角
  borderRadius: {
    none: '0',
    sm: '4px',
    md: '8px',
    lg: '12px',
    xl: '16px',
    '2xl': '24px',
    full: '9999px'
  },
  
  // 阴影
  shadows: {
    sm: '0 1px 2px 0 rgba(0, 0, 0, 0.05)',
    md: '0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06)',
    lg: '0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05)',
    xl: '0 20px 25px -5px rgba(0, 0, 0, 0.1), 0 10px 10px -5px rgba(0, 0, 0, 0.04)',
    '2xl': '0 25px 50px -12px rgba(0, 0, 0, 0.25)',
    inner: 'inset 0 2px 4px 0 rgba(0, 0, 0, 0.06)'
  },
  
  // 断点
  breakpoints: {
    xs: '480px',
    sm: '576px',
    md: '768px',
    lg: '992px',
    xl: '1200px',
    xxl: '1600px'
  },
  
  // Z-index层级
  zIndex: {
    hide: -1,
    auto: 'auto',
    base: 0,
    docked: 10,
    dropdown: 1000,
    sticky: 1100,
    banner: 1200,
    overlay: 1300,
    modal: 1400,
    popover: 1500,
    skipLink: 1600,
    toast: 1700,
    tooltip: 1800
  },
  
  // 过渡动画
  transitions: {
    fast: '150ms ease-in-out',
    normal: '300ms ease-in-out',
    slow: '500ms ease-in-out'
  },
  
  // 医学图像查看器专用配置
  medical: {
    // 窗宽窗位预设
    windowPresets: {
      lung: { width: 1500, center: -600 },
      abdomen: { width: 400, center: 50 },
      bone: { width: 1800, center: 400 },
      brain: { width: 100, center: 50 },
      liver: { width: 150, center: 30 },
      mediastinum: { width: 350, center: 50 }
    },
    
    // 测量工具颜色
    measurementColors: {
      length: '#ff6b6b',
      angle: '#4ecdc4',
      area: '#45b7d1',
      volume: '#96ceb4',
      density: '#ffeaa7'
    },
    
    // 标注颜色
    annotationColors: {
      normal: '#2ecc71',
      suspicious: '#f39c12',
      abnormal: '#e74c3c',
      critical: '#9b59b6'
    }
  }
}

// 暗色主题
export const darkTheme = {
  ...theme,
  colors: {
    ...theme.colors,
    gray: {
      50: '#1f1f1f',
      100: '#262626',
      200: '#434343',
      300: '#595959',
      400: '#8c8c8c',
      500: '#bfbfbf',
      600: '#d9d9d9',
      700: '#f0f0f0',
      800: '#f5f5f5',
      900: '#fafafa'
    }
  }
}

// 主题类型定义
export type Theme = typeof theme
export type ThemeColors = typeof theme.colors
export type ThemeSpacing = typeof theme.spacing

export default theme