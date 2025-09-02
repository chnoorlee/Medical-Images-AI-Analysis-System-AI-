import React from 'react'
import { createGlobalStyle } from 'styled-components'

const GlobalStylesComponent = createGlobalStyle`
  * {
    box-sizing: border-box;
    margin: 0;
    padding: 0;
  }

  html, body {
    height: 100%;
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'PingFang SC', 'Hiragino Sans GB',
      'Microsoft YaHei', 'Helvetica Neue', Helvetica, Arial, sans-serif, 'Apple Color Emoji',
      'Segoe UI Emoji', 'Segoe UI Symbol';
    font-size: 14px;
    line-height: 1.5715;
    color: rgba(0, 0, 0, 0.85);
    background-color: #f5f5f5;
    -webkit-font-smoothing: antialiased;
    -moz-osx-font-smoothing: grayscale;
  }

  #root {
    height: 100%;
  }

  .app {
    height: 100%;
  }

  .app-loading {
    display: flex;
    justify-content: center;
    align-items: center;
    height: 100vh;
    background-color: #f5f5f5;
  }

  /* 自定义滚动条 */
  ::-webkit-scrollbar {
    width: 8px;
    height: 8px;
  }

  ::-webkit-scrollbar-track {
    background: #f1f1f1;
    border-radius: 4px;
  }

  ::-webkit-scrollbar-thumb {
    background: #c1c1c1;
    border-radius: 4px;
  }

  ::-webkit-scrollbar-thumb:hover {
    background: #a8a8a8;
  }

  /* 医学图像查看器样式 */
  .medical-image-viewer {
    background: #000;
    position: relative;
    overflow: hidden;
  }

  .medical-image-viewer canvas {
    cursor: crosshair;
  }

  .medical-image-viewer .tools-overlay {
    position: absolute;
    top: 10px;
    left: 10px;
    z-index: 10;
    background: rgba(0, 0, 0, 0.7);
    border-radius: 4px;
    padding: 8px;
  }

  .medical-image-viewer .info-overlay {
    position: absolute;
    top: 10px;
    right: 10px;
    z-index: 10;
    background: rgba(0, 0, 0, 0.7);
    border-radius: 4px;
    padding: 8px;
    color: white;
    font-size: 12px;
  }

  /* 工作站布局样式 */
  .workstation-layout {
    height: 100vh;
    display: flex;
    flex-direction: column;
  }

  .workstation-header {
    flex-shrink: 0;
    background: #fff;
    border-bottom: 1px solid #f0f0f0;
    padding: 12px 24px;
  }

  .workstation-content {
    flex: 1;
    display: flex;
    overflow: hidden;
  }

  .workstation-sidebar {
    width: 300px;
    flex-shrink: 0;
    background: #fff;
    border-right: 1px solid #f0f0f0;
    overflow-y: auto;
  }

  .workstation-main {
    flex: 1;
    display: flex;
    flex-direction: column;
    overflow: hidden;
  }

  .workstation-viewer {
    flex: 1;
    background: #000;
    position: relative;
  }

  .workstation-tools {
    height: 60px;
    flex-shrink: 0;
    background: #fff;
    border-top: 1px solid #f0f0f0;
    display: flex;
    align-items: center;
    padding: 0 16px;
    gap: 12px;
  }

  /* 响应式设计 */
  @media (max-width: 768px) {
    .workstation-sidebar {
      width: 100%;
      position: absolute;
      top: 0;
      left: 0;
      z-index: 100;
      transform: translateX(-100%);
      transition: transform 0.3s ease;
    }

    .workstation-sidebar.open {
      transform: translateX(0);
    }
  }

  /* 打印样式 */
  @media print {
    .no-print {
      display: none !important;
    }

    .print-only {
      display: block !important;
    }

    body {
      background: white !important;
    }
  }

  /* 动画效果 */
  .fade-in {
    animation: fadeIn 0.3s ease-in;
  }

  @keyframes fadeIn {
    from {
      opacity: 0;
    }
    to {
      opacity: 1;
    }
  }

  .slide-in {
    animation: slideIn 0.3s ease-out;
  }

  @keyframes slideIn {
    from {
      transform: translateY(-20px);
      opacity: 0;
    }
    to {
      transform: translateY(0);
      opacity: 1;
    }
  }

  /* 高对比度模式 */
  @media (prefers-contrast: high) {
    .ant-btn {
      border-width: 2px;
    }

    .ant-card {
      border-width: 2px;
    }
  }

  /* 减少动画模式 */
  @media (prefers-reduced-motion: reduce) {
    * {
      animation-duration: 0.01ms !important;
      animation-iteration-count: 1 !important;
      transition-duration: 0.01ms !important;
    }
  }

  /* 暗色主题支持 */
  @media (prefers-color-scheme: dark) {
    .auto-theme {
      background-color: #141414;
      color: rgba(255, 255, 255, 0.85);
    }
  }
`

export const GlobalStyles: React.FC = () => {
  return <GlobalStylesComponent />
}

export default GlobalStyles