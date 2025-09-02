import React, { Component, ErrorInfo, ReactNode } from 'react'
import { Result, Button } from 'antd'
import styled from 'styled-components'
import { reportReactError } from '../../utils/errorBoundary'

interface Props {
  children: ReactNode
  fallback?: ReactNode
}

interface State {
  hasError: boolean
  error?: Error
  errorInfo?: ErrorInfo
}

const ErrorContainer = styled.div`
  display: flex;
  justify-content: center;
  align-items: center;
  min-height: 400px;
  padding: 20px;
`

class ErrorBoundary extends Component<Props, State> {
  constructor(props: Props) {
    super(props)
    this.state = { hasError: false }
  }

  static getDerivedStateFromError(error: Error): State {
    return {
      hasError: true,
      error
    }
  }

  componentDidCatch(error: Error, errorInfo: ErrorInfo) {
    console.error('ErrorBoundary caught an error:', error, errorInfo)
    
    // 报告错误到监控服务
    reportReactError(error, errorInfo)
    
    this.setState({
      error,
      errorInfo
    })
  }

  handleReload = () => {
    window.location.reload()
  }

  handleReset = () => {
    this.setState({ hasError: false, error: undefined, errorInfo: undefined })
  }

  render() {
    if (this.state.hasError) {
      if (this.props.fallback) {
        return this.props.fallback
      }

      return (
        <ErrorContainer>
          <Result
            status="error"
            title="页面出现错误"
            subTitle="抱歉，页面遇到了一些问题。请尝试刷新页面或联系技术支持。"
            extra={[
              <Button type="primary" key="reload" onClick={this.handleReload}>
                刷新页面
              </Button>,
              <Button key="reset" onClick={this.handleReset}>
                重试
              </Button>
            ]}
          >
            {process.env.NODE_ENV === 'development' && this.state.error && (
              <div style={{ textAlign: 'left', marginTop: 16 }}>
                <details>
                  <summary>错误详情 (开发模式)</summary>
                  <pre style={{ whiteSpace: 'pre-wrap', fontSize: '12px' }}>
                    {this.state.error.toString()}
                    {this.state.errorInfo?.componentStack}
                  </pre>
                </details>
              </div>
            )}
          </Result>
        </ErrorContainer>
      )
    }

    return this.props.children
  }
}

export { ErrorBoundary }