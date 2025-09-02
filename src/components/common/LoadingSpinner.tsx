import React from 'react'
import { Spin, SpinProps } from 'antd'
import styled from 'styled-components'

interface LoadingSpinnerProps extends SpinProps {
  tip?: string
  overlay?: boolean
}

const SpinnerContainer = styled.div<{ overlay?: boolean }>`
  display: flex;
  justify-content: center;
  align-items: center;
  min-height: ${props => props.overlay ? '100vh' : '200px'};
  width: 100%;
  
  ${props => props.overlay && `
    position: fixed;
    top: 0;
    left: 0;
    right: 0;
    bottom: 0;
    background: rgba(255, 255, 255, 0.8);
    z-index: 9999;
  `}
`

const LoadingSpinner: React.FC<LoadingSpinnerProps> = ({
  tip = '加载中...',
  overlay = false,
  size = 'default',
  ...props
}) => {
  return (
    <SpinnerContainer overlay={overlay}>
      <Spin size={size} tip={tip} {...props} />
    </SpinnerContainer>
  )
}

export { LoadingSpinner }