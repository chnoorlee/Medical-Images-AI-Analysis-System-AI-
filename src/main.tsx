import React from 'react'
import ReactDOM from 'react-dom/client'
import { Provider } from 'react-redux'
import { BrowserRouter } from 'react-router-dom'
import { ConfigProvider } from 'antd'
import zhCN from 'antd/locale/zh_CN'
import dayjs from 'dayjs'
import 'dayjs/locale/zh-cn'

import App from './App'
import { store } from './store'
import { GlobalStyles } from './styles/GlobalStyles'
import { theme } from './styles/theme'

import './styles/index.css'

// Configure dayjs locale
dayjs.locale('zh-cn')

// Error boundary for development
if (import.meta.env.DEV) {
  import('./utils/errorBoundary')
}

// Performance monitoring
if (import.meta.env.PROD) {
  import('./utils/performance')
}

ReactDOM.createRoot(document.getElementById('root')!).render(
  <React.StrictMode>
    <Provider store={store}>
      <BrowserRouter>
        <ConfigProvider
          locale={zhCN}
          theme={{
            token: {
              colorPrimary: theme.colors.primary[500],
              colorSuccess: theme.colors.success[500],
              colorWarning: theme.colors.warning[500],
              colorError: theme.colors.error[500],
              colorInfo: theme.colors.info[500],
              borderRadius: 8,
              fontSize: 14,
              fontFamily: theme.fonts.primary,
            },
            components: {
              Button: {
                borderRadius: 8,
                controlHeight: 40,
              },
              Input: {
                borderRadius: 8,
                controlHeight: 40,
              },
              Select: {
                borderRadius: 8,
                controlHeight: 40,
              },
              Table: {
                borderRadius: 8,
                headerBg: theme.colors.gray[50],
              },
              Card: {
                borderRadius: 12,
                boxShadow: '0 2px 8px rgba(0, 0, 0, 0.06)',
              },
              Modal: {
                borderRadius: 12,
              },
            },
          }}
        >
          <GlobalStyles />
          <App />
        </ConfigProvider>
      </BrowserRouter>
    </Provider>
  </React.StrictMode>,
)