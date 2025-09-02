import React, { Suspense, useEffect } from 'react'
import { Routes, Route, Navigate } from 'react-router-dom'
import { Spin } from 'antd'
import { useSelector, useDispatch } from 'react-redux'

import { RootState } from './store'
import { initializeApp } from './store/slices/appSlice'
import { ProtectedRoute } from './components/common/ProtectedRoute'
import { ErrorBoundary } from './components/common/ErrorBoundary'
import { Layout } from './components/layout/Layout'
import { LoadingSpinner } from './components/common/LoadingSpinner'

// Lazy load pages for better performance
const LoginPage = React.lazy(() => import('./pages/auth/LoginPage'))
const DashboardPage = React.lazy(() => import('./pages/dashboard/DashboardPage'))
const WorkstationPage = React.lazy(() => import('./pages/workstation/WorkstationPage'))
const PatientListPage = React.lazy(() => import('./pages/patient/PatientListPage'))
const PatientDetailPage = React.lazy(() => import('./pages/patient/PatientDetailPage'))
const ImageViewerPage = React.lazy(() => import('./pages/image/ImageViewerPage'))
const ReportPage = React.lazy(() => import('./pages/report/ReportPage'))
const ReportDetailPage = React.lazy(() => import('./pages/report/ReportDetailPage'))
const SystemManagePage = React.lazy(() => import('./pages/admin/SystemManagePage'))
const QualityControlPage = React.lazy(() => import('./pages/quality/QualityControlPage'))
const DataManagePage = React.lazy(() => import('./pages/data/DataManagePage'))
const SettingsPage = React.lazy(() => import('./pages/settings/SettingsPage'))
const NotFoundPage = React.lazy(() => import('./pages/error/NotFoundPage'))

function App() {
  const dispatch = useDispatch()
  const { isInitialized, isLoading } = useSelector((state: RootState) => state.app)
  const { isAuthenticated, user } = useSelector((state: RootState) => state.auth)

  useEffect(() => {
    dispatch(initializeApp())
  }, [dispatch])

  // Show loading spinner during app initialization
  if (!isInitialized || isLoading) {
    return (
      <div className="app-loading">
        <LoadingSpinner size="large" tip="正在初始化系统..." />
      </div>
    )
  }

  return (
    <ErrorBoundary>
      <div className="app">
        <Routes>
          {/* Public routes */}
          <Route
            path="/login"
            element={
              <Suspense fallback={<LoadingSpinner />}>
                <LoginPage />
              </Suspense>
            }
          />

          {/* Protected routes */}
          <Route
            path="/"
            element={
              <ProtectedRoute>
                <Layout />
              </ProtectedRoute>
            }
          >
            {/* Dashboard */}
            <Route
              index
              element={
                <Suspense fallback={<LoadingSpinner />}>
                  <DashboardPage />
                </Suspense>
              }
            />

            {/* Workstation */}
            <Route
              path="workstation"
              element={
                <Suspense fallback={<LoadingSpinner />}>
                  <WorkstationPage />
                </Suspense>
              }
            />

            {/* Patient Management */}
            <Route
              path="patients"
              element={
                <Suspense fallback={<LoadingSpinner />}>
                  <PatientListPage />
                </Suspense>
              }
            />
            <Route
              path="patients/:patientId"
              element={
                <Suspense fallback={<LoadingSpinner />}>
                  <PatientDetailPage />
                </Suspense>
              }
            />

            {/* Image Viewer */}
            <Route
              path="images/:imageId"
              element={
                <Suspense fallback={<LoadingSpinner />}>
                  <ImageViewerPage />
                </Suspense>
              }
            />

            {/* Reports */}
            <Route
              path="reports"
              element={
                <Suspense fallback={<LoadingSpinner />}>
                  <ReportPage />
                </Suspense>
              }
            />
            <Route
              path="reports/:reportId"
              element={
                <Suspense fallback={<LoadingSpinner />}>
                  <ReportDetailPage />
                </Suspense>
              }
            />

            {/* Quality Control */}
            <Route
              path="quality"
              element={
                <Suspense fallback={<LoadingSpinner />}>
                  <QualityControlPage />
                </Suspense>
              }
            />

            {/* Data Management */}
            <Route
              path="data"
              element={
                <Suspense fallback={<LoadingSpinner />}>
                  <DataManagePage />
                </Suspense>
              }
            />

            {/* Admin Panel - Only for admin users */}
            {user?.role === 'admin' && (
              <Route
                path="admin"
                element={
                  <Suspense fallback={<LoadingSpinner />}>
                    <SystemManagePage />
                  </Suspense>
                }
              />
            )}

            {/* Settings */}
            <Route
              path="settings"
              element={
                <Suspense fallback={<LoadingSpinner />}>
                  <SettingsPage />
                </Suspense>
              }
            />
          </Route>

          {/* 404 Page */}
          <Route
            path="*"
            element={
              <Suspense fallback={<LoadingSpinner />}>
                <NotFoundPage />
              </Suspense>
            }
          />
        </Routes>
      </div>
    </ErrorBoundary>
  )
}

export default App