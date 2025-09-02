import React from 'react'
import { Navigate, useLocation } from 'react-router-dom'
import { useSelector } from 'react-redux'
import { RootState } from '../../store'
import { LoadingSpinner } from './LoadingSpinner'

interface ProtectedRouteProps {
  children: React.ReactNode
  requiredRole?: string
  requiredPermissions?: string[]
}

const ProtectedRoute: React.FC<ProtectedRouteProps> = ({
  children,
  requiredRole,
  requiredPermissions = []
}) => {
  const location = useLocation()
  const { isAuthenticated, user, isLoading } = useSelector((state: RootState) => state.auth)

  // Show loading spinner while checking authentication
  if (isLoading) {
    return <LoadingSpinner tip="验证用户身份..." overlay />
  }

  // Redirect to login if not authenticated
  if (!isAuthenticated || !user) {
    return <Navigate to="/login" state={{ from: location }} replace />
  }

  // Check role requirement
  if (requiredRole && user.role !== requiredRole) {
    return (
      <Navigate 
        to="/unauthorized" 
        state={{ 
          message: `需要 ${requiredRole} 权限才能访问此页面`,
          from: location 
        }} 
        replace 
      />
    )
  }

  // Check permissions requirement
  if (requiredPermissions.length > 0) {
    const userPermissions = user.permissions || []
    const hasAllPermissions = requiredPermissions.every(permission => 
      userPermissions.includes(permission)
    )

    if (!hasAllPermissions) {
      return (
        <Navigate 
          to="/unauthorized" 
          state={{ 
            message: '您没有足够的权限访问此页面',
            requiredPermissions,
            from: location 
          }} 
          replace 
        />
      )
    }
  }

  // User is authenticated and has required permissions
  return <>{children}</>
}

export { ProtectedRoute }