import React, { useState } from 'react'
import { Outlet, useNavigate, useLocation } from 'react-router-dom'
import { Layout as AntLayout, Menu, Avatar, Dropdown, Badge, Button, Space } from 'antd'
import {
  DashboardOutlined,
  ExperimentOutlined,
  UserOutlined,
  FileImageOutlined,
  FileTextOutlined,
  SettingOutlined,
  LogoutOutlined,
  BellOutlined,
  MenuFoldOutlined,
  MenuUnfoldOutlined,
  TeamOutlined,
  SafetyCertificateOutlined,
  DatabaseOutlined
} from '@ant-design/icons'
import { useSelector, useDispatch } from 'react-redux'
import styled from 'styled-components'
import { RootState } from '../../store'
import { logout } from '../../store/slices/authSlice'

const { Header, Sider, Content } = AntLayout

const StyledLayout = styled(AntLayout)`
  min-height: 100vh;
`

const StyledHeader = styled(Header)`
  background: #fff;
  padding: 0 24px;
  box-shadow: 0 1px 4px rgba(0, 21, 41, 0.08);
  display: flex;
  align-items: center;
  justify-content: space-between;
  z-index: 10;
`

const StyledContent = styled(Content)`
  margin: 24px;
  padding: 24px;
  background: #fff;
  border-radius: 8px;
  min-height: calc(100vh - 112px);
`

const Logo = styled.div`
  height: 32px;
  margin: 16px;
  background: linear-gradient(135deg, #1890ff, #722ed1);
  border-radius: 6px;
  display: flex;
  align-items: center;
  justify-content: center;
  color: white;
  font-weight: bold;
  font-size: 14px;
`

const Layout: React.FC = () => {
  const [collapsed, setCollapsed] = useState(false)
  const navigate = useNavigate()
  const location = useLocation()
  const dispatch = useDispatch()
  const { user } = useSelector((state: RootState) => state.auth)

  const menuItems = [
    {
      key: '/',
      icon: <DashboardOutlined />,
      label: '仪表板'
    },
    {
      key: '/workstation',
      icon: <ExperimentOutlined />,
      label: '医生工作台'
    },
    {
      key: '/patients',
      icon: <TeamOutlined />,
      label: '患者管理'
    },
    {
      key: '/images',
      icon: <FileImageOutlined />,
      label: '影像管理'
    },
    {
      key: '/reports',
      icon: <FileTextOutlined />,
      label: '报告管理'
    },
    {
      key: '/quality',
      icon: <SafetyCertificateOutlined />,
      label: '质量控制'
    },
    {
      key: '/data',
      icon: <DatabaseOutlined />,
      label: '数据管理'
    },
    ...(user?.role === 'admin' ? [{
      key: '/admin',
      icon: <SettingOutlined />,
      label: '系统管理'
    }] : []),
    {
      key: '/settings',
      icon: <SettingOutlined />,
      label: '设置'
    }
  ]

  const handleMenuClick = ({ key }: { key: string }) => {
    navigate(key)
  }

  const handleLogout = () => {
    dispatch(logout())
    navigate('/login')
  }

  const userMenuItems = [
    {
      key: 'profile',
      icon: <UserOutlined />,
      label: '个人资料'
    },
    {
      key: 'settings',
      icon: <SettingOutlined />,
      label: '账户设置'
    },
    {
      type: 'divider' as const
    },
    {
      key: 'logout',
      icon: <LogoutOutlined />,
      label: '退出登录',
      onClick: handleLogout
    }
  ]

  return (
    <StyledLayout>
      <Sider 
        trigger={null} 
        collapsible 
        collapsed={collapsed}
        theme="light"
        width={256}
      >
        <Logo>
          {collapsed ? 'AI' : 'Medical AI'}
        </Logo>
        <Menu
          mode="inline"
          selectedKeys={[location.pathname]}
          items={menuItems}
          onClick={handleMenuClick}
        />
      </Sider>
      
      <AntLayout>
        <StyledHeader>
          <Space>
            <Button
              type="text"
              icon={collapsed ? <MenuUnfoldOutlined /> : <MenuFoldOutlined />}
              onClick={() => setCollapsed(!collapsed)}
            />
            <span style={{ fontSize: '16px', fontWeight: 500 }}>
              医学影像AI分析系统
            </span>
          </Space>
          
          <Space size="middle">
            <Badge count={5}>
              <Button 
                type="text" 
                icon={<BellOutlined />} 
                size="large"
              />
            </Badge>
            
            <Dropdown 
              menu={{ items: userMenuItems }}
              placement="bottomRight"
            >
              <Space style={{ cursor: 'pointer' }}>
                <Avatar 
                  size="small" 
                  icon={<UserOutlined />}
                  src={user?.avatar}
                />
                <span>{user?.name || user?.username}</span>
              </Space>
            </Dropdown>
          </Space>
        </StyledHeader>
        
        <StyledContent>
          <Outlet />
        </StyledContent>
      </AntLayout>
    </StyledLayout>
  )
}

export { Layout }