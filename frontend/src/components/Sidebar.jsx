import React from 'react'
import { NavLink, useLocation } from 'react-router-dom'
import { motion } from 'framer-motion'
import { 
  LayoutDashboard, 
  Users, 
  Target, 
  Mail, 
  BarChart3, 
  Settings,
  X,
  TrendingUp,
  Database
} from 'lucide-react'
import clsx from 'clsx'

const navigation = [
  { name: 'Dashboard', href: '/dashboard', icon: LayoutDashboard },
  { name: 'Clusters', href: '/clusters', icon: Database },
  { name: 'Usuários', href: '/users', icon: Users },
  { name: 'Campanhas', href: '/campaigns', icon: Mail },
  { name: 'Analytics', href: '/analytics', icon: BarChart3 },
]

const Sidebar = ({ sidebarOpen, setSidebarOpen }) => {
  const location = useLocation()

  return (
    <>
      {/* Mobile sidebar */}
      <motion.div
        initial={{ x: -280 }}
        animate={{ x: sidebarOpen ? 0 : -280 }}
        transition={{ type: 'spring', damping: 25, stiffness: 200 }}
        className="fixed inset-y-0 left-0 z-50 w-64 bg-white shadow-xl lg:hidden"
      >
        <div className="flex h-16 items-center justify-between px-6 border-b border-secondary-200">
          <div className="flex items-center space-x-3">
            <div className="flex h-8 w-8 items-center justify-center rounded-lg bg-primary-600">
              <TrendingUp className="h-5 w-5 text-white" />
            </div>
            <span className="text-xl font-bold text-secondary-900">CRM ML</span>
          </div>
          <button
            onClick={() => setSidebarOpen(false)}
            className="p-2 rounded-lg hover:bg-secondary-100 transition-colors"
          >
            <X className="h-5 w-5 text-secondary-500" />
          </button>
        </div>
        <nav className="mt-6 px-3">
          <ul className="space-y-1">
            {navigation.map((item) => {
              const isActive = location.pathname === item.href
              return (
                <li key={item.name}>
                  <NavLink
                    to={item.href}
                    onClick={() => setSidebarOpen(false)}
                    className={clsx(
                      'sidebar-nav-item',
                      isActive && 'active'
                    )}
                  >
                    <item.icon className="mr-3 h-5 w-5 flex-shrink-0" />
                    <span>{item.name}</span>
                    {isActive && (
                      <motion.div
                        layoutId="sidebar-indicator"
                        className="absolute right-3 h-2 w-2 rounded-full bg-primary-600"
                      />
                    )}
                  </NavLink>
                </li>
              )
            })}
          </ul>
        </nav>
        
        {/* Bottom section */}
        <div className="absolute bottom-0 left-0 right-0 p-4 border-t border-secondary-200">
          <div className="flex items-center space-x-3 p-3 rounded-lg bg-secondary-50">
            <div className="h-8 w-8 rounded-full bg-primary-100 flex items-center justify-center">
              <span className="text-xs font-medium text-primary-700">ML</span>
            </div>
            <div className="flex-1 min-w-0">
              <p className="text-sm font-medium text-secondary-900 truncate">
                Marketing Team
              </p>
              <p className="text-xs text-secondary-500 truncate">
                CRM Dashboard v1.0
              </p>
            </div>
          </div>
        </div>
      </motion.div>

      {/* Desktop sidebar */}
      <div className="hidden lg:fixed lg:inset-y-0 lg:flex lg:w-64 lg:flex-col">
        <div className="flex flex-col flex-grow bg-white border-r border-secondary-200">
          {/* Logo */}
          <div className="flex h-16 items-center px-6 border-b border-secondary-200">
            <div className="flex items-center space-x-3">
              <div className="flex h-8 w-8 items-center justify-center rounded-lg bg-primary-600">
                <TrendingUp className="h-5 w-5 text-white" />
              </div>
              <span className="text-xl font-bold text-secondary-900">CRM ML</span>
            </div>
          </div>

          {/* Navigation */}
          <nav className="mt-6 flex-1 px-3">
            <ul className="space-y-1">
              {navigation.map((item) => {
                const isActive = location.pathname === item.href
                return (
                  <li key={item.name}>
                    <NavLink
                      to={item.href}
                      className={clsx(
                        'sidebar-nav-item relative',
                        isActive && 'active'
                      )}
                    >
                      <item.icon className="mr-3 h-5 w-5 flex-shrink-0" />
                      <span>{item.name}</span>
                      {isActive && (
                        <motion.div
                          layoutId="sidebar-indicator"
                          className="absolute right-3 h-2 w-2 rounded-full bg-primary-600"
                        />
                      )}
                    </NavLink>
                  </li>
                )
              })}
            </ul>

            {/* Metrics Section */}
            <div className="mt-8 px-3">
              <h3 className="text-xs font-semibold text-secondary-400 uppercase tracking-wider mb-3">
                Métricas Rápidas
              </h3>
              <div className="space-y-3">
                <div className="bg-primary-50 p-3 rounded-lg">
                  <div className="flex items-center justify-between">
                    <span className="text-sm font-medium text-primary-900">Clusters Ativos</span>
                    <span className="text-lg font-bold text-primary-600">8</span>
                  </div>
                </div>
                <div className="bg-success-50 p-3 rounded-lg">
                  <div className="flex items-center justify-between">
                    <span className="text-sm font-medium text-success-900">Campanhas</span>
                    <span className="text-lg font-bold text-success-600">12</span>
                  </div>
                </div>
                <div className="bg-warning-50 p-3 rounded-lg">
                  <div className="flex items-center justify-between">
                    <span className="text-sm font-medium text-warning-900">Taxa Conv.</span>
                    <span className="text-lg font-bold text-warning-600">4.2%</span>
                  </div>
                </div>
              </div>
            </div>
          </nav>

          {/* Bottom section */}
          <div className="p-4 border-t border-secondary-200">
            <div className="flex items-center space-x-3 p-3 rounded-lg bg-secondary-50">
              <div className="h-8 w-8 rounded-full bg-primary-100 flex items-center justify-center">
                <span className="text-xs font-medium text-primary-700">ML</span>
              </div>
              <div className="flex-1 min-w-0">
                <p className="text-sm font-medium text-secondary-900 truncate">
                  Marketing Team
                </p>
                <p className="text-xs text-secondary-500 truncate">
                  CRM Dashboard v1.0
                </p>
              </div>
              <Settings className="h-4 w-4 text-secondary-400" />
            </div>
          </div>
        </div>
      </div>
    </>
  )
}

export default Sidebar