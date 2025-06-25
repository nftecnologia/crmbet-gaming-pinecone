import React, { useState } from 'react'
import { useLocation } from 'react-router-dom'
import { 
  Menu, 
  Search, 
  Bell, 
  Settings, 
  User,
  ChevronDown,
  RefreshCw,
  Download,
  Filter
} from 'lucide-react'
import { motion, AnimatePresence } from 'framer-motion'

const pageNames = {
  '/dashboard': 'Dashboard',
  '/clusters': 'Clusters',
  '/users': 'Usuários',
  '/campaigns': 'Campanhas',
  '/analytics': 'Analytics'
}

const Header = ({ setSidebarOpen }) => {
  const location = useLocation()
  const [searchValue, setSearchValue] = useState('')
  const [showNotifications, setShowNotifications] = useState(false)
  const [showProfile, setShowProfile] = useState(false)
  
  const currentPageName = pageNames[location.pathname] || 'Dashboard'

  const notifications = [
    {
      id: 1,
      title: 'Nova campanha criada',
      message: 'Campanha "Black Friday 2024" foi criada com sucesso',
      time: '2 min atrás',
      type: 'success'
    },
    {
      id: 2,
      title: 'Cluster atualizado',
      message: 'Cluster "Alto Valor" teve 15 novos usuários adicionados',
      time: '5 min atrás',
      type: 'info'
    },
    {
      id: 3,
      title: 'Meta de conversão atingida',
      message: 'Campanha "Reativação" atingiu 105% da meta',
      time: '1 hora atrás',
      type: 'success'
    }
  ]

  return (
    <header className="bg-white border-b border-secondary-200 sticky top-0 z-30">
      <div className="px-4 sm:px-6 lg:px-8">
        <div className="flex h-16 items-center justify-between">
          {/* Left section */}
          <div className="flex items-center space-x-4">
            {/* Mobile menu button */}
            <button
              type="button"
              className="lg:hidden p-2 rounded-md text-secondary-500 hover:bg-secondary-100 hover:text-secondary-700 focus:outline-none focus:ring-2 focus:ring-primary-500"
              onClick={() => setSidebarOpen(true)}
            >
              <Menu className="h-5 w-5" />
            </button>

            {/* Page title */}
            <div className="flex items-center space-x-3">
              <h1 className="text-2xl font-bold text-secondary-900">
                {currentPageName}
              </h1>
              {location.pathname === '/dashboard' && (
                <div className="flex items-center space-x-2">
                  <div className="h-2 w-2 bg-success-500 rounded-full animate-pulse"></div>
                  <span className="text-sm text-secondary-500">Ao vivo</span>
                </div>
              )}
            </div>
          </div>

          {/* Center section - Search */}
          <div className="hidden md:flex flex-1 max-w-lg mx-8">
            <div className="relative w-full">
              <div className="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none">
                <Search className="h-4 w-4 text-secondary-400" />
              </div>
              <input
                type="text"
                value={searchValue}
                onChange={(e) => setSearchValue(e.target.value)}
                className="block w-full pl-10 pr-3 py-2 border border-secondary-300 rounded-lg bg-white placeholder-secondary-400 focus:outline-none focus:ring-1 focus:ring-primary-500 focus:border-primary-500 text-sm"
                placeholder="Buscar usuários, campanhas, clusters..."
              />
              {searchValue && (
                <div className="absolute top-full left-0 right-0 mt-1 bg-white border border-secondary-200 rounded-lg shadow-lg z-50">
                  <div className="p-2">
                    <div className="text-xs text-secondary-500 px-2 py-1">Resultados da busca</div>
                    <div className="space-y-1">
                      <div className="px-2 py-2 hover:bg-secondary-50 rounded cursor-pointer">
                        <div className="text-sm font-medium">Cluster Alto Valor</div>
                        <div className="text-xs text-secondary-500">3.2k usuários</div>
                      </div>
                      <div className="px-2 py-2 hover:bg-secondary-50 rounded cursor-pointer">
                        <div className="text-sm font-medium">Campanha Black Friday</div>
                        <div className="text-xs text-secondary-500">Ativa • 85% de taxa de abertura</div>
                      </div>
                    </div>
                  </div>
                </div>
              )}
            </div>
          </div>

          {/* Right section */}
          <div className="flex items-center space-x-3">
            {/* Action buttons */}
            <div className="hidden sm:flex items-center space-x-2">
              <button className="btn-ghost btn-sm">
                <RefreshCw className="h-4 w-4 mr-2" />
                Atualizar
              </button>
              <button className="btn-ghost btn-sm">
                <Download className="h-4 w-4 mr-2" />
                Exportar
              </button>
              <button className="btn-ghost btn-sm">
                <Filter className="h-4 w-4 mr-2" />
                Filtros
              </button>
            </div>

            {/* Notifications */}
            <div className="relative">
              <button
                className="p-2 text-secondary-500 hover:text-secondary-700 hover:bg-secondary-100 rounded-lg transition-colors relative"
                onClick={() => setShowNotifications(!showNotifications)}
              >
                <Bell className="h-5 w-5" />
                <span className="absolute top-1 right-1 h-2 w-2 bg-danger-500 rounded-full"></span>
              </button>

              <AnimatePresence>
                {showNotifications && (
                  <motion.div
                    initial={{ opacity: 0, scale: 0.95, y: -10 }}
                    animate={{ opacity: 1, scale: 1, y: 0 }}
                    exit={{ opacity: 0, scale: 0.95, y: -10 }}
                    className="absolute right-0 mt-2 w-80 bg-white rounded-lg shadow-lg border border-secondary-200 z-50"
                  >
                    <div className="p-4 border-b border-secondary-200">
                      <div className="flex items-center justify-between">
                        <h3 className="text-sm font-semibold text-secondary-900">Notificações</h3>
                        <span className="text-xs text-secondary-500">{notifications.length} não lidas</span>
                      </div>
                    </div>
                    <div className="max-h-80 overflow-y-auto">
                      {notifications.map((notification) => (
                        <div key={notification.id} className="p-4 border-b border-secondary-100 hover:bg-secondary-50 cursor-pointer">
                          <div className="flex items-start space-x-3">
                            <div className={`h-2 w-2 rounded-full mt-2 ${
                              notification.type === 'success' ? 'bg-success-500' : 
                              notification.type === 'info' ? 'bg-primary-500' : 'bg-warning-500'
                            }`}></div>
                            <div className="flex-1 min-w-0">
                              <p className="text-sm font-medium text-secondary-900">{notification.title}</p>
                              <p className="text-sm text-secondary-600 mt-1">{notification.message}</p>
                              <p className="text-xs text-secondary-400 mt-2">{notification.time}</p>
                            </div>
                          </div>
                        </div>
                      ))}
                    </div>
                    <div className="p-3 border-t border-secondary-200">
                      <button className="w-full text-center text-sm text-primary-600 hover:text-primary-700 font-medium">
                        Ver todas as notificações
                      </button>
                    </div>
                  </motion.div>
                )}
              </AnimatePresence>
              
              {showNotifications && (
                <div 
                  className="fixed inset-0 z-40" 
                  onClick={() => setShowNotifications(false)}
                />
              )}
            </div>

            {/* Profile dropdown */}
            <div className="relative">
              <button
                className="flex items-center space-x-2 p-2 text-secondary-700 hover:bg-secondary-100 rounded-lg transition-colors"
                onClick={() => setShowProfile(!showProfile)}
              >
                <div className="h-8 w-8 bg-primary-100 rounded-full flex items-center justify-center">
                  <User className="h-4 w-4 text-primary-600" />
                </div>
                <div className="hidden md:block text-left">
                  <div className="text-sm font-medium">Marketing Team</div>
                  <div className="text-xs text-secondary-500">Admin</div>
                </div>
                <ChevronDown className="h-4 w-4" />
              </button>

              <AnimatePresence>
                {showProfile && (
                  <motion.div
                    initial={{ opacity: 0, scale: 0.95, y: -10 }}
                    animate={{ opacity: 1, scale: 1, y: 0 }}
                    exit={{ opacity: 0, scale: 0.95, y: -10 }}
                    className="absolute right-0 mt-2 w-48 bg-white rounded-lg shadow-lg border border-secondary-200 z-50"
                  >
                    <div className="p-2">
                      <button className="w-full text-left px-3 py-2 text-sm text-secondary-700 hover:bg-secondary-100 rounded-md flex items-center space-x-2">
                        <User className="h-4 w-4" />
                        <span>Perfil</span>
                      </button>
                      <button className="w-full text-left px-3 py-2 text-sm text-secondary-700 hover:bg-secondary-100 rounded-md flex items-center space-x-2">
                        <Settings className="h-4 w-4" />
                        <span>Configurações</span>
                      </button>
                      <hr className="my-1 border-secondary-200" />
                      <button className="w-full text-left px-3 py-2 text-sm text-danger-600 hover:bg-danger-50 rounded-md">
                        Sair
                      </button>
                    </div>
                  </motion.div>
                )}
              </AnimatePresence>
              
              {showProfile && (
                <div 
                  className="fixed inset-0 z-40" 
                  onClick={() => setShowProfile(false)}
                />
              )}
            </div>
          </div>
        </div>
      </div>
    </header>
  )
}

export default Header