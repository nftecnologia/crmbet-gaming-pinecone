import React, { useState, useMemo, useCallback, useEffect } from 'react'
import { motion } from 'framer-motion'
import { 
  Search, 
  Filter, 
  Download, 
  MoreVertical,
  User,
  Mail,
  Phone,
  Calendar,
  Target,
  Activity,
  Eye,
  Edit,
  Trash2,
  Settings
} from 'lucide-react'
import VirtualTable from '../components/VirtualTable'
import { useOfflineStorage } from '../services/offlineStorage'
import websocketService from '../services/websocket'

const Users = () => {
  const [searchTerm, setSearchTerm] = useState('')
  const [selectedCluster, setSelectedCluster] = useState('all')
  const [selectedRows, setSelectedRows] = useState(new Set())
  const [loading, setLoading] = useState(false)
  const [hasNextPage, setHasNextPage] = useState(true)
  const [isNextPageLoading, setIsNextPageLoading] = useState(false)

  // Use offline storage for data persistence
  const { 
    data: users, 
    loading: storageLoading, 
    stats: storageStats,
    store: storeUser,
    delete: deleteUser,
    refresh: refreshUsers
  } = useOfflineStorage('users')

  // Generate large dataset for performance testing
  const generateUsers = useCallback((count = 100000) => {
    const clusters = ['Alto Valor', 'Médio Valor', 'Baixo Valor', 'Novo Cliente', 'VIP Premium', 'Inativo']
    const clusterColors = ['primary', 'success', 'warning', 'info', 'purple', 'danger']
    const names = [
      'Ana Silva', 'Carlos Santos', 'Mariana Costa', 'João Oliveira', 'Lucia Fernandes',
      'Roberto Lima', 'Patricia Rocha', 'Fernando Alves', 'Beatriz Souza', 'Ricardo Pereira',
      'Camila Rodrigues', 'Thiago Martins', 'Juliana Barbosa', 'Gabriel Nascimento', 'Larissa Carvalho'
    ]
    
    return Array.from({ length: count }, (_, i) => {
      const clusterIndex = i % clusters.length
      const nameIndex = i % names.length
      
      return {
        id: i + 1,
        name: `${names[nameIndex]} ${i + 1}`,
        email: `user${i + 1}@example.com`,
        phone: `(11) ${String(99999 + i).slice(-5)}-${String(1111 + i).slice(-4)}`,
        cluster: clusters[clusterIndex],
        clusterColor: clusterColors[clusterIndex],
        totalValue: `R$ ${(Math.random() * 10000).toFixed(2)}`,
        lastActivity: `${Math.floor(Math.random() * 24)} horas atrás`,
        joinDate: new Date(2023, Math.floor(Math.random() * 12), Math.floor(Math.random() * 28) + 1).toLocaleDateString('pt-BR'),
        campaigns: Math.floor(Math.random() * 20),
        conversion: `${(Math.random() * 20).toFixed(1)}%`,
        status: Math.random() > 0.2 ? 'active' : 'inactive',
        score: Math.floor(Math.random() * 100),
        createdAt: new Date(2023, Math.floor(Math.random() * 12), Math.floor(Math.random() * 28) + 1).toISOString()
      }
    })
  }, [])

  // Initialize with large dataset if empty
  useEffect(() => {
    if (users.length === 0 && !storageLoading) {
      setLoading(true)
      const largeDataset = generateUsers(100000)
      
      // Store in chunks to avoid blocking UI
      const storeInChunks = async () => {
        const chunkSize = 1000
        for (let i = 0; i < largeDataset.length; i += chunkSize) {
          const chunk = largeDataset.slice(i, i + chunkSize)
          await Promise.all(chunk.map(user => storeUser(user)))
          
          // Update progress
          if (i % 10000 === 0) {
            console.log(`Stored ${i + chunk.length} / ${largeDataset.length} users`)
          }
        }
        setLoading(false)
        refreshUsers()
      }
      
      storeInChunks()
    }
  }, [users.length, storageLoading, generateUsers, storeUser, refreshUsers])

  // WebSocket integration for real-time updates
  useEffect(() => {
    const unsubscribe = websocketService.subscribe('user_update', (data) => {
      console.log('Real-time user update:', data)
      if (data.data) {
        storeUser(data.data)
      }
    })
    
    return unsubscribe
  }, [storeUser])

  // Mock data for backward compatibility (kept smaller for development)
  const mockUsers = [
    {
      id: 1,
      name: 'Ana Silva',
      email: 'ana.silva@email.com',
      phone: '(11) 99999-1111',
      cluster: 'Alto Valor',
      clusterColor: 'primary',
      totalValue: 'R$ 2.450',
      lastActivity: '2 horas atrás',
      joinDate: '15/03/2023',
      campaigns: 8,
      conversion: '12.5%',
      status: 'active'
    },
    {
      id: 2,
      name: 'Carlos Santos',
      email: 'carlos.santos@email.com',
      phone: '(11) 99999-2222',
      cluster: 'Médio Valor',
      clusterColor: 'success',
      totalValue: 'R$ 890',
      lastActivity: '1 dia atrás',
      joinDate: '22/01/2023',
      campaigns: 5,
      conversion: '8.2%',
      status: 'active'
    },
    {
      id: 3,
      name: 'Mariana Costa',
      email: 'mariana.costa@email.com',
      phone: '(11) 99999-3333',
      cluster: 'VIP Premium',
      clusterColor: 'purple',
      totalValue: 'R$ 5.200',
      lastActivity: '30 min atrás',
      joinDate: '08/05/2023',
      campaigns: 12,
      conversion: '18.7%',
      status: 'active'
    },
    {
      id: 4,
      name: 'João Oliveira',
      email: 'joao.oliveira@email.com',
      phone: '(11) 99999-4444',
      cluster: 'Novo Cliente',
      clusterColor: 'info',
      totalValue: 'R$ 120',
      lastActivity: '3 horas atrás',
      joinDate: '20/12/2023',
      campaigns: 2,
      conversion: '15.1%',
      status: 'active'
    },
    {
      id: 5,
      name: 'Lucia Fernandes',
      email: 'lucia.fernandes@email.com',
      phone: '(11) 99999-5555',
      cluster: 'Baixo Valor',
      clusterColor: 'warning',
      totalValue: 'R$ 340',
      lastActivity: '2 dias atrás',
      joinDate: '11/09/2023',
      campaigns: 3,
      conversion: '4.8%',
      status: 'active'
    },
    {
      id: 6,
      name: 'Roberto Lima',
      email: 'roberto.lima@email.com',
      phone: '(11) 99999-6666',
      cluster: 'Inativo',
      clusterColor: 'danger',
      totalValue: 'R$ 0',
      lastActivity: '2 semanas atrás',
      joinDate: '05/07/2023',
      campaigns: 1,
      conversion: '0%',
      status: 'inactive'
    },
    {
      id: 7,
      name: 'Patricia Rocha',
      email: 'patricia.rocha@email.com',
      phone: '(11) 99999-7777',
      cluster: 'Alto Valor',
      clusterColor: 'primary',
      totalValue: 'R$ 1.890',
      lastActivity: '1 hora atrás',
      joinDate: '18/04/2023',
      campaigns: 7,
      conversion: '11.3%',
      status: 'active'
    },
    {
      id: 8,
      name: 'Fernando Alves',
      email: 'fernando.alves@email.com',
      phone: '(11) 99999-8888',
      cluster: 'Médio Valor',
      clusterColor: 'success',
      totalValue: 'R$ 670',
      lastActivity: '4 horas atrás',
      joinDate: '30/06/2023',
      campaigns: 4,
      conversion: '7.9%',
      status: 'active'
    }
  ]

  const clusters = ['Alto Valor', 'Médio Valor', 'Baixo Valor', 'Novo Cliente', 'VIP Premium', 'Inativo']

  // Use actual data or fallback to mock data
  const currentUsers = users.length > 0 ? users : mockUsers

  // Memoized filtered data for performance
  const filteredUsers = useMemo(() => {
    return currentUsers.filter(user => {
      const matchesSearch = user.name?.toLowerCase().includes(searchTerm.toLowerCase()) ||
                           user.email?.toLowerCase().includes(searchTerm.toLowerCase())
      const matchesCluster = selectedCluster === 'all' || user.cluster === selectedCluster
      return matchesSearch && matchesCluster
    })
  }, [currentUsers, searchTerm, selectedCluster])

  // Virtual Table configuration
  const columns = useMemo(() => [
    {
      key: 'name',
      title: 'Usuário',
      width: 280,
      render: (value, user) => (
        <div className="flex items-center space-x-3">
          <div className="h-8 w-8 bg-secondary-100 rounded-full flex items-center justify-center">
            <User className="h-4 w-4 text-secondary-600" />
          </div>
          <div>
            <div className="font-medium text-secondary-900">{user.name}</div>
            <div className="text-sm text-secondary-500 flex items-center space-x-1">
              <Mail className="h-3 w-3" />
              <span>{user.email}</span>
            </div>
          </div>
        </div>
      )
    },
    {
      key: 'cluster',
      title: 'Cluster',
      width: 140,
      render: (value, user) => (
        <span className={`badge ${getClusterBadgeClass(user.clusterColor)}`}>
          {value}
        </span>
      )
    },
    {
      key: 'totalValue',
      title: 'Valor Total',
      width: 120,
      render: (value) => (
        <div className="font-semibold text-secondary-900">{value}</div>
      )
    },
    {
      key: 'campaigns',
      title: 'Campanhas',
      width: 100,
      render: (value) => (
        <div className="text-center">
          <span className="inline-flex items-center justify-center h-6 w-6 rounded-full bg-primary-100 text-xs font-medium text-primary-800">
            {value}
          </span>
        </div>
      )
    },
    {
      key: 'conversion',
      title: 'Conversão',
      width: 100,
      render: (value) => {
        const numValue = parseFloat(value)
        const colorClass = numValue > 10 ? 'text-success-600' :
                          numValue > 5 ? 'text-warning-600' : 'text-danger-600'
        return (
          <div className={`font-medium ${colorClass}`}>
            {value}
          </div>
        )
      }
    },
    {
      key: 'lastActivity',
      title: 'Última Atividade',
      width: 140,
      render: (value) => (
        <div className="text-sm text-secondary-600">{value}</div>
      )
    },
    {
      key: 'status',
      title: 'Status',
      width: 100,
      render: (value) => (
        <span className={`badge ${value === 'active' ? 'badge-success' : 'badge-secondary'}`}>
          {value === 'active' ? 'Ativo' : 'Inativo'}
        </span>
      )
    },
    {
      key: 'actions',
      title: 'Ações',
      width: 120,
      sortable: false,
      render: (_, user) => (
        <div className="flex items-center space-x-2">
          <button 
            className="p-1 hover:bg-secondary-100 rounded text-secondary-400 hover:text-secondary-600"
            onClick={() => handleViewUser(user)}
          >
            <Eye className="h-4 w-4" />
          </button>
          <button 
            className="p-1 hover:bg-secondary-100 rounded text-secondary-400 hover:text-secondary-600"
            onClick={() => handleEditUser(user)}
          >
            <Edit className="h-4 w-4" />
          </button>
          <button 
            className="p-1 hover:bg-danger-50 rounded text-secondary-400 hover:text-danger-600"
            onClick={() => handleDeleteUser(user)}
          >
            <Trash2 className="h-4 w-4" />
          </button>
          <button className="p-1 hover:bg-secondary-100 rounded text-secondary-400 hover:text-secondary-600">
            <MoreVertical className="h-4 w-4" />
          </button>
        </div>
      )
    }
  ], [])

  // Event handlers
  const handleRowClick = useCallback((user) => {
    console.log('Row clicked:', user)
  }, [])

  const handleRowSelect = useCallback((userId, selected) => {
    setSelectedRows(prev => {
      const newSet = new Set(prev)
      if (selected) {
        newSet.add(userId)
      } else {
        newSet.delete(userId)
      }
      return newSet
    })
  }, [])

  const handleViewUser = useCallback((user) => {
    console.log('View user:', user)
  }, [])

  const handleEditUser = useCallback((user) => {
    console.log('Edit user:', user)
  }, [])

  const handleDeleteUser = useCallback(async (user) => {
    if (window.confirm(`Deseja realmente excluir o usuário ${user.name}?`)) {
      await deleteUser(user.id)
      refreshUsers()
    }
  }, [deleteUser, refreshUsers])

  const loadNextPage = useCallback(async () => {
    if (isNextPageLoading) return
    
    setIsNextPageLoading(true)
    
    // Simulate loading more data
    await new Promise(resolve => setTimeout(resolve, 1000))
    
    // In a real app, this would fetch more data from the API
    setIsNextPageLoading(false)
  }, [isNextPageLoading])

  const handleExport = useCallback(() => {
    const dataToExport = filteredUsers.map(user => ({
      Nome: user.name,
      Email: user.email,
      Telefone: user.phone,
      Cluster: user.cluster,
      'Valor Total': user.totalValue,
      Campanhas: user.campaigns,
      Conversão: user.conversion,
      Status: user.status === 'active' ? 'Ativo' : 'Inativo'
    }))
    
    const csv = [
      Object.keys(dataToExport[0]).join(','),
      ...dataToExport.map(row => Object.values(row).join(','))
    ].join('\n')
    
    const blob = new Blob([csv], { type: 'text/csv' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = `usuarios-${new Date().toISOString().split('T')[0]}.csv`
    a.click()
    URL.revokeObjectURL(url)
  }, [filteredUsers])

  const getClusterBadgeClass = (color) => {
    const colorMap = {
      primary: 'badge-info',
      success: 'badge-success',
      warning: 'badge-warning',
      danger: 'badge-danger',
      info: 'bg-sky-100 text-sky-800',
      purple: 'bg-purple-100 text-purple-800'
    }
    return colorMap[color] || 'badge-secondary'
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between">
        <div>
          <h1 className="text-2xl font-bold text-secondary-900">Usuários</h1>
          <p className="mt-1 text-sm text-secondary-600">
            Gerencie usuários organizados por clusters de ML
          </p>
        </div>
        <div className="mt-4 sm:mt-0 flex items-center space-x-3">
          <button 
            onClick={handleExport}
            className="btn-secondary btn-md"
          >
            <Download className="h-4 w-4 mr-2" />
            Exportar
          </button>
          {storageStats.isOnline && (
            <button className="btn-primary btn-md">
              <Settings className="h-4 w-4 mr-2" />
              Configurações
            </button>
          )}
        </div>
      </div>

      {/* Search and Filters */}
      <div className="flex flex-col sm:flex-row gap-4">
        <div className="flex-1 relative">
          <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 h-4 w-4 text-secondary-400" />
          <input
            type="text"
            placeholder="Buscar usuários por nome ou email..."
            value={searchTerm}
            onChange={(e) => setSearchTerm(e.target.value)}
            className="form-input pl-10"
          />
        </div>
        <div className="flex items-center space-x-3">
          <select
            value={selectedCluster}
            onChange={(e) => setSelectedCluster(e.target.value)}
            className="form-input min-w-[160px]"
          >
            <option value="all">Todos os clusters</option>
            {clusters.map(cluster => (
              <option key={cluster} value={cluster}>{cluster}</option>
            ))}
          </select>
          <button className="btn-ghost btn-md">
            <Filter className="h-4 w-4 mr-2" />
            Filtros
          </button>
        </div>
      </div>

      {/* Stats Cards */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        <div className="card p-4">
          <div className="flex items-center">
            <div className="p-2 bg-primary-100 rounded-lg">
              <User className="h-5 w-5 text-primary-600" />
            </div>
            <div className="ml-3">
              <p className="text-sm font-medium text-secondary-600">Total de Usuários</p>
              <p className="text-2xl font-bold text-secondary-900">{filteredUsers.length}</p>
            </div>
          </div>
        </div>
        <div className="card p-4">
          <div className="flex items-center">
            <div className="p-2 bg-success-100 rounded-lg">
              <Activity className="h-5 w-5 text-success-600" />
            </div>
            <div className="ml-3">
              <p className="text-sm font-medium text-secondary-600">Usuários Ativos</p>
              <p className="text-2xl font-bold text-secondary-900">
                {filteredUsers.filter(u => u.status === 'active').length}
              </p>
            </div>
          </div>
        </div>
        <div className="card p-4">
          <div className="flex items-center">
            <div className="p-2 bg-warning-100 rounded-lg">
              <Target className="h-5 w-5 text-warning-600" />
            </div>
            <div className="ml-3">
              <p className="text-sm font-medium text-secondary-600">Taxa Conv. Média</p>
              <p className="text-2xl font-bold text-secondary-900">9.2%</p>
            </div>
          </div>
        </div>
        <div className="card p-4">
          <div className="flex items-center">
            <div className="p-2 bg-purple-100 rounded-lg">
              <Mail className="h-5 w-5 text-purple-600" />
            </div>
            <div className="ml-3">
              <p className="text-sm font-medium text-secondary-600">Campanhas Ativas</p>
              <p className="text-2xl font-bold text-secondary-900">47</p>
            </div>
          </div>
        </div>
      </div>

      {/* Ultra-Performance Virtual Table */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        className="space-y-4"
      >
        {/* Performance Stats */}
        {process.env.NODE_ENV === 'development' && (
          <div className="bg-gray-800 text-white p-4 rounded-lg text-sm font-mono">
            <div className="grid grid-cols-4 gap-4">
              <div>
                <div className="text-gray-400">Total Records</div>
                <div className="text-xl font-bold">{currentUsers.length.toLocaleString()}</div>
              </div>
              <div>
                <div className="text-gray-400">Filtered</div>
                <div className="text-xl font-bold">{filteredUsers.length.toLocaleString()}</div>
              </div>
              <div>
                <div className="text-gray-400">Selected</div>
                <div className="text-xl font-bold">{selectedRows.size}</div>
              </div>
              <div>
                <div className="text-gray-400">Storage</div>
                <div className="text-xl font-bold">
                  {storageStats.isOnline ? 'Online' : 'Offline'}
                </div>
              </div>
            </div>
          </div>
        )}

        {/* Loading State */}
        {(loading || storageLoading) && (
          <div className="card p-8 text-center">
            <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600 mx-auto mb-4"></div>
            <p className="text-gray-600 font-medium">
              {loading ? 'Gerando dados de teste...' : 'Carregando usuários...'}
            </p>
            <p className="text-sm text-gray-500 mt-2">
              {loading ? 'Isso pode levar alguns segundos' : 'Otimizando performance...'}
            </p>
          </div>
        )}

        {/* Virtual Table */}
        {!loading && !storageLoading && (
          <VirtualTable
            data={filteredUsers}
            columns={columns}
            height={600}
            rowHeight={64}
            onRowClick={handleRowClick}
            onRowSelect={handleRowSelect}
            selectedRows={selectedRows}
            hasNextPage={hasNextPage}
            isNextPageLoading={isNextPageLoading}
            loadNextPage={loadNextPage}
            searchable={false} // Search is handled externally
            sortable={true}
            filterable={false}
            className="shadow-lg"
          />
        )}
      </motion.div>

      {/* Empty State */}
      {filteredUsers.length === 0 && (
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          className="text-center py-12"
        >
          <User className="mx-auto h-12 w-12 text-secondary-400" />
          <h3 className="mt-4 text-lg font-medium text-secondary-900">
            Nenhum usuário encontrado
          </h3>
          <p className="mt-2 text-sm text-secondary-500">
            Tente ajustar os filtros de busca ou cluster selecionado
          </p>
        </motion.div>
      )}
    </div>
  )
}

export default Users