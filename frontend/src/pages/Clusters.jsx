import React, { useState } from 'react'
import { motion } from 'framer-motion'
import { 
  Plus, 
  Search, 
  Filter, 
  MoreVertical,
  Users,
  TrendingUp,
  TrendingDown,
  Eye,
  Edit,
  Trash2,
  Target,
  Activity
} from 'lucide-react'

const Clusters = () => {
  const [searchTerm, setSearchTerm] = useState('')
  const [selectedFilter, setSelectedFilter] = useState('all')

  // Mock data for clusters
  const clusters = [
    {
      id: 1,
      name: 'Alto Valor',
      description: 'Clientes com alto valor de vida útil e alta frequência de compra',
      users: 3240,
      avgValue: 'R$ 850',
      conversion: '8.5%',
      growth: '+12.3%',
      trend: 'up',
      color: 'primary',
      status: 'active',
      lastUpdated: '2 horas atrás',
      campaigns: 5
    },
    {
      id: 2,
      name: 'Médio Valor',
      description: 'Clientes com comportamento de compra moderado e potencial de crescimento',
      users: 5680,
      avgValue: 'R$ 420',
      conversion: '5.2%',
      growth: '+8.1%',
      trend: 'up',
      color: 'success',
      status: 'active',
      lastUpdated: '3 horas atrás',
      campaigns: 8
    },
    {
      id: 3,
      name: 'Baixo Valor',
      description: 'Clientes com baixo ticket médio mas alta frequência de interação',
      users: 8920,
      avgValue: 'R$ 180',
      conversion: '3.1%',
      growth: '-2.4%',
      trend: 'down',
      color: 'warning',
      status: 'active',
      lastUpdated: '1 hora atrás',
      campaigns: 3
    },
    {
      id: 4,
      name: 'Novo Cliente',
      description: 'Usuários recém-cadastrados com potencial ainda a ser explorado',
      users: 2150,
      avgValue: 'R$ 95',
      conversion: '12.8%',
      growth: '+45.2%',
      trend: 'up',
      color: 'info',
      status: 'active',
      lastUpdated: '30 min atrás',
      campaigns: 12
    },
    {
      id: 5,
      name: 'Inativo',
      description: 'Clientes sem atividade nos últimos 90 dias - necessita reativação',
      users: 1890,
      avgValue: 'R$ 0',
      conversion: '0.8%',
      growth: '+5.6%',
      trend: 'up',
      color: 'danger',
      status: 'inactive',
      lastUpdated: '6 horas atrás',
      campaigns: 2
    },
    {
      id: 6,
      name: 'VIP Premium',
      description: 'Clientes premium com histórico de compras de alto valor',
      users: 450,
      avgValue: 'R$ 2,150',
      conversion: '15.2%',
      growth: '+18.7%',
      trend: 'up',
      color: 'purple',
      status: 'active',
      lastUpdated: '1 hora atrás',
      campaigns: 7
    }
  ]

  const filteredClusters = clusters.filter(cluster => {
    const matchesSearch = cluster.name.toLowerCase().includes(searchTerm.toLowerCase()) ||
                         cluster.description.toLowerCase().includes(searchTerm.toLowerCase())
    const matchesFilter = selectedFilter === 'all' || cluster.status === selectedFilter
    return matchesSearch && matchesFilter
  })

  const getColorClasses = (color) => {
    const colorMap = {
      primary: 'bg-primary-50 text-primary-700 border-primary-200',
      success: 'bg-success-50 text-success-700 border-success-200',
      warning: 'bg-warning-50 text-warning-700 border-warning-200',
      danger: 'bg-danger-50 text-danger-700 border-danger-200',
      info: 'bg-sky-50 text-sky-700 border-sky-200',
      purple: 'bg-purple-50 text-purple-700 border-purple-200'
    }
    return colorMap[color] || colorMap.primary
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between">
        <div>
          <h1 className="text-2xl font-bold text-secondary-900">Clusters de Usuários</h1>
          <p className="mt-1 text-sm text-secondary-600">
            Gerencie e analise os segmentos de usuários baseados em Machine Learning
          </p>
        </div>
        <div className="mt-4 sm:mt-0">
          <button className="btn-primary btn-md">
            <Plus className="h-4 w-4 mr-2" />
            Novo Cluster
          </button>
        </div>
      </div>

      {/* Search and Filters */}
      <div className="flex flex-col sm:flex-row gap-4">
        <div className="flex-1 relative">
          <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 h-4 w-4 text-secondary-400" />
          <input
            type="text"
            placeholder="Buscar clusters..."
            value={searchTerm}
            onChange={(e) => setSearchTerm(e.target.value)}
            className="form-input pl-10"
          />
        </div>
        <div className="flex items-center space-x-3">
          <select
            value={selectedFilter}
            onChange={(e) => setSelectedFilter(e.target.value)}
            className="form-input min-w-[140px]"
          >
            <option value="all">Todos os status</option>
            <option value="active">Ativos</option>
            <option value="inactive">Inativos</option>
          </select>
          <button className="btn-ghost btn-md">
            <Filter className="h-4 w-4 mr-2" />
            Filtros
          </button>
        </div>
      </div>

      {/* Clusters Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-2 xl:grid-cols-3 gap-6">
        {filteredClusters.map((cluster, index) => (
          <motion.div
            key={cluster.id}
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: index * 0.1 }}
            className="card hover:shadow-lg transition-all duration-200 group"
          >
            <div className="card-header">
              <div className="flex items-start justify-between">
                <div className="flex items-center space-x-3">
                  <div className={`p-2 rounded-lg border ${getColorClasses(cluster.color)}`}>
                    <Target className="h-5 w-5" />
                  </div>
                  <div>
                    <h3 className="text-lg font-semibold text-secondary-900 group-hover:text-primary-600 transition-colors">
                      {cluster.name}
                    </h3>
                    <div className="flex items-center space-x-2 mt-1">
                      <span className={`badge ${cluster.status === 'active' ? 'badge-success' : 'badge-secondary'}`}>
                        {cluster.status === 'active' ? 'Ativo' : 'Inativo'}
                      </span>
                      <span className="text-xs text-secondary-500">
                        {cluster.campaigns} campanhas
                      </span>
                    </div>
                  </div>
                </div>
                <div className="relative">
                  <button className="p-2 hover:bg-secondary-100 rounded-lg transition-colors">
                    <MoreVertical className="h-4 w-4 text-secondary-400" />
                  </button>
                </div>
              </div>
            </div>

            <div className="card-body space-y-4">
              <p className="text-sm text-secondary-600 line-clamp-2">
                {cluster.description}
              </p>

              {/* Metrics */}
              <div className="grid grid-cols-2 gap-4">
                <div className="text-center p-3 bg-secondary-50 rounded-lg">
                  <div className="flex items-center justify-center mb-1">
                    <Users className="h-4 w-4 text-secondary-400 mr-1" />
                    <span className="text-xs text-secondary-500">Usuários</span>
                  </div>
                  <p className="text-lg font-bold text-secondary-900">
                    {cluster.users.toLocaleString()}
                  </p>
                </div>
                <div className="text-center p-3 bg-secondary-50 rounded-lg">
                  <div className="flex items-center justify-center mb-1">
                    <Activity className="h-4 w-4 text-secondary-400 mr-1" />
                    <span className="text-xs text-secondary-500">Ticket Médio</span>
                  </div>
                  <p className="text-lg font-bold text-secondary-900">
                    {cluster.avgValue}
                  </p>
                </div>
              </div>

              <div className="flex items-center justify-between">
                <div className="text-center">
                  <p className="text-xs text-secondary-500">Conversão</p>
                  <p className="text-sm font-semibold text-secondary-900">{cluster.conversion}</p>
                </div>
                <div className="text-center">
                  <p className="text-xs text-secondary-500">Crescimento</p>
                  <div className={`flex items-center justify-center text-sm font-semibold ${
                    cluster.trend === 'up' ? 'text-success-600' : 'text-danger-600'
                  }`}>
                    {cluster.trend === 'up' ? (
                      <TrendingUp className="h-3 w-3 mr-1" />
                    ) : (
                      <TrendingDown className="h-3 w-3 mr-1" />
                    )}
                    {cluster.growth}
                  </div>
                </div>
              </div>

              <div className="text-xs text-secondary-500 border-t border-secondary-200 pt-3">
                Atualizado {cluster.lastUpdated}
              </div>
            </div>

            <div className="card-footer">
              <div className="flex items-center justify-between">
                <div className="flex items-center space-x-2">
                  <button className="btn-ghost btn-sm">
                    <Eye className="h-4 w-4 mr-1" />
                    Visualizar
                  </button>
                  <button className="btn-ghost btn-sm">
                    <Edit className="h-4 w-4 mr-1" />
                    Editar
                  </button>
                </div>
                <button className="btn-ghost btn-sm text-danger-600 hover:bg-danger-50">
                  <Trash2 className="h-4 w-4" />
                </button>
              </div>
            </div>
          </motion.div>
        ))}
      </div>

      {/* Empty State */}
      {filteredClusters.length === 0 && (
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          className="text-center py-12"
        >
          <Target className="mx-auto h-12 w-12 text-secondary-400" />
          <h3 className="mt-4 text-lg font-medium text-secondary-900">
            Nenhum cluster encontrado
          </h3>
          <p className="mt-2 text-sm text-secondary-500">
            {searchTerm ? 'Tente ajustar os filtros de busca' : 'Comece criando seu primeiro cluster'}
          </p>
          {!searchTerm && (
            <button className="mt-4 btn-primary btn-md">
              <Plus className="h-4 w-4 mr-2" />
              Criar Cluster
            </button>
          )}
        </motion.div>
      )}
    </div>
  )
}

export default Clusters