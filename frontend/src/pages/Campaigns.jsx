import React, { useState } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { 
  Plus, 
  Search, 
  Filter, 
  MoreVertical,
  Mail,
  Users,
  TrendingUp,
  Calendar,
  Eye,
  Edit,
  Trash2,
  Play,
  Pause,
  Copy,
  Send,
  Target,
  Clock,
  CheckCircle,
  AlertCircle
} from 'lucide-react'

const Campaigns = () => {
  const [searchTerm, setSearchTerm] = useState('')
  const [selectedStatus, setSelectedStatus] = useState('all')
  const [showCreateModal, setShowCreateModal] = useState(false)

  // Mock data for campaigns
  const campaigns = [
    {
      id: 1,
      name: 'Black Friday 2024',
      description: 'Campanha especial para o Black Friday com descontos exclusivos',
      type: 'promotional',
      status: 'active',
      cluster: 'Alto Valor',
      clusterColor: 'primary',
      startDate: '2024-11-20',
      endDate: '2024-11-30',
      sent: 2450,
      delivered: 2385,
      opened: 1673,
      clicked: 412,
      converted: 103,
      revenue: 'R$ 15.450',
      openRate: '70.2%',
      clickRate: '24.6%',
      conversionRate: '4.2%',
      createdAt: '2 dias atrás',
      lastActivity: '30 min atrás'
    },
    {
      id: 2,
      name: 'Reativação de Inativos',
      description: 'Campanha para reativar usuários que não compram há mais de 90 dias',
      type: 'reactivation',
      status: 'active',
      cluster: 'Inativo',
      clusterColor: 'danger',
      startDate: '2024-11-15',
      endDate: '2024-12-15',
      sent: 1890,
      delivered: 1820,
      opened: 728,
      clicked: 146,
      converted: 32,
      revenue: 'R$ 2.890',
      openRate: '40.0%',
      clickRate: '20.1%',
      conversionRate: '2.2%',
      createdAt: '1 semana atrás',
      lastActivity: '2 horas atrás'
    },
    {
      id: 3,
      name: 'Boas-vindas Novos Clientes',
      description: 'Sequência de emails de boas-vindas para novos usuários cadastrados',
      type: 'welcome',
      status: 'active',
      cluster: 'Novo Cliente',
      clusterColor: 'info',
      startDate: '2024-11-01',
      endDate: '2024-12-31',
      sent: 2150,
      delivered: 2098,
      opened: 1469,
      clicked: 588,
      converted: 275,
      revenue: 'R$ 8.450',
      openRate: '70.1%',
      clickRate: '40.0%',
      conversionRate: '13.1%',
      createdAt: '3 semanas atrás',
      lastActivity: '1 hora atrás'
    },
    {
      id: 4,
      name: 'Produtos Relacionados',
      description: 'Recomendações personalizadas baseadas no histórico de compras',
      type: 'recommendation',
      status: 'scheduled',
      cluster: 'Médio Valor',
      clusterColor: 'success',
      startDate: '2024-12-01',
      endDate: '2024-12-31',
      sent: 0,
      delivered: 0,
      opened: 0,
      clicked: 0,
      converted: 0,
      revenue: 'R$ 0',
      openRate: '0%',
      clickRate: '0%',
      conversionRate: '0%',
      createdAt: '5 dias atrás',
      lastActivity: 'Agendada'
    },
    {
      id: 5,
      name: 'Pesquisa de Satisfação',
      description: 'Coleta de feedback dos clientes VIP sobre produtos e serviços',
      type: 'survey',
      status: 'completed',
      cluster: 'VIP Premium',
      clusterColor: 'purple',
      startDate: '2024-10-15',
      endDate: '2024-10-30',
      sent: 450,
      delivered: 445,
      opened: 356,
      clicked: 189,
      converted: 94,
      revenue: 'N/A',
      openRate: '80.0%',
      clickRate: '53.1%',
      conversionRate: '20.9%',
      createdAt: '1 mês atrás',
      lastActivity: '2 semanas atrás'
    },
    {
      id: 6,
      name: 'Carrinho Abandonado',
      description: 'Recuperação de vendas para carrinhos abandonados',
      type: 'recovery',
      status: 'paused',
      cluster: 'Baixo Valor',
      clusterColor: 'warning',
      startDate: '2024-11-10',
      endDate: '2024-12-10',
      sent: 892,
      delivered: 864,
      opened: 432,
      clicked: 86,
      converted: 25,
      revenue: 'R$ 1.250',
      openRate: '50.0%',
      clickRate: '19.9%',
      conversionRate: '2.9%',
      createdAt: '2 semanas atrás',
      lastActivity: '3 dias atrás'
    }
  ]

  const filteredCampaigns = campaigns.filter(campaign => {
    const matchesSearch = campaign.name.toLowerCase().includes(searchTerm.toLowerCase()) ||
                         campaign.description.toLowerCase().includes(searchTerm.toLowerCase())
    const matchesStatus = selectedStatus === 'all' || campaign.status === selectedStatus
    return matchesSearch && matchesStatus
  })

  const getStatusColor = (status) => {
    const statusMap = {
      active: 'badge-success',
      scheduled: 'badge-info',
      completed: 'badge-secondary',
      paused: 'badge-warning',
      draft: 'bg-secondary-100 text-secondary-800'
    }
    return statusMap[status] || 'badge-secondary'
  }

  const getStatusText = (status) => {
    const statusMap = {
      active: 'Ativa',
      scheduled: 'Agendada',
      completed: 'Concluída',
      paused: 'Pausada',
      draft: 'Rascunho'
    }
    return statusMap[status] || status
  }

  const getTypeIcon = (type) => {
    const iconMap = {
      promotional: Mail,
      reactivation: TrendingUp,
      welcome: Users,
      recommendation: Target,
      survey: CheckCircle,
      recovery: AlertCircle
    }
    return iconMap[type] || Mail
  }

  const getClusterBadgeClass = (color) => {
    const colorMap = {
      primary: 'bg-primary-100 text-primary-800',
      success: 'bg-success-100 text-success-800',
      warning: 'bg-warning-100 text-warning-800',
      danger: 'bg-danger-100 text-danger-800',
      info: 'bg-sky-100 text-sky-800',
      purple: 'bg-purple-100 text-purple-800'
    }
    return colorMap[color] || 'bg-secondary-100 text-secondary-800'
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between">
        <div>
          <h1 className="text-2xl font-bold text-secondary-900">Campanhas</h1>
          <p className="mt-1 text-sm text-secondary-600">
            Gerencie campanhas de marketing segmentadas por clusters de ML
          </p>
        </div>
        <div className="mt-4 sm:mt-0">
          <button 
            onClick={() => setShowCreateModal(true)}
            className="btn-primary btn-md"
          >
            <Plus className="h-4 w-4 mr-2" />
            Nova Campanha
          </button>
        </div>
      </div>

      {/* Search and Filters */}
      <div className="flex flex-col sm:flex-row gap-4">
        <div className="flex-1 relative">
          <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 h-4 w-4 text-secondary-400" />
          <input
            type="text"
            placeholder="Buscar campanhas..."
            value={searchTerm}
            onChange={(e) => setSearchTerm(e.target.value)}
            className="form-input pl-10"
          />
        </div>
        <div className="flex items-center space-x-3">
          <select
            value={selectedStatus}
            onChange={(e) => setSelectedStatus(e.target.value)}
            className="form-input min-w-[140px]"
          >
            <option value="all">Todos os status</option>
            <option value="active">Ativas</option>
            <option value="scheduled">Agendadas</option>
            <option value="paused">Pausadas</option>
            <option value="completed">Concluídas</option>
          </select>
          <button className="btn-ghost btn-md">
            <Filter className="h-4 w-4 mr-2" />
            Filtros
          </button>
        </div>
      </div>

      {/* Summary Cards */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        <div className="card p-4">
          <div className="flex items-center">
            <div className="p-2 bg-primary-100 rounded-lg">
              <Mail className="h-5 w-5 text-primary-600" />
            </div>
            <div className="ml-3">
              <p className="text-sm font-medium text-secondary-600">Campanhas Ativas</p>
              <p className="text-2xl font-bold text-secondary-900">
                {campaigns.filter(c => c.status === 'active').length}
              </p>
            </div>
          </div>
        </div>
        <div className="card p-4">
          <div className="flex items-center">
            <div className="p-2 bg-success-100 rounded-lg">
              <Send className="h-5 w-5 text-success-600" />
            </div>
            <div className="ml-3">
              <p className="text-sm font-medium text-secondary-600">Emails Enviados</p>
              <p className="text-2xl font-bold text-secondary-900">
                {campaigns.reduce((sum, c) => sum + c.sent, 0).toLocaleString()}
              </p>
            </div>
          </div>
        </div>
        <div className="card p-4">
          <div className="flex items-center">
            <div className="p-2 bg-warning-100 rounded-lg">
              <TrendingUp className="h-5 w-5 text-warning-600" />
            </div>
            <div className="ml-3">
              <p className="text-sm font-medium text-secondary-600">Taxa Abertura Média</p>
              <p className="text-2xl font-bold text-secondary-900">62.1%</p>
            </div>
          </div>
        </div>
        <div className="card p-4">
          <div className="flex items-center">
            <div className="p-2 bg-purple-100 rounded-lg">
              <Target className="h-5 w-5 text-purple-600" />
            </div>
            <div className="ml-3">
              <p className="text-sm font-medium text-secondary-600">Receita Total</p>
              <p className="text-2xl font-bold text-secondary-900">R$ 28.0k</p>
            </div>
          </div>
        </div>
      </div>

      {/* Campaigns Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {filteredCampaigns.map((campaign, index) => {
          const TypeIcon = getTypeIcon(campaign.type)
          return (
            <motion.div
              key={campaign.id}
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: index * 0.1 }}
              className="card hover:shadow-lg transition-all duration-200 group"
            >
              <div className="card-header">
                <div className="flex items-start justify-between">
                  <div className="flex items-start space-x-3">
                    <div className="p-2 bg-primary-50 rounded-lg border border-primary-200">
                      <TypeIcon className="h-5 w-5 text-primary-600" />
                    </div>
                    <div className="flex-1">
                      <h3 className="text-lg font-semibold text-secondary-900 group-hover:text-primary-600 transition-colors">
                        {campaign.name}
                      </h3>
                      <div className="flex items-center space-x-2 mt-1">
                        <span className={`badge ${getStatusColor(campaign.status)}`}>
                          {getStatusText(campaign.status)}
                        </span>
                        <span className={`badge ${getClusterBadgeClass(campaign.clusterColor)}`}>
                          {campaign.cluster}
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
                  {campaign.description}
                </p>

                {/* Date Range */}
                <div className="flex items-center text-sm text-secondary-500">
                  <Calendar className="h-4 w-4 mr-2" />
                  <span>{campaign.startDate} - {campaign.endDate}</span>
                </div>

                {/* Metrics */}
                <div className="grid grid-cols-2 gap-4">
                  <div className="text-center p-3 bg-secondary-50 rounded-lg">
                    <p className="text-xs text-secondary-500 mb-1">Enviados</p>
                    <p className="text-lg font-bold text-secondary-900">
                      {campaign.sent.toLocaleString()}
                    </p>
                  </div>
                  <div className="text-center p-3 bg-secondary-50 rounded-lg">
                    <p className="text-xs text-secondary-500 mb-1">Taxa Abertura</p>
                    <p className="text-lg font-bold text-primary-600">
                      {campaign.openRate}
                    </p>
                  </div>
                </div>

                <div className="grid grid-cols-3 gap-3 text-center text-sm">
                  <div>
                    <p className="text-secondary-500">Cliques</p>
                    <p className="font-semibold text-secondary-900">{campaign.clickRate}</p>
                  </div>
                  <div>
                    <p className="text-secondary-500">Conversão</p>
                    <p className="font-semibold text-secondary-900">{campaign.conversionRate}</p>
                  </div>
                  <div>
                    <p className="text-secondary-500">Receita</p>
                    <p className="font-semibold text-success-600">{campaign.revenue}</p>
                  </div>
                </div>

                <div className="text-xs text-secondary-500 border-t border-secondary-200 pt-3 flex items-center justify-between">
                  <span>Criada {campaign.createdAt}</span>
                  <div className="flex items-center space-x-1">
                    <Clock className="h-3 w-3" />
                    <span>{campaign.lastActivity}</span>
                  </div>
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
                  <div className="flex items-center space-x-2">
                    {campaign.status === 'active' ? (
                      <button className="btn-ghost btn-sm text-warning-600 hover:bg-warning-50">
                        <Pause className="h-4 w-4" />
                      </button>
                    ) : campaign.status === 'paused' ? (
                      <button className="btn-ghost btn-sm text-success-600 hover:bg-success-50">
                        <Play className="h-4 w-4" />
                      </button>
                    ) : null}
                    <button className="btn-ghost btn-sm">
                      <Copy className="h-4 w-4" />
                    </button>
                    <button className="btn-ghost btn-sm text-danger-600 hover:bg-danger-50">
                      <Trash2 className="h-4 w-4" />
                    </button>
                  </div>
                </div>
              </div>
            </motion.div>
          )
        })}
      </div>

      {/* Empty State */}
      {filteredCampaigns.length === 0 && (
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          className="text-center py-12"
        >
          <Mail className="mx-auto h-12 w-12 text-secondary-400" />
          <h3 className="mt-4 text-lg font-medium text-secondary-900">
            Nenhuma campanha encontrada
          </h3>
          <p className="mt-2 text-sm text-secondary-500">
            {searchTerm ? 'Tente ajustar os filtros de busca' : 'Comece criando sua primeira campanha'}
          </p>
          {!searchTerm && (
            <button 
              onClick={() => setShowCreateModal(true)}
              className="mt-4 btn-primary btn-md"
            >
              <Plus className="h-4 w-4 mr-2" />
              Criar Campanha
            </button>
          )}
        </motion.div>
      )}

      {/* Create Campaign Modal (Placeholder) */}
      <AnimatePresence>
        {showCreateModal && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-secondary-900/50 backdrop-blur-sm"
            onClick={() => setShowCreateModal(false)}
          >
            <motion.div
              initial={{ scale: 0.95, opacity: 0 }}
              animate={{ scale: 1, opacity: 1 }}
              exit={{ scale: 0.95, opacity: 0 }}
              onClick={(e) => e.stopPropagation()}
              className="bg-white rounded-xl p-6 max-w-md w-full"
            >
              <h3 className="text-lg font-semibold text-secondary-900 mb-4">
                Nova Campanha
              </h3>
              <p className="text-sm text-secondary-600 mb-6">
                O formulário de criação de campanha será implementado em breve com drag-and-drop e seleção de clusters.
              </p>
              <div className="flex items-center justify-end space-x-3">
                <button 
                  onClick={() => setShowCreateModal(false)}
                  className="btn-secondary btn-md"
                >
                  Cancelar
                </button>
                <button 
                  onClick={() => setShowCreateModal(false)}
                  className="btn-primary btn-md"
                >
                  Criar Campanha
                </button>
              </div>
            </motion.div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  )
}

export default Campaigns