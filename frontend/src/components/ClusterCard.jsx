import React from 'react'
import { motion } from 'framer-motion'
import { 
  Users, 
  TrendingUp, 
  TrendingDown, 
  Target, 
  Activity,
  MoreVertical,
  Eye,
  Edit,
  Trash2
} from 'lucide-react'
import clsx from 'clsx'

const ClusterCard = ({ 
  cluster, 
  index = 0, 
  onView, 
  onEdit, 
  onDelete,
  className = ''
}) => {
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
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ delay: index * 0.1 }}
      className={clsx(
        'card hover:shadow-lg transition-all duration-200 group',
        className
      )}
    >
      {/* Header */}
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

      {/* Body */}
      <div className="card-body space-y-4">
        <p className="text-sm text-secondary-600 line-clamp-2">
          {cluster.description}
        </p>

        {/* Main Metrics Grid */}
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

        {/* Performance Metrics */}
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

        {/* Last Updated */}
        <div className="text-xs text-secondary-500 border-t border-secondary-200 pt-3">
          Atualizado {cluster.lastUpdated}
        </div>
      </div>

      {/* Footer */}
      <div className="card-footer">
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-2">
            <button 
              onClick={() => onView?.(cluster)}
              className="btn-ghost btn-sm"
            >
              <Eye className="h-4 w-4 mr-1" />
              Visualizar
            </button>
            <button 
              onClick={() => onEdit?.(cluster)}
              className="btn-ghost btn-sm"
            >
              <Edit className="h-4 w-4 mr-1" />
              Editar
            </button>
          </div>
          <button 
            onClick={() => onDelete?.(cluster)}
            className="btn-ghost btn-sm text-danger-600 hover:bg-danger-50"
          >
            <Trash2 className="h-4 w-4" />
          </button>
        </div>
      </div>
    </motion.div>
  )
}

export default ClusterCard