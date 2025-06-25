import React, { useState } from 'react'
import { motion } from 'framer-motion'
import { 
  User, 
  Mail, 
  Phone, 
  Eye, 
  Edit, 
  Trash2, 
  MoreVertical,
  ChevronLeft,
  ChevronRight,
  ArrowUpDown
} from 'lucide-react'

const UserTable = ({ 
  users = [], 
  onView, 
  onEdit, 
  onDelete,
  itemsPerPage = 10,
  showPagination = true,
  sortable = true 
}) => {
  const [currentPage, setCurrentPage] = useState(1)
  const [sortConfig, setSortConfig] = useState({ key: null, direction: 'asc' })

  // Sorting function
  const sortedUsers = React.useMemo(() => {
    if (!sortConfig.key) return users

    return [...users].sort((a, b) => {
      const aVal = a[sortConfig.key]
      const bVal = b[sortConfig.key]

      if (sortConfig.key === 'totalValue') {
        const aNum = parseFloat(aVal.replace(/[R$,\s]/g, ''))
        const bNum = parseFloat(bVal.replace(/[R$,\s]/g, ''))
        return sortConfig.direction === 'asc' ? aNum - bNum : bNum - aNum
      }

      if (typeof aVal === 'string') {
        return sortConfig.direction === 'asc' 
          ? aVal.localeCompare(bVal)
          : bVal.localeCompare(aVal)
      }

      return sortConfig.direction === 'asc' ? aVal - bVal : bVal - aVal
    })
  }, [users, sortConfig])

  // Pagination
  const totalPages = Math.ceil(sortedUsers.length / itemsPerPage)
  const startIndex = (currentPage - 1) * itemsPerPage
  const paginatedUsers = sortedUsers.slice(startIndex, startIndex + itemsPerPage)

  const handleSort = (key) => {
    if (!sortable) return
    
    setSortConfig(prev => ({
      key,
      direction: prev.key === key && prev.direction === 'asc' ? 'desc' : 'asc'
    }))
  }

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

  const SortableHeader = ({ children, sortKey, className = '' }) => (
    <th 
      className={`cursor-pointer hover:bg-secondary-100 transition-colors ${className}`}
      onClick={() => handleSort(sortKey)}
    >
      <div className="flex items-center space-x-1">
        <span>{children}</span>
        {sortable && (
          <ArrowUpDown className={`h-3 w-3 ${
            sortConfig.key === sortKey ? 'text-primary-500' : 'text-secondary-400'
          }`} />
        )}
      </div>
    </th>
  )

  if (users.length === 0) {
    return (
      <div className="card">
        <div className="card-body text-center py-12">
          <User className="mx-auto h-12 w-12 text-secondary-400" />
          <h3 className="mt-4 text-lg font-medium text-secondary-900">
            Nenhum usuário encontrado
          </h3>
          <p className="mt-2 text-sm text-secondary-500">
            Não há usuários para exibir nesta tabela.
          </p>
        </div>
      </div>
    )
  }

  return (
    <div className="card">
      <div className="overflow-x-auto">
        <table className="data-table">
          <thead>
            <tr>
              <SortableHeader sortKey="name">Usuário</SortableHeader>
              <SortableHeader sortKey="cluster">Cluster</SortableHeader>
              <SortableHeader sortKey="totalValue">Valor Total</SortableHeader>
              <SortableHeader sortKey="campaigns">Campanhas</SortableHeader>
              <SortableHeader sortKey="conversion">Conversão</SortableHeader>
              <SortableHeader sortKey="lastActivity">Última Atividade</SortableHeader>
              <SortableHeader sortKey="status">Status</SortableHeader>
              <th>Ações</th>
            </tr>
          </thead>
          <tbody>
            {paginatedUsers.map((user, index) => (
              <motion.tr
                key={user.id}
                initial={{ opacity: 0, x: -20 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ delay: index * 0.05 }}
                className="hover:bg-secondary-50"
              >
                <td>
                  <div className="flex items-center space-x-3">
                    <div className="h-8 w-8 bg-secondary-100 rounded-full flex items-center justify-center flex-shrink-0">
                      <User className="h-4 w-4 text-secondary-600" />
                    </div>
                    <div className="min-w-0 flex-1">
                      <div className="font-medium text-secondary-900 truncate">
                        {user.name}
                      </div>
                      <div className="text-sm text-secondary-500 flex items-center space-x-2">
                        <Mail className="h-3 w-3 flex-shrink-0" />
                        <span className="truncate">{user.email}</span>
                      </div>
                      <div className="text-sm text-secondary-500 flex items-center space-x-2">
                        <Phone className="h-3 w-3 flex-shrink-0" />
                        <span className="truncate">{user.phone}</span>
                      </div>
                    </div>
                  </div>
                </td>
                <td>
                  <span className={`badge ${getClusterBadgeClass(user.clusterColor)}`}>
                    {user.cluster}
                  </span>
                </td>
                <td>
                  <div className="font-semibold text-secondary-900">
                    {user.totalValue}
                  </div>
                </td>
                <td>
                  <div className="text-center">
                    <span className="inline-flex items-center justify-center h-6 w-6 rounded-full bg-primary-100 text-xs font-medium text-primary-800">
                      {user.campaigns}
                    </span>
                  </div>
                </td>
                <td>
                  <div className={`font-medium ${
                    parseFloat(user.conversion) > 10 ? 'text-success-600' :
                    parseFloat(user.conversion) > 5 ? 'text-warning-600' : 'text-danger-600'
                  }`}>
                    {user.conversion}
                  </div>
                </td>
                <td>
                  <div className="text-sm text-secondary-600">
                    {user.lastActivity}
                  </div>
                </td>
                <td>
                  <span className={`badge ${user.status === 'active' ? 'badge-success' : 'badge-secondary'}`}>
                    {user.status === 'active' ? 'Ativo' : 'Inativo'}
                  </span>
                </td>
                <td>
                  <div className="flex items-center space-x-2">
                    <button 
                      onClick={() => onView?.(user)}
                      className="p-1 hover:bg-secondary-100 rounded text-secondary-400 hover:text-secondary-600 transition-colors"
                      title="Visualizar usuário"
                    >
                      <Eye className="h-4 w-4" />
                    </button>
                    <button 
                      onClick={() => onEdit?.(user)}
                      className="p-1 hover:bg-secondary-100 rounded text-secondary-400 hover:text-secondary-600 transition-colors"
                      title="Editar usuário"
                    >
                      <Edit className="h-4 w-4" />
                    </button>
                    <button 
                      onClick={() => onDelete?.(user)}
                      className="p-1 hover:bg-danger-50 rounded text-secondary-400 hover:text-danger-600 transition-colors"
                      title="Remover usuário"
                    >
                      <Trash2 className="h-4 w-4" />
                    </button>
                    <button className="p-1 hover:bg-secondary-100 rounded text-secondary-400 hover:text-secondary-600 transition-colors">
                      <MoreVertical className="h-4 w-4" />
                    </button>
                  </div>
                </td>
              </motion.tr>
            ))}
          </tbody>
        </table>
      </div>

      {/* Pagination */}
      {showPagination && totalPages > 1 && (
        <div className="card-footer">
          <div className="flex items-center justify-between">
            <div className="text-sm text-secondary-700">
              Mostrando {startIndex + 1} a {Math.min(startIndex + itemsPerPage, sortedUsers.length)} de {sortedUsers.length} usuários
            </div>
            <div className="flex items-center space-x-2">
              <button
                onClick={() => setCurrentPage(prev => Math.max(prev - 1, 1))}
                disabled={currentPage === 1}
                className="btn-ghost btn-sm disabled:opacity-50 disabled:cursor-not-allowed"
              >
                <ChevronLeft className="h-4 w-4" />
                Anterior
              </button>
              <div className="flex items-center space-x-1">
                {Array.from({ length: Math.min(totalPages, 5) }, (_, i) => {
                  let page
                  if (totalPages <= 5) {
                    page = i + 1
                  } else if (currentPage <= 3) {
                    page = i + 1
                  } else if (currentPage >= totalPages - 2) {
                    page = totalPages - 4 + i
                  } else {
                    page = currentPage - 2 + i
                  }
                  
                  return (
                    <button
                      key={page}
                      onClick={() => setCurrentPage(page)}
                      className={`w-8 h-8 text-sm font-medium rounded-lg transition-colors ${
                        currentPage === page
                          ? 'bg-primary-600 text-white'
                          : 'text-secondary-600 hover:bg-secondary-100'
                      }`}
                    >
                      {page}
                    </button>
                  )
                })}
              </div>
              <button
                onClick={() => setCurrentPage(prev => Math.min(prev + 1, totalPages))}
                disabled={currentPage === totalPages}
                className="btn-ghost btn-sm disabled:opacity-50 disabled:cursor-not-allowed"
              >
                Próximo
                <ChevronRight className="h-4 w-4" />
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}

export default UserTable