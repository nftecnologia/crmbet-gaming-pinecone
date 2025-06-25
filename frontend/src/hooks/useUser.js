import { useState, useEffect, useCallback } from 'react'
import { usersApi } from '../services/api'
import toast from 'react-hot-toast'

export const useUser = (userId = null) => {
  const [user, setUser] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)

  const fetchUser = useCallback(async () => {
    if (!userId) return

    setLoading(true)
    setError(null)

    try {
      const data = await usersApi.getById(userId)
      setUser(data)
    } catch (err) {
      setError(err)
      console.error('Error fetching user:', err)
    } finally {
      setLoading(false)
    }
  }, [userId])

  useEffect(() => {
    fetchUser()
  }, [fetchUser])

  const updateUser = useCallback(async (updateData) => {
    if (!userId) return

    setLoading(true)
    try {
      const updatedUser = await usersApi.update(userId, updateData)
      setUser(updatedUser)
      toast.success('Usuário atualizado com sucesso')
      return updatedUser
    } catch (err) {
      setError(err)
      toast.error('Erro ao atualizar usuário')
      throw err
    } finally {
      setLoading(false)
    }
  }, [userId])

  const deleteUser = useCallback(async () => {
    if (!userId) return

    setLoading(true)
    try {
      await usersApi.delete(userId)
      toast.success('Usuário removido com sucesso')
    } catch (err) {
      setError(err)
      toast.error('Erro ao remover usuário')
      throw err
    } finally {
      setLoading(false)
    }
  }, [userId])

  return {
    user,
    loading,
    error,
    fetchUser,
    updateUser,
    deleteUser,
  }
}

export const useUsers = (filters = {}) => {
  const [users, setUsers] = useState([])
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)
  const [pagination, setPagination] = useState({
    page: 1,
    limit: 10,
    total: 0,
    totalPages: 0,
  })

  const fetchUsers = useCallback(async (params = {}) => {
    setLoading(true)
    setError(null)

    try {
      const data = await usersApi.getAll({ ...filters, ...params })
      setUsers(data.users || data)
      
      if (data.pagination) {
        setPagination(data.pagination)
      }
    } catch (err) {
      setError(err)
      console.error('Error fetching users:', err)
    } finally {
      setLoading(false)
    }
  }, [filters])

  useEffect(() => {
    fetchUsers()
  }, [fetchUsers])

  const createUser = useCallback(async (userData) => {
    setLoading(true)
    try {
      const newUser = await usersApi.create(userData)
      setUsers(prev => [newUser, ...prev])
      toast.success('Usuário criado com sucesso')
      return newUser
    } catch (err) {
      setError(err)
      toast.error('Erro ao criar usuário')
      throw err
    } finally {
      setLoading(false)
    }
  }, [])

  const updateUser = useCallback(async (userId, updateData) => {
    setLoading(true)
    try {
      const updatedUser = await usersApi.update(userId, updateData)
      setUsers(prev => 
        prev.map(user => 
          user.id === userId ? updatedUser : user
        )
      )
      toast.success('Usuário atualizado com sucesso')
      return updatedUser
    } catch (err) {
      setError(err)
      toast.error('Erro ao atualizar usuário')
      throw err
    } finally {
      setLoading(false)
    }
  }, [])

  const deleteUser = useCallback(async (userId) => {
    setLoading(true)
    try {
      await usersApi.delete(userId)
      setUsers(prev => prev.filter(user => user.id !== userId))
      toast.success('Usuário removido com sucesso')
    } catch (err) {
      setError(err)
      toast.error('Erro ao remover usuário')
      throw err
    } finally {
      setLoading(false)
    }
  }, [])

  const bulkDelete = useCallback(async (userIds) => {
    setLoading(true)
    try {
      const promises = userIds.map(id => usersApi.delete(id))
      await Promise.all(promises)
      
      setUsers(prev => prev.filter(user => !userIds.includes(user.id)))
      toast.success(`${userIds.length} usuário(s) removido(s) com sucesso`)
    } catch (err) {
      setError(err)
      toast.error('Erro ao remover usuários')
      throw err
    } finally {
      setLoading(false)
    }
  }, [])

  const searchUsers = useCallback(async (searchTerm, searchFilters = {}) => {
    const searchParams = {
      search: searchTerm,
      ...searchFilters,
    }
    return fetchUsers(searchParams)
  }, [fetchUsers])

  const refreshUsers = useCallback(() => {
    fetchUsers()
  }, [fetchUsers])

  return {
    users,
    loading,
    error,
    pagination,
    fetchUsers,
    createUser,
    updateUser,
    deleteUser,
    bulkDelete,
    searchUsers,
    refreshUsers,
  }
}

export const useUsersByCluster = (clusterId) => {
  const [users, setUsers] = useState([])
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)

  const fetchUsersByCluster = useCallback(async () => {
    if (!clusterId) return

    setLoading(true)
    setError(null)

    try {
      const data = await usersApi.getByClusters(clusterId)
      setUsers(data)
    } catch (err) {
      setError(err)
      console.error('Error fetching users by cluster:', err)
    } finally {
      setLoading(false)
    }
  }, [clusterId])

  useEffect(() => {
    fetchUsersByCluster()
  }, [fetchUsersByCluster])

  return {
    users,
    loading,
    error,
    refreshUsers: fetchUsersByCluster,
  }
}

export const useUserExport = () => {
  const [exporting, setExporting] = useState(false)
  const [error, setError] = useState(null)

  const exportUsers = useCallback(async (filters = {}, format = 'csv') => {
    setExporting(true)
    setError(null)

    try {
      const params = {
        ...filters,
        format,
      }

      const blob = await usersApi.export(params)
      
      // Create download link
      const url = window.URL.createObjectURL(blob)
      const link = document.createElement('a')
      link.href = url
      link.download = `usuarios_${new Date().toISOString().split('T')[0]}.${format}`
      document.body.appendChild(link)
      link.click()
      link.remove()
      window.URL.revokeObjectURL(url)

      toast.success('Exportação concluída com sucesso')
    } catch (err) {
      setError(err)
      toast.error('Erro ao exportar usuários')
      throw err
    } finally {
      setExporting(false)
    }
  }, [])

  return {
    exporting,
    error,
    exportUsers,
  }
}

export const useUserFilters = () => {
  const [filters, setFilters] = useState({
    search: '',
    cluster: '',
    status: '',
    dateRange: null,
    sortBy: 'createdAt',
    sortOrder: 'desc',
  })

  const updateFilter = useCallback((key, value) => {
    setFilters(prev => ({ ...prev, [key]: value }))
  }, [])

  const updateFilters = useCallback((newFilters) => {
    setFilters(prev => ({ ...prev, ...newFilters }))
  }, [])

  const resetFilters = useCallback(() => {
    setFilters({
      search: '',
      cluster: '',
      status: '',
      dateRange: null,
      sortBy: 'createdAt',
      sortOrder: 'desc',
    })
  }, [])

  const hasActiveFilters = useCallback(() => {
    return filters.search || 
           filters.cluster || 
           filters.status || 
           filters.dateRange ||
           filters.sortBy !== 'createdAt' ||
           filters.sortOrder !== 'desc'
  }, [filters])

  return {
    filters,
    updateFilter,
    updateFilters,
    resetFilters,
    hasActiveFilters,
  }
}

export const useUserSelection = (users = []) => {
  const [selectedUsers, setSelectedUsers] = useState(new Set())

  const toggleUser = useCallback((userId) => {
    setSelectedUsers(prev => {
      const newSet = new Set(prev)
      if (newSet.has(userId)) {
        newSet.delete(userId)
      } else {
        newSet.add(userId)
      }
      return newSet
    })
  }, [])

  const selectAll = useCallback(() => {
    setSelectedUsers(new Set(users.map(user => user.id)))
  }, [users])

  const selectNone = useCallback(() => {
    setSelectedUsers(new Set())
  }, [])

  const isSelected = useCallback((userId) => {
    return selectedUsers.has(userId)
  }, [selectedUsers])

  const isAllSelected = useCallback(() => {
    return users.length > 0 && users.every(user => selectedUsers.has(user.id))
  }, [users, selectedUsers])

  const isIndeterminate = useCallback(() => {
    return selectedUsers.size > 0 && !isAllSelected()
  }, [selectedUsers.size, isAllSelected])

  const selectedCount = selectedUsers.size

  const getSelectedUsers = useCallback(() => {
    return users.filter(user => selectedUsers.has(user.id))
  }, [users, selectedUsers])

  return {
    selectedUsers: Array.from(selectedUsers),
    selectedCount,
    toggleUser,
    selectAll,
    selectNone,
    isSelected,
    isAllSelected,
    isIndeterminate,
    getSelectedUsers,
  }
}