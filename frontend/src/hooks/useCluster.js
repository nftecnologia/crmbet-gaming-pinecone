import { useState, useEffect, useCallback } from 'react'
import { clustersApi } from '../services/api'
import toast from 'react-hot-toast'

export const useCluster = (clusterId = null) => {
  const [cluster, setCluster] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)

  const fetchCluster = useCallback(async () => {
    if (!clusterId) return

    setLoading(true)
    setError(null)

    try {
      const data = await clustersApi.getById(clusterId)
      setCluster(data)
    } catch (err) {
      setError(err)
      console.error('Error fetching cluster:', err)
    } finally {
      setLoading(false)
    }
  }, [clusterId])

  useEffect(() => {
    fetchCluster()
  }, [fetchCluster])

  const updateCluster = useCallback(async (updateData) => {
    if (!clusterId) return

    setLoading(true)
    try {
      const updatedCluster = await clustersApi.update(clusterId, updateData)
      setCluster(updatedCluster)
      toast.success('Cluster atualizado com sucesso')
      return updatedCluster
    } catch (err) {
      setError(err)
      toast.error('Erro ao atualizar cluster')
      throw err
    } finally {
      setLoading(false)
    }
  }, [clusterId])

  const deleteCluster = useCallback(async () => {
    if (!clusterId) return

    setLoading(true)
    try {
      await clustersApi.delete(clusterId)
      toast.success('Cluster removido com sucesso')
    } catch (err) {
      setError(err)
      toast.error('Erro ao remover cluster')
      throw err
    } finally {
      setLoading(false)
    }
  }, [clusterId])

  const recalculateCluster = useCallback(async () => {
    if (!clusterId) return

    setLoading(true)
    try {
      const result = await clustersApi.recalculate(clusterId)
      await fetchCluster() // Refresh cluster data
      toast.success('Cluster recalculado com sucesso')
      return result
    } catch (err) {
      setError(err)
      toast.error('Erro ao recalcular cluster')
      throw err
    } finally {
      setLoading(false)
    }
  }, [clusterId, fetchCluster])

  return {
    cluster,
    loading,
    error,
    fetchCluster,
    updateCluster,
    deleteCluster,
    recalculateCluster,
  }
}

export const useClusters = (filters = {}) => {
  const [clusters, setClusters] = useState([])
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)
  const [pagination, setPagination] = useState({
    page: 1,
    limit: 10,
    total: 0,
    totalPages: 0,
  })

  const fetchClusters = useCallback(async (params = {}) => {
    setLoading(true)
    setError(null)

    try {
      const data = await clustersApi.getAll({ ...filters, ...params })
      setClusters(data.clusters || data)
      
      if (data.pagination) {
        setPagination(data.pagination)
      }
    } catch (err) {
      setError(err)
      console.error('Error fetching clusters:', err)
    } finally {
      setLoading(false)
    }
  }, [filters])

  useEffect(() => {
    fetchClusters()
  }, [fetchClusters])

  const createCluster = useCallback(async (clusterData) => {
    setLoading(true)
    try {
      const newCluster = await clustersApi.create(clusterData)
      setClusters(prev => [newCluster, ...prev])
      toast.success('Cluster criado com sucesso')
      return newCluster
    } catch (err) {
      setError(err)
      toast.error('Erro ao criar cluster')
      throw err
    } finally {
      setLoading(false)
    }
  }, [])

  const updateCluster = useCallback(async (clusterId, updateData) => {
    setLoading(true)
    try {
      const updatedCluster = await clustersApi.update(clusterId, updateData)
      setClusters(prev => 
        prev.map(cluster => 
          cluster.id === clusterId ? updatedCluster : cluster
        )
      )
      toast.success('Cluster atualizado com sucesso')
      return updatedCluster
    } catch (err) {
      setError(err)
      toast.error('Erro ao atualizar cluster')
      throw err
    } finally {
      setLoading(false)
    }
  }, [])

  const deleteCluster = useCallback(async (clusterId) => {
    setLoading(true)
    try {
      await clustersApi.delete(clusterId)
      setClusters(prev => prev.filter(cluster => cluster.id !== clusterId))
      toast.success('Cluster removido com sucesso')
    } catch (err) {
      setError(err)
      toast.error('Erro ao remover cluster')
      throw err
    } finally {
      setLoading(false)
    }
  }, [])

  const refreshClusters = useCallback(() => {
    fetchClusters()
  }, [fetchClusters])

  return {
    clusters,
    loading,
    error,
    pagination,
    fetchClusters,
    createCluster,
    updateCluster,
    deleteCluster,
    refreshClusters,
  }
}

export const useClusterDistribution = () => {
  const [distribution, setDistribution] = useState([])
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)

  const fetchDistribution = useCallback(async () => {
    setLoading(true)
    setError(null)

    try {
      const data = await clustersApi.getDistribution()
      setDistribution(data)
    } catch (err) {
      setError(err)
      console.error('Error fetching cluster distribution:', err)
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => {
    fetchDistribution()
  }, [fetchDistribution])

  return {
    distribution,
    loading,
    error,
    refreshDistribution: fetchDistribution,
  }
}

export const useClusterMetrics = (clusterId) => {
  const [metrics, setMetrics] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)

  const fetchMetrics = useCallback(async () => {
    if (!clusterId) return

    setLoading(true)
    setError(null)

    try {
      const data = await clustersApi.getMetrics(clusterId)
      setMetrics(data)
    } catch (err) {
      setError(err)
      console.error('Error fetching cluster metrics:', err)
    } finally {
      setLoading(false)
    }
  }, [clusterId])

  useEffect(() => {
    fetchMetrics()
  }, [fetchMetrics])

  return {
    metrics,
    loading,
    error,
    refreshMetrics: fetchMetrics,
  }
}