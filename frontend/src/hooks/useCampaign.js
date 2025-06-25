import { useState, useEffect, useCallback } from 'react'
import { campaignsApi } from '../services/api'
import toast from 'react-hot-toast'

export const useCampaign = (campaignId = null) => {
  const [campaign, setCampaign] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)

  const fetchCampaign = useCallback(async () => {
    if (!campaignId) return

    setLoading(true)
    setError(null)

    try {
      const data = await campaignsApi.getById(campaignId)
      setCampaign(data)
    } catch (err) {
      setError(err)
      console.error('Error fetching campaign:', err)
    } finally {
      setLoading(false)
    }
  }, [campaignId])

  useEffect(() => {
    fetchCampaign()
  }, [fetchCampaign])

  const updateCampaign = useCallback(async (updateData) => {
    if (!campaignId) return

    setLoading(true)
    try {
      const updatedCampaign = await campaignsApi.update(campaignId, updateData)
      setCampaign(updatedCampaign)
      toast.success('Campanha atualizada com sucesso')
      return updatedCampaign
    } catch (err) {
      setError(err)
      toast.error('Erro ao atualizar campanha')
      throw err
    } finally {
      setLoading(false)
    }
  }, [campaignId])

  const deleteCampaign = useCallback(async () => {
    if (!campaignId) return

    setLoading(true)
    try {
      await campaignsApi.delete(campaignId)
      toast.success('Campanha removida com sucesso')
    } catch (err) {
      setError(err)
      toast.error('Erro ao remover campanha')
      throw err
    } finally {
      setLoading(false)
    }
  }, [campaignId])

  const startCampaign = useCallback(async () => {
    if (!campaignId) return

    setLoading(true)
    try {
      const result = await campaignsApi.start(campaignId)
      await fetchCampaign() // Refresh campaign data
      toast.success('Campanha iniciada com sucesso')
      return result
    } catch (err) {
      setError(err)
      toast.error('Erro ao iniciar campanha')
      throw err
    } finally {
      setLoading(false)
    }
  }, [campaignId, fetchCampaign])

  const pauseCampaign = useCallback(async () => {
    if (!campaignId) return

    setLoading(true)
    try {
      const result = await campaignsApi.pause(campaignId)
      await fetchCampaign() // Refresh campaign data
      toast.success('Campanha pausada com sucesso')
      return result
    } catch (err) {
      setError(err)
      toast.error('Erro ao pausar campanha')
      throw err
    } finally {
      setLoading(false)
    }
  }, [campaignId, fetchCampaign])

  const stopCampaign = useCallback(async () => {
    if (!campaignId) return

    setLoading(true)
    try {
      const result = await campaignsApi.stop(campaignId)
      await fetchCampaign() // Refresh campaign data
      toast.success('Campanha finalizada com sucesso')
      return result
    } catch (err) {
      setError(err)
      toast.error('Erro ao finalizar campanha')
      throw err
    } finally {
      setLoading(false)
    }
  }, [campaignId, fetchCampaign])

  const cloneCampaign = useCallback(async () => {
    if (!campaignId) return

    setLoading(true)
    try {
      const clonedCampaign = await campaignsApi.clone(campaignId)
      toast.success('Campanha duplicada com sucesso')
      return clonedCampaign
    } catch (err) {
      setError(err)
      toast.error('Erro ao duplicar campanha')
      throw err
    } finally {
      setLoading(false)
    }
  }, [campaignId])

  return {
    campaign,
    loading,
    error,
    fetchCampaign,
    updateCampaign,
    deleteCampaign,
    startCampaign,
    pauseCampaign,
    stopCampaign,
    cloneCampaign,
  }
}

export const useCampaigns = (filters = {}) => {
  const [campaigns, setCampaigns] = useState([])
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)
  const [pagination, setPagination] = useState({
    page: 1,
    limit: 10,
    total: 0,
    totalPages: 0,
  })

  const fetchCampaigns = useCallback(async (params = {}) => {
    setLoading(true)
    setError(null)

    try {
      const data = await campaignsApi.getAll({ ...filters, ...params })
      setCampaigns(data.campaigns || data)
      
      if (data.pagination) {
        setPagination(data.pagination)
      }
    } catch (err) {
      setError(err)
      console.error('Error fetching campaigns:', err)
    } finally {
      setLoading(false)
    }
  }, [filters])

  useEffect(() => {
    fetchCampaigns()
  }, [fetchCampaigns])

  const createCampaign = useCallback(async (campaignData) => {
    setLoading(true)
    try {
      const newCampaign = await campaignsApi.create(campaignData)
      setCampaigns(prev => [newCampaign, ...prev])
      toast.success('Campanha criada com sucesso')
      return newCampaign
    } catch (err) {
      setError(err)
      toast.error('Erro ao criar campanha')
      throw err
    } finally {
      setLoading(false)
    }
  }, [])

  const updateCampaign = useCallback(async (campaignId, updateData) => {
    setLoading(true)
    try {
      const updatedCampaign = await campaignsApi.update(campaignId, updateData)
      setCampaigns(prev => 
        prev.map(campaign => 
          campaign.id === campaignId ? updatedCampaign : campaign
        )
      )
      toast.success('Campanha atualizada com sucesso')
      return updatedCampaign
    } catch (err) {
      setError(err)
      toast.error('Erro ao atualizar campanha')
      throw err
    } finally {
      setLoading(false)
    }
  }, [])

  const deleteCampaign = useCallback(async (campaignId) => {
    setLoading(true)
    try {
      await campaignsApi.delete(campaignId)
      setCampaigns(prev => prev.filter(campaign => campaign.id !== campaignId))
      toast.success('Campanha removida com sucesso')
    } catch (err) {
      setError(err)
      toast.error('Erro ao remover campanha')
      throw err
    } finally {
      setLoading(false)
    }
  }, [])

  const bulkAction = useCallback(async (action, campaignIds) => {
    setLoading(true)
    try {
      const promises = campaignIds.map(id => {
        switch (action) {
          case 'start':
            return campaignsApi.start(id)
          case 'pause':
            return campaignsApi.pause(id)
          case 'stop':
            return campaignsApi.stop(id)
          case 'delete':
            return campaignsApi.delete(id)
          default:
            throw new Error(`Ação desconhecida: ${action}`)
        }
      })

      await Promise.all(promises)
      
      if (action === 'delete') {
        setCampaigns(prev => prev.filter(campaign => !campaignIds.includes(campaign.id)))
      } else {
        await fetchCampaigns() // Refresh all campaigns
      }

      toast.success(`${campaignIds.length} campanha(s) ${action === 'start' ? 'iniciada(s)' : 
                     action === 'pause' ? 'pausada(s)' : 
                     action === 'stop' ? 'finalizada(s)' : 'removida(s)'} com sucesso`)
    } catch (err) {
      setError(err)
      toast.error(`Erro ao ${action === 'start' ? 'iniciar' : 
                   action === 'pause' ? 'pausar' : 
                   action === 'stop' ? 'finalizar' : 'remover'} campanhas`)
      throw err
    } finally {
      setLoading(false)
    }
  }, [fetchCampaigns])

  const refreshCampaigns = useCallback(() => {
    fetchCampaigns()
  }, [fetchCampaigns])

  return {
    campaigns,
    loading,
    error,
    pagination,
    fetchCampaigns,
    createCampaign,
    updateCampaign,
    deleteCampaign,
    bulkAction,
    refreshCampaigns,
  }
}

export const useCampaignMetrics = (campaignId) => {
  const [metrics, setMetrics] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)

  const fetchMetrics = useCallback(async () => {
    if (!campaignId) return

    setLoading(true)
    setError(null)

    try {
      const data = await campaignsApi.getMetrics(campaignId)
      setMetrics(data)
    } catch (err) {
      setError(err)
      console.error('Error fetching campaign metrics:', err)
    } finally {
      setLoading(false)
    }
  }, [campaignId])

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

export const useCampaignPreview = () => {
  const [preview, setPreview] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)

  const generatePreview = useCallback(async (campaignData) => {
    setLoading(true)
    setError(null)

    try {
      const data = await campaignsApi.preview(campaignData)
      setPreview(data)
      return data
    } catch (err) {
      setError(err)
      toast.error('Erro ao gerar preview da campanha')
      throw err
    } finally {
      setLoading(false)
    }
  }, [])

  return {
    preview,
    loading,
    error,
    generatePreview,
  }
}

export const useCampaignReport = (campaignId) => {
  const [report, setReport] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)

  const fetchReport = useCallback(async () => {
    if (!campaignId) return

    setLoading(true)
    setError(null)

    try {
      const data = await campaignsApi.getReport(campaignId)
      setReport(data)
    } catch (err) {
      setError(err)
      console.error('Error fetching campaign report:', err)
    } finally {
      setLoading(false)
    }
  }, [campaignId])

  useEffect(() => {
    fetchReport()
  }, [fetchReport])

  return {
    report,
    loading,
    error,
    refreshReport: fetchReport,
  }
}