import axios from 'axios'
import toast from 'react-hot-toast'

// Base API configuration
const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000/api'

// Create axios instance
const api = axios.create({
  baseURL: API_BASE_URL,
  timeout: 30000,
  headers: {
    'Content-Type': 'application/json',
  },
})

// Request interceptor
api.interceptors.request.use(
  (config) => {
    // Add auth token if available
    const token = localStorage.getItem('authToken')
    if (token) {
      config.headers.Authorization = `Bearer ${token}`
    }
    
    // Add request timestamp for analytics
    config.metadata = { startTime: new Date() }
    
    return config
  },
  (error) => {
    return Promise.reject(error)
  }
)

// Response interceptor
api.interceptors.response.use(
  (response) => {
    // Calculate request duration
    if (response.config.metadata) {
      const duration = new Date() - response.config.metadata.startTime
      console.log(`API Request: ${response.config.method?.toUpperCase()} ${response.config.url} - ${duration}ms`)
    }
    
    return response
  },
  (error) => {
    // Handle common errors
    if (error.response) {
      const { status, data } = error.response
      
      switch (status) {
        case 401:
          localStorage.removeItem('authToken')
          toast.error('Sessão expirada. Faça login novamente.')
          // Redirect to login if needed
          break
        case 403:
          toast.error('Acesso negado')
          break
        case 404:
          toast.error('Recurso não encontrado')
          break
        case 422:
          // Validation errors
          if (data.errors) {
            Object.values(data.errors).flat().forEach(msg => {
              toast.error(msg)
            })
          } else {
            toast.error(data.message || 'Erro de validação')
          }
          break
        case 500:
          toast.error('Erro interno do servidor')
          break
        default:
          toast.error(data.message || 'Erro na requisição')
      }
    } else if (error.request) {
      toast.error('Erro de conexão com o servidor')
    } else {
      toast.error('Erro inesperado')
    }
    
    return Promise.reject(error)
  }
)

// Helper function to handle API responses
const handleResponse = (response) => response.data

// Helper function to handle API errors
const handleError = (error) => {
  throw error
}

// Auth API
export const authApi = {
  login: (credentials) => api.post('/auth/login', credentials).then(handleResponse),
  logout: () => api.post('/auth/logout').then(handleResponse),
  refreshToken: () => api.post('/auth/refresh').then(handleResponse),
  profile: () => api.get('/auth/profile').then(handleResponse),
}

// Users API
export const usersApi = {
  getAll: (params = {}) => api.get('/users', { params }).then(handleResponse),
  getById: (id) => api.get(`/users/${id}`).then(handleResponse),
  create: (data) => api.post('/users', data).then(handleResponse),
  update: (id, data) => api.put(`/users/${id}`, data).then(handleResponse),
  delete: (id) => api.delete(`/users/${id}`).then(handleResponse),
  getByClusters: (clusterId) => api.get(`/users/clusters/${clusterId}`).then(handleResponse),
  export: (params = {}) => api.get('/users/export', { 
    params, 
    responseType: 'blob' 
  }).then(response => response.data),
}

// Clusters API
export const clustersApi = {
  getAll: (params = {}) => api.get('/clusters', { params }).then(handleResponse),
  getById: (id) => api.get(`/clusters/${id}`).then(handleResponse),
  create: (data) => api.post('/clusters', data).then(handleResponse),
  update: (id, data) => api.put(`/clusters/${id}`, data).then(handleResponse),
  delete: (id) => api.delete(`/clusters/${id}`).then(handleResponse),
  getMetrics: (id) => api.get(`/clusters/${id}/metrics`).then(handleResponse),
  recalculate: (id) => api.post(`/clusters/${id}/recalculate`).then(handleResponse),
  getDistribution: () => api.get('/clusters/distribution').then(handleResponse),
}

// Campaigns API
export const campaignsApi = {
  getAll: (params = {}) => api.get('/campaigns', { params }).then(handleResponse),
  getById: (id) => api.get(`/campaigns/${id}`).then(handleResponse),
  create: (data) => api.post('/campaigns', data).then(handleResponse),
  update: (id, data) => api.put(`/campaigns/${id}`, data).then(handleResponse),
  delete: (id) => api.delete(`/campaigns/${id}`).then(handleResponse),
  start: (id) => api.post(`/campaigns/${id}/start`).then(handleResponse),
  pause: (id) => api.post(`/campaigns/${id}/pause`).then(handleResponse),
  stop: (id) => api.post(`/campaigns/${id}/stop`).then(handleResponse),
  clone: (id) => api.post(`/campaigns/${id}/clone`).then(handleResponse),
  getMetrics: (id) => api.get(`/campaigns/${id}/metrics`).then(handleResponse),
  getReport: (id) => api.get(`/campaigns/${id}/report`).then(handleResponse),
  preview: (data) => api.post('/campaigns/preview', data).then(handleResponse),
}

// Analytics API
export const analyticsApi = {
  getDashboard: (params = {}) => api.get('/analytics/dashboard', { params }).then(handleResponse),
  getRevenue: (params = {}) => api.get('/analytics/revenue', { params }).then(handleResponse),
  getConversion: (params = {}) => api.get('/analytics/conversion', { params }).then(handleResponse),
  getUserActivity: (params = {}) => api.get('/analytics/user-activity', { params }).then(handleResponse),
  getCampaignPerformance: (params = {}) => api.get('/analytics/campaign-performance', { params }).then(handleResponse),
  getClusterPerformance: (params = {}) => api.get('/analytics/cluster-performance', { params }).then(handleResponse),
  getFunnel: (params = {}) => api.get('/analytics/funnel', { params }).then(handleResponse),
  getInsights: () => api.get('/analytics/insights').then(handleResponse),
  getRecommendations: () => api.get('/analytics/recommendations').then(handleResponse),
  export: (type, params = {}) => api.get(`/analytics/export/${type}`, { 
    params, 
    responseType: 'blob' 
  }).then(response => response.data),
}

// ML API
export const mlApi = {
  getModels: () => api.get('/ml/models').then(handleResponse),
  getModelStatus: (id) => api.get(`/ml/models/${id}/status`).then(handleResponse),
  trainModel: (data) => api.post('/ml/train', data).then(handleResponse),
  predict: (data) => api.post('/ml/predict', data).then(handleResponse),
  getPredictions: (params = {}) => api.get('/ml/predictions', { params }).then(handleResponse),
  getFeatureImportance: (modelId) => api.get(`/ml/models/${modelId}/features`).then(handleResponse),
}

// Notifications API
export const notificationsApi = {
  getAll: (params = {}) => api.get('/notifications', { params }).then(handleResponse),
  getUnread: () => api.get('/notifications/unread').then(handleResponse),
  markAsRead: (id) => api.patch(`/notifications/${id}/read`).then(handleResponse),
  markAllAsRead: () => api.patch('/notifications/read-all').then(handleResponse),
  delete: (id) => api.delete(`/notifications/${id}`).then(handleResponse),
  getSettings: () => api.get('/notifications/settings').then(handleResponse),
  updateSettings: (data) => api.put('/notifications/settings', data).then(handleResponse),
}

// Settings API
export const settingsApi = {
  get: () => api.get('/settings').then(handleResponse),
  update: (data) => api.put('/settings', data).then(handleResponse),
  getIntegrations: () => api.get('/settings/integrations').then(handleResponse),
  updateIntegration: (id, data) => api.put(`/settings/integrations/${id}`, data).then(handleResponse),
  testIntegration: (id) => api.post(`/settings/integrations/${id}/test`).then(handleResponse),
}

// Health check
export const healthApi = {
  check: () => api.get('/health').then(handleResponse),
  detailed: () => api.get('/health/detailed').then(handleResponse),
}

// Generic API helpers
export const apiHelpers = {
  // Upload file helper
  uploadFile: async (file, endpoint, onProgress) => {
    const formData = new FormData()
    formData.append('file', file)
    
    return api.post(endpoint, formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
      onUploadProgress: (progressEvent) => {
        if (onProgress) {
          const progress = Math.round((progressEvent.loaded * 100) / progressEvent.total)
          onProgress(progress)
        }
      },
    }).then(handleResponse)
  },
  
  // Download file helper
  downloadFile: async (url, filename) => {
    try {
      const response = await api.get(url, { responseType: 'blob' })
      const blob = new Blob([response.data])
      const downloadUrl = window.URL.createObjectURL(blob)
      const link = document.createElement('a')
      link.href = downloadUrl
      link.download = filename
      document.body.appendChild(link)
      link.click()
      link.remove()
      window.URL.revokeObjectURL(downloadUrl)
    } catch (error) {
      toast.error('Erro ao baixar arquivo')
      throw error
    }
  },
  
  // Batch requests helper
  batch: (requests) => Promise.allSettled(requests),
  
  // Cancel token helper
  createCancelToken: () => axios.CancelToken.source(),
}

// Export the main api instance for direct use if needed
export default api