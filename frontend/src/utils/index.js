import { format, parseISO, isValid, startOfDay, endOfDay, subDays, subMonths } from 'date-fns'
import { ptBR } from 'date-fns/locale'

// Date utilities
export const dateUtils = {
  format: (date, pattern = 'dd/MM/yyyy') => {
    if (!date) return ''
    const dateObj = typeof date === 'string' ? parseISO(date) : date
    if (!isValid(dateObj)) return ''
    return format(dateObj, pattern, { locale: ptBR })
  },

  formatDateTime: (date) => {
    return dateUtils.format(date, 'dd/MM/yyyy HH:mm')
  },

  formatRelative: (date) => {
    if (!date) return ''
    const dateObj = typeof date === 'string' ? parseISO(date) : date
    if (!isValid(dateObj)) return ''
    
    const now = new Date()
    const diffInMinutes = Math.floor((now - dateObj) / (1000 * 60))
    
    if (diffInMinutes < 1) return 'agora'
    if (diffInMinutes < 60) return `${diffInMinutes} min atrás`
    
    const diffInHours = Math.floor(diffInMinutes / 60)
    if (diffInHours < 24) return `${diffInHours}h atrás`
    
    const diffInDays = Math.floor(diffInHours / 24)
    if (diffInDays < 7) return `${diffInDays} dia${diffInDays > 1 ? 's' : ''} atrás`
    
    return dateUtils.format(dateObj)
  },

  getDateRange: (range) => {
    const end = endOfDay(new Date())
    let start
    
    switch (range) {
      case '7d':
        start = startOfDay(subDays(new Date(), 7))
        break
      case '30d':
        start = startOfDay(subDays(new Date(), 30))
        break
      case '90d':
        start = startOfDay(subDays(new Date(), 90))
        break
      case '12m':
        start = startOfDay(subMonths(new Date(), 12))
        break
      default:
        start = startOfDay(subDays(new Date(), 30))
    }
    
    return { start, end }
  },
}

// Number utilities
export const numberUtils = {
  format: (number, options = {}) => {
    if (typeof number !== 'number') return '0'
    
    return new Intl.NumberFormat('pt-BR', {
      minimumFractionDigits: 0,
      maximumFractionDigits: 2,
      ...options,
    }).format(number)
  },

  formatCurrency: (value, currency = 'BRL') => {
    if (typeof value !== 'number') return 'R$ 0,00'
    
    return new Intl.NumberFormat('pt-BR', {
      style: 'currency',
      currency,
    }).format(value)
  },

  formatPercentage: (value, decimals = 1) => {
    if (typeof value !== 'number') return '0%'
    return `${(value * 100).toFixed(decimals)}%`
  },

  formatCompact: (number) => {
    if (typeof number !== 'number') return '0'
    
    const abbrev = ['', 'K', 'M', 'B', 'T']
    const tier = Math.log10(Math.abs(number)) / 3 | 0
    
    if (tier === 0) return number.toString()
    
    const suffix = abbrev[tier]
    const scale = Math.pow(10, tier * 3)
    const scaled = number / scale
    
    return scaled.toFixed(1) + suffix
  },
}

// String utilities
export const stringUtils = {
  capitalize: (str) => {
    if (typeof str !== 'string') return ''
    return str.charAt(0).toUpperCase() + str.slice(1).toLowerCase()
  },

  truncate: (str, length = 50, suffix = '...') => {
    if (typeof str !== 'string') return ''
    if (str.length <= length) return str
    return str.substring(0, length) + suffix
  },

  slug: (str) => {
    if (typeof str !== 'string') return ''
    return str
      .toLowerCase()
      .normalize('NFD')
      .replace(/[\u0300-\u036f]/g, '')
      .replace(/[^\w\s-]/g, '')
      .replace(/[\s_-]+/g, '-')
      .replace(/^-+|-+$/g, '')
  },

  initials: (name) => {
    if (typeof name !== 'string') return ''
    return name
      .split(' ')
      .map(word => word.charAt(0))
      .join('')
      .toUpperCase()
      .substring(0, 2)
  },

  mask: {
    phone: (value) => {
      if (!value) return ''
      const cleaned = value.replace(/\D/g, '')
      const match = cleaned.match(/^(\d{2})(\d{5})(\d{4})$/)
      if (match) {
        return `(${match[1]}) ${match[2]}-${match[3]}`
      }
      return value
    },

    cpf: (value) => {
      if (!value) return ''
      const cleaned = value.replace(/\D/g, '')
      const match = cleaned.match(/^(\d{3})(\d{3})(\d{3})(\d{2})$/)
      if (match) {
        return `${match[1]}.${match[2]}.${match[3]}-${match[4]}`
      }
      return value
    },

    cnpj: (value) => {
      if (!value) return ''
      const cleaned = value.replace(/\D/g, '')
      const match = cleaned.match(/^(\d{2})(\d{3})(\d{3})(\d{4})(\d{2})$/)
      if (match) {
        return `${match[1]}.${match[2]}.${match[3]}/${match[4]}-${match[5]}`
      }
      return value
    },
  },
}

// Array utilities
export const arrayUtils = {
  groupBy: (array, key) => {
    return array.reduce((groups, item) => {
      const group = (groups[item[key]] = groups[item[key]] || [])
      group.push(item)
      return groups
    }, {})
  },

  sortBy: (array, key, direction = 'asc') => {
    return [...array].sort((a, b) => {
      const aVal = a[key]
      const bVal = b[key]
      
      if (aVal < bVal) return direction === 'asc' ? -1 : 1
      if (aVal > bVal) return direction === 'asc' ? 1 : -1
      return 0
    })
  },

  unique: (array, key = null) => {
    if (key) {
      const seen = new Set()
      return array.filter(item => {
        const val = item[key]
        if (seen.has(val)) return false
        seen.add(val)
        return true
      })
    }
    return [...new Set(array)]
  },

  chunk: (array, size) => {
    const chunks = []
    for (let i = 0; i < array.length; i += size) {
      chunks.push(array.slice(i, i + size))
    }
    return chunks
  },
}

// Validation utilities
export const validationUtils = {
  email: (email) => {
    const pattern = /^[^\s@]+@[^\s@]+\.[^\s@]+$/
    return pattern.test(email)
  },

  phone: (phone) => {
    const cleaned = phone.replace(/\D/g, '')
    return cleaned.length === 11
  },

  cpf: (cpf) => {
    const cleaned = cpf.replace(/\D/g, '')
    if (cleaned.length !== 11) return false
    
    // Check for known invalid patterns
    if (/^(\d)\1{10}$/.test(cleaned)) return false
    
    // Validate check digits
    let sum = 0
    for (let i = 0; i < 9; i++) {
      sum += parseInt(cleaned.charAt(i)) * (10 - i)
    }
    let digit = 11 - (sum % 11)
    if (digit === 10 || digit === 11) digit = 0
    if (digit !== parseInt(cleaned.charAt(9))) return false
    
    sum = 0
    for (let i = 0; i < 10; i++) {
      sum += parseInt(cleaned.charAt(i)) * (11 - i)
    }
    digit = 11 - (sum % 11)
    if (digit === 10 || digit === 11) digit = 0
    if (digit !== parseInt(cleaned.charAt(10))) return false
    
    return true
  },

  cnpj: (cnpj) => {
    const cleaned = cnpj.replace(/\D/g, '')
    if (cleaned.length !== 14) return false
    
    // Check for known invalid patterns
    if (/^(\d)\1{13}$/.test(cleaned)) return false
    
    // Validate check digits
    const weights1 = [5, 4, 3, 2, 9, 8, 7, 6, 5, 4, 3, 2]
    const weights2 = [6, 5, 4, 3, 2, 9, 8, 7, 6, 5, 4, 3, 2]
    
    let sum = 0
    for (let i = 0; i < 12; i++) {
      sum += parseInt(cleaned.charAt(i)) * weights1[i]
    }
    let digit = 11 - (sum % 11)
    if (digit === 10 || digit === 11) digit = 0
    if (digit !== parseInt(cleaned.charAt(12))) return false
    
    sum = 0
    for (let i = 0; i < 13; i++) {
      sum += parseInt(cleaned.charAt(i)) * weights2[i]
    }
    digit = 11 - (sum % 11)
    if (digit === 10 || digit === 11) digit = 0
    if (digit !== parseInt(cleaned.charAt(13))) return false
    
    return true
  },
}

// Color utilities
export const colorUtils = {
  getStatusColor: (status) => {
    const colors = {
      active: 'success',
      inactive: 'secondary',
      pending: 'warning',
      completed: 'success',
      cancelled: 'danger',
      draft: 'secondary',
      scheduled: 'info',
      paused: 'warning',
    }
    return colors[status] || 'secondary'
  },

  getClusterColor: (index) => {
    const colors = ['primary', 'success', 'warning', 'danger', 'info', 'purple']
    return colors[index % colors.length]
  },

  hexToRgb: (hex) => {
    const result = /^#?([a-f\d]{2})([a-f\d]{2})([a-f\d]{2})$/i.exec(hex)
    return result ? {
      r: parseInt(result[1], 16),
      g: parseInt(result[2], 16),
      b: parseInt(result[3], 16)
    } : null
  },

  rgbToHex: (r, g, b) => {
    return "#" + ((1 << 24) + (r << 16) + (g << 8) + b).toString(16).slice(1)
  },
}

// Local storage utilities
export const storageUtils = {
  get: (key, defaultValue = null) => {
    try {
      const item = localStorage.getItem(key)
      return item ? JSON.parse(item) : defaultValue
    } catch (error) {
      console.error('Error reading from localStorage:', error)
      return defaultValue
    }
  },

  set: (key, value) => {
    try {
      localStorage.setItem(key, JSON.stringify(value))
      return true
    } catch (error) {
      console.error('Error writing to localStorage:', error)
      return false
    }
  },

  remove: (key) => {
    try {
      localStorage.removeItem(key)
      return true
    } catch (error) {
      console.error('Error removing from localStorage:', error)
      return false
    }
  },

  clear: () => {
    try {
      localStorage.clear()
      return true
    } catch (error) {
      console.error('Error clearing localStorage:', error)
      return false
    }
  },
}

// URL utilities
export const urlUtils = {
  buildQuery: (params) => {
    const query = new URLSearchParams()
    Object.entries(params).forEach(([key, value]) => {
      if (value !== null && value !== undefined && value !== '') {
        query.append(key, value.toString())
      }
    })
    return query.toString()
  },

  parseQuery: (queryString = window.location.search) => {
    const params = new URLSearchParams(queryString)
    const result = {}
    for (const [key, value] of params) {
      result[key] = value
    }
    return result
  },
}

// Download utilities
export const downloadUtils = {
  json: (data, filename = 'data.json') => {
    const blob = new Blob([JSON.stringify(data, null, 2)], {
      type: 'application/json'
    })
    downloadUtils.blob(blob, filename)
  },

  csv: (data, filename = 'data.csv') => {
    if (!Array.isArray(data) || data.length === 0) return
    
    const headers = Object.keys(data[0])
    const csvContent = [
      headers.join(','),
      ...data.map(row => headers.map(header => 
        JSON.stringify(row[header] || '')
      ).join(','))
    ].join('\n')
    
    const blob = new Blob([csvContent], { type: 'text/csv' })
    downloadUtils.blob(blob, filename)
  },

  blob: (blob, filename) => {
    const url = window.URL.createObjectURL(blob)
    const link = document.createElement('a')
    link.href = url
    link.download = filename
    document.body.appendChild(link)
    link.click()
    document.body.removeChild(link)
    window.URL.revokeObjectURL(url)
  },
}

// Debounce utility
export const debounce = (func, wait, immediate = false) => {
  let timeout
  return function executedFunction(...args) {
    const later = () => {
      timeout = null
      if (!immediate) func(...args)
    }
    const callNow = immediate && !timeout
    clearTimeout(timeout)
    timeout = setTimeout(later, wait)
    if (callNow) func(...args)
  }
}

// Throttle utility
export const throttle = (func, limit) => {
  let inThrottle
  return function(...args) {
    if (!inThrottle) {
      func.apply(this, args)
      inThrottle = true
      setTimeout(() => inThrottle = false, limit)
    }
  }
}

// Error handling utilities
export const errorUtils = {
  getErrorMessage: (error) => {
    if (typeof error === 'string') return error
    if (error?.response?.data?.message) return error.response.data.message
    if (error?.message) return error.message
    return 'Erro desconhecido'
  },

  isNetworkError: (error) => {
    return !error?.response && error?.request
  },

  isValidationError: (error) => {
    return error?.response?.status === 422
  },

  isAuthError: (error) => {
    return error?.response?.status === 401
  },
}

export default {
  dateUtils,
  numberUtils,
  stringUtils,
  arrayUtils,
  validationUtils,
  colorUtils,
  storageUtils,
  urlUtils,
  downloadUtils,
  debounce,
  throttle,
  errorUtils,
}