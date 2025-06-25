import React, { Component, Suspense } from 'react'
import { AlertTriangle, RefreshCw, Home, Bug, Wifi, WifiOff } from 'lucide-react'

/**
 * Enterprise-Grade Error Boundary with Auto Recovery
 * Handles network failures, component crashes, and provides graceful degradation
 */
class ErrorBoundary extends Component {
  constructor(props) {
    super(props)
    
    this.state = {
      hasError: false,
      error: null,
      errorInfo: null,
      errorId: null,
      retryCount: 0,
      isRetrying: false,
      networkError: false,
      criticalError: false,
      errorHistory: []
    }
    
    this.maxRetries = props.maxRetries || 3
    this.retryDelay = props.retryDelay || 1000
    this.enableAutoRetry = props.enableAutoRetry !== false
    this.reportErrors = props.reportErrors !== false
    
    // Bind methods
    this.retry = this.retry.bind(this)
    this.handleNetworkChange = this.handleNetworkChange.bind(this)
    this.reportError = this.reportError.bind(this)
  }

  static getDerivedStateFromError(error) {
    // Update state to trigger error UI
    const errorId = `error_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`
    
    return {
      hasError: true,
      error,
      errorId,
      networkError: ErrorBoundary.isNetworkError(error),
      criticalError: ErrorBoundary.isCriticalError(error)
    }
  }

  static isNetworkError(error) {
    return (
      error?.message?.includes('fetch') ||
      error?.message?.includes('Network') ||
      error?.message?.includes('Failed to fetch') ||
      error?.name === 'NetworkError' ||
      !navigator.onLine
    )
  }

  static isCriticalError(error) {
    const criticalPatterns = [
      'ChunkLoadError',
      'Loading chunk',
      'Loading CSS chunk',
      'Script error',
      'SecurityError'
    ]
    
    return criticalPatterns.some(pattern => 
      error?.message?.includes(pattern) ||
      error?.name?.includes(pattern)
    )
  }

  componentDidCatch(error, errorInfo) {
    const errorDetails = {
      error,
      errorInfo,
      errorId: this.state.errorId,
      timestamp: new Date().toISOString(),
      url: window.location.href,
      userAgent: navigator.userAgent,
      online: navigator.onLine,
      component: this.props.fallbackComponent || 'Unknown',
      retryCount: this.state.retryCount
    }

    // Add to error history
    this.setState(prevState => ({
      errorInfo,
      errorHistory: [...prevState.errorHistory, errorDetails].slice(-10) // Keep last 10 errors
    }))

    // Report error to monitoring service
    if (this.reportErrors) {
      this.reportError(errorDetails)
    }

    // Setup network monitoring for network errors
    if (this.state.networkError) {
      window.addEventListener('online', this.handleNetworkChange)
      window.addEventListener('offline', this.handleNetworkChange)
    }

    // Auto-retry for non-critical errors
    if (this.enableAutoRetry && !this.state.criticalError && this.state.retryCount < this.maxRetries) {
      this.scheduleRetry()
    }

    console.error('ErrorBoundary caught an error:', error, errorInfo)
  }

  componentWillUnmount() {
    window.removeEventListener('online', this.handleNetworkChange)
    window.removeEventListener('offline', this.handleNetworkChange)
    
    if (this.retryTimeout) {
      clearTimeout(this.retryTimeout)
    }
  }

  handleNetworkChange() {
    if (navigator.onLine && this.state.networkError) {
      console.log('Network restored, attempting automatic recovery...')
      this.retry()
    }
  }

  scheduleRetry() {
    const delay = this.retryDelay * Math.pow(2, this.state.retryCount) // Exponential backoff
    
    this.retryTimeout = setTimeout(() => {
      this.retry()
    }, delay)
  }

  async retry() {
    if (this.state.isRetrying) return

    this.setState({ isRetrying: true })

    try {
      // For network errors, check connectivity first
      if (this.state.networkError) {
        await this.checkConnectivity()
      }

      // For chunk loading errors, reload the page
      if (this.state.criticalError && this.state.error?.name === 'ChunkLoadError') {
        window.location.reload()
        return
      }

      // Reset error state
      setTimeout(() => {
        this.setState({
          hasError: false,
          error: null,
          errorInfo: null,
          isRetrying: false,
          retryCount: this.state.retryCount + 1
        })
      }, 100)

    } catch (retryError) {
      console.error('Retry failed:', retryError)
      
      this.setState(prevState => ({
        isRetrying: false,
        retryCount: prevState.retryCount + 1
      }))

      // Schedule another retry if under limit
      if (this.state.retryCount < this.maxRetries) {
        this.scheduleRetry()
      }
    }
  }

  async checkConnectivity() {
    try {
      const response = await fetch('/api/health', {
        method: 'HEAD',
        cache: 'no-cache',
        timeout: 5000
      })
      
      if (!response.ok) {
        throw new Error('Server not responding')
      }
    } catch (error) {
      throw new Error('Connectivity check failed')
    }
  }

  reportError(errorDetails) {
    try {
      // Report to error monitoring service (Sentry, LogRocket, etc.)
      if (window.Sentry) {
        window.Sentry.captureException(errorDetails.error, {
          extra: errorDetails,
          tags: {
            component: 'ErrorBoundary',
            errorId: errorDetails.errorId
          }
        })
      }

      // Report to custom analytics
      if (window.analytics) {
        window.analytics.track('Error Boundary Triggered', {
          errorId: errorDetails.errorId,
          errorMessage: errorDetails.error.message,
          component: errorDetails.component,
          retryCount: errorDetails.retryCount
        })
      }

      // Report to backend API
      fetch('/api/errors', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(errorDetails)
      }).catch(err => {
        console.warn('Failed to report error to backend:', err)
      })

    } catch (reportingError) {
      console.error('Error reporting failed:', reportingError)
    }
  }

  renderNetworkError() {
    const { isRetrying, retryCount } = this.state
    const isOnline = navigator.onLine

    return (
      <div className="min-h-screen bg-gray-50 flex items-center justify-center p-4">
        <div className="max-w-md w-full bg-white rounded-lg shadow-lg p-6 text-center">
          {isOnline ? (
            <Wifi className="w-16 h-16 text-orange-500 mx-auto mb-4" />
          ) : (
            <WifiOff className="w-16 h-16 text-red-500 mx-auto mb-4" />
          )}
          
          <h2 className="text-xl font-semibold text-gray-900 mb-2">
            {isOnline ? 'Conexão Instável' : 'Sem Conexão'}
          </h2>
          
          <p className="text-gray-600 mb-6">
            {isOnline 
              ? 'Problemas de conectividade detectados. Tentando reconectar...'
              : 'Verifique sua conexão com a internet e tente novamente.'
            }
          </p>

          {retryCount > 0 && (
            <div className="bg-blue-50 border border-blue-200 rounded-md p-3 mb-4">
              <p className="text-sm text-blue-700">
                Tentativa {retryCount} de {this.maxRetries}
              </p>
            </div>
          )}

          <div className="space-y-3">
            <button
              onClick={this.retry}
              disabled={isRetrying}
              className="w-full bg-blue-600 text-white py-2 px-4 rounded-md hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center"
            >
              {isRetrying ? (
                <>
                  <RefreshCw className="w-4 h-4 mr-2 animate-spin" />
                  Reconectando...
                </>
              ) : (
                <>
                  <RefreshCw className="w-4 h-4 mr-2" />
                  Tentar Novamente
                </>
              )}
            </button>

            <button
              onClick={() => window.location.href = '/'}
              className="w-full bg-gray-100 text-gray-700 py-2 px-4 rounded-md hover:bg-gray-200 flex items-center justify-center"
            >
              <Home className="w-4 h-4 mr-2" />
              Ir para Início
            </button>
          </div>

          {!isOnline && (
            <div className="mt-6 p-4 bg-yellow-50 border border-yellow-200 rounded-md">
              <p className="text-sm text-yellow-700">
                <strong>Modo Offline:</strong> Algumas funcionalidades podem estar limitadas.
                Os dados serão sincronizados quando a conexão for restaurada.
              </p>
            </div>
          )}
        </div>
      </div>
    )
  }

  renderCriticalError() {
    const { error, errorId, retryCount } = this.state

    return (
      <div className="min-h-screen bg-red-50 flex items-center justify-center p-4">
        <div className="max-w-lg w-full bg-white rounded-lg shadow-lg p-6">
          <div className="flex items-center mb-4">
            <AlertTriangle className="w-8 h-8 text-red-500 mr-3" />
            <h2 className="text-xl font-semibold text-gray-900">
              Erro Crítico Detectado
            </h2>
          </div>

          <div className="bg-red-50 border border-red-200 rounded-md p-4 mb-6">
            <p className="text-sm text-red-700 font-medium mb-2">
              {error?.message || 'Erro desconhecido'}
            </p>
            <p className="text-xs text-red-600">
              ID do Erro: {errorId}
            </p>
          </div>

          <div className="space-y-4">
            <p className="text-gray-600 text-sm">
              Um erro crítico impediu o carregamento do aplicativo. 
              Nossa equipe foi notificada automaticamente.
            </p>

            <div className="flex space-x-3">
              <button
                onClick={() => window.location.reload()}
                className="flex-1 bg-red-600 text-white py-2 px-4 rounded-md hover:bg-red-700 flex items-center justify-center"
              >
                <RefreshCw className="w-4 h-4 mr-2" />
                Recarregar Página
              </button>

              <button
                onClick={() => window.location.href = '/'}
                className="flex-1 bg-gray-100 text-gray-700 py-2 px-4 rounded-md hover:bg-gray-200 flex items-center justify-center"
              >
                <Home className="w-4 h-4 mr-2" />
                Início
              </button>
            </div>

            {process.env.NODE_ENV === 'development' && (
              <details className="mt-4">
                <summary className="cursor-pointer text-sm text-gray-500 hover:text-gray-700">
                  Detalhes Técnicos (Desenvolvimento)
                </summary>
                <pre className="mt-2 p-3 bg-gray-100 rounded text-xs overflow-auto">
                  {error?.stack}
                </pre>
              </details>
            )}
          </div>
        </div>
      </div>
    )
  }

  renderGenericError() {
    const { error, errorId, retryCount, isRetrying } = this.state

    return (
      <div className="min-h-screen bg-gray-50 flex items-center justify-center p-4">
        <div className="max-w-md w-full bg-white rounded-lg shadow-lg p-6 text-center">
          <Bug className="w-16 h-16 text-orange-500 mx-auto mb-4" />
          
          <h2 className="text-xl font-semibold text-gray-900 mb-2">
            Algo deu errado
          </h2>
          
          <p className="text-gray-600 mb-6">
            Ocorreu um erro inesperado. Nossa equipe foi notificada.
          </p>

          {retryCount > 0 && retryCount < this.maxRetries && (
            <div className="bg-orange-50 border border-orange-200 rounded-md p-3 mb-4">
              <p className="text-sm text-orange-700">
                Tentativa {retryCount} de {this.maxRetries}
              </p>
            </div>
          )}

          {retryCount >= this.maxRetries && (
            <div className="bg-red-50 border border-red-200 rounded-md p-3 mb-4">
              <p className="text-sm text-red-700">
                Máximo de tentativas atingido. Recarregue a página ou entre em contato com o suporte.
              </p>
            </div>
          )}

          <div className="space-y-3">
            {retryCount < this.maxRetries && (
              <button
                onClick={this.retry}
                disabled={isRetrying}
                className="w-full bg-orange-600 text-white py-2 px-4 rounded-md hover:bg-orange-700 disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center"
              >
                {isRetrying ? (
                  <>
                    <RefreshCw className="w-4 h-4 mr-2 animate-spin" />
                    Tentando...
                  </>
                ) : (
                  <>
                    <RefreshCw className="w-4 h-4 mr-2" />
                    Tentar Novamente
                  </>
                )}
              </button>
            )}

            <button
              onClick={() => window.location.reload()}
              className="w-full bg-gray-100 text-gray-700 py-2 px-4 rounded-md hover:bg-gray-200 flex items-center justify-center"
            >
              <RefreshCw className="w-4 h-4 mr-2" />
              Recarregar Página
            </button>

            <button
              onClick={() => window.location.href = '/'}
              className="w-full bg-gray-100 text-gray-700 py-2 px-4 rounded-md hover:bg-gray-200 flex items-center justify-center"
            >
              <Home className="w-4 h-4 mr-2" />
              Ir para Início
            </button>
          </div>

          <div className="mt-6 text-xs text-gray-500">
            ID do Erro: {errorId}
          </div>
        </div>
      </div>
    )
  }

  render() {
    const { hasError, networkError, criticalError, children } = this.state

    if (hasError) {
      // Custom fallback component
      if (this.props.fallback) {
        return this.props.fallback
      }

      // Different error UIs based on error type
      if (networkError) {
        return this.renderNetworkError()
      }

      if (criticalError) {
        return this.renderCriticalError()
      }

      return this.renderGenericError()
    }

    // Wrap children in Suspense for code splitting support
    return (
      <Suspense fallback={this.props.loadingFallback || <LoadingFallback />}>
        {children}
      </Suspense>
    )
  }
}

// Loading fallback component
const LoadingFallback = () => (
  <div className="min-h-screen bg-gray-50 flex items-center justify-center">
    <div className="text-center">
      <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600 mx-auto mb-4"></div>
      <p className="text-gray-600">Carregando...</p>
    </div>
  </div>
)

// Higher-order component for easy usage
export const withErrorBoundary = (Component, errorBoundaryProps = {}) => {
  const WrappedComponent = (props) => (
    <ErrorBoundary {...errorBoundaryProps}>
      <Component {...props} />
    </ErrorBoundary>
  )

  WrappedComponent.displayName = `withErrorBoundary(${Component.displayName || Component.name})`
  return WrappedComponent
}

// Hook for manual error reporting
export const useErrorHandler = () => {
  const handleError = useCallback((error, errorContext = {}) => {
    const errorDetails = {
      error,
      errorContext,
      timestamp: new Date().toISOString(),
      url: window.location.href,
      userAgent: navigator.userAgent
    }

    // Report error
    if (window.Sentry) {
      window.Sentry.captureException(error, { extra: errorDetails })
    }

    console.error('Manual error report:', error, errorDetails)
  }, [])

  return handleError
}

export default ErrorBoundary