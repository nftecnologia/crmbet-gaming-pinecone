import React, { Suspense, lazy, useEffect } from 'react'
import { BrowserRouter as Router, Routes, Route, Navigate } from 'react-router-dom'
import { Toaster } from 'react-hot-toast'
import ErrorBoundary from './components/ErrorBoundary'
import Layout from './components/Layout'
import websocketService from './services/websocket'
import offlineStorage from './services/offlineStorage'

// Lazy load pages for better performance
const Dashboard = lazy(() => import('./pages/Dashboard'))
const Clusters = lazy(() => import('./pages/Clusters'))
const Users = lazy(() => import('./pages/Users'))
const Campaigns = lazy(() => import('./pages/Campaigns'))
const Analytics = lazy(() => import('./pages/Analytics'))

// Performance monitoring
const startTime = performance.now()

// Loading fallback component
const PageLoadingFallback = () => (
  <div className="min-h-screen bg-secondary-50 flex items-center justify-center">
    <div className="text-center">
      <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600 mx-auto mb-4"></div>
      <p className="text-gray-600 font-medium">Carregando página...</p>
      <p className="text-sm text-gray-500 mt-2">Otimizando performance...</p>
    </div>
  </div>
)

function App() {
  useEffect(() => {
    // Initialize services
    const initializeServices = async () => {
      try {
        // Initialize offline storage
        await offlineStorage.init()
        
        // Connect WebSocket if online
        if (navigator.onLine) {
          const wsUrl = process.env.REACT_APP_WS_URL || 'ws://localhost:8000/ws'
          await websocketService.connect(wsUrl)
        }
        
        // Performance monitoring
        if (__PERFORMANCE_MONITORING__) {
          const loadTime = performance.now() - startTime
          console.log(`App initialized in ${loadTime.toFixed(2)}ms`)
          
          // Report to analytics
          if (window.analytics) {
            window.analytics.track('App Loaded', {
              loadTime,
              buildTime: __BUILD_TIME__,
              userAgent: navigator.userAgent,
              viewport: {
                width: window.innerWidth,
                height: window.innerHeight
              }
            })
          }
        }
        
      } catch (error) {
        console.error('Failed to initialize services:', error)
      }
    }

    initializeServices()

    // Cleanup on unmount
    return () => {
      websocketService.disconnect()
    }
  }, [])

  // Performance optimization: Preload critical resources
  useEffect(() => {
    // Preload critical images
    const criticalImages = [
      '/assets/logo.png',
      '/assets/icons/dashboard.svg'
    ]
    
    criticalImages.forEach(src => {
      const img = new Image()
      img.src = src
    })
    
    // Prefetch likely next pages
    const prefetchPages = [
      () => import('./pages/Users'),
      () => import('./pages/Analytics')
    ]
    
    // Prefetch after a short delay to not block initial render
    setTimeout(() => {
      prefetchPages.forEach(importFn => {
        importFn().catch(() => {
          // Ignore prefetch errors
        })
      })
    }, 2000)
  }, [])

  return (
    <ErrorBoundary
      maxRetries={3}
      enableAutoRetry={true}
      reportErrors={true}
      fallbackComponent="App"
    >
      <Router>
        <div className="App min-h-screen bg-secondary-50">
          <Layout>
            <Suspense fallback={<PageLoadingFallback />}>
              <ErrorBoundary
                maxRetries={2}
                enableAutoRetry={true}
                loadingFallback={<PageLoadingFallback />}
              >
                <Routes>
                  <Route path="/" element={<Navigate to="/dashboard" replace />} />
                  <Route path="/dashboard" element={<Dashboard />} />
                  <Route path="/clusters" element={<Clusters />} />
                  <Route path="/users" element={<Users />} />
                  <Route path="/campaigns" element={<Campaigns />} />
                  <Route path="/analytics" element={<Analytics />} />
                  <Route path="*" element={<Navigate to="/dashboard" replace />} />
                </Routes>
              </ErrorBoundary>
            </Suspense>
          </Layout>
          
          {/* Toast Notifications */}
          <Toaster
            position="top-right"
            toastOptions={{
              duration: 4000,
              className: 'font-medium',
              style: {
                background: '#ffffff',
                color: '#1e293b',
                boxShadow: '0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05)',
                border: '1px solid #e2e8f0',
              },
              success: {
                iconTheme: {
                  primary: '#22c55e',
                  secondary: '#ffffff',
                },
              },
              error: {
                iconTheme: {
                  primary: '#ef4444',
                  secondary: '#ffffff',
                },
              },
            }}
          />
          
          {/* Performance Monitoring (Development Only) */}
          {process.env.NODE_ENV === 'development' && (
            <div className="fixed bottom-4 right-4 bg-black bg-opacity-75 text-white text-xs px-3 py-2 rounded-md font-mono">
              <div>Build: {__BUILD_TIME__}</div>
              <div>Online: {navigator.onLine ? '✓' : '✗'}</div>
              <div>WS: {websocketService.isConnected ? '✓' : '✗'}</div>
            </div>
          )}
        </div>
      </Router>
    </ErrorBoundary>
  )
}

export default App