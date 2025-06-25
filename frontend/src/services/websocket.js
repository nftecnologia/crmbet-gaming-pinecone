import throttle from 'lodash.throttle'
import debounce from 'lodash.debounce'

/**
 * Ultra-Performance WebSocket Service with Smart Throttling
 * Handles massive real-time data updates with zero lag tolerance
 */
class WebSocketService {
  constructor() {
    this.ws = null
    this.url = null
    this.protocols = null
    this.listeners = new Map()
    this.messageQueue = []
    this.isConnected = false
    this.isConnecting = false
    this.reconnectAttempts = 0
    this.maxReconnectAttempts = 10
    this.reconnectInterval = 1000
    this.maxReconnectInterval = 30000
    this.heartbeatInterval = null
    this.heartbeatTimeout = null
    this.lastHeartbeat = null
    this.messageStats = {
      sent: 0,
      received: 0,
      throttled: 0,
      errors: 0
    }

    // Smart throttling configurations
    this.throttleConfigs = {
      // High-frequency data (price updates, live metrics)
      'price_update': { delay: 100, maxWait: 500 },
      'metrics_update': { delay: 250, maxWait: 1000 },
      
      // Medium-frequency data (user actions, campaign updates)
      'user_action': { delay: 500, maxWait: 2000 },
      'campaign_update': { delay: 750, maxWait: 3000 },
      
      // Low-frequency data (system notifications, config changes)
      'notification': { delay: 1000, maxWait: 5000 },
      'config_update': { delay: 2000, maxWait: 10000 },
      
      // Default for unknown message types
      'default': { delay: 500, maxWait: 2000 }
    }

    // Create throttled handlers for each message type
    this.throttledHandlers = new Map()
    this.setupThrottledHandlers()

    // Bind methods
    this.connect = this.connect.bind(this)
    this.disconnect = this.disconnect.bind(this)
    this.send = this.send.bind(this)
    this.subscribe = this.subscribe.bind(this)
    this.unsubscribe = this.unsubscribe.bind(this)
  }

  /**
   * Setup throttled message handlers based on message types
   */
  setupThrottledHandlers() {
    Object.entries(this.throttleConfigs).forEach(([type, config]) => {
      this.throttledHandlers.set(type, throttle(
        (data) => this.processMessage(type, data),
        config.delay,
        { maxWait: config.maxWait, leading: true, trailing: true }
      ))
    })
  }

  /**
   * Connect to WebSocket with automatic retry logic
   */
  async connect(url, protocols = null) {
    if (this.isConnected || this.isConnecting) {
      console.warn('WebSocket already connected or connecting')
      return Promise.resolve()
    }

    this.url = url
    this.protocols = protocols
    this.isConnecting = true

    return new Promise((resolve, reject) => {
      try {
        this.ws = new WebSocket(url, protocols)
        
        // Connection timeout
        const connectionTimeout = setTimeout(() => {
          this.ws.close()
          reject(new Error('Connection timeout'))
        }, 10000)

        this.ws.onopen = (event) => {
          clearTimeout(connectionTimeout)
          this.isConnected = true
          this.isConnecting = false
          this.reconnectAttempts = 0
          this.reconnectInterval = 1000
          
          console.log('WebSocket connected successfully')
          this.emit('connected', event)
          this.startHeartbeat()
          this.flushMessageQueue()
          resolve(event)
        }

        this.ws.onmessage = (event) => {
          this.messageStats.received++
          this.lastHeartbeat = Date.now()
          
          try {
            const data = JSON.parse(event.data)
            this.handleIncomingMessage(data)
          } catch (error) {
            console.error('Failed to parse WebSocket message:', error)
            this.messageStats.errors++
          }
        }

        this.ws.onclose = (event) => {
          clearTimeout(connectionTimeout)
          this.isConnected = false
          this.isConnecting = false
          this.stopHeartbeat()
          
          console.log('WebSocket connection closed:', event.code, event.reason)
          this.emit('disconnected', event)
          
          // Auto-reconnect if not intentionally closed
          if (event.code !== 1000 && this.reconnectAttempts < this.maxReconnectAttempts) {
            this.scheduleReconnect()
          }
        }

        this.ws.onerror = (error) => {
          clearTimeout(connectionTimeout)
          this.messageStats.errors++
          console.error('WebSocket error:', error)
          this.emit('error', error)
          
          if (this.isConnecting) {
            reject(error)
          }
        }

      } catch (error) {
        this.isConnecting = false
        reject(error)
      }
    })
  }

  /**
   * Handle incoming messages with smart throttling
   */
  handleIncomingMessage(data) {
    const messageType = data.type || 'default'
    const handler = this.throttledHandlers.get(messageType) || 
                   this.throttledHandlers.get('default')
    
    if (handler) {
      handler(data)
    } else {
      this.processMessage(messageType, data)
    }
  }

  /**
   * Process message after throttling
   */
  processMessage(type, data) {
    const listeners = this.listeners.get(type) || []
    const globalListeners = this.listeners.get('*') || []
    
    // Emit to specific type listeners
    listeners.forEach(callback => {
      try {
        callback(data)
      } catch (error) {
        console.error(`Error in listener for ${type}:`, error)
      }
    })
    
    // Emit to global listeners
    globalListeners.forEach(callback => {
      try {
        callback(type, data)
      } catch (error) {
        console.error('Error in global listener:', error)
      }
    })
  }

  /**
   * Schedule reconnection with exponential backoff
   */
  scheduleReconnect() {
    this.reconnectAttempts++
    const delay = Math.min(
      this.reconnectInterval * Math.pow(2, this.reconnectAttempts),
      this.maxReconnectInterval
    )
    
    console.log(`Reconnecting in ${delay}ms (attempt ${this.reconnectAttempts}/${this.maxReconnectAttempts})`)
    
    setTimeout(() => {
      if (!this.isConnected && this.url) {
        this.connect(this.url, this.protocols).catch(error => {
          console.error('Reconnection failed:', error)
        })
      }
    }, delay)
  }

  /**
   * Start heartbeat mechanism
   */
  startHeartbeat() {
    this.heartbeatInterval = setInterval(() => {
      if (this.isConnected) {
        this.send({ type: 'ping', timestamp: Date.now() })
        
        // Check for heartbeat timeout
        this.heartbeatTimeout = setTimeout(() => {
          console.warn('Heartbeat timeout - connection may be lost')
          this.ws?.close()
        }, 5000)
      }
    }, 30000) // Send ping every 30 seconds
  }

  /**
   * Stop heartbeat mechanism
   */
  stopHeartbeat() {
    if (this.heartbeatInterval) {
      clearInterval(this.heartbeatInterval)
      this.heartbeatInterval = null
    }
    if (this.heartbeatTimeout) {
      clearTimeout(this.heartbeatTimeout)
      this.heartbeatTimeout = null
    }
  }

  /**
   * Send message with queuing for offline scenarios
   */
  send(data) {
    const message = JSON.stringify(data)
    
    if (this.isConnected && this.ws.readyState === WebSocket.OPEN) {
      try {
        this.ws.send(message)
        this.messageStats.sent++
        return true
      } catch (error) {
        console.error('Failed to send message:', error)
        this.messageStats.errors++
        this.messageQueue.push(data)
        return false
      }
    } else {
      // Queue message for later sending
      this.messageQueue.push(data)
      return false
    }
  }

  /**
   * Flush queued messages when connection is restored
   */
  flushMessageQueue() {
    while (this.messageQueue.length > 0 && this.isConnected) {
      const message = this.messageQueue.shift()
      this.send(message)
    }
  }

  /**
   * Subscribe to specific message types
   */
  subscribe(messageType, callback) {
    if (!this.listeners.has(messageType)) {
      this.listeners.set(messageType, [])
    }
    this.listeners.get(messageType).push(callback)
    
    return () => this.unsubscribe(messageType, callback)
  }

  /**
   * Unsubscribe from message types
   */
  unsubscribe(messageType, callback) {
    const listeners = this.listeners.get(messageType)
    if (listeners) {
      const index = listeners.indexOf(callback)
      if (index > -1) {
        listeners.splice(index, 1)
      }
      if (listeners.length === 0) {
        this.listeners.delete(messageType)
      }
    }
  }

  /**
   * Emit events to listeners
   */
  emit(event, data) {
    const listeners = this.listeners.get(event) || []
    listeners.forEach(callback => {
      try {
        callback(data)
      } catch (error) {
        console.error(`Error in ${event} listener:`, error)
      }
    })
  }

  /**
   * Disconnect WebSocket
   */
  disconnect() {
    if (this.ws) {
      this.stopHeartbeat()
      this.ws.close(1000, 'Intentional disconnect')
      this.ws = null
    }
    this.isConnected = false
    this.isConnecting = false
    this.reconnectAttempts = this.maxReconnectAttempts // Prevent auto-reconnect
  }

  /**
   * Get connection statistics
   */
  getStats() {
    return {
      ...this.messageStats,
      isConnected: this.isConnected,
      reconnectAttempts: this.reconnectAttempts,
      queuedMessages: this.messageQueue.length,
      activeListeners: Array.from(this.listeners.keys()).length,
      uptime: this.lastHeartbeat ? Date.now() - this.lastHeartbeat : 0
    }
  }

  /**
   * Update throttle configuration for specific message types
   */
  updateThrottleConfig(messageType, config) {
    this.throttleConfigs[messageType] = config
    
    // Recreate throttled handler
    this.throttledHandlers.set(messageType, throttle(
      (data) => this.processMessage(messageType, data),
      config.delay,
      { maxWait: config.maxWait, leading: true, trailing: true }
    ))
  }

  /**
   * Clear all message handlers and reset throttling
   */
  clearThrottling() {
    this.throttledHandlers.forEach(handler => handler.cancel())
    this.setupThrottledHandlers()
  }
}

// Singleton instance
const websocketService = new WebSocketService()

export default websocketService

// React hook for easy WebSocket usage
export const useWebSocket = (url, options = {}) => {
  const [isConnected, setIsConnected] = React.useState(false)
  const [error, setError] = React.useState(null)
  const [stats, setStats] = React.useState(websocketService.getStats())
  
  React.useEffect(() => {
    let unsubscribers = []
    
    // Subscribe to connection events
    unsubscribers.push(
      websocketService.subscribe('connected', () => {
        setIsConnected(true)
        setError(null)
      })
    )
    
    unsubscribers.push(
      websocketService.subscribe('disconnected', () => {
        setIsConnected(false)
      })
    )
    
    unsubscribers.push(
      websocketService.subscribe('error', (err) => {
        setError(err)
      })
    )
    
    // Connect if URL provided
    if (url && !websocketService.isConnected) {
      websocketService.connect(url, options.protocols).catch(setError)
    }
    
    // Update stats periodically
    const statsInterval = setInterval(() => {
      setStats(websocketService.getStats())
    }, 1000)
    
    return () => {
      unsubscribers.forEach(unsub => unsub())
      clearInterval(statsInterval)
      if (options.disconnectOnUnmount) {
        websocketService.disconnect()
      }
    }
  }, [url, options.protocols, options.disconnectOnUnmount])
  
  return {
    isConnected,
    error,
    stats,
    send: websocketService.send,
    subscribe: websocketService.subscribe,
    unsubscribe: websocketService.unsubscribe
  }
}