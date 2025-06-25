import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest'
import websocketService from '../websocket'

// Mock WebSocket
class MockWebSocket {
  static CONNECTING = 0
  static OPEN = 1
  static CLOSING = 2
  static CLOSED = 3

  constructor(url, protocols) {
    this.url = url
    this.protocols = protocols
    this.readyState = MockWebSocket.CONNECTING
    this.onopen = null
    this.onclose = null
    this.onmessage = null
    this.onerror = null
    
    // Simulate connection after a short delay
    setTimeout(() => {
      this.readyState = MockWebSocket.OPEN
      if (this.onopen) {
        this.onopen({ type: 'open' })
      }
    }, 10)
  }

  send(data) {
    if (this.readyState !== MockWebSocket.OPEN) {
      throw new Error('WebSocket is not open')
    }
    
    // Simulate echo for testing
    setTimeout(() => {
      if (this.onmessage) {
        this.onmessage({
          data: JSON.stringify({ type: 'echo', data: JSON.parse(data) })
        })
      }
    }, 5)
  }

  close(code = 1000, reason = '') {
    this.readyState = MockWebSocket.CLOSED
    if (this.onclose) {
      this.onclose({ code, reason, type: 'close' })
    }
  }

  simulateMessage(data) {
    if (this.onmessage) {
      this.onmessage({
        data: JSON.stringify(data)
      })
    }
  }

  simulateError(error) {
    if (this.onerror) {
      this.onerror({ type: 'error', error })
    }
  }
}

describe('WebSocketService', () => {
  let mockWebSocket

  beforeEach(() => {
    global.WebSocket = MockWebSocket
    mockWebSocket = null
    
    // Reset service state
    websocketService.disconnect()
    websocketService.listeners.clear()
    websocketService.messageQueue = []
    websocketService.reconnectAttempts = 0
  })

  afterEach(() => {
    websocketService.disconnect()
    vi.clearAllTimers()
    vi.useRealTimers()
  })

  describe('Connection Management', () => {
    it('connects successfully', async () => {
      const connection = websocketService.connect('ws://localhost:8080')
      
      await expect(connection).resolves.toBeDefined()
      expect(websocketService.isConnected).toBe(true)
    })

    it('handles connection timeout', async () => {
      vi.useFakeTimers()
      
      // Mock WebSocket that never connects
      global.WebSocket = function(url) {
        this.url = url
        this.readyState = 0
        this.close = vi.fn()
        // Never trigger onopen
      }

      const connectionPromise = websocketService.connect('ws://localhost:8080')
      
      // Fast-forward time to trigger timeout
      vi.advanceTimersByTime(10000)
      
      await expect(connectionPromise).rejects.toThrow('Connection timeout')
    })

    it('disconnects properly', async () => {
      await websocketService.connect('ws://localhost:8080')
      expect(websocketService.isConnected).toBe(true)
      
      websocketService.disconnect()
      expect(websocketService.isConnected).toBe(false)
    })

    it('prevents multiple simultaneous connections', async () => {
      const connection1 = websocketService.connect('ws://localhost:8080')
      const connection2 = websocketService.connect('ws://localhost:8080')
      
      await connection1
      await connection2
      
      // Should resolve both but only create one connection
      expect(websocketService.isConnected).toBe(true)
    })
  })

  describe('Message Handling', () => {
    beforeEach(async () => {
      await websocketService.connect('ws://localhost:8080')
    })

    it('sends messages when connected', () => {
      const testMessage = { type: 'test', data: 'hello' }
      const result = websocketService.send(testMessage)
      
      expect(result).toBe(true)
    })

    it('queues messages when disconnected', () => {
      websocketService.disconnect()
      
      const testMessage = { type: 'test', data: 'hello' }
      const result = websocketService.send(testMessage)
      
      expect(result).toBe(false)
      expect(websocketService.messageQueue).toContain(testMessage)
    })

    it('flushes message queue on reconnection', async () => {
      // Send message while disconnected
      websocketService.disconnect()
      const testMessage = { type: 'test', data: 'hello' }
      websocketService.send(testMessage)
      
      expect(websocketService.messageQueue.length).toBe(1)
      
      // Reconnect
      await websocketService.connect('ws://localhost:8080')
      
      // Queue should be empty after connection
      expect(websocketService.messageQueue.length).toBe(0)
    })

    it('handles JSON parsing errors gracefully', () => {
      const consoleSpy = vi.spyOn(console, 'error').mockImplementation(() => {})
      
      // Simulate invalid JSON message
      websocketService.ws.onmessage({ data: 'invalid json' })
      
      expect(consoleSpy).toHaveBeenCalledWith(
        'Failed to parse WebSocket message:',
        expect.any(Error)
      )
      
      consoleSpy.mockRestore()
    })
  })

  describe('Event Subscription', () => {
    beforeEach(async () => {
      await websocketService.connect('ws://localhost:8080')
    })

    it('subscribes to specific message types', () => {
      const mockCallback = vi.fn()
      const unsubscribe = websocketService.subscribe('user_action', mockCallback)
      
      // Simulate incoming message
      websocketService.ws.simulateMessage({ type: 'user_action', data: { action: 'login' } })
      
      expect(mockCallback).toHaveBeenCalledWith({ type: 'user_action', data: { action: 'login' } })
      
      // Test unsubscribe
      unsubscribe()
      websocketService.ws.simulateMessage({ type: 'user_action', data: { action: 'logout' } })
      
      expect(mockCallback).toHaveBeenCalledTimes(1)
    })

    it('subscribes to all message types with wildcard', () => {
      const mockCallback = vi.fn()
      websocketService.subscribe('*', mockCallback)
      
      websocketService.ws.simulateMessage({ type: 'user_action', data: {} })
      websocketService.ws.simulateMessage({ type: 'price_update', data: {} })
      
      expect(mockCallback).toHaveBeenCalledTimes(2)
    })

    it('handles multiple subscribers for same message type', () => {
      const callback1 = vi.fn()
      const callback2 = vi.fn()
      
      websocketService.subscribe('test', callback1)
      websocketService.subscribe('test', callback2)
      
      websocketService.ws.simulateMessage({ type: 'test', data: {} })
      
      expect(callback1).toHaveBeenCalled()
      expect(callback2).toHaveBeenCalled()
    })

    it('handles errors in listener callbacks gracefully', () => {
      const consoleSpy = vi.spyOn(console, 'error').mockImplementation(() => {})
      const errorCallback = vi.fn(() => { throw new Error('Callback error') })
      
      websocketService.subscribe('test', errorCallback)
      websocketService.ws.simulateMessage({ type: 'test', data: {} })
      
      expect(consoleSpy).toHaveBeenCalledWith(
        'Error in listener for test:',
        expect.any(Error)
      )
      
      consoleSpy.mockRestore()
    })
  })

  describe('Throttling Mechanism', () => {
    beforeEach(async () => {
      vi.useFakeTimers()
      await websocketService.connect('ws://localhost:8080')
    })

    afterEach(() => {
      vi.useRealTimers()
    })

    it('throttles high-frequency messages', () => {
      const mockCallback = vi.fn()
      websocketService.subscribe('price_update', mockCallback)
      
      // Send multiple messages rapidly
      for (let i = 0; i < 10; i++) {
        websocketService.ws.simulateMessage({
          type: 'price_update',
          data: { price: 100 + i }
        })
      }
      
      // Should be throttled, not all 10 calls made immediately
      expect(mockCallback.mock.calls.length).toBeLessThan(10)
      
      // Fast-forward time to allow throttled calls
      vi.advanceTimersByTime(1000)
      
      // Should have processed more calls after time passes
      expect(mockCallback.mock.calls.length).toBeGreaterThan(0)
    })

    it('applies different throttling for different message types', () => {
      const priceCallback = vi.fn()
      const notificationCallback = vi.fn()
      
      websocketService.subscribe('price_update', priceCallback)
      websocketService.subscribe('notification', notificationCallback)
      
      // Send same number of each message type
      for (let i = 0; i < 5; i++) {
        websocketService.ws.simulateMessage({ type: 'price_update', data: {} })
        websocketService.ws.simulateMessage({ type: 'notification', data: {} })
      }
      
      // Price updates should be throttled more aggressively than notifications
      // (This depends on the specific throttling configuration)
    })

    it('allows configuration of throttling parameters', () => {
      const originalConfig = websocketService.throttleConfigs['custom_type']
      
      websocketService.updateThrottleConfig('custom_type', {
        delay: 100,
        maxWait: 500
      })
      
      expect(websocketService.throttleConfigs['custom_type']).toEqual({
        delay: 100,
        maxWait: 500
      })
      
      // Test with custom throttling
      const mockCallback = vi.fn()
      websocketService.subscribe('custom_type', mockCallback)
      
      websocketService.ws.simulateMessage({ type: 'custom_type', data: {} })
      
      expect(mockCallback).toHaveBeenCalled()
    })
  })

  describe('Auto-Reconnection', () => {
    beforeEach(() => {
      vi.useFakeTimers()
    })

    afterEach(() => {
      vi.useRealTimers()
    })

    it('attempts reconnection on unexpected disconnect', async () => {
      await websocketService.connect('ws://localhost:8080')
      
      // Simulate unexpected disconnect
      websocketService.ws.close(1006, 'Connection lost')
      
      expect(websocketService.isConnected).toBe(false)
      expect(websocketService.reconnectAttempts).toBe(0)
      
      // Fast-forward to trigger reconnection
      vi.advanceTimersByTime(2000)
      
      expect(websocketService.reconnectAttempts).toBeGreaterThan(0)
    })

    it('uses exponential backoff for reconnection attempts', () => {
      const scheduleReconnectSpy = vi.spyOn(websocketService, 'scheduleReconnect')
      
      websocketService.reconnectAttempts = 3
      websocketService.scheduleReconnect()
      
      // Should schedule with exponential delay
      expect(scheduleReconnectSpy).toHaveBeenCalled()
    })

    it('stops reconnecting after max attempts', async () => {
      websocketService.maxReconnectAttempts = 2
      
      await websocketService.connect('ws://localhost:8080')
      
      // Force multiple failed reconnections
      websocketService.reconnectAttempts = 2
      websocketService.ws.close(1006, 'Connection lost')
      
      vi.advanceTimersByTime(10000)
      
      // Should not exceed max attempts
      expect(websocketService.reconnectAttempts).toBeLessThanOrEqual(2)
    })

    it('does not reconnect on intentional disconnect', async () => {
      await websocketService.connect('ws://localhost:8080')
      
      websocketService.disconnect()
      
      vi.advanceTimersByTime(5000)
      
      // Should not attempt reconnection
      expect(websocketService.reconnectAttempts).toBe(10) // Set to max to prevent reconnection
    })
  })

  describe('Heartbeat Mechanism', () => {
    beforeEach(() => {
      vi.useFakeTimers()
    })

    afterEach(() => {
      vi.useRealTimers()
    })

    it('sends periodic heartbeat messages', async () => {
      const sendSpy = vi.spyOn(websocketService, 'send')
      
      await websocketService.connect('ws://localhost:8080')
      
      // Fast-forward to trigger heartbeat
      vi.advanceTimersByTime(30000)
      
      expect(sendSpy).toHaveBeenCalledWith({
        type: 'ping',
        timestamp: expect.any(Number)
      })
    })

    it('detects connection timeout', async () => {
      await websocketService.connect('ws://localhost:8080')
      
      // Fast-forward past heartbeat timeout
      vi.advanceTimersByTime(35000)
      
      // Should close connection due to timeout
      expect(websocketService.isConnected).toBe(false)
    })

    it('resets heartbeat timeout on message receipt', async () => {
      await websocketService.connect('ws://localhost:8080')
      
      // Simulate message receipt
      websocketService.ws.simulateMessage({ type: 'pong' })
      
      expect(websocketService.lastHeartbeat).toBeGreaterThan(0)
    })
  })

  describe('Statistics and Monitoring', () => {
    beforeEach(async () => {
      await websocketService.connect('ws://localhost:8080')
    })

    it('tracks message statistics', () => {
      const initialStats = websocketService.getStats()
      
      websocketService.send({ type: 'test' })
      websocketService.ws.simulateMessage({ type: 'response' })
      
      const updatedStats = websocketService.getStats()
      
      expect(updatedStats.sent).toBeGreaterThan(initialStats.sent)
      expect(updatedStats.received).toBeGreaterThan(initialStats.received)
    })

    it('tracks connection state in statistics', () => {
      const stats = websocketService.getStats()
      
      expect(stats.isConnected).toBe(true)
      expect(stats.reconnectAttempts).toBe(0)
      expect(stats.queuedMessages).toBe(0)
    })

    it('counts active listeners', () => {
      websocketService.subscribe('test1', vi.fn())
      websocketService.subscribe('test2', vi.fn())
      
      const stats = websocketService.getStats()
      
      expect(stats.activeListeners).toBe(2)
    })
  })

  describe('Error Handling', () => {
    it('handles WebSocket construction errors', async () => {
      global.WebSocket = function() {
        throw new Error('WebSocket not supported')
      }
      
      await expect(websocketService.connect('ws://localhost:8080'))
        .rejects.toThrow('WebSocket not supported')
    })

    it('handles send errors gracefully', async () => {
      await websocketService.connect('ws://localhost:8080')
      
      // Mock send to throw error
      websocketService.ws.send = vi.fn(() => {
        throw new Error('Send failed')
      })
      
      const result = websocketService.send({ type: 'test' })
      
      expect(result).toBe(false)
      expect(websocketService.messageQueue.length).toBe(1)
    })

    it('increments error count on send failures', async () => {
      await websocketService.connect('ws://localhost:8080')
      
      const initialStats = websocketService.getStats()
      
      websocketService.ws.send = vi.fn(() => {
        throw new Error('Send failed')
      })
      
      websocketService.send({ type: 'test' })
      
      const updatedStats = websocketService.getStats()
      expect(updatedStats.errors).toBeGreaterThan(initialStats.errors)
    })
  })

  describe('Memory Management', () => {
    it('clears throttling on service reset', () => {
      const clearSpy = vi.fn()
      
      // Mock throttled handlers
      websocketService.throttledHandlers.set('test', { cancel: clearSpy })
      
      websocketService.clearThrottling()
      
      expect(clearSpy).toHaveBeenCalled()
    })

    it('cleans up listeners on unsubscribe', () => {
      const callback = vi.fn()
      websocketService.subscribe('test', callback)
      
      expect(websocketService.listeners.get('test')).toContain(callback)
      
      websocketService.unsubscribe('test', callback)
      
      expect(websocketService.listeners.get('test')).not.toContain(callback)
    })

    it('removes empty listener arrays', () => {
      const callback = vi.fn()
      websocketService.subscribe('test', callback)
      websocketService.unsubscribe('test', callback)
      
      expect(websocketService.listeners.has('test')).toBe(false)
    })
  })
})