import { openDB } from 'idb'
import debounce from 'lodash.debounce'

/**
 * Ultra-Performance Offline Storage Service
 * Implements offline-first architecture with intelligent sync
 */
class OfflineStorageService {
  constructor() {
    this.db = null
    this.isOnline = navigator.onLine
    this.syncQueue = []
    this.conflictResolution = new Map()
    this.syncInProgress = false
    this.syncCallbacks = new Map()
    this.storageStats = {
      totalRecords: 0,
      pendingSync: 0,
      lastSync: null,
      conflicts: 0,
      storageUsed: 0
    }

    // Database configuration
    this.dbConfig = {
      name: 'crmbet-offline',
      version: 1,
      stores: {
        users: { keyPath: 'id', indexes: ['email', 'status', 'updatedAt'] },
        clusters: { keyPath: 'id', indexes: ['name', 'status', 'updatedAt'] },
        campaigns: { keyPath: 'id', indexes: ['name', 'status', 'updatedAt'] },
        analytics: { keyPath: 'id', indexes: ['date', 'type', 'updatedAt'] },
        syncQueue: { keyPath: 'id', indexes: ['timestamp', 'action', 'status'] },
        metadata: { keyPath: 'key' }
      }
    }

    this.init()
    this.setupEventListeners()
    this.debouncedSync = debounce(this.performSync.bind(this), 2000)
  }

  /**
   * Initialize IndexedDB database
   */
  async init() {
    try {
      this.db = await openDB(this.dbConfig.name, this.dbConfig.version, {
        upgrade: (db, oldVersion, newVersion, transaction) => {
          // Create object stores
          Object.entries(this.dbConfig.stores).forEach(([storeName, config]) => {
            if (!db.objectStoreNames.contains(storeName)) {
              const store = db.createObjectStore(storeName, { keyPath: config.keyPath })
              
              // Create indexes
              config.indexes?.forEach(indexName => {
                store.createIndex(indexName, indexName, { unique: false })
              })
            }
          })
        }
      })

      // Initialize metadata
      await this.initializeMetadata()
      await this.updateStorageStats()
      
      console.log('OfflineStorage initialized successfully')
    } catch (error) {
      console.error('Failed to initialize OfflineStorage:', error)
    }
  }

  /**
   * Initialize metadata store
   */
  async initializeMetadata() {
    const tx = this.db.transaction('metadata', 'readwrite')
    const store = tx.objectStore('metadata')
    
    const defaults = [
      { key: 'lastSync', value: null },
      { key: 'syncStrategy', value: 'incremental' },
      { key: 'maxCacheSize', value: 100 * 1024 * 1024 }, // 100MB
      { key: 'compressionEnabled', value: true }
    ]

    for (const item of defaults) {
      const existing = await store.get(item.key)
      if (!existing) {
        await store.put(item)
      }
    }
  }

  /**
   * Setup event listeners for online/offline detection
   */
  setupEventListeners() {
    window.addEventListener('online', () => {
      this.isOnline = true
      console.log('Connection restored - triggering sync')
      this.debouncedSync()
    })

    window.addEventListener('offline', () => {
      this.isOnline = false
      console.log('Connection lost - enabling offline mode')
    })

    // Sync on visibility change (when user returns to tab)
    document.addEventListener('visibilitychange', () => {
      if (!document.hidden && this.isOnline) {
        this.debouncedSync()
      }
    })
  }

  /**
   * Store data with automatic conflict detection
   */
  async store(storeName, data, options = {}) {
    if (!this.db) await this.init()

    const tx = this.db.transaction([storeName, 'metadata'], 'readwrite')
    const store = tx.objectStore(storeName)
    const metaStore = tx.objectStore('metadata')

    try {
      // Add metadata
      const timestamp = Date.now()
      const enrichedData = {
        ...data,
        _offline: true,
        _timestamp: timestamp,
        _version: (data._version || 0) + 1,
        _modified: !options.fromSync
      }

      // Check for conflicts if data exists
      if (data.id) {
        const existing = await store.get(data.id)
        if (existing && existing._version && enrichedData._version <= existing._version) {
          // Potential conflict detected
          await this.handleConflict(storeName, existing, enrichedData)
          return
        }
      }

      await store.put(enrichedData)
      
      // Queue for sync if modified locally
      if (!options.fromSync && enrichedData._modified) {
        await this.addToSyncQueue(storeName, 'upsert', enrichedData)
      }

      await this.updateStorageStats()
      
      console.log(`Stored ${storeName} record:`, enrichedData.id)
    } catch (error) {
      console.error(`Failed to store ${storeName} record:`, error)
      throw error
    }
  }

  /**
   * Retrieve data with caching strategy
   */
  async get(storeName, id, options = {}) {
    if (!this.db) await this.init()

    const tx = this.db.transaction(storeName, 'readonly')
    const store = tx.objectStore(storeName')
    
    try {
      const result = await store.get(id)
      
      if (result) {
        // Update last accessed timestamp
        if (options.updateAccess) {
          result._lastAccessed = Date.now()
          await this.store(storeName, result, { fromSync: true })
        }
        
        return this.sanitizeData(result)
      }
      
      return null
    } catch (error) {
      console.error(`Failed to get ${storeName} record:`, error)
      return null
    }
  }

  /**
   * Query data with advanced filtering
   */
  async query(storeName, options = {}) {
    if (!this.db) await this.init()

    const tx = this.db.transaction(storeName, 'readonly')
    const store = tx.objectStore(storeName)
    
    try {
      let cursor
      
      // Use index if specified
      if (options.index) {
        const index = store.index(options.index)
        cursor = await index.openCursor(options.range)
      } else {
        cursor = await store.openCursor()
      }

      const results = []
      
      if (cursor) {
        do {
          const record = cursor.value
          
          // Apply filters
          if (this.matchesFilters(record, options.filters)) {
            results.push(this.sanitizeData(record))
          }
          
          // Limit results
          if (options.limit && results.length >= options.limit) {
            break
          }
          
        } while (await cursor.continue())
      }

      // Apply sorting
      if (options.sort) {
        results.sort((a, b) => {
          const aVal = a[options.sort.field]
          const bVal = b[options.sort.field]
          const direction = options.sort.direction === 'desc' ? -1 : 1
          
          if (aVal < bVal) return -1 * direction
          if (aVal > bVal) return 1 * direction
          return 0
        })
      }

      return results
    } catch (error) {
      console.error(`Failed to query ${storeName}:`, error)
      return []
    }
  }

  /**
   * Delete data with sync queue management
   */
  async delete(storeName, id, options = {}) {
    if (!this.db) await this.init()

    const tx = this.db.transaction([storeName, 'syncQueue'], 'readwrite')
    const store = tx.objectStore(storeName)
    const syncStore = tx.objectStore('syncQueue')

    try {
      await store.delete(id)
      
      // Queue for sync deletion
      if (!options.fromSync) {
        await this.addToSyncQueue(storeName, 'delete', { id })
      }

      await this.updateStorageStats()
      
      console.log(`Deleted ${storeName} record:`, id)
    } catch (error) {
      console.error(`Failed to delete ${storeName} record:`, error)
      throw error
    }
  }

  /**
   * Add operation to sync queue
   */
  async addToSyncQueue(storeName, action, data) {
    const tx = this.db.transaction('syncQueue', 'readwrite')
    const store = tx.objectStore('syncQueue')

    const queueItem = {
      id: `${storeName}_${action}_${data.id}_${Date.now()}`,
      storeName,
      action,
      data,
      timestamp: Date.now(),
      status: 'pending',
      retries: 0,
      maxRetries: 3
    }

    await store.put(queueItem)
    this.storageStats.pendingSync++
    
    // Trigger sync if online
    if (this.isOnline) {
      this.debouncedSync()
    }
  }

  /**
   * Perform synchronization with server
   */
  async performSync() {
    if (this.syncInProgress || !this.isOnline) return

    this.syncInProgress = true
    console.log('Starting sync process...')

    try {
      const tx = this.db.transaction('syncQueue', 'readwrite')
      const store = tx.objectStore('syncQueue')
      const pendingItems = await store.index('status').getAll('pending')

      for (const item of pendingItems) {
        try {
          await this.syncItem(item)
          
          // Mark as completed
          item.status = 'completed'
          await store.put(item)
          
        } catch (error) {
          console.error(`Failed to sync item ${item.id}:`, error)
          
          // Increment retry count
          item.retries++
          
          if (item.retries >= item.maxRetries) {
            item.status = 'failed'
            console.error(`Max retries exceeded for item ${item.id}`)
          }
          
          await store.put(item)
        }
      }

      // Update last sync timestamp
      await this.setMetadata('lastSync', Date.now())
      this.storageStats.lastSync = Date.now()

      // Clean up completed items
      await this.cleanupSyncQueue()
      
      console.log('Sync completed successfully')
      
    } catch (error) {
      console.error('Sync process failed:', error)
    } finally {
      this.syncInProgress = false
      await this.updateStorageStats()
    }
  }

  /**
   * Sync individual item with server
   */
  async syncItem(item) {
    const { storeName, action, data } = item
    
    // This would integrate with your API service
    const apiEndpoint = this.getApiEndpoint(storeName)
    
    switch (action) {
      case 'upsert':
        // POST or PUT request to server
        const response = await fetch(`${apiEndpoint}/${data.id || ''}`, {
          method: data.id ? 'PUT' : 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(this.sanitizeData(data))
        })
        
        if (response.ok) {
          const serverData = await response.json()
          // Update local data with server response
          await this.store(storeName, serverData, { fromSync: true })
        }
        break
        
      case 'delete':
        await fetch(`${apiEndpoint}/${data.id}`, { 
          method: 'DELETE' 
        })
        break
    }
  }

  /**
   * Handle data conflicts
   */
  async handleConflict(storeName, localData, serverData) {
    this.storageStats.conflicts++
    
    // Default strategy: server wins
    let resolvedData = serverData
    
    // Custom conflict resolution
    const resolver = this.conflictResolution.get(storeName)
    if (resolver) {
      resolvedData = await resolver(localData, serverData)
    }
    
    // Store resolved data
    await this.store(storeName, resolvedData, { fromSync: true })
    
    console.log(`Conflict resolved for ${storeName}:`, resolvedData.id)
  }

  /**
   * Utility methods
   */
  sanitizeData(data) {
    if (!data) return data
    
    const sanitized = { ...data }
    delete sanitized._offline
    delete sanitized._timestamp
    delete sanitized._version
    delete sanitized._modified
    delete sanitized._lastAccessed
    
    return sanitized
  }

  matchesFilters(record, filters) {
    if (!filters) return true
    
    return Object.entries(filters).every(([key, value]) => {
      if (typeof value === 'function') {
        return value(record[key])
      }
      return record[key] === value
    })
  }

  getApiEndpoint(storeName) {
    const endpoints = {
      users: '/api/users',
      clusters: '/api/clusters',
      campaigns: '/api/campaigns',
      analytics: '/api/analytics'
    }
    return endpoints[storeName] || `/api/${storeName}`
  }

  async updateStorageStats() {
    try {
      // Calculate storage usage
      const estimate = await navigator.storage?.estimate?.()
      this.storageStats.storageUsed = estimate?.usage || 0
      
      // Count pending sync items
      const tx = this.db.transaction('syncQueue', 'readonly')
      const store = tx.objectStore('syncQueue')
      const pendingCount = await store.index('status').count('pending')
      this.storageStats.pendingSync = pendingCount
      
    } catch (error) {
      console.error('Failed to update storage stats:', error)
    }
  }

  async setMetadata(key, value) {
    const tx = this.db.transaction('metadata', 'readwrite')
    const store = tx.objectStore('metadata')
    await store.put({ key, value })
  }

  async getMetadata(key) {
    const tx = this.db.transaction('metadata', 'readonly')
    const store = tx.objectStore('metadata')
    const result = await store.get(key)
    return result?.value
  }

  async cleanupSyncQueue() {
    const tx = this.db.transaction('syncQueue', 'readwrite')
    const store = tx.objectStore('syncQueue')
    
    // Remove completed items older than 24 hours
    const cutoff = Date.now() - (24 * 60 * 60 * 1000)
    const completedItems = await store.index('status').getAll('completed')
    
    for (const item of completedItems) {
      if (item.timestamp < cutoff) {
        await store.delete(item.id)
      }
    }
  }

  // Public API methods
  getStats() {
    return { ...this.storageStats, isOnline: this.isOnline }
  }

  setConflictResolver(storeName, resolver) {
    this.conflictResolution.set(storeName, resolver)
  }

  async clearStore(storeName) {
    const tx = this.db.transaction(storeName, 'readwrite')
    const store = tx.objectStore(storeName)
    await store.clear()
    await this.updateStorageStats()
  }

  async exportData() {
    const data = {}
    
    for (const storeName of Object.keys(this.dbConfig.stores)) {
      data[storeName] = await this.query(storeName)
    }
    
    return data
  }
}

// Singleton instance
const offlineStorage = new OfflineStorageService()

export default offlineStorage

// React hook for offline storage
export const useOfflineStorage = (storeName) => {
  const [data, setData] = React.useState([])
  const [loading, setLoading] = React.useState(true)
  const [stats, setStats] = React.useState(offlineStorage.getStats())

  const refresh = React.useCallback(async () => {
    setLoading(true)
    try {
      const results = await offlineStorage.query(storeName)
      setData(results)
    } catch (error) {
      console.error(`Failed to load ${storeName}:`, error)
    } finally {
      setLoading(false)
    }
  }, [storeName])

  React.useEffect(() => {
    refresh()
    
    // Update stats periodically  
    const statsInterval = setInterval(() => {
      setStats(offlineStorage.getStats())
    }, 5000)
    
    return () => clearInterval(statsInterval)
  }, [refresh])

  return {
    data,
    loading,
    stats,
    store: (item) => offlineStorage.store(storeName, item),
    get: (id) => offlineStorage.get(storeName, id),
    delete: (id) => offlineStorage.delete(storeName, id),
    query: (options) => offlineStorage.query(storeName, options),
    refresh
  }
}