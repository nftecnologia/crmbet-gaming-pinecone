/**
 * Redis Configuration - ENTERPRISE CLUSTERING ULTRA-PERFORMANCE
 * 
 * Sistema Redis enterprise-grade otimizado para:
 * - 100k+ requests/segundo
 * - 10M+ usuários simultâneos  
 * - Multi-layer caching (L1-L4)
 * - Clustering com failover automático
 * - Memory optimization para datasets gigantes
 * - Cache invalidation inteligente
 * 
 * @author CRM Ultra-Performance Team
 */

const redis = require('redis');
const { Cluster } = require('ioredis');
const logger = require('../utils/logger');

// Configuração enterprise para dados massivos
const REDIS_ENTERPRISE_CONFIG = {
  // Clustering configuration
  CLUSTER: {
    ENABLED: process.env.REDIS_CLUSTER_ENABLED === 'true',
    NODES: process.env.REDIS_CLUSTER_NODES ? 
      process.env.REDIS_CLUSTER_NODES.split(',').map(node => {
        const [host, port] = node.split(':');
        return { host, port: parseInt(port) || 6379 };
      }) : [
        { host: process.env.REDIS_HOST || 'localhost', port: parseInt(process.env.REDIS_PORT) || 6379 },
        { host: process.env.REDIS_HOST_2 || 'localhost', port: parseInt(process.env.REDIS_PORT_2) || 6380 },
        { host: process.env.REDIS_HOST_3 || 'localhost', port: parseInt(process.env.REDIS_PORT_3) || 6381 }
      ]
  },
  
  // Performance settings
  PERFORMANCE: {
    POOL_SIZE: parseInt(process.env.REDIS_POOL_SIZE) || 100,
    PIPELINE_BATCH_SIZE: parseInt(process.env.REDIS_PIPELINE_BATCH) || 100,
    CONNECT_TIMEOUT: parseInt(process.env.REDIS_CONNECT_TIMEOUT) || 3000,
    COMMAND_TIMEOUT: parseInt(process.env.REDIS_COMMAND_TIMEOUT) || 5000,
    RETRY_ATTEMPTS: parseInt(process.env.REDIS_RETRY_ATTEMPTS) || 3,
    RETRY_DELAY: parseInt(process.env.REDIS_RETRY_DELAY) || 200
  },
  
  // Multi-layer cache TTL configuration
  CACHE_LAYERS: {
    L1_MEMORY: {
      TTL: parseInt(process.env.REDIS_L1_TTL) || 60, // 1 minute
      MAX_SIZE: parseInt(process.env.REDIS_L1_MAX_SIZE) || 10000
    },
    L2_REDIS_HOT: {
      TTL: parseInt(process.env.REDIS_L2_TTL) || 300, // 5 minutes
      DB: parseInt(process.env.REDIS_L2_DB) || 0
    },
    L3_REDIS_WARM: {
      TTL: parseInt(process.env.REDIS_L3_TTL) || 1800, // 30 minutes
      DB: parseInt(process.env.REDIS_L3_DB) || 1
    },
    L4_REDIS_COLD: {
      TTL: parseInt(process.env.REDIS_L4_TTL) || 7200, // 2 hours
      DB: parseInt(process.env.REDIS_L4_DB) || 2
    }
  },
  
  // Memory optimization
  MEMORY: {
    MAX_MEMORY: process.env.REDIS_MAX_MEMORY || '2gb',
    MAX_MEMORY_POLICY: process.env.REDIS_MAX_MEMORY_POLICY || 'allkeys-lru',
    LAZY_FREE: process.env.REDIS_LAZY_FREE !== 'false'
  }
};

// Configuração base para conexões Redis
const baseRedisConfig = {
  password: process.env.REDIS_PASSWORD || undefined,
  connectTimeout: REDIS_ENTERPRISE_CONFIG.PERFORMANCE.CONNECT_TIMEOUT,
  commandTimeout: REDIS_ENTERPRISE_CONFIG.PERFORMANCE.COMMAND_TIMEOUT,
  lazyConnect: true,
  maxRetriesPerRequest: REDIS_ENTERPRISE_CONFIG.PERFORMANCE.RETRY_ATTEMPTS,
  retryDelayOnFailover: REDIS_ENTERPRISE_CONFIG.PERFORMANCE.RETRY_DELAY,
  retryDelayOnClusterDown: REDIS_ENTERPRISE_CONFIG.PERFORMANCE.RETRY_DELAY,
  enableAutoPipelining: true,
  family: 4,
  keepAlive: true,
  
  // Enterprise retry strategy
  retryStrategy: (times) => {
    if (times > REDIS_ENTERPRISE_CONFIG.PERFORMANCE.RETRY_ATTEMPTS) {
      logger.error('Redis max retry attempts exceeded');
      return null;
    }
    
    const delay = Math.min(times * REDIS_ENTERPRISE_CONFIG.PERFORMANCE.RETRY_DELAY, 3000);
    logger.warn(`Redis retry attempt ${times}, delay: ${delay}ms`);
    return delay;
  }
};

// Redis Cluster Manager para enterprise clustering
class RedisClusterManager {
  constructor() {
    this.cluster = null;
    this.nodes = [];
    this.isClusterMode = REDIS_ENTERPRISE_CONFIG.CLUSTER.ENABLED;
    this.initialize();
  }

  initialize() {
    if (this.isClusterMode) {
      this.initializeCluster();
    } else {
      this.initializeSingleNode();
    }
  }

  initializeCluster() {
    try {
      this.cluster = new Cluster(REDIS_ENTERPRISE_CONFIG.CLUSTER.NODES, {
        ...baseRedisConfig,
        enableOfflineQueue: false,
        redisOptions: {
          ...baseRedisConfig,
          password: process.env.REDIS_PASSWORD
        },
        clusterRetryDelayOnFailover: REDIS_ENTERPRISE_CONFIG.PERFORMANCE.RETRY_DELAY,
        clusterRetryDelayOnClusterDown: REDIS_ENTERPRISE_CONFIG.PERFORMANCE.RETRY_DELAY,
        scaleReads: 'slave',
        maxRedirections: 3
      });

      this.setupClusterEventListeners();
      logger.info('Redis cluster initialized with nodes:', REDIS_ENTERPRISE_CONFIG.CLUSTER.NODES);
    } catch (error) {
      logger.error('Failed to initialize Redis cluster:', error);
      throw error;
    }
  }

  initializeSingleNode() {
    const singleNodeConfig = {
      host: process.env.REDIS_HOST || 'localhost',
      port: process.env.REDIS_PORT || 6379,
      db: process.env.REDIS_DB || 0,
      ...baseRedisConfig
    };

    this.cluster = redis.createClient(singleNodeConfig);
    this.setupSingleNodeEventListeners();
    logger.info('Redis single node initialized:', singleNodeConfig.host + ':' + singleNodeConfig.port);
  }

  setupClusterEventListeners() {
    this.cluster.on('connect', () => {
      logger.info('Redis cluster connected');
    });

    this.cluster.on('ready', () => {
      logger.info('Redis cluster ready');
      this.optimizeClusterSettings();
    });

    this.cluster.on('error', (err) => {
      logger.error('Redis cluster error:', err);
    });

    this.cluster.on('close', () => {
      logger.warn('Redis cluster connection closed');
    });

    this.cluster.on('reconnecting', () => {
      logger.info('Redis cluster reconnecting...');
    });

    this.cluster.on('node error', (err, node) => {
      logger.error(`Redis cluster node error [${node.options.host}:${node.options.port}]:`, err);
    });
  }

  setupSingleNodeEventListeners() {
    this.cluster.on('connect', () => {
      logger.info('Redis connected');
    });

    this.cluster.on('ready', () => {
      logger.info('Redis ready');
      this.optimizeSettings();
    });

    this.cluster.on('error', (err) => {
      logger.error('Redis error:', err);
    });

    this.cluster.on('end', () => {
      logger.warn('Redis connection ended');
    });

    this.cluster.on('reconnecting', () => {
      logger.info('Redis reconnecting...');
    });
  }

  async optimizeClusterSettings() {
    try {
      // Configurações de performance para cluster
      await this.cluster.config('SET', 'maxmemory-policy', REDIS_ENTERPRISE_CONFIG.MEMORY.MAX_MEMORY_POLICY);
      await this.cluster.config('SET', 'timeout', '0');
      await this.cluster.config('SET', 'tcp-keepalive', '60');
      
      if (REDIS_ENTERPRISE_CONFIG.MEMORY.LAZY_FREE) {
        await this.cluster.config('SET', 'lazyfree-lazy-eviction', 'yes');
        await this.cluster.config('SET', 'lazyfree-lazy-expire', 'yes');
        await this.cluster.config('SET', 'lazyfree-lazy-server-del', 'yes');
      }
      
      logger.info('Redis cluster performance optimizations applied');
    } catch (error) {
      logger.warn('Some Redis cluster optimizations failed:', error.message);
    }
  }

  async optimizeSettings() {
    try {
      // Configurações de performance para single node
      await this.cluster.configSet('maxmemory-policy', REDIS_ENTERPRISE_CONFIG.MEMORY.MAX_MEMORY_POLICY);
      await this.cluster.configSet('timeout', '0');
      await this.cluster.configSet('tcp-keepalive', '60');
      
      if (REDIS_ENTERPRISE_CONFIG.MEMORY.LAZY_FREE) {
        await this.cluster.configSet('lazyfree-lazy-eviction', 'yes');
        await this.cluster.configSet('lazyfree-lazy-expire', 'yes');
        await this.cluster.configSet('lazyfree-lazy-server-del', 'yes');
      }
      
      logger.info('Redis performance optimizations applied');
    } catch (error) {
      logger.warn('Some Redis optimizations failed:', error.message);
    }
  }

  getClient() {
    return this.cluster;
  }

  async healthCheck() {
    try {
      const start = Date.now();
      await this.cluster.ping();
      const latency = Date.now() - start;

      let info = {};
      if (this.isClusterMode) {
        const nodes = this.cluster.nodes();
        info = {
          clusterMode: true,
          nodeCount: nodes.length,
          latency: `${latency}ms`
        };
      } else {
        const memoryInfo = await this.cluster.info('memory');
        info = {
          clusterMode: false,
          latency: `${latency}ms`,
          memory: this.parseRedisInfo(memoryInfo)
        };
      }

      return {
        status: 'healthy',
        ...info
      };
    } catch (error) {
      return {
        status: 'unhealthy',
        error: error.message
      };
    }
  }

  parseRedisInfo(infoString) {
    const lines = infoString.split('\r\n');
    const info = {};
    lines.forEach(line => {
      const [key, value] = line.split(':');
      if (key && value) info[key] = value;
    });
    return {
      used: info.used_memory_human,
      peak: info.used_memory_peak_human,
      fragmentation: info.mem_fragmentation_ratio
    };
  }
}

// Instanciar cluster manager
const clusterManager = new RedisClusterManager();
const client = clusterManager.getClient();

// Multi-Layer Cache System Ultra-Performance (L1-L4)
class MultiLayerCacheManager {
  constructor() {
    this.l1Cache = new Map(); // L1: In-memory cache (ultra-fast)
    this.l2Client = null;     // L2: Redis Hot Cache (fast)
    this.l3Client = null;     // L3: Redis Warm Cache (medium)
    this.l4Client = null;     // L4: Redis Cold Cache (slower but persistent)
    
    this.initializeLayers();
    this.setupCleanupTimers();
    
    // Metrics tracking
    this.metrics = {
      l1: { hits: 0, misses: 0, sets: 0, evictions: 0 },
      l2: { hits: 0, misses: 0, sets: 0 },
      l3: { hits: 0, misses: 0, sets: 0 },
      l4: { hits: 0, misses: 0, sets: 0 },
      totalRequests: 0
    };
  }

  initializeLayers() {
    // L2 - Hot Cache (DB 0)
    this.l2Client = clusterManager.isClusterMode ? 
      client : 
      redis.createClient({ ...baseRedisConfig, db: REDIS_ENTERPRISE_CONFIG.CACHE_LAYERS.L2_REDIS_HOT.DB });
    
    // L3 - Warm Cache (DB 1)
    if (!clusterManager.isClusterMode) {
      this.l3Client = redis.createClient({ ...baseRedisConfig, db: REDIS_ENTERPRISE_CONFIG.CACHE_LAYERS.L3_REDIS_WARM.DB });
      this.l4Client = redis.createClient({ ...baseRedisConfig, db: REDIS_ENTERPRISE_CONFIG.CACHE_LAYERS.L4_REDIS_COLD.DB });
    } else {
      // Em cluster mode, usar namespaces em vez de DBs
      this.l3Client = client;
      this.l4Client = client;
    }

    logger.info('Multi-layer cache system initialized (L1-L4)');
  }

  setupCleanupTimers() {
    // L1 cleanup a cada 5 minutos
    setInterval(() => {
      this.cleanupL1Cache();
    }, 5 * 60 * 1000);

    // Metrics reset a cada hora
    setInterval(() => {
      this.resetMetrics();
    }, 60 * 60 * 1000);
  }

  cleanupL1Cache() {
    const maxSize = REDIS_ENTERPRISE_CONFIG.CACHE_LAYERS.L1_MEMORY.MAX_SIZE;
    if (this.l1Cache.size > maxSize) {
      const keysToDelete = this.l1Cache.size - maxSize;
      const keys = Array.from(this.l1Cache.keys());
      
      for (let i = 0; i < keysToDelete; i++) {
        this.l1Cache.delete(keys[i]);
        this.metrics.l1.evictions++;
      }
      
      logger.debug(`L1 cache cleanup: removed ${keysToDelete} entries`);
    }
  }

  generateKey(key, layer) {
    if (clusterManager.isClusterMode && layer !== 'l2') {
      return `${layer}:${key}`;
    }
    return key;
  }

  async get(key) {
    this.metrics.totalRequests++;
    const start = Date.now();

    try {
      // L1 - Memory Cache (fastest)
      if (this.l1Cache.has(key)) {
        const item = this.l1Cache.get(key);
        if (item.expires > Date.now()) {
          this.metrics.l1.hits++;
          logger.debug(`L1 cache hit: ${key} (${Date.now() - start}ms)`);
          return item.value;
        } else {
          this.l1Cache.delete(key);
        }
      }
      this.metrics.l1.misses++;

      // L2 - Redis Hot Cache
      let value = await this.getFromRedis(this.l2Client, this.generateKey(key, 'l2'));
      if (value !== null) {
        this.metrics.l2.hits++;
        this.setL1(key, value, REDIS_ENTERPRISE_CONFIG.CACHE_LAYERS.L1_MEMORY.TTL);
        logger.debug(`L2 cache hit: ${key} (${Date.now() - start}ms)`);
        return value;
      }
      this.metrics.l2.misses++;

      // L3 - Redis Warm Cache
      value = await this.getFromRedis(this.l3Client, this.generateKey(key, 'l3'));
      if (value !== null) {
        this.metrics.l3.hits++;
        // Promover para L2 e L1
        await this.setL2(key, value, REDIS_ENTERPRISE_CONFIG.CACHE_LAYERS.L2_REDIS_HOT.TTL);
        this.setL1(key, value, REDIS_ENTERPRISE_CONFIG.CACHE_LAYERS.L1_MEMORY.TTL);
        logger.debug(`L3 cache hit: ${key} (${Date.now() - start}ms)`);
        return value;
      }
      this.metrics.l3.misses++;

      // L4 - Redis Cold Cache
      value = await this.getFromRedis(this.l4Client, this.generateKey(key, 'l4'));
      if (value !== null) {
        this.metrics.l4.hits++;
        // Promover para L3, L2 e L1
        await this.setL3(key, value, REDIS_ENTERPRISE_CONFIG.CACHE_LAYERS.L3_REDIS_WARM.TTL);
        await this.setL2(key, value, REDIS_ENTERPRISE_CONFIG.CACHE_LAYERS.L2_REDIS_HOT.TTL);
        this.setL1(key, value, REDIS_ENTERPRISE_CONFIG.CACHE_LAYERS.L1_MEMORY.TTL);
        logger.debug(`L4 cache hit: ${key} (${Date.now() - start}ms)`);
        return value;
      }
      this.metrics.l4.misses++;

      logger.debug(`Cache miss: ${key} (${Date.now() - start}ms)`);
      return null;

    } catch (error) {
      logger.error(`Cache get error for key ${key}:`, error);
      return null;
    }
  }

  async set(key, value, ttl = null) {
    try {
      const l1TTL = ttl || REDIS_ENTERPRISE_CONFIG.CACHE_LAYERS.L1_MEMORY.TTL;
      const l2TTL = ttl || REDIS_ENTERPRISE_CONFIG.CACHE_LAYERS.L2_REDIS_HOT.TTL;
      const l3TTL = ttl || REDIS_ENTERPRISE_CONFIG.CACHE_LAYERS.L3_REDIS_WARM.TTL;
      const l4TTL = ttl || REDIS_ENTERPRISE_CONFIG.CACHE_LAYERS.L4_REDIS_COLD.TTL;

      // Set em todas as camadas simultaneamente para máxima performance
      const promises = [
        this.setL1(key, value, l1TTL),
        this.setL2(key, value, l2TTL),
        this.setL3(key, value, l3TTL),
        this.setL4(key, value, l4TTL)
      ];

      await Promise.allSettled(promises);
      logger.debug(`Cache set: ${key} (all layers)`);
      return true;

    } catch (error) {
      logger.error(`Cache set error for key ${key}:`, error);
      return false;
    }
  }

  setL1(key, value, ttl) {
    this.l1Cache.set(key, {
      value,
      expires: Date.now() + (ttl * 1000)
    });
    this.metrics.l1.sets++;
  }

  async setL2(key, value, ttl) {
    try {
      await this.setToRedis(this.l2Client, this.generateKey(key, 'l2'), value, ttl);
      this.metrics.l2.sets++;
    } catch (error) {
      logger.error(`L2 cache set error:`, error);
    }
  }

  async setL3(key, value, ttl) {
    try {
      await this.setToRedis(this.l3Client, this.generateKey(key, 'l3'), value, ttl);
      this.metrics.l3.sets++;
    } catch (error) {
      logger.error(`L3 cache set error:`, error);
    }
  }

  async setL4(key, value, ttl) {
    try {
      await this.setToRedis(this.l4Client, this.generateKey(key, 'l4'), value, ttl);
      this.metrics.l4.sets++;
    } catch (error) {
      logger.error(`L4 cache set error:`, error);
    }
  }

  async getFromRedis(client, key) {
    try {
      const value = await client.get(key);
      return value ? JSON.parse(value) : null;
    } catch (error) {
      logger.error(`Redis get error for key ${key}:`, error);
      return null;
    }
  }

  async setToRedis(client, key, value, ttl) {
    try {
      const serialized = JSON.stringify(value);
      if (ttl > 0) {
        await client.setex(key, ttl, serialized);
      } else {
        await client.set(key, serialized);
      }
    } catch (error) {
      logger.error(`Redis set error for key ${key}:`, error);
    }
  }

  async del(key) {
    try {
      // Deletar de todas as camadas
      this.l1Cache.delete(key);
      
      const promises = [
        this.l2Client.del(this.generateKey(key, 'l2')),
        this.l3Client.del(this.generateKey(key, 'l3')),
        this.l4Client.del(this.generateKey(key, 'l4'))
      ];

      await Promise.allSettled(promises);
      logger.debug(`Cache delete: ${key} (all layers)`);
      return true;

    } catch (error) {
      logger.error(`Cache delete error for key ${key}:`, error);
      return false;
    }
  }

  async invalidatePattern(pattern) {
    try {
      // L1 - Manual pattern matching
      const l1Keys = Array.from(this.l1Cache.keys()).filter(key => 
        new RegExp(pattern.replace('*', '.*')).test(key)
      );
      l1Keys.forEach(key => this.l1Cache.delete(key));

      // L2, L3, L4 - Redis pattern
      const promises = [
        this.invalidateRedisPattern(this.l2Client, this.generateKey(pattern, 'l2')),
        this.invalidateRedisPattern(this.l3Client, this.generateKey(pattern, 'l3')),
        this.invalidateRedisPattern(this.l4Client, this.generateKey(pattern, 'l4'))
      ];

      await Promise.allSettled(promises);
      logger.info(`Cache pattern invalidated: ${pattern}`);
      return true;

    } catch (error) {
      logger.error(`Cache pattern invalidation error for ${pattern}:`, error);
      return false;
    }
  }

  async invalidateRedisPattern(client, pattern) {
    try {
      const keys = await client.keys(pattern);
      if (keys.length > 0) {
        await client.del(keys);
      }
    } catch (error) {
      logger.error(`Redis pattern invalidation error:`, error);
    }
  }

  getMetrics() {
    const total = this.metrics.totalRequests;
    return {
      ...this.metrics,
      hitRates: {
        l1: total > 0 ? ((this.metrics.l1.hits / total) * 100).toFixed(2) + '%' : '0%',
        l2: total > 0 ? ((this.metrics.l2.hits / total) * 100).toFixed(2) + '%' : '0%',
        l3: total > 0 ? ((this.metrics.l3.hits / total) * 100).toFixed(2) + '%' : '0%',
        l4: total > 0 ? ((this.metrics.l4.hits / total) * 100).toFixed(2) + '%' : '0%',
        overall: total > 0 ? (((this.metrics.l1.hits + this.metrics.l2.hits + this.metrics.l3.hits + this.metrics.l4.hits) / total) * 100).toFixed(2) + '%' : '0%'
      }
    };
  }

  resetMetrics() {
    this.metrics = {
      l1: { hits: 0, misses: 0, sets: 0, evictions: 0 },
      l2: { hits: 0, misses: 0, sets: 0 },
      l3: { hits: 0, misses: 0, sets: 0 },
      l4: { hits: 0, misses: 0, sets: 0 },
      totalRequests: 0
    };
    logger.debug('Cache metrics reset');
  }
}

// Instanciar multi-layer cache manager
const multiCache = new MultiLayerCacheManager();

/**
 * Conectar ao sistema Redis enterprise ultra-performance
 */
const connectRedis = async () => {
  try {
    logger.info('Initializing Redis enterprise cluster system...');

    // Conectar cliente principal
    if (!clusterManager.isClusterMode) {
      await client.connect();
    }
    
    // Conectar clientes de camadas
    if (!clusterManager.isClusterMode) {
      if (multiCache.l2Client !== client) await multiCache.l2Client.connect();
      if (multiCache.l3Client !== client) await multiCache.l3Client.connect();
      if (multiCache.l4Client !== client) await multiCache.l4Client.connect();
    }
    
    // Testar conectividade
    const pong = await client.ping();
    logger.info('Redis cluster ping response:', pong);
    
    // Health check completo
    const health = await clusterManager.healthCheck();
    logger.info('Redis cluster health:', health);
    
    logger.info('Redis enterprise system initialized successfully');
    return true;
    
  } catch (error) {
    logger.error('Redis connection failed:', error);
    throw error;
  }
};

/**
 * Enterprise Cache Manager Ultra-Performance
 * Wrapper inteligente que usa multi-layer caching automaticamente
 */
class UltraPerformanceCacheManager {
  constructor(defaultTTL = 3600) {
    this.defaultTTL = defaultTTL;
    this.keyPrefix = 'crmbet:';
    this.multiCache = multiCache;
  }

  /**
   * Gerar chave com prefixo
   */
  getKey(key) {
    return `${this.keyPrefix}${key}`;
  }

  /**
   * Definir valor no cache (todas as camadas)
   */
  async set(key, value, ttl = this.defaultTTL) {
    try {
      const cacheKey = this.getKey(key);
      const result = await this.multiCache.set(cacheKey, value, ttl);
      
      if (result) {
        logger.debug(`Ultra cache set: ${key} (TTL: ${ttl}s, All layers)`);
      }
      
      return result;
    } catch (error) {
      logger.error('Ultra cache set error:', error);
      return false;
    }
  }

  /**
   * Obter valor do cache (multi-layer lookup)
   */
  async get(key) {
    try {
      const cacheKey = this.getKey(key);
      const value = await this.multiCache.get(cacheKey);
      
      if (value === null) {
        logger.debug(`Ultra cache miss: ${key}`);
      } else {
        logger.debug(`Ultra cache hit: ${key}`);
      }
      
      return value;
    } catch (error) {
      logger.error('Ultra cache get error:', error);
      return null;
    }
  }

  /**
   * Deletar do cache (todas as camadas)
   */
  async del(key) {
    try {
      const cacheKey = this.getKey(key);
      const result = await this.multiCache.del(cacheKey);
      
      if (result) {
        logger.debug(`Ultra cache delete: ${key} (All layers)`);
      }
      
      return result;
    } catch (error) {
      logger.error('Ultra cache delete error:', error);
      return false;
    }
  }

  /**
   * Verificar se existe no cache (L1 first, fallback para Redis)
   */
  async exists(key) {
    try {
      const cacheKey = this.getKey(key);
      
      // Check L1 first (fastest)
      if (this.multiCache.l1Cache.has(cacheKey)) {
        const item = this.multiCache.l1Cache.get(cacheKey);
        if (item.expires > Date.now()) {
          return true;
        }
      }
      
      // Fallback para Redis
      const result = await client.exists(cacheKey);
      return result === 1;
    } catch (error) {
      logger.error('Ultra cache exists error:', error);
      return false;
    }
  }

  /**
   * Incrementar valor (Redis direto para atomicidade)
   */
  async incr(key, amount = 1, ttl = this.defaultTTL) {
    try {
      const cacheKey = this.getKey(key);
      
      // Usar Redis direto para operações atômicas
      const result = await client.incrBy(cacheKey, amount);
      
      if (ttl > 0) {
        await client.expire(cacheKey, ttl);
      }
      
      // Invalidar caches em memória para consistência
      await this.multiCache.del(cacheKey);
      
      return result;
    } catch (error) {
      logger.error('Ultra cache incr error:', error);
      return null;
    }
  }

  /**
   * Definir TTL para chave existente
   */
  async expire(key, ttl) {
    try {
      const cacheKey = this.getKey(key);
      const result = await client.expire(cacheKey, ttl);
      return result === 1;
    } catch (error) {
      logger.error('Ultra cache expire error:', error);
      return false;
    }
  }

  /**
   * Obter múltiplas chaves (otimizado para pipeline)
   */
  async mget(keys) {
    try {
      // Tentar L1 cache primeiro
      const l1Results = [];
      const missedKeys = [];
      const missedIndexes = [];

      keys.forEach((key, index) => {
        const cacheKey = this.getKey(key);
        if (this.multiCache.l1Cache.has(cacheKey)) {
          const item = this.multiCache.l1Cache.get(cacheKey);
          if (item.expires > Date.now()) {
            l1Results[index] = item.value;
            return;
          }
        }
        l1Results[index] = null;
        missedKeys.push(cacheKey);
        missedIndexes.push(index);
      });

      // Para chaves não encontradas em L1, usar Redis pipeline
      if (missedKeys.length > 0) {
        const pipeline = client.pipeline ? client.pipeline() : client.multi();
        
        missedKeys.forEach(key => {
          pipeline.get(key);
        });
        
        const redisResults = await pipeline.exec();
        
        redisResults.forEach((result, index) => {
          const value = result[1];
          if (value !== null) {
            try {
              const parsed = JSON.parse(value);
              const originalIndex = missedIndexes[index];
              l1Results[originalIndex] = parsed;
              
              // Promover para L1
              const originalKey = keys[originalIndex];
              this.multiCache.setL1(this.getKey(originalKey), parsed, REDIS_ENTERPRISE_CONFIG.CACHE_LAYERS.L1_MEMORY.TTL);
            } catch (error) {
              logger.error(`Parse error for key ${missedKeys[index]}:`, error);
            }
          }
        });
      }

      return l1Results;
    } catch (error) {
      logger.error('Ultra cache mget error:', error);
      return keys.map(() => null);
    }
  }

  /**
   * Definir múltiplas chaves (pipeline otimizado)
   */
  async mset(keyValuePairs, ttl = this.defaultTTL) {
    try {
      // Set em todas as camadas
      const promises = Object.entries(keyValuePairs).map(([key, value]) => 
        this.set(key, value, ttl)
      );

      const results = await Promise.allSettled(promises);
      const successCount = results.filter(r => r.status === 'fulfilled' && r.value).length;
      
      logger.debug(`Ultra cache mset: ${successCount}/${Object.keys(keyValuePairs).length} successful`);
      return successCount === Object.keys(keyValuePairs).length;
    } catch (error) {
      logger.error('Ultra cache mset error:', error);
      return false;
    }
  }

  /**
   * Limpar cache por pattern (todas as camadas)
   */
  async clear(pattern = '*') {
    try {
      const cachePattern = this.getKey(pattern);
      const result = await this.multiCache.invalidatePattern(cachePattern);
      
      if (result) {
        logger.info(`Ultra cache pattern cleared: ${pattern}`);
      }
      
      return result;
    } catch (error) {
      logger.error('Ultra cache clear error:', error);
      return false;
    }
  }

  /**
   * Obter métricas de performance do cache
   */
  getMetrics() {
    return this.multiCache.getMetrics();
  }

  /**
   * Cache warming - preload de dados críticos
   */
  async warmCache(warmingData) {
    try {
      logger.info('Starting cache warming...');
      
      const promises = Object.entries(warmingData).map(([key, data]) => {
        const { value, ttl = this.defaultTTL } = data;
        return this.set(key, value, ttl);
      });

      const results = await Promise.allSettled(promises);
      const successCount = results.filter(r => r.status === 'fulfilled' && r.value).length;
      
      logger.info(`Cache warming completed: ${successCount}/${Object.keys(warmingData).length} items warmed`);
      return successCount;
    } catch (error) {
      logger.error('Cache warming failed:', error);
      return 0;
    }
  }
}

// Instância global do ultra-performance cache manager
const cache = new UltraPerformanceCacheManager();

/**
 * Health check ultra-completo do sistema Redis
 */
const healthCheck = async () => {
  const start = Date.now();
  const health = {
    status: 'healthy',
    timestamp: new Date().toISOString(),
    duration: 0,
    components: {}
  };

  try {
    // Cluster health check
    health.components.cluster = await clusterManager.healthCheck();
    
    // Multi-layer cache metrics
    health.components.cacheMetrics = cache.getMetrics();
    
    // Detailed performance metrics
    if (!clusterManager.isClusterMode) {
      const info = await client.info('memory');
      const cpuInfo = await client.info('cpu');
      const statsInfo = await client.info('stats');
      
      health.components.performance = {
        memory: clusterManager.parseRedisInfo(info),
        cpu: clusterManager.parseRedisInfo(cpuInfo),
        stats: clusterManager.parseRedisInfo(statsInfo)
      };
    }
    
    // L1 Cache health
    health.components.l1Cache = {
      size: multiCache.l1Cache.size,
      maxSize: REDIS_ENTERPRISE_CONFIG.CACHE_LAYERS.L1_MEMORY.MAX_SIZE,
      usage: `${((multiCache.l1Cache.size / REDIS_ENTERPRISE_CONFIG.CACHE_LAYERS.L1_MEMORY.MAX_SIZE) * 100).toFixed(1)}%`
    };

  } catch (error) {
    health.status = 'unhealthy';
    health.error = error.message;
  }

  health.duration = Date.now() - start;
  return health;
};

/**
 * Enterprise Pub/Sub para comunicação massiva entre serviços
 */
class EnterprisePubSub {
  constructor() {
    this.subscribers = new Map();
    this.publisher = client;
    this.isClusterMode = clusterManager.isClusterMode;
    this.messageBuffer = [];
    this.batchSize = REDIS_ENTERPRISE_CONFIG.PERFORMANCE.PIPELINE_BATCH_SIZE;
    this.setupBatchProcessing();
  }

  setupBatchProcessing() {
    // Process batched messages every 10ms for ultra-performance
    setInterval(() => {
      this.processBatchedMessages();
    }, 10);
  }

  async subscribe(channel, callback) {
    try {
      // Create dedicated subscriber for this channel
      const subscriber = this.isClusterMode ? 
        clusterManager.cluster.duplicate() : 
        client.duplicate();
      
      if (!this.isClusterMode) {
        await subscriber.connect();
      }

      await subscriber.subscribe(channel, (message, receivedChannel) => {
        try {
          const parsedMessage = JSON.parse(message);
          callback(parsedMessage, receivedChannel);
        } catch (error) {
          logger.error('Message parse error:', error);
          callback(message, receivedChannel);
        }
      });

      this.subscribers.set(channel, subscriber);
      logger.info(`Enterprise pub/sub subscribed to channel: ${channel}`);
    } catch (error) {
      logger.error('Enterprise subscribe error:', error);
    }
  }

  async publish(channel, message, options = {}) {
    try {
      const { batch = false, priority = 'normal' } = options;
      
      const messageData = {
        channel,
        message: JSON.stringify(message),
        timestamp: Date.now(),
        priority
      };

      if (batch) {
        this.messageBuffer.push(messageData);
        
        if (this.messageBuffer.length >= this.batchSize) {
          await this.processBatchedMessages();
        }
      } else {
        await this.publisher.publish(channel, messageData.message);
        logger.debug(`Enterprise pub/sub published to channel: ${channel}`);
      }
    } catch (error) {
      logger.error('Enterprise publish error:', error);
    }
  }

  async processBatchedMessages() {
    if (this.messageBuffer.length === 0) return;

    try {
      // Sort by priority and timestamp
      this.messageBuffer.sort((a, b) => {
        const priorityOrder = { high: 3, normal: 2, low: 1 };
        const aPriority = priorityOrder[a.priority] || 2;
        const bPriority = priorityOrder[b.priority] || 2;
        
        if (aPriority !== bPriority) {
          return bPriority - aPriority;
        }
        return a.timestamp - b.timestamp;
      });

      // Use pipeline for batch publishing
      const pipeline = this.publisher.pipeline ? this.publisher.pipeline() : this.publisher.multi();
      
      this.messageBuffer.forEach(({ channel, message }) => {
        pipeline.publish(channel, message);
      });

      await pipeline.exec();
      
      logger.debug(`Enterprise pub/sub processed ${this.messageBuffer.length} batched messages`);
      this.messageBuffer = [];
    } catch (error) {
      logger.error('Batched message processing error:', error);
    }
  }

  async unsubscribe(channel) {
    try {
      const subscriber = this.subscribers.get(channel);
      if (subscriber) {
        await subscriber.unsubscribe(channel);
        await subscriber.quit();
        this.subscribers.delete(channel);
        logger.info(`Enterprise pub/sub unsubscribed from channel: ${channel}`);
      }
    } catch (error) {
      logger.error('Enterprise unsubscribe error:', error);
    }
  }

  async publishBroadcast(message, excludeChannels = []) {
    try {
      // Broadcast to all known subscribers except excluded ones
      const channels = Array.from(this.subscribers.keys()).filter(
        channel => !excludeChannels.includes(channel)
      );

      const promises = channels.map(channel => 
        this.publish(channel, message, { batch: true })
      );

      await Promise.allSettled(promises);
      logger.info(`Enterprise broadcast sent to ${channels.length} channels`);
    } catch (error) {
      logger.error('Enterprise broadcast error:', error);
    }
  }

  getMetrics() {
    return {
      activeSubscribers: this.subscribers.size,
      bufferedMessages: this.messageBuffer.length,
      batchSize: this.batchSize,
      subscriberChannels: Array.from(this.subscribers.keys())
    };
  }
}

const pubsub = new EnterprisePubSub();

/**
 * Cache warming automático no startup
 */
const performCacheWarming = async () => {
  try {
    logger.info('Starting automatic cache warming...');
    
    // Dados críticos para warming
    const criticalData = {
      'system:config': { 
        value: { initialized: true, timestamp: Date.now() },
        ttl: 3600 
      },
      'system:health': { 
        value: { status: 'warming', timestamp: Date.now() },
        ttl: 300 
      }
    };

    await cache.warmCache(criticalData);
    logger.info('Automatic cache warming completed');
  } catch (error) {
    logger.error('Cache warming failed:', error);
  }
};

// Executar cache warming no startup
setTimeout(performCacheWarming, 5000);

module.exports = {
  // Core clients
  redis: client,
  clusterManager,
  
  // Connection
  connectRedis,
  
  // Ultra-performance cache
  cache,
  multiCache,
  
  // Enterprise pub/sub
  pubsub,
  
  // Health & monitoring
  healthCheck,
  
  // Configuration
  REDIS_ENTERPRISE_CONFIG,
  
  // Utilities
  performCacheWarming,
  
  // Legacy compatibility
  client // Para compatibilidade com código existente
};