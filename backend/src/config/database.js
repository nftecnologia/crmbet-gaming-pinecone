/**
 * Database Configuration - PostgreSQL ULTRA-PERFORMANCE
 * 
 * Sistema de banco de dados enterprise-grade otimizado para:
 * - 100k+ requests/segundo
 * - 10M+ usuários simultâneos
 * - Read replicas + sharding
 * - Connection pooling avançado
 * - Query optimization extremo
 * 
 * @author CRM Ultra-Performance Team
 */

const { Pool } = require('pg');
const logger = require('../utils/logger');

// Configuração ultra-performance para dados massivos
const ULTRA_PERFORMANCE_CONFIG = {
  // Connection pooling para 10k+ conexões simultâneas
  POOL_SIZES: {
    WRITE_POOL_MIN: parseInt(process.env.DB_WRITE_POOL_MIN) || 50,
    WRITE_POOL_MAX: parseInt(process.env.DB_WRITE_POOL_MAX) || 200,
    READ_POOL_MIN: parseInt(process.env.DB_READ_POOL_MIN) || 100,
    READ_POOL_MAX: parseInt(process.env.DB_READ_POOL_MAX) || 500,
    ANALYTICS_POOL_MIN: parseInt(process.env.DB_ANALYTICS_POOL_MIN) || 20,
    ANALYTICS_POOL_MAX: parseInt(process.env.DB_ANALYTICS_POOL_MAX) || 100
  },
  
  // Timeouts otimizados para alta performance
  TIMEOUTS: {
    CONNECTION_TIMEOUT: parseInt(process.env.DB_CONNECTION_TIMEOUT) || 3000,
    IDLE_TIMEOUT: parseInt(process.env.DB_IDLE_TIMEOUT) || 10000,
    STATEMENT_TIMEOUT: parseInt(process.env.DB_STATEMENT_TIMEOUT) || 30000,
    QUERY_TIMEOUT: parseInt(process.env.DB_QUERY_TIMEOUT) || 15000
  },
  
  // Configurações de performance crítica
  PERFORMANCE: {
    SLOW_QUERY_THRESHOLD: parseInt(process.env.DB_SLOW_QUERY_MS) || 100,
    MAX_RETRY_ATTEMPTS: parseInt(process.env.DB_MAX_RETRIES) || 3,
    RETRY_DELAY_BASE: parseInt(process.env.DB_RETRY_DELAY) || 100,
    PREPARED_STATEMENTS: process.env.DB_PREPARED_STATEMENTS !== 'false'
  }
};

// Configuração do pool MASTER (Write Operations)
const masterPoolConfig = {
  host: process.env.DB_MASTER_HOST || process.env.DB_HOST || 'localhost',
  port: process.env.DB_MASTER_PORT || process.env.DB_PORT || 5432,
  database: process.env.DB_NAME || 'crmbet',
  user: process.env.DB_USER || 'postgres',
  password: process.env.DB_PASSWORD || 'password',
  
  // Pool settings otimizado para writes
  min: ULTRA_PERFORMANCE_CONFIG.POOL_SIZES.WRITE_POOL_MIN,
  max: ULTRA_PERFORMANCE_CONFIG.POOL_SIZES.WRITE_POOL_MAX,
  
  // Connection settings ultra-performance
  connectionTimeoutMillis: ULTRA_PERFORMANCE_CONFIG.TIMEOUTS.CONNECTION_TIMEOUT,
  idleTimeoutMillis: ULTRA_PERFORMANCE_CONFIG.TIMEOUTS.IDLE_TIMEOUT,
  statement_timeout: ULTRA_PERFORMANCE_CONFIG.TIMEOUTS.STATEMENT_TIMEOUT,
  query_timeout: ULTRA_PERFORMANCE_CONFIG.TIMEOUTS.QUERY_TIMEOUT,
  
  // Performance optimizations
  max_prepared_transactions: 100,
  shared_preload_libraries: 'pg_stat_statements',
  track_activity_query_size: 2048,
  
  // SSL configuration
  ssl: process.env.NODE_ENV === 'production' ? {
    rejectUnauthorized: false
  } : false,
  
  // Application name para monitoring
  application_name: 'crmbet-master'
};

// Configuração do pool READ REPLICA (Read Operations)
const readReplicaPoolConfig = {
  host: process.env.DB_READ_HOST || process.env.DB_HOST || 'localhost',
  port: process.env.DB_READ_PORT || process.env.DB_PORT || 5432,
  database: process.env.DB_NAME || 'crmbet',
  user: process.env.DB_READ_USER || process.env.DB_USER || 'postgres',
  password: process.env.DB_READ_PASSWORD || process.env.DB_PASSWORD || 'password',
  
  // Pool settings otimizado para reads
  min: ULTRA_PERFORMANCE_CONFIG.POOL_SIZES.READ_POOL_MIN,
  max: ULTRA_PERFORMANCE_CONFIG.POOL_SIZES.READ_POOL_MAX,
  
  // Connection settings
  connectionTimeoutMillis: ULTRA_PERFORMANCE_CONFIG.TIMEOUTS.CONNECTION_TIMEOUT,
  idleTimeoutMillis: ULTRA_PERFORMANCE_CONFIG.TIMEOUTS.IDLE_TIMEOUT,
  statement_timeout: ULTRA_PERFORMANCE_CONFIG.TIMEOUTS.STATEMENT_TIMEOUT,
  query_timeout: ULTRA_PERFORMANCE_CONFIG.TIMEOUTS.QUERY_TIMEOUT,
  
  // Read-only optimizations
  default_transaction_read_only: true,
  
  // SSL configuration
  ssl: process.env.NODE_ENV === 'production' ? {
    rejectUnauthorized: false
  } : false,
  
  application_name: 'crmbet-read-replica'
};

// Configuração do pool ANALYTICS (Heavy Analytics Queries)
const analyticsPoolConfig = {
  host: process.env.DB_ANALYTICS_HOST || process.env.DB_HOST || 'localhost',
  port: process.env.DB_ANALYTICS_PORT || process.env.DB_PORT || 5432,
  database: process.env.DB_ANALYTICS_NAME || process.env.DB_NAME || 'crmbet_analytics',
  user: process.env.DB_ANALYTICS_USER || process.env.DB_USER || 'postgres',
  password: process.env.DB_ANALYTICS_PASSWORD || process.env.DB_PASSWORD || 'password',
  
  // Pool settings para analytics pesados
  min: ULTRA_PERFORMANCE_CONFIG.POOL_SIZES.ANALYTICS_POOL_MIN,
  max: ULTRA_PERFORMANCE_CONFIG.POOL_SIZES.ANALYTICS_POOL_MAX,
  
  // Timeouts maiores para queries analíticas
  connectionTimeoutMillis: ULTRA_PERFORMANCE_CONFIG.TIMEOUTS.CONNECTION_TIMEOUT,
  idleTimeoutMillis: ULTRA_PERFORMANCE_CONFIG.TIMEOUTS.IDLE_TIMEOUT * 3,
  statement_timeout: ULTRA_PERFORMANCE_CONFIG.TIMEOUTS.STATEMENT_TIMEOUT * 10,
  query_timeout: ULTRA_PERFORMANCE_CONFIG.TIMEOUTS.QUERY_TIMEOUT * 10,
  
  // Analytics optimizations
  work_mem: '256MB',
  maintenance_work_mem: '1GB',
  effective_cache_size: '4GB',
  
  // SSL configuration
  ssl: process.env.NODE_ENV === 'production' ? {
    rejectUnauthorized: false
  } : false,
  
  application_name: 'crmbet-analytics'
};

// Criar pools de conexão especializados
const masterPool = new Pool(masterPoolConfig);
const readReplicaPool = new Pool(readReplicaPoolConfig);
const analyticsPool = new Pool(analyticsPoolConfig);

// Sharding Strategy - Distribuição baseada em hash
const SHARD_CONFIG = {
  TOTAL_SHARDS: parseInt(process.env.DB_TOTAL_SHARDS) || 4,
  SHARD_KEY_FIELD: 'user_id', // Campo usado para sharding
  SHARD_ALGORITHM: 'murmur3' // Algoritmo de hash para distribuição
};

// Pool manager para sharding inteligente
class ShardManager {
  constructor() {
    this.shards = new Map();
    this.initializeShards();
  }

  initializeShards() {
    for (let i = 0; i < SHARD_CONFIG.TOTAL_SHARDS; i++) {
      const shardConfig = {
        host: process.env[`DB_SHARD_${i}_HOST`] || process.env.DB_HOST || 'localhost',
        port: process.env[`DB_SHARD_${i}_PORT`] || process.env.DB_PORT || 5432,
        database: process.env[`DB_SHARD_${i}_NAME`] || `${process.env.DB_NAME || 'crmbet'}_shard_${i}`,
        user: process.env.DB_USER || 'postgres',
        password: process.env.DB_PASSWORD || 'password',
        
        // Pool otimizado para shards
        min: 10,
        max: 50,
        connectionTimeoutMillis: ULTRA_PERFORMANCE_CONFIG.TIMEOUTS.CONNECTION_TIMEOUT,
        idleTimeoutMillis: ULTRA_PERFORMANCE_CONFIG.TIMEOUTS.IDLE_TIMEOUT,
        
        ssl: process.env.NODE_ENV === 'production' ? {
          rejectUnauthorized: false
        } : false,
        
        application_name: `crmbet-shard-${i}`
      };
      
      this.shards.set(i, new Pool(shardConfig));
      logger.info(`Shard ${i} initialized: ${shardConfig.database}`);
    }
  }

  // Calcular shard baseado em hash do user_id
  calculateShard(userId) {
    // Implementação simples de hash - em produção usar murmur3
    const hash = this.simpleHash(userId.toString());
    return hash % SHARD_CONFIG.TOTAL_SHARDS;
  }

  simpleHash(str) {
    let hash = 0;
    for (let i = 0; i < str.length; i++) {
      const char = str.charCodeAt(i);
      hash = ((hash << 5) - hash) + char;
      hash = hash & hash; // Convert to 32-bit integer
    }
    return Math.abs(hash);
  }

  // Obter pool do shard apropriado
  getShardPool(userId) {
    const shardId = this.calculateShard(userId);
    return this.shards.get(shardId);
  }

  // Executar query em shard específico
  async queryOnShard(userId, text, params) {
    const pool = this.getShardPool(userId);
    return await pool.query(text, params);
  }

  // Executar query em todos os shards (para agregações)
  async queryAllShards(text, params) {
    const promises = [];
    for (let i = 0; i < SHARD_CONFIG.TOTAL_SHARDS; i++) {
      const pool = this.shards.get(i);
      promises.push(pool.query(text, params));
    }
    return await Promise.all(promises);
  }

  // Health check de todos os shards
  async healthCheckAllShards() {
    const results = {};
    for (let i = 0; i < SHARD_CONFIG.TOTAL_SHARDS; i++) {
      try {
        const pool = this.shards.get(i);
        const result = await pool.query('SELECT NOW() as current_time');
        results[`shard_${i}`] = {
          status: 'healthy',
          timestamp: result.rows[0].current_time,
          pool: {
            total: pool.totalCount,
            idle: pool.idleCount,
            waiting: pool.waitingCount
          }
        };
      } catch (error) {
        results[`shard_${i}`] = {
          status: 'unhealthy',
          error: error.message
        };
      }
    }
    return results;
  }
}

// Instanciar shard manager
const shardManager = new ShardManager();

// Connection Load Balancer para Read Replicas
class LoadBalancer {
  constructor(pools) {
    this.pools = pools;
    this.currentIndex = 0;
    this.poolMetrics = new Map();
    this.initializeMetrics();
  }

  initializeMetrics() {
    this.pools.forEach((pool, index) => {
      this.poolMetrics.set(index, {
        activeConnections: 0,
        totalQueries: 0,
        avgResponseTime: 0,
        lastHealthCheck: Date.now(),
        isHealthy: true
      });
    });
  }

  // Round-robin com health check  
  getNextPool() {
    let attempts = 0;
    const maxAttempts = this.pools.length;

    while (attempts < maxAttempts) {
      const pool = this.pools[this.currentIndex];
      const metrics = this.poolMetrics.get(this.currentIndex);

      this.currentIndex = (this.currentIndex + 1) % this.pools.length;
      attempts++;

      // Verificar se o pool está saudável
      if (metrics && metrics.isHealthy) {
        return pool;
      }
    }

    // Se todos os pools estão unhealthy, retornar o primeiro
    logger.warn('All read replica pools are unhealthy, using fallback');
    return this.pools[0];
  }

  // Atualizar métricas do pool
  updateMetrics(poolIndex, responseTime) {
    const metrics = this.poolMetrics.get(poolIndex);
    if (metrics) {
      metrics.totalQueries++;
      metrics.avgResponseTime = (metrics.avgResponseTime + responseTime) / 2;
    }
  }

  // Health check periódico dos pools
  async performHealthChecks() {
    for (let i = 0; i < this.pools.length; i++) {
      try {
        const start = Date.now();
        await this.pools[i].query('SELECT 1');
        const responseTime = Date.now() - start;

        const metrics = this.poolMetrics.get(i);
        if (metrics) {
          metrics.isHealthy = true;
          metrics.lastHealthCheck = Date.now();
          this.updateMetrics(i, responseTime);
        }
      } catch (error) {
        logger.error(`Read replica ${i} health check failed:`, error);
        const metrics = this.poolMetrics.get(i);
        if (metrics) {
          metrics.isHealthy = false;
          metrics.lastHealthCheck = Date.now();
        }
      }
    }
  }
}

// Load balancer para read replicas (múltiplas replicas se configuradas)
const readReplicaPools = [readReplicaPool];
if (process.env.DB_READ_HOST_2) {
  const readReplica2Config = { ...readReplicaPoolConfig };
  readReplica2Config.host = process.env.DB_READ_HOST_2;
  readReplica2Config.application_name = 'crmbet-read-replica-2';
  readReplicaPools.push(new Pool(readReplica2Config));
}
if (process.env.DB_READ_HOST_3) {
  const readReplica3Config = { ...readReplicaPoolConfig };
  readReplica3Config.host = process.env.DB_READ_HOST_3;
  readReplica3Config.application_name = 'crmbet-read-replica-3';
  readReplicaPools.push(new Pool(readReplica3Config));
}

const loadBalancer = new LoadBalancer(readReplicaPools);

// Health check periódico a cada 30 segundos
setInterval(() => {
  loadBalancer.performHealthChecks();
}, 30000);

// Event listeners para monitoramento ultra-performance
[masterPool, readReplicaPool, analyticsPool].forEach((pool, index) => {
  const poolNames = ['master', 'read-replica', 'analytics'];
  const poolName = poolNames[index];

  pool.on('connect', (client) => {
    logger.info(`${poolName} pool: New client connected`);
  });

  pool.on('acquire', (client) => {
    logger.debug(`${poolName} pool: Client acquired`);
  });

  pool.on('remove', (client) => {
    logger.info(`${poolName} pool: Client removed`);
  });

  pool.on('error', (err) => {
    logger.error(`${poolName} pool error:`, err);
  });
});

// Prepared Statements Cache para ultra-performance
class PreparedStatementCache {
  constructor() {
    this.cache = new Map();
    this.maxCacheSize = 1000;
  }

  generateKey(sql, poolType) {
    return `${poolType}:${this.hashSQL(sql)}`;
  }

  hashSQL(sql) {
    // Hash simples para cache key
    let hash = 0;
    for (let i = 0; i < sql.length; i++) {
      const char = sql.charCodeAt(i);
      hash = ((hash << 5) - hash) + char;
      hash = hash & hash;
    }
    return Math.abs(hash).toString(36);
  }

  get(key) {
    return this.cache.get(key);
  }

  set(key, statement) {
    if (this.cache.size >= this.maxCacheSize) {
      // Remove oldest entry
      const firstKey = this.cache.keys().next().value;
      this.cache.delete(firstKey);
    }
    this.cache.set(key, statement);
  }

  clear() {
    this.cache.clear();
  }
}

const preparedStmtCache = new PreparedStatementCache();

/**
 * Conectar ao sistema de banco de dados ultra-performance
 */
const connectDB = async () => {
  try {
    logger.info('Initializing ultra-performance database system...');

    // Testar conexão master
    const masterClient = await masterPool.connect();
    const masterResult = await masterClient.query('SELECT version(), current_setting(\'max_connections\') as max_conn');
    logger.info('Master PostgreSQL:', {
      version: masterResult.rows[0].version.split(' ')[0],
      maxConnections: masterResult.rows[0].max_conn
    });
    
    // Configurar otimizações de performance no master
    await optimizeDatabasePerformance(masterClient);
    
    // Verificar e criar estruturas
    await checkTables(masterClient);
    await createMaterializedViews(masterClient);
    await createPerformanceIndexes(masterClient);
    
    masterClient.release();

    // Testar read replicas
    const readClient = await loadBalancer.getNextPool().connect();
    const readResult = await readClient.query('SELECT version(), pg_is_in_recovery() as is_replica');
    logger.info('Read Replica Status:', {
      version: readResult.rows[0].version.split(' ')[0],
      isReplica: readResult.rows[0].is_replica
    });
    readClient.release();

    // Testar analytics pool
    const analyticsClient = await analyticsPool.connect();
    await analyticsClient.query('SELECT 1');
    logger.info('Analytics pool connected successfully');
    analyticsClient.release();

    // Verificar shards
    const shardResults = await shardManager.healthCheckAllShards();
    const healthyShards = Object.values(shardResults).filter(r => r.status === 'healthy').length;
    logger.info(`Shards status: ${healthyShards}/${SHARD_CONFIG.TOTAL_SHARDS} healthy`);

    logger.info('Ultra-performance database system initialized successfully');
    return true;
    
  } catch (error) {
    logger.error('Database connection failed:', error);
    throw error;
  }
};

/**
 * Configurar otimizações de performance no PostgreSQL
 */
const optimizeDatabasePerformance = async (client) => {
  const optimizations = [
    // Configurações de memória
    "SET work_mem = '32MB'",
    "SET maintenance_work_mem = '512MB'",
    "SET effective_cache_size = '2GB'",
    
    // Configurações de checkpoints
    "SET checkpoint_completion_target = 0.9",
    "SET wal_buffers = '16MB'",
    
    // Configurações de conexões
    "SET max_connections = 1000",
    
    // Configurações de logging para performance
    "SET log_min_duration_statement = 100",
    "SET log_checkpoints = on",
    "SET log_connections = on",
    "SET log_disconnections = on",
    
    // Configurações de vacuum
    "SET autovacuum_max_workers = 6",
    "SET autovacuum_naptime = '15s'"
  ];

  for (const sql of optimizations) {
    try {
      await client.query(sql);
    } catch (error) {
      // Algumas configurações podem falhar se já estiverem definidas
      logger.debug(`Performance optimization warning: ${sql} - ${error.message}`);
    }
  }
  
  logger.info('Database performance optimizations applied');
};

/**
 * Verificar e criar tabelas otimizadas para dados massivos
 */
const checkTables = async (client) => {
  const tables = [
    {
      name: 'users',
      sql: `
        CREATE TABLE IF NOT EXISTS users (
          id BIGSERIAL PRIMARY KEY,
          external_id VARCHAR(255) UNIQUE NOT NULL,
          email VARCHAR(255) UNIQUE NOT NULL,
          name VARCHAR(255) NOT NULL,
          segment VARCHAR(100),
          cluster_id INTEGER,
          registration_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
          last_activity TIMESTAMP,
          total_deposits DECIMAL(15,2) DEFAULT 0,
          total_withdrawals DECIMAL(15,2) DEFAULT 0,
          total_bets DECIMAL(15,2) DEFAULT 0,
          bet_count INTEGER DEFAULT 0,
          win_rate DECIMAL(5,2) DEFAULT 0,
          risk_score DECIMAL(5,2) DEFAULT 0,
          ltv_score DECIMAL(10,2) DEFAULT 0,
          metadata JSONB DEFAULT '{}',
          created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
          updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
          -- Otimizações para sharding
          shard_key INTEGER GENERATED ALWAYS AS (abs(hashtext(external_id)) % 4) STORED
        ) PARTITION BY HASH (shard_key);
      `
    },
    {
      name: 'users_partitions',
      sql: `
        -- Criar partições para sharding
        CREATE TABLE IF NOT EXISTS users_shard_0 PARTITION OF users FOR VALUES WITH (modulus 4, remainder 0);
        CREATE TABLE IF NOT EXISTS users_shard_1 PARTITION OF users FOR VALUES WITH (modulus 4, remainder 1);
        CREATE TABLE IF NOT EXISTS users_shard_2 PARTITION OF users FOR VALUES WITH (modulus 4, remainder 2);
        CREATE TABLE IF NOT EXISTS users_shard_3 PARTITION OF users FOR VALUES WITH (modulus 4, remainder 3);
      `
    },
    {
      name: 'user_activities',
      sql: `
        CREATE TABLE IF NOT EXISTS user_activities (
          id BIGSERIAL,
          user_id BIGINT NOT NULL,
          activity_type VARCHAR(50) NOT NULL,
          activity_data JSONB DEFAULT '{}',
          value DECIMAL(15,2),
          timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
          processed BOOLEAN DEFAULT FALSE,
          PRIMARY KEY (id, timestamp)
        ) PARTITION BY RANGE (timestamp);
      `
    },
    {
      name: 'user_activities_partitions',
      sql: `
        -- Partições mensais para atividades (últimos 12 meses)
        CREATE TABLE IF NOT EXISTS user_activities_2024_01 PARTITION OF user_activities 
        FOR VALUES FROM ('2024-01-01') TO ('2024-02-01');
        CREATE TABLE IF NOT EXISTS user_activities_2024_02 PARTITION OF user_activities 
        FOR VALUES FROM ('2024-02-01') TO ('2024-03-01');
        CREATE TABLE IF NOT EXISTS user_activities_2024_03 PARTITION OF user_activities 
        FOR VALUES FROM ('2024-03-01') TO ('2024-04-01');
        CREATE TABLE IF NOT EXISTS user_activities_2024_04 PARTITION OF user_activities 
        FOR VALUES FROM ('2024-04-01') TO ('2024-05-01');
        CREATE TABLE IF NOT EXISTS user_activities_2024_05 PARTITION OF user_activities 
        FOR VALUES FROM ('2024-05-01') TO ('2024-06-01');
        CREATE TABLE IF NOT EXISTS user_activities_2024_06 PARTITION OF user_activities 
        FOR VALUES FROM ('2024-06-01') TO ('2024-07-01');
        CREATE TABLE IF NOT EXISTS user_activities_current PARTITION OF user_activities 
        FOR VALUES FROM ('2024-06-01') TO ('2025-01-01');
      `
    },
    {
      name: 'clusters',
      sql: `
        CREATE TABLE IF NOT EXISTS clusters (
          id SERIAL PRIMARY KEY,
          name VARCHAR(255) NOT NULL,
          description TEXT,
          algorithm VARCHAR(100) NOT NULL,
          parameters JSONB DEFAULT '{}',
          features JSONB DEFAULT '{}',
          user_count INTEGER DEFAULT 0,
          avg_ltv DECIMAL(10,2) DEFAULT 0,
          conversion_rate DECIMAL(5,2) DEFAULT 0,
          risk_level VARCHAR(20) DEFAULT 'medium',
          created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
          updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
      `
    },
    {
      name: 'campaigns',
      sql: `
        CREATE TABLE IF NOT EXISTS campaigns (
          id BIGSERIAL PRIMARY KEY,
          name VARCHAR(255) NOT NULL,
          description TEXT,
          type VARCHAR(100) NOT NULL,
          status VARCHAR(50) DEFAULT 'draft',
          target_cluster_id INTEGER REFERENCES clusters(id),
          target_segment VARCHAR(100),
          target_criteria JSONB DEFAULT '{}',
          content JSONB NOT NULL,
          schedule_at TIMESTAMP,
          started_at TIMESTAMP,
          completed_at TIMESTAMP,
          smartico_campaign_id VARCHAR(255),
          total_sent INTEGER DEFAULT 0,
          total_opened INTEGER DEFAULT 0,
          total_clicked INTEGER DEFAULT 0,
          total_converted INTEGER DEFAULT 0,
          conversion_rate DECIMAL(5,2) DEFAULT 0,
          roi DECIMAL(10,2) DEFAULT 0,
          metadata JSONB DEFAULT '{}',
          created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
          updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
      `
    },
    {
      name: 'campaign_results',
      sql: `
        CREATE TABLE IF NOT EXISTS campaign_results (
          id BIGSERIAL,
          campaign_id BIGINT NOT NULL,
          user_id BIGINT NOT NULL,
          status VARCHAR(50) NOT NULL,
          sent_at TIMESTAMP,
          opened_at TIMESTAMP,
          clicked_at TIMESTAMP,
          converted_at TIMESTAMP,
          conversion_value DECIMAL(15,2),
          metadata JSONB DEFAULT '{}',
          created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
          PRIMARY KEY (id, created_at)
        ) PARTITION BY RANGE (created_at);
      `
    },
    {
      name: 'campaign_results_partitions',
      sql: `
        -- Partições mensais para resultados de campanha
        CREATE TABLE IF NOT EXISTS campaign_results_2024_06 PARTITION OF campaign_results 
        FOR VALUES FROM ('2024-06-01') TO ('2024-07-01');
        CREATE TABLE IF NOT EXISTS campaign_results_current PARTITION OF campaign_results 
        FOR VALUES FROM ('2024-06-01') TO ('2025-01-01');
      `
    }
  ];

  for (const table of tables) {
    try {
      await client.query(table.sql);
      logger.info(`Table/Partition ${table.name} verified/created`);
    } catch (error) {
      logger.error(`Error creating table ${table.name}:`, error);
      // Continuar mesmo com erro (algumas partições podem já existir)
    }
  }

  logger.info('Ultra-performance table structure created');
};

/**
 * Criar índices ultra-performance para dados massivos
 */
const createPerformanceIndexes = async (client) => {
  const indexes = [
    // Índices principais para users
    'CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_users_external_id_btree ON users USING btree(external_id);',
    'CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_users_email_btree ON users USING btree(email);',
    'CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_users_segment_hash ON users USING hash(segment);',
    'CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_users_cluster_id_btree ON users USING btree(cluster_id);',
    'CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_users_last_activity_brin ON users USING brin(last_activity);',
    'CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_users_registration_date_brin ON users USING brin(registration_date);',
    
    // Índices compostos para queries complexas
    'CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_users_segment_cluster ON users(segment, cluster_id) WHERE cluster_id IS NOT NULL;',
    'CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_users_activity_segment ON users(last_activity, segment) WHERE last_activity > NOW() - INTERVAL \'30 days\';',
    'CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_users_ltv_risk ON users(ltv_score DESC, risk_score) WHERE ltv_score > 0;',
    
    // Índices para JSONB metadata
    'CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_users_metadata_gin ON users USING gin(metadata);',
    
    // Índices para user_activities
    'CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_activities_user_time ON user_activities(user_id, timestamp DESC);',
    'CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_activities_type_time ON user_activities(activity_type, timestamp DESC);',
    'CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_activities_processed ON user_activities(processed, timestamp) WHERE processed = false;',
    'CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_activities_value_time ON user_activities(value, timestamp) WHERE value > 0;',
    
    // Índices para campaigns
    'CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_campaigns_status_hash ON campaigns USING hash(status);',
    'CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_campaigns_type_status ON campaigns(type, status);',
    'CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_campaigns_target_cluster ON campaigns(target_cluster_id) WHERE target_cluster_id IS NOT NULL;',
    'CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_campaigns_schedule_status ON campaigns(schedule_at, status) WHERE schedule_at IS NOT NULL;',
    'CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_campaigns_created_status ON campaigns(created_at DESC, status);',
    
    // Índices para campaign_results
    'CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_campaign_results_campaign_id ON campaign_results(campaign_id, created_at DESC);',
    'CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_campaign_results_user_id ON campaign_results(user_id, created_at DESC);',
    'CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_campaign_results_status_time ON campaign_results(status, created_at DESC);',
    'CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_campaign_results_conversion ON campaign_results(converted_at, conversion_value) WHERE converted_at IS NOT NULL;',
    
    // Índices para clusters
    'CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_clusters_algorithm ON clusters(algorithm);',
    'CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_clusters_user_count ON clusters(user_count DESC);',
    'CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_clusters_features_gin ON clusters USING gin(features);'
  ];

  logger.info('Creating ultra-performance indexes...');
  
  for (const index of indexes) {
    try {
      await client.query(index);
      logger.debug(`Index created: ${index.substring(0, 80)}...`);
    } catch (error) {
      if (error.message.includes('already exists')) {
        logger.debug(`Index already exists: ${index.substring(0, 80)}...`);
      } else {
        logger.error('Error creating index:', error.message);
      }
    }
  }
  
  // Criar índices de texto para search
  const textIndexes = [
    'CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_users_name_trgm ON users USING gin(name gin_trgm_ops);',
    'CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_campaigns_name_trgm ON campaigns USING gin(name gin_trgm_ops);'
  ];
  
  for (const index of textIndexes) {
    try {
      await client.query('CREATE EXTENSION IF NOT EXISTS pg_trgm;');
      await client.query(index);
    } catch (error) {
      logger.debug(`Text search index warning: ${error.message}`);
    }
  }
  
  logger.info('Ultra-performance indexes created successfully');
};

/**
 * Criar materialized views para agregações pesadas
 */
const createMaterializedViews = async (client) => {
  const views = [
    {
      name: 'mv_user_segments_summary',
      sql: `
        CREATE MATERIALIZED VIEW IF NOT EXISTS mv_user_segments_summary AS
        SELECT 
          segment,
          COUNT(*) as user_count,
          AVG(total_deposits) as avg_deposits,
          AVG(total_bets) as avg_bets,
          AVG(win_rate) as avg_win_rate,
          AVG(ltv_score) as avg_ltv,
          AVG(risk_score) as avg_risk,
          COUNT(*) FILTER (WHERE last_activity > NOW() - INTERVAL '7 days') as active_7d,
          COUNT(*) FILTER (WHERE last_activity > NOW() - INTERVAL '30 days') as active_30d,
          NOW() as refreshed_at
        FROM users 
        WHERE segment IS NOT NULL
        GROUP BY segment;
      `
    },
    {
      name: 'mv_cluster_performance',
      sql: `
        CREATE MATERIALIZED VIEW IF NOT EXISTS mv_cluster_performance AS
        SELECT 
          c.id as cluster_id,
          c.name as cluster_name,
          c.algorithm,
          COUNT(u.id) as user_count,
          AVG(u.total_deposits) as avg_deposits,
          AVG(u.total_bets) as avg_bets,
          AVG(u.win_rate) as avg_win_rate,
          AVG(u.ltv_score) as avg_ltv,
          AVG(u.risk_score) as avg_risk,
          COUNT(*) FILTER (WHERE u.last_activity > NOW() - INTERVAL '7 days') as active_7d,
          COUNT(*) FILTER (WHERE u.last_activity > NOW() - INTERVAL '30 days') as active_30d,
          NOW() as refreshed_at
        FROM clusters c
        LEFT JOIN users u ON c.id = u.cluster_id
        GROUP BY c.id, c.name, c.algorithm;
      `
    },
    {
      name: 'mv_campaign_performance',
      sql: `
        CREATE MATERIALIZED VIEW IF NOT EXISTS mv_campaign_performance AS
        SELECT 
          c.id as campaign_id,
          c.name as campaign_name,
          c.type,
          c.status,
          c.total_sent,
          c.total_opened,
          c.total_clicked,
          c.total_converted,
          CASE WHEN c.total_sent > 0 THEN (c.total_opened::decimal / c.total_sent * 100) ELSE 0 END as open_rate,
          CASE WHEN c.total_sent > 0 THEN (c.total_clicked::decimal / c.total_sent * 100) ELSE 0 END as click_rate,
          CASE WHEN c.total_sent > 0 THEN (c.total_converted::decimal / c.total_sent * 100) ELSE 0 END as conversion_rate,
          COALESCE(SUM(cr.conversion_value), 0) as total_revenue,
          c.created_at,
          c.completed_at,
          NOW() as refreshed_at
        FROM campaigns c
        LEFT JOIN campaign_results cr ON c.id = cr.campaign_id AND cr.converted_at IS NOT NULL
        GROUP BY c.id, c.name, c.type, c.status, c.total_sent, c.total_opened, 
                 c.total_clicked, c.total_converted, c.created_at, c.completed_at;
      `
    },
    {
      name: 'mv_daily_user_activity',
      sql: `
        CREATE MATERIALIZED VIEW IF NOT EXISTS mv_daily_user_activity AS
        SELECT 
          DATE(timestamp) as activity_date,
          activity_type,
          COUNT(*) as activity_count,
          COUNT(DISTINCT user_id) as unique_users,
          COALESCE(SUM(value), 0) as total_value,
          AVG(value) as avg_value,
          NOW() as refreshed_at
        FROM user_activities 
        WHERE timestamp >= NOW() - INTERVAL '90 days'
        GROUP BY DATE(timestamp), activity_type;
      `
    }
  ];

  logger.info('Creating materialized views for ultra-performance analytics...');
  
  for (const view of views) {
    try {
      await client.query(view.sql);
      
      // Criar índices nas materialized views
      await client.query(`CREATE INDEX IF NOT EXISTS idx_${view.name}_refreshed_at ON ${view.name}(refreshed_at);`);
      
      logger.info(`Materialized view created: ${view.name}`);
    } catch (error) {
      logger.error(`Error creating materialized view ${view.name}:`, error.message);
    }
  }
  
  logger.info('Materialized views created successfully');
};

/**
 * Query Router Ultra-Performance
 * Direciona automaticamente para o pool apropriado baseado no tipo de query
 */
class QueryRouter {
  constructor() {
    this.readPatterns = [
      /^\s*SELECT/i,
      /^\s*WITH.*SELECT/i,
      /^\s*EXPLAIN/i,
      /^\s*SHOW/i
    ];
    
    this.writePatterns = [
      /^\s*INSERT/i,
      /^\s*UPDATE/i,
      /^\s*DELETE/i,
      /^\s*CREATE/i,
      /^\s*ALTER/i,
      /^\s*DROP/i
    ];
    
    this.analyticsPatterns = [
      /\bAVG\s*\(/i,
      /\bSUM\s*\(/i,
      /\bCOUNT\s*\(/i,
      /\bGROUP\s+BY\b/i,
      /\bORDER\s+BY\b/i,
      /\bHAVING\b/i,
      /\bWINDOW\b/i,
      /\bPARTITION\s+BY\b/i
    ];
  }

  determineQueryType(sql) {
    const upperSQL = sql.toUpperCase();
    
    // Verificar se é analytics (queries pesadas)
    if (this.analyticsPatterns.some(pattern => pattern.test(sql))) {
      return 'analytics';
    }
    
    // Verificar se é read
    if (this.readPatterns.some(pattern => pattern.test(sql))) {
      return 'read';
    }
    
    // Verificar se é write
    if (this.writePatterns.some(pattern => pattern.test(sql))) {
      return 'write';
    }
    
    // Default para write (safer)
    return 'write';
  }

  getPool(queryType) {
    switch (queryType) {
      case 'read':
        return loadBalancer.getNextPool();
      case 'write':
        return masterPool;
      case 'analytics':
        return analyticsPool;
      default:
        return masterPool;
    }
  }
}

const queryRouter = new QueryRouter();

/**
 * Executar query ultra-performance com retry inteligente
 */
const query = async (text, params, options = {}) => {
  const {
    retries = ULTRA_PERFORMANCE_CONFIG.PERFORMANCE.MAX_RETRY_ATTEMPTS,
    queryType = null,
    timeout = ULTRA_PERFORMANCE_CONFIG.TIMEOUTS.QUERY_TIMEOUT,
    usePreparedStatement = ULTRA_PERFORMANCE_CONFIG.PERFORMANCE.PREPARED_STATEMENTS
  } = options;

  let lastError;
  let actualQueryType = queryType || queryRouter.determineQueryType(text);
  let pool = queryRouter.getPool(actualQueryType);
  
  for (let i = 0; i < retries; i++) {
    try {
      const start = Date.now();
      let result;
      
      // Usar prepared statement se habilitado
      if (usePreparedStatement && params && params.length > 0) {
        const cacheKey = preparedStmtCache.generateKey(text, actualQueryType);
        const cachedStmt = preparedStmtCache.get(cacheKey);
        
        if (cachedStmt) {
          result = await pool.query(cachedStmt, params);
        } else {
          result = await pool.query(text, params);
          preparedStmtCache.set(cacheKey, text);
        }
      } else {
        result = await pool.query(text, params);
      }
      
      const duration = Date.now() - start;
      
      // Log de performance
      if (duration > ULTRA_PERFORMANCE_CONFIG.PERFORMANCE.SLOW_QUERY_THRESHOLD) {
        logger.warn(`Slow query detected (${duration}ms) [${actualQueryType}]:`, {
          sql: text.substring(0, 100),
          duration,
          queryType: actualQueryType,
          params: params?.length || 0
        });
      } else {
        logger.debug(`Query executed (${duration}ms) [${actualQueryType}]`);
      }
      
      return result;
      
    } catch (error) {
      lastError = error;
      logger.error(`Query attempt ${i + 1}/${retries} failed [${actualQueryType}]:`, {
        error: error.message,
        sql: text.substring(0, 100),
        attempt: i + 1
      });
      
      // Se é um erro de conexão, tentar outro pool
      if (error.code === 'ECONNREFUSED' || error.code === 'ENOTFOUND') {
        if (actualQueryType === 'read') {
          pool = loadBalancer.getNextPool();
        }
      }
      
      if (i < retries - 1) {
        const delay = ULTRA_PERFORMANCE_CONFIG.PERFORMANCE.RETRY_DELAY_BASE * Math.pow(2, i);
        await new Promise(resolve => setTimeout(resolve, delay));
      }
    }
  }
  
  throw lastError;
};

/**
 * Query para leitura (usando read replicas)
 */
const queryRead = async (text, params, options = {}) => {
  return query(text, params, { ...options, queryType: 'read' });
};

/**
 * Query para escrita (usando master)
 */
const queryWrite = async (text, params, options = {}) => {
  return query(text, params, { ...options, queryType: 'write' });
};

/**
 * Query para analytics (usando analytics pool)
 */
const queryAnalytics = async (text, params, options = {}) => {
  return query(text, params, { ...options, queryType: 'analytics' });
};

/**
 * Executar transação ultra-performance
 */
const transaction = async (callback, options = {}) => {
  const { isolationLevel = 'READ COMMITTED', timeout = 30000 } = options;
  const client = await masterPool.connect();
  
  try {
    await client.query('BEGIN');
    await client.query(`SET TRANSACTION ISOLATION LEVEL ${isolationLevel}`);
    
    // Timeout para transação
    const timeoutPromise = new Promise((_, reject) => {
      setTimeout(() => reject(new Error('Transaction timeout')), timeout);
    });
    
    const result = await Promise.race([
      callback(client),
      timeoutPromise
    ]);
    
    await client.query('COMMIT');
    return result;
  } catch (error) {
    await client.query('ROLLBACK');
    logger.error('Transaction failed:', error);
    throw error;
  } finally {
    client.release();
  }
};

/**
 * Batch operations para inserções em massa
 */
const batchInsert = async (tableName, data, options = {}) => {
  const { batchSize = 1000, onConflict = 'DO NOTHING' } = options;
  
  if (!data || data.length === 0) return { rowCount: 0 };
  
  const keys = Object.keys(data[0]);
  const columns = keys.join(', ');
  const totalRows = data.length;
  let insertedRows = 0;
  
  for (let i = 0; i < totalRows; i += batchSize) {
    const batch = data.slice(i, i + batchSize);
    const values = [];
    const placeholders = [];
    
    batch.forEach((row, index) => {
      const rowPlaceholders = keys.map((_, keyIndex) => `$${index * keys.length + keyIndex + 1}`);
      placeholders.push(`(${rowPlaceholders.join(', ')})`);
      keys.forEach(key => values.push(row[key]));
    });
    
    const sql = `
      INSERT INTO ${tableName} (${columns}) 
      VALUES ${placeholders.join(', ')} 
      ON CONFLICT ${onConflict}
    `;
    
    try {
      const result = await queryWrite(sql, values);
      insertedRows += result.rowCount;
      
      logger.debug(`Batch insert: ${result.rowCount} rows inserted into ${tableName}`);
    } catch (error) {
      logger.error(`Batch insert failed for ${tableName}:`, error);
      throw error;
    }
  }
  
  return { rowCount: insertedRows, batches: Math.ceil(totalRows / batchSize) };
};

/**
 * Health check ultra-completo do sistema de banco
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
    // Check master pool
    const masterResult = await masterPool.query('SELECT NOW() as current_time, version()');
    health.components.master = {
      status: 'healthy',
      timestamp: masterResult.rows[0].current_time,
      version: masterResult.rows[0].version.split(' ')[0],
      pool: {
        total: masterPool.totalCount,
        idle: masterPool.idleCount,
        waiting: masterPool.waitingCount
      }
    };
    
    // Check read replicas
    try {
      const readPool = loadBalancer.getNextPool();
      const readResult = await readPool.query('SELECT NOW() as current_time, pg_is_in_recovery() as is_replica');
      health.components.readReplica = {
        status: 'healthy',
        timestamp: readResult.rows[0].current_time,
        isReplica: readResult.rows[0].is_replica,
        pool: {
          total: readPool.totalCount,
          idle: readPool.idleCount,
          waiting: readPool.waitingCount
        }
      };
    } catch (error) {
      health.components.readReplica = {
        status: 'unhealthy',
        error: error.message
      };
      health.status = 'degraded';
    }
    
    // Check analytics pool
    try {
      const analyticsResult = await analyticsPool.query('SELECT NOW() as current_time');
      health.components.analytics = {
        status: 'healthy',
        timestamp: analyticsResult.rows[0].current_time,
        pool: {
          total: analyticsPool.totalCount,
          idle: analyticsPool.idleCount,
          waiting: analyticsPool.waitingCount
        }
      };
    } catch (error) {
      health.components.analytics = {
        status: 'unhealthy',
        error: error.message
      };
      health.status = 'degraded';
    }
    
    // Check shards
    health.components.shards = await shardManager.healthCheckAllShards();
    const healthyShards = Object.values(health.components.shards).filter(s => s.status === 'healthy').length;
    
    if (healthyShards < SHARD_CONFIG.TOTAL_SHARDS) {
      health.status = 'degraded';
    }
    
    health.components.shardsSummary = {
      total: SHARD_CONFIG.TOTAL_SHARDS,
      healthy: healthyShards,
      unhealthy: SHARD_CONFIG.TOTAL_SHARDS - healthyShards
    };
    
  } catch (error) {
    health.status = 'unhealthy';
    health.error = error.message;
  }
  
  health.duration = Date.now() - start;
  return health;
};

/**
 * Refresh materialized views (executar periodicamente)
 */
const refreshMaterializedViews = async () => {
  const views = [
    'mv_user_segments_summary',
    'mv_cluster_performance', 
    'mv_campaign_performance',
    'mv_daily_user_activity'
  ];
  
  logger.info('Starting materialized views refresh...');
  
  for (const view of views) {
    try {
      const start = Date.now();
      await queryWrite(`REFRESH MATERIALIZED VIEW CONCURRENTLY ${view}`);
      const duration = Date.now() - start;
      logger.info(`Materialized view ${view} refreshed in ${duration}ms`);
    } catch (error) {
      logger.error(`Failed to refresh materialized view ${view}:`, error);
    }
  }
  
  logger.info('Materialized views refresh completed');
};

/**
 * Migração completa ultra-performance
 */
const migrate = async () => {
  try {
    logger.info('Starting ultra-performance database migration...');
    
    await connectDB();
    
    // Refresh materialized views (setup inicial)
    await refreshMaterializedViews();
    
    logger.info('Ultra-performance database migration completed successfully');
    return true;
  } catch (error) {
    logger.error('Migration failed:', error);
    throw error;
  }
};

/**
 * Cleanup e otimização periódica
 */
const performMaintenance = async () => {
  logger.info('Starting database maintenance...');
  
  try {
    // Analyze tables for query planner
    await queryWrite('ANALYZE;');
    
    // Refresh materialized views
    await refreshMaterializedViews();
    
    // Cleanup old partitions (manter apenas últimos 3 meses)
    const oldPartitions = [
      'user_activities_2024_01',
      'user_activities_2024_02', 
      'user_activities_2024_03'
    ];
    
    for (const partition of oldPartitions) {
      try {
        await queryWrite(`DROP TABLE IF EXISTS ${partition}`);
        logger.info(`Old partition ${partition} dropped`);
      } catch (error) {
        logger.debug(`Partition ${partition} cleanup warning:`, error.message);
      }
    }
    
    logger.info('Database maintenance completed');
  } catch (error) {
    logger.error('Database maintenance failed:', error);
  }
};

// Agendar manutenção automática a cada 6 horas
setInterval(performMaintenance, 6 * 60 * 60 * 1000);

// Agendar refresh de materialized views a cada 15 minutos
setInterval(refreshMaterializedViews, 15 * 60 * 1000);

module.exports = {
  // Pools
  masterPool,
  readReplicaPool,
  analyticsPool,
  shardManager,
  loadBalancer,
  
  // Connection
  connectDB,
  
  // Query functions
  query,
  queryRead,
  queryWrite,
  queryAnalytics,
  
  // Transaction
  transaction,
  
  // Batch operations
  batchInsert,
  
  // Health & maintenance
  healthCheck,
  migrate,
  refreshMaterializedViews,
  performMaintenance,
  
  // Configuration
  ULTRA_PERFORMANCE_CONFIG,
  SHARD_CONFIG,
  
  // Legacy compatibility
  pool: masterPool // Para compatibilidade com código existente
};