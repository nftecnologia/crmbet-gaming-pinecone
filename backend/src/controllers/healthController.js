/**
 * Health Controller
 * 
 * Controller para health checks e monitoramento
 * do sistema e suas dependências
 * 
 * @author CRM Team
 */

const { healthCheck: dbHealthCheck } = require('../config/database');
const { healthCheck: redisHealthCheck } = require('../config/redis');
const { healthCheck: rabbitmqHealthCheck } = require('../config/rabbitmq');
const logger = require('../utils/logger');
const { asyncHandler } = require('../middleware/errorHandler');

/**
 * @swagger
 * /health:
 *   get:
 *     tags: [Health]
 *     summary: Health check básico
 *     description: Retorna status básico do sistema
 *     responses:
 *       200:
 *         description: Sistema saudável
 *         content:
 *           application/json:
 *             schema:
 *               type: object
 *               properties:
 *                 status:
 *                   type: string
 *                   example: "healthy"
 *                 timestamp:
 *                   type: string
 *                   format: date-time
 *                 uptime:
 *                   type: number
 *                   description: Uptime em segundos
 *       503:
 *         description: Sistema com problemas
 *         content:
 *           application/json:
 *             schema:
 *               $ref: '#/components/schemas/Error'
 */
const healthCheck = asyncHandler(async (req, res) => {
  const startTime = Date.now();
  
  try {
    const health = {
      status: 'healthy',
      timestamp: new Date().toISOString(),
      uptime: process.uptime(),
      environment: process.env.NODE_ENV || 'development',
      version: process.env.npm_package_version || '1.0.0',
      responseTime: `${Date.now() - startTime}ms`
    };
    
    res.status(200).json(health);
    
  } catch (error) {
    logger.error('Health check failed:', error);
    
    res.status(503).json({
      status: 'unhealthy',
      timestamp: new Date().toISOString(),
      error: error.message
    });
  }
});

/**
 * @swagger
 * /api/v1/health:
 *   get:
 *     tags: [Health]
 *     summary: Health check detalhado
 *     description: Retorna status detalhado do sistema e dependências
 *     security:
 *       - bearerAuth: []
 *     responses:
 *       200:
 *         description: Status detalhado do sistema
 *         content:
 *           application/json:
 *             schema:
 *               $ref: '#/components/schemas/HealthCheck'
 *       503:
 *         description: Sistema ou dependências com problemas
 *         content:
 *           application/json:
 *             schema:
 *               $ref: '#/components/schemas/Error'
 */
const detailedHealth = asyncHandler(async (req, res) => {
  const startTime = Date.now();
  const healthStatus = {
    status: 'healthy',
    timestamp: new Date().toISOString(),
    environment: process.env.NODE_ENV || 'development',
    version: process.env.npm_package_version || '1.0.0',
    uptime: process.uptime(),
    memory: {
      used: Math.round(process.memoryUsage().heapUsed / 1024 / 1024),
      total: Math.round(process.memoryUsage().heapTotal / 1024 / 1024),
      limit: Math.round((process.memoryUsage().heapTotal + process.memoryUsage().external) / 1024 / 1024)
    },
    cpu: {
      usage: process.cpuUsage(),
      loadAverage: require('os').loadavg()
    },
    services: {},
    responseTime: null
  };
  
  let overallHealthy = true;
  const serviceChecks = [];
  
  // Check Database
  serviceChecks.push(
    dbHealthCheck()
      .then(result => {
        healthStatus.services.database = result;
        if (result.status !== 'healthy') {
          overallHealthy = false;
        }
      })
      .catch(error => {
        healthStatus.services.database = {
          status: 'unhealthy',
          error: error.message
        };
        overallHealthy = false;
      })
  );
  
  // Check Redis
  serviceChecks.push(
    redisHealthCheck()
      .then(result => {
        healthStatus.services.redis = result;
        if (result.status !== 'healthy') {
          overallHealthy = false;
        }
      })
      .catch(error => {
        healthStatus.services.redis = {
          status: 'unhealthy',
          error: error.message
        };
        overallHealthy = false;
      })
  );
  
  // Check RabbitMQ
  serviceChecks.push(
    rabbitmqHealthCheck()
      .then(result => {
        healthStatus.services.rabbitmq = result;
        if (result.status !== 'healthy') {
          overallHealthy = false;
        }
      })
      .catch(error => {
        healthStatus.services.rabbitmq = {
          status: 'unhealthy',
          error: error.message
        };
        overallHealthy = false;
      })
  );
  
  // Aguardar todos os checks
  await Promise.all(serviceChecks);
  
  // Finalizar health check
  healthStatus.status = overallHealthy ? 'healthy' : 'unhealthy';
  healthStatus.responseTime = `${Date.now() - startTime}ms`;
  
  // Log do resultado
  logger.structured('info', 'Health check performed', {
    status: healthStatus.status,
    responseTime: healthStatus.responseTime,
    services: Object.keys(healthStatus.services).reduce((acc, service) => {
      acc[service] = healthStatus.services[service].status;
      return acc;
    }, {}),
    userId: req.user?.id
  });
  
  const statusCode = overallHealthy ? 200 : 503;
  res.status(statusCode).json(healthStatus);
});

/**
 * @swagger
 * /api/v1/health/metrics:
 *   get:
 *     tags: [Health]
 *     summary: Métricas do sistema
 *     description: Retorna métricas detalhadas de performance e uso
 *     security:
 *       - bearerAuth: []
 *     responses:
 *       200:
 *         description: Métricas do sistema
 *         content:
 *           application/json:
 *             schema:
 *               type: object
 *               properties:
 *                 timestamp:
 *                   type: string
 *                   format: date-time
 *                 process:
 *                   type: object
 *                   properties:
 *                     pid:
 *                       type: integer
 *                     uptime:
 *                       type: number
 *                     memory:
 *                       type: object
 *                     cpu:
 *                       type: object
 *                 system:
 *                   type: object
 *                   properties:
 *                     platform:
 *                       type: string
 *                     arch:
 *                       type: string
 *                     loadAverage:
 *                       type: array
 *                       items:
 *                         type: number
 *                     freeMemory:
 *                       type: number
 *                     totalMemory:
 *                       type: number
 */
const getMetrics = asyncHandler(async (req, res) => {
  const os = require('os');
  const process = require('process');
  
  const metrics = {
    timestamp: new Date().toISOString(),
    process: {
      pid: process.pid,
      uptime: process.uptime(),
      memory: {
        rss: process.memoryUsage().rss,
        heapTotal: process.memoryUsage().heapTotal,
        heapUsed: process.memoryUsage().heapUsed,
        external: process.memoryUsage().external,
        arrayBuffers: process.memoryUsage().arrayBuffers
      },
      cpu: process.cpuUsage(),
      versions: process.versions
    },
    system: {
      platform: os.platform(),
      arch: os.arch(),
      hostname: os.hostname(),
      loadAverage: os.loadavg(),
      freeMemory: os.freemem(),
      totalMemory: os.totalmem(),
      uptime: os.uptime(),
      cpus: os.cpus().length
    },
    node: {
      version: process.version,
      environment: process.env.NODE_ENV
    }
  };
  
  logger.debug('Metrics requested', {
    userId: req.user?.id,
    ip: req.ip
  });
  
  res.status(200).json(metrics);
});

/**
 * @swagger
 * /api/v1/health/readiness:
 *   get:
 *     tags: [Health]
 *     summary: Readiness probe
 *     description: Verifica se o sistema está pronto para receber tráfego
 *     responses:
 *       200:
 *         description: Sistema pronto
 *       503:
 *         description: Sistema não está pronto
 */
const readinessProbe = asyncHandler(async (req, res) => {
  const checks = [];
  let ready = true;
  
  // Verificar se todas as dependências estão funcionando
  try {
    // Database readiness
    const dbResult = await dbHealthCheck();
    checks.push({ service: 'database', status: dbResult.status });
    if (dbResult.status !== 'healthy') ready = false;
    
    // Redis readiness
    const redisResult = await redisHealthCheck();
    checks.push({ service: 'redis', status: redisResult.status });
    if (redisResult.status !== 'healthy') ready = false;
    
    // RabbitMQ readiness
    const rabbitmqResult = await rabbitmqHealthCheck();
    checks.push({ service: 'rabbitmq', status: rabbitmqResult.status });
    if (rabbitmqResult.status !== 'healthy') ready = false;
    
  } catch (error) {
    ready = false;
    logger.error('Readiness probe failed:', error);
  }
  
  const response = {
    ready,
    timestamp: new Date().toISOString(),
    checks
  };
  
  const statusCode = ready ? 200 : 503;
  res.status(statusCode).json(response);
});

/**
 * @swagger
 * /api/v1/health/liveness:
 *   get:
 *     tags: [Health]
 *     summary: Liveness probe
 *     description: Verifica se o sistema está ativo e responsivo
 *     responses:
 *       200:
 *         description: Sistema ativo
 *       503:
 *         description: Sistema com problemas
 */
const livenessProbe = asyncHandler(async (req, res) => {
  // Checks básicos de liveness
  const alive = process.uptime() > 0;
  const memoryUsage = process.memoryUsage();
  const memoryUsagePercent = (memoryUsage.heapUsed / memoryUsage.heapTotal) * 100;
  
  // Verificar se o uso de memória está muito alto
  const memoryThreshold = 90; // 90%
  const memoryOk = memoryUsagePercent < memoryThreshold;
  
  const response = {
    alive: alive && memoryOk,
    timestamp: new Date().toISOString(),
    uptime: process.uptime(),
    memory: {
      usagePercent: Math.round(memoryUsagePercent),
      threshold: memoryThreshold,
      ok: memoryOk
    }
  };
  
  const statusCode = (alive && memoryOk) ? 200 : 503;
  res.status(statusCode).json(response);
});

module.exports = {
  healthCheck,
  detailedHealth,
  getMetrics,
  readinessProbe,
  livenessProbe
};