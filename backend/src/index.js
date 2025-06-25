/**
 * CRM Inteligente - Backend API Server
 * 
 * Servidor principal com Express, PostgreSQL, Redis e RabbitMQ
 * Integração com Smartico CRM via API
 * 
 * @author CRM Team
 * @version 1.0.0
 */

const express = require('express');
const cors = require('cors');
const helmet = require('helmet');
const morgan = require('morgan');
const compression = require('compression');
const swaggerUi = require('swagger-ui-express');
const swaggerJSDoc = require('swagger-jsdoc');
const cron = require('node-cron');

// Configurações
require('dotenv').config();
const { connectDB, pool } = require('./config/database');
const { connectRedis, redis } = require('./config/redis');
const { connectRabbitMQ, channel } = require('./config/rabbitmq');
const swaggerConfig = require('./config/swagger');

// Middleware
const authMiddleware = require('./middleware/auth');
const rateLimitMiddleware = require('./middleware/rateLimit');
const validationMiddleware = require('./middleware/validation');
const errorHandler = require('./middleware/errorHandler');

// Controllers
const userController = require('./controllers/userController');
const clusterController = require('./controllers/clusterController');
const campaignController = require('./controllers/campaignController');
const healthController = require('./controllers/healthController');

// Services
const smarticoService = require('./services/smarticoService');

// Utilitários
const logger = require('./utils/logger');

const app = express();
const PORT = process.env.PORT || 3000;

// ===== CONFIGURAÇÃO DE SEGURANÇA =====
app.use(helmet({
  contentSecurityPolicy: {
    directives: {
      defaultSrc: ["'self'"],
      styleSrc: ["'self'", "'unsafe-inline'", "https:"],
      scriptSrc: ["'self'", "https:"],
      imgSrc: ["'self'", "data:", "https:"],
    },
  },
  hsts: {
    maxAge: 31536000,
    includeSubDomains: true,
    preload: true
  }
}));

// ===== MIDDLEWARE GLOBAL =====
app.use(cors({
  origin: process.env.ALLOWED_ORIGINS?.split(',') || ['http://localhost:3001'],
  credentials: true,
  optionsSuccessStatus: 200
}));

app.use(compression());
app.use(express.json({ limit: '10mb' }));
app.use(express.urlencoded({ extended: true, limit: '10mb' }));

// Logging
app.use(morgan('combined', {
  stream: { write: message => logger.info(message.trim()) }
}));

// Rate limiting global
app.use(rateLimitMiddleware.globalLimit);

// ===== SWAGGER DOCUMENTATION =====
const swaggerSpec = swaggerJSDoc(swaggerConfig);
app.use('/api/docs', swaggerUi.serve, swaggerUi.setup(swaggerSpec, {
  customCss: '.swagger-ui .topbar { display: none }',
  customSiteTitle: 'CRM ML API Documentation'
}));

// ===== HEALTH CHECK =====
app.get('/health', healthController.healthCheck);
app.get('/api/v1/health', healthController.detailedHealth);

// ===== ROTAS DA API =====

// Webhook do Smartico (sem autenticação)
app.post('/api/v1/webhooks/smartico', 
  rateLimitMiddleware.webhookLimit,
  validationMiddleware.validateSmarticoWebhook,
  smarticoService.handleWebhook
);

// Rotas autenticadas
app.use('/api/v1', authMiddleware.authenticate);

// User routes
app.get('/api/v1/user/:id/segment', 
  validationMiddleware.validateUserId,
  userController.getUserSegment
);

// Cluster routes
app.get('/api/v1/clusters', 
  rateLimitMiddleware.apiLimit,
  clusterController.getClusters
);

app.get('/api/v1/clusters/:id/users', 
  validationMiddleware.validateClusterId,
  rateLimitMiddleware.apiLimit,
  clusterController.getClusterUsers
);

// Campaign routes
app.post('/api/v1/campaigns', 
  rateLimitMiddleware.campaignLimit,
  validationMiddleware.validateCampaign,
  campaignController.createCampaign
);

app.get('/api/v1/campaigns/:id/results', 
  validationMiddleware.validateCampaignId,
  rateLimitMiddleware.apiLimit,
  campaignController.getCampaignResults
);

app.get('/api/v1/campaigns', 
  rateLimitMiddleware.apiLimit,
  campaignController.getCampaigns
);

app.put('/api/v1/campaigns/:id', 
  validationMiddleware.validateCampaignId,
  validationMiddleware.validateCampaignUpdate,
  rateLimitMiddleware.campaignLimit,
  campaignController.updateCampaign
);

app.delete('/api/v1/campaigns/:id', 
  validationMiddleware.validateCampaignId,
  rateLimitMiddleware.campaignLimit,
  campaignController.deleteCampaign
);

// ===== ERROR HANDLING =====
// 404 Handler
app.use('*', (req, res) => {
  res.status(404).json({
    error: 'Endpoint not found',
    message: `Route ${req.method} ${req.originalUrl} not found`,
    timestamp: new Date().toISOString()
  });
});

// Global error handler
app.use(errorHandler.handle);

// ===== CRON JOBS =====
// Sincronização com Smartico a cada 5 minutos
cron.schedule('*/5 * * * *', async () => {
  try {
    await smarticoService.syncData();
    logger.info('Smartico sync completed successfully');
  } catch (error) {
    logger.error('Smartico sync failed:', error);
  }
});

// Limpeza de cache a cada hora
cron.schedule('0 * * * *', async () => {
  try {
    await redis.flushdb();
    logger.info('Cache cleanup completed');
  } catch (error) {
    logger.error('Cache cleanup failed:', error);
  }
});

// ===== GRACEFUL SHUTDOWN =====
const gracefulShutdown = async (signal) => {
  logger.info(`Received ${signal}. Starting graceful shutdown...`);
  
  server.close(async () => {
    try {
      // Fechar conexões
      if (pool) await pool.end();
      if (redis) await redis.quit();
      if (channel) await channel.close();
      
      logger.info('All connections closed successfully');
      process.exit(0);
    } catch (error) {
      logger.error('Error during shutdown:', error);
      process.exit(1);
    }
  });
  
  // Force shutdown após 10 segundos
  setTimeout(() => {
    logger.error('Force shutdown after timeout');
    process.exit(1);
  }, 10000);
};

process.on('SIGTERM', () => gracefulShutdown('SIGTERM'));
process.on('SIGINT', () => gracefulShutdown('SIGINT'));

// ===== STARTUP =====
const startServer = async () => {
  try {
    // Conectar ao banco de dados
    await connectDB();
    logger.info('PostgreSQL connected successfully');
    
    // Conectar ao Redis
    await connectRedis();
    logger.info('Redis connected successfully');
    
    // Conectar ao RabbitMQ
    await connectRabbitMQ();
    logger.info('RabbitMQ connected successfully');
    
    // Inicializar Smartico
    await smarticoService.initialize();
    logger.info('Smartico service initialized');
    
    // Iniciar servidor
    const server = app.listen(PORT, () => {
      logger.info(`🚀 CRM Backend API running on port ${PORT}`);
      logger.info(`📚 API Documentation: http://localhost:${PORT}/api/docs`);
      logger.info(`🔧 Health Check: http://localhost:${PORT}/health`);
    });
    
    // Timeout para requisições
    server.timeout = 30000;
    
    return server;
    
  } catch (error) {
    logger.error('Failed to start server:', error);
    process.exit(1);
  }
};

// Iniciar apenas se não estiver sendo importado
if (require.main === module) {
  startServer();
}

module.exports = app;