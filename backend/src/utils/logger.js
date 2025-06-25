/**
 * Logger Configuration - Winston
 * 
 * Sistema de logging robusto com Winston,
 * rotação de logs e diferentes níveis
 * 
 * @author CRM Team
 */

const winston = require('winston');
const path = require('path');

// Configurações do logger
const logConfig = {
  level: process.env.LOG_LEVEL || 'info',
  format: winston.format.combine(
    winston.format.timestamp({
      format: 'YYYY-MM-DD HH:mm:ss'
    }),
    winston.format.errors({ stack: true }),
    winston.format.json(),
    winston.format.prettyPrint()
  ),
  
  // Diretório de logs
  logDir: process.env.LOG_DIR || './logs',
  
  // Configurações de rotação
  rotation: {
    maxSize: '20m',
    maxFiles: '14d'
  }
};

// Criar diretório de logs se não existir
const fs = require('fs');
if (!fs.existsSync(logConfig.logDir)) {
  fs.mkdirSync(logConfig.logDir, { recursive: true });
}

// Transports
const transports = [
  // Console transport para desenvolvimento
  new winston.transports.Console({
    level: logConfig.level,
    format: winston.format.combine(
      winston.format.colorize(),
      winston.format.simple(),
      winston.format.printf(({ timestamp, level, message, ...meta }) => {
        let msg = `${timestamp} [${level}]: ${message}`;
        if (Object.keys(meta).length > 0) {
          msg += ` ${JSON.stringify(meta, null, 2)}`;
        }
        return msg;
      })
    )
  }),
  
  // File transport para todos os logs
  new winston.transports.File({
    filename: path.join(logConfig.logDir, 'combined.log'),
    level: 'info',
    maxsize: 20 * 1024 * 1024, // 20MB
    maxFiles: 5,
    format: logConfig.format
  }),
  
  // File transport para erros
  new winston.transports.File({
    filename: path.join(logConfig.logDir, 'error.log'),
    level: 'error',
    maxsize: 20 * 1024 * 1024, // 20MB
    maxFiles: 5,
    format: logConfig.format
  }),
  
  // File transport para warnings
  new winston.transports.File({
    filename: path.join(logConfig.logDir, 'warning.log'),
    level: 'warn',
    maxsize: 10 * 1024 * 1024, // 10MB
    maxFiles: 3,
    format: logConfig.format
  })
];

// Adicionar transport HTTP para produção (opcional)
if (process.env.LOG_HTTP_URL) {
  transports.push(
    new winston.transports.Http({
      host: process.env.LOG_HTTP_HOST,
      port: process.env.LOG_HTTP_PORT,
      path: process.env.LOG_HTTP_PATH,
      level: 'error'
    })
  );
}

// Criar logger
const logger = winston.createLogger({
  level: logConfig.level,
  format: logConfig.format,
  defaultMeta: {
    service: 'crmbet-backend',
    environment: process.env.NODE_ENV || 'development',
    version: process.env.npm_package_version || '1.0.0'
  },
  transports,
  
  // Exception handling
  exceptionHandlers: [
    new winston.transports.File({
      filename: path.join(logConfig.logDir, 'exceptions.log')
    })
  ],
  
  // Rejection handling
  rejectionHandlers: [
    new winston.transports.File({
      filename: path.join(logConfig.logDir, 'rejections.log')
    })
  ]
});

// Adicionar métodos customizados
logger.request = (req, message = 'HTTP Request') => {
  logger.info(message, {
    method: req.method,
    url: req.url,
    ip: req.ip,
    userAgent: req.get('User-Agent'),
    timestamp: new Date().toISOString()
  });
};

logger.response = (req, res, responseTime) => {
  logger.info('HTTP Response', {
    method: req.method,
    url: req.url,
    statusCode: res.statusCode,
    responseTime: `${responseTime}ms`,
    ip: req.ip,
    timestamp: new Date().toISOString()
  });
};

logger.database = (query, duration, error = null) => {
  if (error) {
    logger.error('Database Query Error', {
      query: query.substring(0, 200),
      duration: `${duration}ms`,
      error: error.message,
      stack: error.stack
    });
  } else {
    logger.debug('Database Query', {
      query: query.substring(0, 200),
      duration: `${duration}ms`
    });
  }
};

logger.security = (event, details = {}) => {
  logger.warn('Security Event', {
    event,
    ...details,
    timestamp: new Date().toISOString()
  });
};

logger.performance = (operation, duration, metadata = {}) => {
  const level = duration > 1000 ? 'warn' : 'info';
  logger[level]('Performance Metric', {
    operation,
    duration: `${duration}ms`,
    ...metadata
  });
};

logger.business = (event, data = {}) => {
  logger.info('Business Event', {
    event,
    ...data,
    timestamp: new Date().toISOString()
  });
};

logger.integration = (service, action, success, details = {}) => {
  const level = success ? 'info' : 'error';
  logger[level]('Integration Event', {
    service,
    action,
    success,
    ...details,
    timestamp: new Date().toISOString()
  });
};

// Stream para Morgan (HTTP logging)
logger.stream = {
  write: (message) => {
    logger.info(message.trim());
  }
};

// Função para logging estruturado
logger.structured = (level, message, metadata = {}) => {
  logger[level](message, {
    ...metadata,
    timestamp: new Date().toISOString(),
    pid: process.pid,
    hostname: require('os').hostname()
  });
};

// Middleware para capturar logs não tratados
if (process.env.NODE_ENV === 'production') {
  process.on('uncaughtException', (error) => {
    logger.error('Uncaught Exception', {
      error: error.message,
      stack: error.stack
    });
    process.exit(1);
  });
  
  process.on('unhandledRejection', (reason, promise) => {
    logger.error('Unhandled Rejection', {
      reason: reason.toString(),
      promise: promise.toString()
    });
  });
}

// Exportar logger
module.exports = logger;