/**
 * Error Handler Middleware
 * 
 * Sistema robusto de tratamento de erros com
 * logging, notificações e resposta padronizada
 * 
 * @author CRM Team
 */

const logger = require('../utils/logger');

/**
 * Tipos de erro customizados
 */
class AppError extends Error {
  constructor(message, statusCode = 500, code = 'INTERNAL_ERROR', details = null) {
    super(message);
    this.statusCode = statusCode;
    this.code = code;
    this.details = details;
    this.isOperational = true;
    
    Error.captureStackTrace(this, this.constructor);
  }
}

class ValidationError extends AppError {
  constructor(message, details = null) {
    super(message, 400, 'VALIDATION_ERROR', details);
  }
}

class AuthenticationError extends AppError {
  constructor(message = 'Authentication failed') {
    super(message, 401, 'AUTHENTICATION_ERROR');
  }
}

class AuthorizationError extends AppError {
  constructor(message = 'Insufficient permissions') {
    super(message, 403, 'AUTHORIZATION_ERROR');
  }
}

class NotFoundError extends AppError {
  constructor(resource = 'Resource') {
    super(`${resource} not found`, 404, 'NOT_FOUND');
  }
}

class ConflictError extends AppError {
  constructor(message, details = null) {
    super(message, 409, 'CONFLICT', details);
  }
}

class RateLimitError extends AppError {
  constructor(message = 'Rate limit exceeded') {
    super(message, 429, 'RATE_LIMIT_EXCEEDED');
  }
}

class ExternalServiceError extends AppError {
  constructor(service, message, originalError = null) {
    super(`${service} service error: ${message}`, 502, 'EXTERNAL_SERVICE_ERROR', {
      service,
      originalError: originalError?.message
    });
  }
}

/**
 * Mapear erros de biblioteca para erros customizados
 */
const mapLibraryErrors = (error) => {
  // Erros do PostgreSQL
  if (error.code) {
    switch (error.code) {
      case '23505': // unique_violation
        return new ConflictError('Resource already exists', {
          constraint: error.constraint,
          detail: error.detail
        });
      case '23503': // foreign_key_violation
        return new ValidationError('Referenced resource does not exist', {
          constraint: error.constraint,
          detail: error.detail
        });
      case '23502': // not_null_violation
        return new ValidationError('Required field is missing', {
          column: error.column,
          table: error.table
        });
      case '22001': // string_data_right_truncation
        return new ValidationError('Data too long for field', {
          column: error.column
        });
      default:
        return new AppError('Database error', 500, 'DATABASE_ERROR', {
          code: error.code,
          detail: error.detail
        });
    }
  }
  
  // Erros do JWT
  if (error.name === 'JsonWebTokenError') {
    return new AuthenticationError('Invalid token');
  }
  if (error.name === 'TokenExpiredError') {
    return new AuthenticationError('Token expired');
  }
  if (error.name === 'NotBeforeError') {
    return new AuthenticationError('Token not active');
  }
  
  // Erros do Joi
  if (error.isJoi) {
    const details = error.details.map(detail => ({
      field: detail.path.join('.'),
      message: detail.message,
      value: detail.context?.value
    }));
    return new ValidationError('Validation failed', details);
  }
  
  // Erros do Axios (chamadas HTTP externas)
  if (error.isAxiosError) {
    const service = error.config?.baseURL || 'External API';
    return new ExternalServiceError(service, error.message, error);
  }
  
  // Erro de syntax JSON
  if (error instanceof SyntaxError && error.message.includes('JSON')) {
    return new ValidationError('Invalid JSON format');
  }
  
  return error;
};

/**
 * Determinar se o erro deve ser logado
 */
const shouldLogError = (error) => {
  // Não logar erros de cliente (4xx) exceto 401 e 403
  if (error.statusCode >= 400 && error.statusCode < 500) {
    return error.statusCode === 401 || error.statusCode === 403;
  }
  
  // Logar todos os erros de servidor (5xx)
  return error.statusCode >= 500;
};

/**
 * Gerar ID único para o erro
 */
const generateErrorId = () => {
  return `err_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
};

/**
 * Formatar resposta de erro
 */
const formatErrorResponse = (error, req, errorId) => {
  const isDevelopment = process.env.NODE_ENV === 'development';
  
  const response = {
    error: error.code || 'INTERNAL_ERROR',
    message: error.message,
    timestamp: new Date().toISOString(),
    path: req.originalUrl,
    method: req.method
  };
  
  // Adicionar ID do erro para rastreamento
  if (errorId) {
    response.errorId = errorId;
  }
  
  // Adicionar detalhes se disponível
  if (error.details) {
    response.details = error.details;
  }
  
  // Em desenvolvimento, incluir stack trace
  if (isDevelopment && error.stack) {
    response.stack = error.stack;
  }
  
  // Para erros de rate limit, incluir informações de retry
  if (error.code === 'RATE_LIMIT_EXCEEDED') {
    response.retryAfter = 60; // 1 minuto
  }
  
  return response;
};

/**
 * Notificar sobre erros críticos
 */
const notifyCriticalError = async (error, req, errorId) => {
  try {
    // Apenas para erros 5xx
    if (error.statusCode < 500) return;
    
    const notification = {
      type: 'critical_error',
      errorId,
      message: error.message,
      stack: error.stack,
      request: {
        method: req.method,
        url: req.originalUrl,
        ip: req.ip,
        userAgent: req.get('User-Agent'),
        userId: req.user?.id
      },
      timestamp: new Date().toISOString()
    };
    
    // Aqui você pode implementar notificações via:
    // - Slack/Discord webhook
    // - Email
    // - SMS
    // - Sistema de monitoramento (Sentry, etc.)
    
    logger.error('Critical error notification', notification);
    
  } catch (notificationError) {
    logger.error('Failed to send critical error notification:', notificationError);
  }
};

/**
 * Middleware principal de tratamento de erros
 */
const handle = async (error, req, res, next) => {
  try {
    // Mapear erro para erro customizado
    const mappedError = mapLibraryErrors(error);
    
    // Garantir que temos um AppError
    const appError = mappedError instanceof AppError ? 
      mappedError : 
      new AppError(mappedError.message || 'Internal server error', 500);
    
    // Gerar ID único para o erro
    const errorId = generateErrorId();
    
    // Log do erro se necessário
    if (shouldLogError(appError)) {
      logger.error('Application error', {
        errorId,
        code: appError.code,
        message: appError.message,
        statusCode: appError.statusCode,
        stack: appError.stack,
        request: {
          method: req.method,
          url: req.originalUrl,
          ip: req.ip,
          userAgent: req.get('User-Agent'),
          userId: req.user?.id,
          body: req.body,
          query: req.query,
          params: req.params
        },
        details: appError.details
      });
      
      // Notificar sobre erros críticos
      await notifyCriticalError(appError, req, errorId);
    }
    
    // Métricas de erro
    const tags = {
      statusCode: appError.statusCode,
      errorCode: appError.code,
      path: req.route?.path || req.originalUrl,
      method: req.method
    };
    
    // Aqui você pode implementar métricas personalizadas
    logger.structured('warn', 'Error metric', {
      metric: 'error_count',
      tags
    });
    
    // Formatar resposta
    const errorResponse = formatErrorResponse(appError, req, errorId);
    
    // Enviar resposta
    res.status(appError.statusCode).json(errorResponse);
    
  } catch (handlerError) {
    // Fallback em caso de erro no próprio handler
    logger.error('Error in error handler:', handlerError);
    
    res.status(500).json({
      error: 'INTERNAL_ERROR',
      message: 'An unexpected error occurred',
      timestamp: new Date().toISOString()
    });
  }
};

/**
 * Handler para rotas não encontradas
 */
const notFound = (req, res, next) => {
  const error = new NotFoundError(`Route ${req.method} ${req.originalUrl}`);
  next(error);
};

/**
 * Wrapper para async route handlers
 */
const asyncHandler = (fn) => {
  return (req, res, next) => {
    Promise.resolve(fn(req, res, next)).catch(next);
  };
};

/**
 * Middleware para capturar erros não tratados
 */
const setupGlobalErrorHandlers = () => {
  // Capturar exceções não tratadas
  process.on('uncaughtException', (error) => {
    logger.error('Uncaught Exception:', {
      error: error.message,
      stack: error.stack
    });
    
    // Graceful shutdown
    process.exit(1);
  });
  
  // Capturar promises rejeitadas
  process.on('unhandledRejection', (reason, promise) => {
    logger.error('Unhandled Rejection:', {
      reason: reason.toString(),
      promise: promise.toString()
    });
    
    // Não fazer exit aqui, apenas logar
  });
};

module.exports = {
  handle,
  notFound,
  asyncHandler,
  setupGlobalErrorHandlers,
  
  // Classes de erro
  AppError,
  ValidationError,
  AuthenticationError,
  AuthorizationError,
  NotFoundError,
  ConflictError,
  RateLimitError,
  ExternalServiceError
};