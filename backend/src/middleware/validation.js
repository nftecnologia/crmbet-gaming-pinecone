/**
 * Validation Middleware
 * 
 * Sistema robusto de validação usando Joi
 * com sanitização e validação customizada
 * 
 * @author CRM Team
 */

const Joi = require('joi');
const logger = require('../utils/logger');

/**
 * Schemas de validação Joi
 */
const schemas = {
  // User schemas
  userId: Joi.number().integer().positive().required(),
  
  userSegment: Joi.string().valid('high_value', 'medium_value', 'low_value', 'new_user', 'inactive'),
  
  // Cluster schemas
  clusterId: Joi.number().integer().positive().required(),
  
  // Campaign schemas
  campaignId: Joi.number().integer().positive().required(),
  
  campaignCreate: Joi.object({
    name: Joi.string().trim().min(3).max(255).required()
      .messages({
        'string.min': 'Nome deve ter pelo menos 3 caracteres',
        'string.max': 'Nome deve ter no máximo 255 caracteres',
        'any.required': 'Nome é obrigatório'
      }),
    
    description: Joi.string().trim().max(1000).optional().allow(''),
    
    type: Joi.string().valid('email', 'sms', 'push', 'in_app').required()
      .messages({
        'any.only': 'Tipo deve ser: email, sms, push ou in_app',
        'any.required': 'Tipo é obrigatório'
      }),
    
    target_cluster_id: Joi.number().integer().positive().optional().allow(null),
    
    target_segment: Joi.string().valid('high_value', 'medium_value', 'low_value', 'new_user', 'inactive').optional().allow(null),
    
    target_criteria: Joi.object({
      min_deposits: Joi.number().min(0).optional(),
      max_deposits: Joi.number().min(0).optional(),
      min_bets: Joi.number().min(0).optional(),
      max_bets: Joi.number().min(0).optional(),
      min_days_since_registration: Joi.number().integer().min(0).optional(),
      max_days_since_registration: Joi.number().integer().min(0).optional(),
      min_last_activity_days: Joi.number().integer().min(0).optional(),
      max_last_activity_days: Joi.number().integer().min(0).optional(),
      win_rate_min: Joi.number().min(0).max(100).optional(),
      win_rate_max: Joi.number().min(0).max(100).optional()
    }).optional().default({}),
    
    content: Joi.object({
      subject: Joi.string().trim().min(1).max(500).required()
        .messages({
          'string.min': 'Assunto é obrigatório',
          'string.max': 'Assunto deve ter no máximo 500 caracteres'
        }),
      
      message: Joi.string().trim().min(1).max(10000).required()
        .messages({
          'string.min': 'Mensagem é obrigatória',
          'string.max': 'Mensagem deve ter no máximo 10000 caracteres'
        }),
      
      template_id: Joi.string().trim().max(100).optional().allow(null),
      
      variables: Joi.object().optional().default({})
    }).required(),
    
    schedule_at: Joi.date().iso().min('now').optional().allow(null)
      .messages({
        'date.min': 'Data de agendamento deve ser no futuro'
      })
  }),
  
  campaignUpdate: Joi.object({
    name: Joi.string().trim().min(3).max(255).optional(),
    description: Joi.string().trim().max(1000).optional().allow(''),
    status: Joi.string().valid('draft', 'scheduled', 'running', 'completed', 'cancelled').optional(),
    schedule_at: Joi.date().iso().min('now').optional().allow(null)
  }).min(1),
  
  // Smartico webhook schema
  smarticoWebhook: Joi.object({
    event_type: Joi.string().required(),
    user_id: Joi.string().required(),
    data: Joi.object().required(),
    timestamp: Joi.date().iso().required(),
    signature: Joi.string().required()
  }),
  
  // Login schema
  login: Joi.object({
    email: Joi.string().email().required()
      .messages({
        'string.email': 'Email deve ter formato válido',
        'any.required': 'Email é obrigatório'
      }),
    
    password: Joi.string().min(6).required()
      .messages({
        'string.min': 'Senha deve ter pelo menos 6 caracteres',
        'any.required': 'Senha é obrigatória'
      })
  }),
  
  // Refresh token schema
  refreshToken: Joi.object({
    refresh_token: Joi.string().required()
      .messages({
        'any.required': 'Refresh token é obrigatório'
      })
  }),
  
  // Query parameters
  queryParams: {
    page: Joi.number().integer().min(1).default(1),
    limit: Joi.number().integer().min(1).max(100).default(20),
    sortBy: Joi.string().valid('id', 'name', 'created_at', 'updated_at').default('created_at'),
    sortOrder: Joi.string().valid('asc', 'desc').default('desc'),
    search: Joi.string().trim().max(100).optional()
  }
};

/**
 * Middleware de validação genérico
 */
const validate = (schema, source = 'body') => {
  return (req, res, next) => {
    try {
      const dataToValidate = source === 'query' ? req.query : 
                           source === 'params' ? req.params : 
                           req.body;
      
      const { error, value } = schema.validate(dataToValidate, {
        abortEarly: false,
        stripUnknown: true,
        convert: true
      });
      
      if (error) {
        const details = error.details.map(detail => ({
          field: detail.path.join('.'),
          message: detail.message,
          value: detail.context?.value
        }));
        
        logger.security('Validation failed', {
          source,
          errors: details,
          ip: req.ip,
          path: req.path,
          userId: req.user?.id
        });
        
        return res.status(400).json({
          error: 'ValidationError',
          message: 'Validation failed',
          details,
          timestamp: new Date().toISOString()
        });
      }
      
      // Substituir dados validados e sanitizados
      if (source === 'query') {
        req.query = value;
      } else if (source === 'params') {
        req.params = value;
      } else {
        req.body = value;
      }
      
      next();
      
    } catch (error) {
      logger.error('Validation middleware error:', error);
      return res.status(500).json({
        error: 'InternalError',
        message: 'Validation processing failed',
        timestamp: new Date().toISOString()
      });
    }
  };
};

/**
 * Validadores específicos para rotas
 */
const validateUserId = validate(Joi.object({
  id: schemas.userId
}), 'params');

const validateClusterId = validate(Joi.object({
  id: schemas.clusterId
}), 'params');

const validateCampaignId = validate(Joi.object({
  id: schemas.campaignId
}), 'params');

const validateCampaign = validate(schemas.campaignCreate);

const validateCampaignUpdate = validate(schemas.campaignUpdate);

const validateSmarticoWebhook = validate(schemas.smarticoWebhook);

const validateLogin = validate(schemas.login);

const validateRefreshToken = validate(schemas.refreshToken);

const validateQueryParams = validate(Joi.object(schemas.queryParams), 'query');

/**
 * Middleware de sanitização adicional
 */
const sanitize = (req, res, next) => {
  try {
    // Sanitizar strings de entrada
    const sanitizeString = (str) => {
      if (typeof str !== 'string') return str;
      
      // Remover caracteres perigosos
      return str
        .replace(/<script\b[^<]*(?:(?!<\/script>)<[^<]*)*<\/script>/gi, '')
        .replace(/javascript:/gi, '')
        .replace(/on\w+\s*=/gi, '')
        .trim();
    };
    
    const sanitizeObject = (obj) => {
      if (typeof obj !== 'object' || obj === null) return obj;
      
      if (Array.isArray(obj)) {
        return obj.map(sanitizeObject);
      }
      
      const sanitized = {};
      for (const [key, value] of Object.entries(obj)) {
        if (typeof value === 'string') {
          sanitized[key] = sanitizeString(value);
        } else if (typeof value === 'object') {
          sanitized[key] = sanitizeObject(value);
        } else {
          sanitized[key] = value;
        }
      }
      return sanitized;
    };
    
    // Sanitizar body, query e params
    if (req.body) {
      req.body = sanitizeObject(req.body);
    }
    
    if (req.query) {
      req.query = sanitizeObject(req.query);
    }
    
    if (req.params) {
      req.params = sanitizeObject(req.params);
    }
    
    next();
    
  } catch (error) {
    logger.error('Sanitization error:', error);
    return res.status(500).json({
      error: 'InternalError',
      message: 'Data sanitization failed',
      timestamp: new Date().toISOString()
    });
  }
};

/**
 * Validação de arquivo de upload
 */
const validateFileUpload = (options = {}) => {
  const {
    maxSize = 5 * 1024 * 1024, // 5MB
    allowedTypes = ['image/jpeg', 'image/png', 'image/gif'],
    maxFiles = 1
  } = options;
  
  return (req, res, next) => {
    try {
      if (!req.files || req.files.length === 0) {
        return res.status(400).json({
          error: 'ValidationError',
          message: 'No files uploaded',
          timestamp: new Date().toISOString()
        });
      }
      
      if (req.files.length > maxFiles) {
        return res.status(400).json({
          error: 'ValidationError',
          message: `Maximum ${maxFiles} files allowed`,
          timestamp: new Date().toISOString()
        });
      }
      
      for (const file of req.files) {
        // Verificar tamanho
        if (file.size > maxSize) {
          return res.status(400).json({
            error: 'ValidationError',
            message: `File size exceeds ${maxSize} bytes`,
            details: { filename: file.originalname, size: file.size },
            timestamp: new Date().toISOString()
          });
        }
        
        // Verificar tipo
        if (!allowedTypes.includes(file.mimetype)) {
          return res.status(400).json({
            error: 'ValidationError',
            message: `File type not allowed. Allowed types: ${allowedTypes.join(', ')}`,
            details: { filename: file.originalname, type: file.mimetype },
            timestamp: new Date().toISOString()
          });
        }
      }
      
      next();
      
    } catch (error) {
      logger.error('File validation error:', error);
      return res.status(500).json({
        error: 'InternalError',
        message: 'File validation failed',
        timestamp: new Date().toISOString()
      });
    }
  };
};

/**
 * Validação de consistência de dados
 */
const validateDataConsistency = (req, res, next) => {
  try {
    const { body } = req;
    
    // Validações específicas de negócio
    if (body.target_criteria) {
      const criteria = body.target_criteria;
      
      // Min/Max deposits
      if (criteria.min_deposits !== undefined && criteria.max_deposits !== undefined) {
        if (criteria.min_deposits > criteria.max_deposits) {
          return res.status(400).json({
            error: 'ValidationError',
            message: 'Min deposits cannot be greater than max deposits',
            timestamp: new Date().toISOString()
          });
        }
      }
      
      // Win rate validation
      if (criteria.win_rate_min !== undefined && criteria.win_rate_max !== undefined) {
        if (criteria.win_rate_min > criteria.win_rate_max) {
          return res.status(400).json({
            error: 'ValidationError',
            message: 'Min win rate cannot be greater than max win rate',
            timestamp: new Date().toISOString()
          });
        }
      }
    }
    
    // Validar que pelo menos um critério de targeting foi fornecido
    if (body.type && !body.target_cluster_id && !body.target_segment && 
        (!body.target_criteria || Object.keys(body.target_criteria).length === 0)) {
      return res.status(400).json({
        error: 'ValidationError',
        message: 'At least one targeting criteria must be provided (cluster_id, segment, or criteria)',
        timestamp: new Date().toISOString()
      });
    }
    
    next();
    
  } catch (error) {
    logger.error('Data consistency validation error:', error);
    return res.status(500).json({
      error: 'InternalError',
      message: 'Data consistency validation failed',
      timestamp: new Date().toISOString()
    });
  }
};

module.exports = {
  validate,
  validateUserId,
  validateClusterId,
  validateCampaignId,
  validateCampaign,
  validateCampaignUpdate,
  validateSmarticoWebhook,
  validateLogin,
  validateRefreshToken,
  validateQueryParams,
  sanitize,
  validateFileUpload,
  validateDataConsistency,
  schemas
};