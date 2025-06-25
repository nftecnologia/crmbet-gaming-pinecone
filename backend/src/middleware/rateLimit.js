/**
 * Rate Limiting Middleware
 * 
 * Sistema avançado de rate limiting com Redis,
 * diferentes políticas e proteção contra abuso
 * 
 * @author CRM Team
 */

const rateLimit = require('express-rate-limit');
const { cache } = require('../config/redis');
const logger = require('../utils/logger');

/**
 * Rate limiter customizado usando Redis
 */
class RedisRateLimiter {
  constructor(options = {}) {
    this.windowMs = options.windowMs || 60 * 1000; // 1 minuto
    this.max = options.max || 100;
    this.keyGenerator = options.keyGenerator || ((req) => req.ip);
    this.skipSuccessfulRequests = options.skipSuccessfulRequests || false;
    this.skipFailedRequests = options.skipFailedRequests || false;
    this.onLimitReached = options.onLimitReached || (() => {});
    this.message = options.message || 'Too many requests';
  }

  async middleware(req, res, next) {
    try {
      const key = `ratelimit:${this.keyGenerator(req)}`;
      const windowStart = Math.floor(Date.now() / this.windowMs) * this.windowMs;
      const windowKey = `${key}:${windowStart}`;
      
      // Obter contador atual
      const current = await cache.get(windowKey) || 0;
      
      // Verificar se excedeu o limite
      if (current >= this.max) {
        logger.security('Rate limit exceeded', {
          key: this.keyGenerator(req),
          current,
          max: this.max,
          ip: req.ip,
          userAgent: req.get('User-Agent'),
          path: req.path
        });
        
        this.onLimitReached(req, res);
        
        return res.status(429).json({
          error: 'TooManyRequests',
          message: this.message,
          retryAfter: Math.ceil(this.windowMs / 1000),
          timestamp: new Date().toISOString()
        });
      }
      
      // Incrementar contador
      const newCount = await cache.incr(windowKey, 1, Math.ceil(this.windowMs / 1000));
      
      // Headers informativos
      res.set({
        'X-RateLimit-Limit': this.max,
        'X-RateLimit-Remaining': Math.max(0, this.max - newCount),
        'X-RateLimit-Reset': windowStart + this.windowMs,
        'X-RateLimit-Window': this.windowMs
      });
      
      next();
      
    } catch (error) {
      logger.error('Rate limiter error:', error);
      // Em caso de erro, permitir a requisição
      next();
    }
  }
  
  handler() {
    return this.middleware.bind(this);
  }
}

/**
 * Rate limiter global (por IP)
 */
const globalLimit = new RedisRateLimiter({
  windowMs: 60 * 60 * 1000, // 1 hora
  max: 1000, // 1000 requests por hora por IP
  keyGenerator: (req) => `global:${req.ip}`,
  message: 'Too many requests from this IP, please try again later',
  onLimitReached: (req, res) => {
    logger.security('Global rate limit exceeded', {
      ip: req.ip,
      userAgent: req.get('User-Agent')
    });
  }
}).handler();

/**
 * Rate limiter para API (por usuário autenticado)
 */
const apiLimit = new RedisRateLimiter({
  windowMs: 60 * 60 * 1000, // 1 hora
  max: 500, // 500 requests por hora por usuário
  keyGenerator: (req) => req.user ? `api:user:${req.user.id}` : `api:ip:${req.ip}`,
  message: 'Too many API requests, please try again later',
  onLimitReached: (req, res) => {
    logger.security('API rate limit exceeded', {
      userId: req.user?.id,
      ip: req.ip,
      userAgent: req.get('User-Agent')
    });
  }
}).handler();

/**
 * Rate limiter para webhooks
 */
const webhookLimit = new RedisRateLimiter({
  windowMs: 60 * 1000, // 1 minuto
  max: 100, // 100 requests por minuto por IP
  keyGenerator: (req) => `webhook:${req.ip}`,
  message: 'Too many webhook requests, please try again later',
  onLimitReached: (req, res) => {
    logger.security('Webhook rate limit exceeded', {
      ip: req.ip,
      userAgent: req.get('User-Agent'),
      path: req.path
    });
  }
}).handler();

/**
 * Rate limiter para campanhas (mais restritivo)
 */
const campaignLimit = new RedisRateLimiter({
  windowMs: 60 * 60 * 1000, // 1 hora
  max: 50, // 50 requests por hora por usuário
  keyGenerator: (req) => req.user ? `campaign:user:${req.user.id}` : `campaign:ip:${req.ip}`,
  message: 'Too many campaign requests, please try again later',
  onLimitReached: (req, res) => {
    logger.security('Campaign rate limit exceeded', {
      userId: req.user?.id,
      ip: req.ip,
      userAgent: req.get('User-Agent')
    });
  }
}).handler();

/**
 * Rate limiter para login (proteção brute force)
 */
const loginLimit = new RedisRateLimiter({
  windowMs: 15 * 60 * 1000, // 15 minutos
  max: 5, // 5 tentativas por 15 minutos por IP
  keyGenerator: (req) => `login:${req.ip}`,
  message: 'Too many login attempts, please try again later',
  skipSuccessfulRequests: true,
  onLimitReached: (req, res) => {
    logger.security('Login rate limit exceeded', {
      ip: req.ip,
      userAgent: req.get('User-Agent'),
      email: req.body.email
    });
  }
}).handler();

/**
 * Rate limiter adaptativo baseado no segmento do usuário
 */
const adaptiveLimit = (req, res, next) => {
  if (!req.user) {
    return next();
  }
  
  const segment = req.user.segment;
  let maxRequests = 500; // Padrão
  
  // Ajustar limite baseado no segmento
  switch (segment) {
    case 'high_value':
      maxRequests = 1000;
      break;
    case 'medium_value':
      maxRequests = 750;
      break;
    case 'low_value':
      maxRequests = 300;
      break;
    case 'new_user':
      maxRequests = 200;
      break;
    case 'inactive':
      maxRequests = 100;
      break;
  }
  
  const limiter = new RedisRateLimiter({
    windowMs: 60 * 60 * 1000, // 1 hora
    max: maxRequests,
    keyGenerator: (req) => `adaptive:user:${req.user.id}`,
    message: `Too many requests for your user segment (${segment})`,
    onLimitReached: (req, res) => {
      logger.security('Adaptive rate limit exceeded', {
        userId: req.user.id,
        segment,
        maxRequests,
        ip: req.ip
      });
    }
  });
  
  return limiter.handler()(req, res, next);
};

/**
 * Middleware para detectar e bloquear ataques DDoS
 */
const ddosProtection = async (req, res, next) => {
  try {
    const ip = req.ip;
    const currentTime = Date.now();
    const timeWindow = 10000; // 10 segundos
    const maxRequests = 50; // Máximo 50 requests em 10 segundos
    
    const key = `ddos:${ip}`;
    const requests = await cache.get(key) || [];
    
    // Filtrar requests dentro da janela de tempo
    const recentRequests = requests.filter(timestamp => 
      currentTime - timestamp < timeWindow
    );
    
    if (recentRequests.length >= maxRequests) {
      // Bloquear IP por 1 hora
      await cache.set(`blocked:${ip}`, true, 3600);
      
      logger.security('DDoS attack detected - IP blocked', {
        ip,
        requestCount: recentRequests.length,
        timeWindow,
        userAgent: req.get('User-Agent')
      });
      
      return res.status(429).json({
        error: 'Blocked',
        message: 'Your IP has been temporarily blocked due to suspicious activity',
        timestamp: new Date().toISOString()
      });
    }
    
    // Adicionar timestamp atual
    recentRequests.push(currentTime);
    await cache.set(key, recentRequests, Math.ceil(timeWindow / 1000));
    
    next();
    
  } catch (error) {
    logger.error('DDoS protection error:', error);
    next();
  }
};

/**
 * Verificar se IP está bloqueado
 */
const checkBlocked = async (req, res, next) => {
  try {
    const ip = req.ip;
    const isBlocked = await cache.exists(`blocked:${ip}`);
    
    if (isBlocked) {
      logger.security('Blocked IP attempted access', {
        ip,
        path: req.path,
        userAgent: req.get('User-Agent')
      });
      
      return res.status(403).json({
        error: 'Blocked',
        message: 'Your IP has been blocked',
        timestamp: new Date().toISOString()
      });
    }
    
    next();
    
  } catch (error) {
    logger.error('Block check error:', error);
    next();
  }
};

/**
 * Middleware para monitorar padrões suspeitos
 */
const suspiciousPatternDetection = async (req, res, next) => {
  try {
    const ip = req.ip;
    const userAgent = req.get('User-Agent');
    const path = req.path;
    
    // Detectar múltiplos User-Agents do mesmo IP
    const agentKey = `agents:${ip}`;
    const agents = await cache.get(agentKey) || [];
    
    if (!agents.includes(userAgent)) {
      agents.push(userAgent);
      await cache.set(agentKey, agents, 3600); // 1 hora
      
      if (agents.length > 10) {
        logger.security('Suspicious pattern - multiple user agents', {
          ip,
          agentCount: agents.length,
          currentAgent: userAgent
        });
      }
    }
    
    // Detectar acesso a muitos endpoints diferentes
    const pathKey = `paths:${ip}`;
    const paths = await cache.get(pathKey) || [];
    
    if (!paths.includes(path)) {
      paths.push(path);
      await cache.set(pathKey, paths, 3600); // 1 hora
      
      if (paths.length > 50) {
        logger.security('Suspicious pattern - many different endpoints', {
          ip,
          pathCount: paths.length,
          currentPath: path
        });
      }
    }
    
    next();
    
  } catch (error) {
    logger.error('Suspicious pattern detection error:', error);
    next();
  }
};

/**
 * Rate limiter para upload de arquivos
 */
const uploadLimit = new RedisRateLimiter({
  windowMs: 60 * 60 * 1000, // 1 hora
  max: 20, // 20 uploads por hora por usuário
  keyGenerator: (req) => req.user ? `upload:user:${req.user.id}` : `upload:ip:${req.ip}`,
  message: 'Too many file uploads, please try again later',
  onLimitReached: (req, res) => {
    logger.security('Upload rate limit exceeded', {
      userId: req.user?.id,
      ip: req.ip
    });
  }
}).handler();

module.exports = {
  globalLimit,
  apiLimit,
  webhookLimit,
  campaignLimit,
  loginLimit,
  adaptiveLimit,
  ddosProtection,
  checkBlocked,
  suspiciousPatternDetection,
  uploadLimit,
  RedisRateLimiter
};