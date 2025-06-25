/**
 * Authentication Middleware
 * 
 * Middleware robusto de autenticação JWT com
 * refresh tokens, rate limiting e segurança
 * 
 * @author CRM Team
 */

const jwt = require('jsonwebtoken');
const bcrypt = require('bcryptjs');
const { cache } = require('../config/redis');
const logger = require('../utils/logger');

// Configurações JWT
const jwtConfig = {
  secret: process.env.JWT_SECRET || 'your-super-secret-key',
  refreshSecret: process.env.JWT_REFRESH_SECRET || 'your-refresh-secret-key',
  accessTokenExpiry: process.env.JWT_ACCESS_EXPIRY || '1h',
  refreshTokenExpiry: process.env.JWT_REFRESH_EXPIRY || '7d',
  issuer: process.env.JWT_ISSUER || 'crmbet-api',
  audience: process.env.JWT_AUDIENCE || 'crmbet-client'
};

/**
 * Gerar token de acesso
 */
const generateAccessToken = (payload) => {
  return jwt.sign(payload, jwtConfig.secret, {
    expiresIn: jwtConfig.accessTokenExpiry,
    issuer: jwtConfig.issuer,
    audience: jwtConfig.audience,
    algorithm: 'HS256'
  });
};

/**
 * Gerar refresh token
 */
const generateRefreshToken = (payload) => {
  return jwt.sign(payload, jwtConfig.refreshSecret, {
    expiresIn: jwtConfig.refreshTokenExpiry,
    issuer: jwtConfig.issuer,
    audience: jwtConfig.audience,
    algorithm: 'HS256'
  });
};

/**
 * Verificar token de acesso
 */
const verifyAccessToken = (token) => {
  try {
    return jwt.verify(token, jwtConfig.secret, {
      issuer: jwtConfig.issuer,
      audience: jwtConfig.audience,
      algorithms: ['HS256']
    });
  } catch (error) {
    throw new Error(`Invalid access token: ${error.message}`);
  }
};

/**
 * Verificar refresh token
 */
const verifyRefreshToken = (token) => {
  try {
    return jwt.verify(token, jwtConfig.refreshSecret, {
      issuer: jwtConfig.issuer,
      audience: jwtConfig.audience,
      algorithms: ['HS256']
    });
  } catch (error) {
    throw new Error(`Invalid refresh token: ${error.message}`);
  }
};

/**
 * Hash password
 */
const hashPassword = async (password) => {
  const saltRounds = parseInt(process.env.BCRYPT_ROUNDS) || 12;
  return await bcrypt.hash(password, saltRounds);
};

/**
 * Verificar password
 */
const verifyPassword = async (password, hashedPassword) => {
  return await bcrypt.compare(password, hashedPassword);
};

/**
 * Middleware de autenticação principal
 */
const authenticate = async (req, res, next) => {
  try {
    const authHeader = req.headers.authorization;
    
    if (!authHeader) {
      logger.security('Missing authorization header', {
        ip: req.ip,
        userAgent: req.get('User-Agent'),
        path: req.path
      });
      
      return res.status(401).json({
        error: 'Unauthorized',
        message: 'Authorization header is required',
        timestamp: new Date().toISOString()
      });
    }
    
    const [scheme, token] = authHeader.split(' ');
    
    if (scheme !== 'Bearer' || !token) {
      logger.security('Invalid authorization format', {
        ip: req.ip,
        userAgent: req.get('User-Agent'),
        path: req.path,
        scheme
      });
      
      return res.status(401).json({
        error: 'Unauthorized',
        message: 'Invalid authorization format. Use Bearer <token>',
        timestamp: new Date().toISOString()
      });
    }
    
    // Verificar se o token está na blacklist
    const isBlacklisted = await cache.exists(`blacklist:${token}`);
    if (isBlacklisted) {
      logger.security('Blacklisted token used', {
        ip: req.ip,
        userAgent: req.get('User-Agent'),
        path: req.path
      });
      
      return res.status(401).json({
        error: 'Unauthorized',
        message: 'Token has been revoked',
        timestamp: new Date().toISOString()
      });
    }
    
    // Verificar token
    const decoded = verifyAccessToken(token);
    
    // Verificar se o usuário ainda existe e está ativo
    const userCacheKey = `user:${decoded.userId}`;
    let user = await cache.get(userCacheKey);
    
    if (!user) {
      // Buscar no banco se não estiver no cache
      const { query } = require('../config/database');
      const result = await query(
        'SELECT id, external_id, email, name, segment, cluster_id FROM users WHERE id = $1 AND active = true',
        [decoded.userId]
      );
      
      if (result.rows.length === 0) {
        logger.security('Token with invalid user', {
          ip: req.ip,
          userId: decoded.userId,
          path: req.path
        });
        
        return res.status(401).json({
          error: 'Unauthorized',
          message: 'User not found or inactive',
          timestamp: new Date().toISOString()
        });
      }
      
      user = result.rows[0];
      // Cachear por 5 minutos
      await cache.set(userCacheKey, user, 300);
    }
    
    // Adicionar informações do usuário à requisição
    req.user = {
      id: user.id,
      externalId: user.external_id,
      email: user.email,
      name: user.name,
      segment: user.segment,
      clusterId: user.cluster_id
    };
    
    req.token = token;
    req.tokenPayload = decoded;
    
    // Log successful authentication
    logger.debug('User authenticated', {
      userId: user.id,
      email: user.email,
      ip: req.ip,
      path: req.path
    });
    
    next();
    
  } catch (error) {
    logger.security('Authentication failed', {
      ip: req.ip,
      userAgent: req.get('User-Agent'),
      path: req.path,
      error: error.message
    });
    
    if (error.name === 'TokenExpiredError') {
      return res.status(401).json({
        error: 'TokenExpired',
        message: 'Access token has expired',
        timestamp: new Date().toISOString()
      });
    }
    
    if (error.name === 'JsonWebTokenError') {
      return res.status(401).json({
        error: 'InvalidToken',
        message: 'Invalid access token',
        timestamp: new Date().toISOString()
      });
    }
    
    return res.status(401).json({
      error: 'Unauthorized',
      message: 'Authentication failed',
      timestamp: new Date().toISOString()
    });
  }
};

/**
 * Middleware de autorização por role
 */
const authorize = (roles = []) => {
  return (req, res, next) => {
    try {
      if (!req.user) {
        return res.status(401).json({
          error: 'Unauthorized',
          message: 'Authentication required',
          timestamp: new Date().toISOString()
        });
      }
      
      // Se não especificar roles, apenas precisa estar autenticado
      if (roles.length === 0) {
        return next();
      }
      
      const userRole = req.tokenPayload.role || 'user';
      
      if (!roles.includes(userRole)) {
        logger.security('Insufficient permissions', {
          userId: req.user.id,
          userRole,
          requiredRoles: roles,
          path: req.path,
          ip: req.ip
        });
        
        return res.status(403).json({
          error: 'Forbidden',
          message: 'Insufficient permissions',
          timestamp: new Date().toISOString()
        });
      }
      
      next();
      
    } catch (error) {
      logger.error('Authorization error:', error);
      return res.status(500).json({
        error: 'InternalError',
        message: 'Authorization check failed',
        timestamp: new Date().toISOString()
      });
    }
  };
};

/**
 * Login de usuário
 */
const login = async (email, password) => {
  try {
    const { query } = require('../config/database');
    
    // Buscar usuário
    const result = await query(
      'SELECT id, external_id, email, name, password_hash, segment, cluster_id, role FROM users WHERE email = $1 AND active = true',
      [email]
    );
    
    if (result.rows.length === 0) {
      throw new Error('Invalid credentials');
    }
    
    const user = result.rows[0];
    
    // Verificar password
    const isValidPassword = await verifyPassword(password, user.password_hash);
    if (!isValidPassword) {
      throw new Error('Invalid credentials');
    }
    
    // Gerar tokens
    const tokenPayload = {
      userId: user.id,
      email: user.email,
      role: user.role || 'user'
    };
    
    const accessToken = generateAccessToken(tokenPayload);
    const refreshToken = generateRefreshToken(tokenPayload);
    
    // Salvar refresh token no cache
    await cache.set(`refresh:${user.id}`, refreshToken, 7 * 24 * 3600); // 7 dias
    
    // Cachear dados do usuário
    const userCache = {
      id: user.id,
      external_id: user.external_id,
      email: user.email,
      name: user.name,
      segment: user.segment,
      cluster_id: user.cluster_id
    };
    await cache.set(`user:${user.id}`, userCache, 300); // 5 minutos
    
    logger.business('User login', {
      userId: user.id,
      email: user.email
    });
    
    return {
      user: userCache,
      tokens: {
        accessToken,
        refreshToken,
        expiresIn: jwtConfig.accessTokenExpiry
      }
    };
    
  } catch (error) {
    logger.security('Login failed', { email, error: error.message });
    throw error;
  }
};

/**
 * Refresh token
 */
const refreshAccessToken = async (refreshToken) => {
  try {
    const decoded = verifyRefreshToken(refreshToken);
    
    // Verificar se o refresh token ainda está válido no cache
    const cachedToken = await cache.get(`refresh:${decoded.userId}`);
    if (cachedToken !== refreshToken) {
      throw new Error('Invalid refresh token');
    }
    
    // Gerar novo access token
    const tokenPayload = {
      userId: decoded.userId,
      email: decoded.email,
      role: decoded.role
    };
    
    const newAccessToken = generateAccessToken(tokenPayload);
    
    logger.debug('Access token refreshed', {
      userId: decoded.userId
    });
    
    return {
      accessToken: newAccessToken,
      expiresIn: jwtConfig.accessTokenExpiry
    };
    
  } catch (error) {
    logger.security('Token refresh failed', { error: error.message });
    throw error;
  }
};

/**
 * Logout (invalidar tokens)
 */
const logout = async (userId, accessToken) => {
  try {
    // Adicionar access token à blacklist
    const decoded = jwt.decode(accessToken);
    const expiry = decoded.exp - Math.floor(Date.now() / 1000);
    if (expiry > 0) {
      await cache.set(`blacklist:${accessToken}`, true, expiry);
    }
    
    // Remover refresh token
    await cache.del(`refresh:${userId}`);
    
    // Remover cache do usuário
    await cache.del(`user:${userId}`);
    
    logger.business('User logout', { userId });
    
  } catch (error) {
    logger.error('Logout error:', error);
    throw error;
  }
};

module.exports = {
  authenticate,
  authorize,
  login,
  refreshAccessToken,
  logout,
  generateAccessToken,
  generateRefreshToken,
  verifyAccessToken,
  verifyRefreshToken,
  hashPassword,
  verifyPassword
};