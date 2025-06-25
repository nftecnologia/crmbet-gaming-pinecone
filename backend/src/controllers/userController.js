/**
 * User Controller
 * 
 * Controller para gestão de usuários e segmentação ML
 * com cache inteligente e analytics
 * 
 * @author CRM Team
 */

const userService = require('../services/userService');
const { cache } = require('../config/redis');
const logger = require('../utils/logger');
const { asyncHandler, NotFoundError, ValidationError } = require('../middleware/errorHandler');

/**
 * @swagger
 * /api/v1/user/{id}/segment:
 *   get:
 *     tags: [Users]
 *     summary: Obter segmento ML do usuário
 *     description: Retorna o segmento de Machine Learning do usuário com dados de comportamento
 *     security:
 *       - bearerAuth: []
 *     parameters:
 *       - in: path
 *         name: id
 *         required: true
 *         schema:
 *           type: integer
 *         description: ID do usuário
 *     responses:
 *       200:
 *         description: Segmento do usuário
 *         content:
 *           application/json:
 *             schema:
 *               type: object
 *               properties:
 *                 user_id:
 *                   type: integer
 *                   example: 123
 *                 segment:
 *                   type: string
 *                   enum: [high_value, medium_value, low_value, new_user, inactive]
 *                   example: "high_value"
 *                 cluster_id:
 *                   type: integer
 *                   nullable: true
 *                   example: 5
 *                 confidence_score:
 *                   type: number
 *                   format: float
 *                   description: Confiança na classificação (0-1)
 *                   example: 0.95
 *                 segment_features:
 *                   type: object
 *                   properties:
 *                     total_deposits:
 *                       type: number
 *                       example: 5000.00
 *                     total_bets:
 *                       type: number
 *                       example: 15000.00
 *                     win_rate:
 *                       type: number
 *                       example: 65.5
 *                     bet_frequency:
 *                       type: number
 *                       example: 12.5
 *                     avg_bet_size:
 *                       type: number
 *                       example: 125.00
 *                     days_since_last_activity:
 *                       type: integer
 *                       example: 2
 *                     registration_days:
 *                       type: integer
 *                       example: 180
 *                 segment_description:
 *                   type: string
 *                   example: "High-value customer with consistent betting patterns"
 *                 recommendations:
 *                   type: array
 *                   items:
 *                     type: object
 *                     properties:
 *                       type:
 *                         type: string
 *                         example: "campaign"
 *                       title:
 *                         type: string
 *                         example: "VIP Bonus Offer"
 *                       description:
 *                         type: string
 *                         example: "Offer exclusive VIP bonuses"
 *                       priority:
 *                         type: string
 *                         enum: [high, medium, low]
 *                         example: "high"
 *                 last_updated:
 *                   type: string
 *                   format: date-time
 *       404:
 *         $ref: '#/components/responses/NotFound'
 *       401:
 *         $ref: '#/components/responses/Unauthorized'
 *       500:
 *         $ref: '#/components/responses/ServerError'
 */
const getUserSegment = asyncHandler(async (req, res) => {
  const userId = parseInt(req.params.id);
  const cacheKey = `user_segment:${userId}`;
  
  try {
    // Verificar cache primeiro
    let segmentData = await cache.get(cacheKey);
    
    if (!segmentData) {
      // Buscar dados do usuário e calcular segmento
      segmentData = await userService.getUserSegment(userId);
      
      if (!segmentData) {
        throw new NotFoundError('User');
      }
      
      // Cachear por 1 hora
      await cache.set(cacheKey, segmentData, 3600);
      
      logger.business('User segment calculated', {
        userId,
        segment: segmentData.segment,
        clusterId: segmentData.cluster_id,
        confidenceScore: segmentData.confidence_score
      });
    }
    
    // Log da consulta
    logger.debug('User segment retrieved', {
      userId,
      segment: segmentData.segment,
      fromCache: !!segmentData,
      requesterId: req.user.id
    });
    
    res.status(200).json(segmentData);
    
  } catch (error) {
    logger.error('Error getting user segment:', {
      userId,
      error: error.message,
      requesterId: req.user.id
    });
    throw error;
  }
});

/**
 * @swagger
 * /api/v1/users/segments/stats:
 *   get:
 *     tags: [Users]
 *     summary: Estatísticas de segmentação
 *     description: Retorna estatísticas agregadas dos segmentos de usuários
 *     security:
 *       - bearerAuth: []
 *     responses:
 *       200:
 *         description: Estatísticas de segmentação
 *         content:
 *           application/json:
 *             schema:
 *               type: object
 *               properties:
 *                 total_users:
 *                   type: integer
 *                   example: 10000
 *                 segments:
 *                   type: object
 *                   properties:
 *                     high_value:
 *                       type: object
 *                       properties:
 *                         count:
 *                           type: integer
 *                           example: 500
 *                         percentage:
 *                           type: number
 *                           example: 5.0
 *                         avg_ltv:
 *                           type: number
 *                           example: 2500.00
 *                     medium_value:
 *                       type: object
 *                       properties:
 *                         count:
 *                           type: integer
 *                           example: 2000
 *                         percentage:
 *                           type: number
 *                           example: 20.0
 *                         avg_ltv:
 *                           type: number
 *                           example: 800.00
 *                     low_value:
 *                       type: object
 *                       properties:
 *                         count:
 *                           type: integer
 *                           example: 4000
 *                         percentage:
 *                           type: number
 *                           example: 40.0
 *                         avg_ltv:
 *                           type: number
 *                           example: 200.00
 *                     new_user:
 *                       type: object
 *                       properties:
 *                         count:
 *                           type: integer
 *                           example: 2500
 *                         percentage:
 *                           type: number
 *                           example: 25.0
 *                         avg_ltv:
 *                           type: number
 *                           example: 50.00
 *                     inactive:
 *                       type: object
 *                       properties:
 *                         count:
 *                           type: integer
 *                           example: 1000
 *                         percentage:
 *                           type: number
 *                           example: 10.0
 *                         avg_ltv:
 *                           type: number
 *                           example: 0.00
 *                 last_updated:
 *                   type: string
 *                   format: date-time
 */
const getSegmentStats = asyncHandler(async (req, res) => {
  const cacheKey = 'segment_stats';
  
  try {
    // Verificar cache primeiro (cachear por 30 minutos)
    let stats = await cache.get(cacheKey);
    
    if (!stats) {
      stats = await userService.getSegmentStats();
      await cache.set(cacheKey, stats, 1800); // 30 minutos
    }
    
    logger.debug('Segment stats retrieved', {
      totalUsers: stats.total_users,
      requesterId: req.user.id
    });
    
    res.status(200).json(stats);
    
  } catch (error) {
    logger.error('Error getting segment stats:', {
      error: error.message,
      requesterId: req.user.id
    });
    throw error;
  }
});

/**
 * @swagger
 * /api/v1/users/{id}/behavior:
 *   get:
 *     tags: [Users]
 *     summary: Análise comportamental do usuário
 *     description: Retorna análise detalhada do comportamento do usuário
 *     security:
 *       - bearerAuth: []
 *     parameters:
 *       - in: path
 *         name: id
 *         required: true
 *         schema:
 *           type: integer
 *         description: ID do usuário
 *       - in: query
 *         name: period
 *         schema:
 *           type: string
 *           enum: [7d, 30d, 90d, 1y]
 *           default: 30d
 *         description: Período de análise
 *     responses:
 *       200:
 *         description: Análise comportamental
 *         content:
 *           application/json:
 *             schema:
 *               type: object
 *               properties:
 *                 user_id:
 *                   type: integer
 *                 period:
 *                   type: string
 *                 behavior_metrics:
 *                   type: object
 *                   properties:
 *                     activity_score:
 *                       type: number
 *                       description: Score de atividade (0-100)
 *                     engagement_score:
 *                       type: number
 *                       description: Score de engajamento (0-100)
 *                     risk_score:
 *                       type: number
 *                       description: Score de risco (0-100)
 *                     loyalty_score:
 *                       type: number
 *                       description: Score de lealdade (0-100)
 *                 patterns:
 *                   type: object
 *                   properties:
 *                     peak_activity_hours:
 *                       type: array
 *                       items:
 *                         type: integer
 *                     preferred_games:
 *                       type: array
 *                       items:
 *                         type: string
 *                     bet_size_trend:
 *                       type: string
 *                       enum: [increasing, decreasing, stable]
 *                     session_duration_avg:
 *                       type: number
 *                 predictions:
 *                   type: object
 *                   properties:
 *                     churn_probability:
 *                       type: number
 *                       description: Probabilidade de churn (0-1)
 *                     next_deposit_probability:
 *                       type: number
 *                       description: Probabilidade de próximo depósito (0-1)
 *                     ltv_prediction:
 *                       type: number
 *                       description: Predição de LTV
 */
const getUserBehavior = asyncHandler(async (req, res) => {
  const userId = parseInt(req.params.id);
  const period = req.query.period || '30d';
  const cacheKey = `user_behavior:${userId}:${period}`;
  
  try {
    // Verificar cache primeiro
    let behaviorData = await cache.get(cacheKey);
    
    if (!behaviorData) {
      behaviorData = await userService.getUserBehaviorAnalysis(userId, period);
      
      if (!behaviorData) {
        throw new NotFoundError('User');
      }
      
      // Cachear por 2 horas
      await cache.set(cacheKey, behaviorData, 7200);
    }
    
    logger.business('User behavior analysis retrieved', {
      userId,
      period,
      activityScore: behaviorData.behavior_metrics?.activity_score,
      churnProbability: behaviorData.predictions?.churn_probability,
      requesterId: req.user.id
    });
    
    res.status(200).json(behaviorData);
    
  } catch (error) {
    logger.error('Error getting user behavior:', {
      userId,
      period,
      error: error.message,
      requesterId: req.user.id
    });
    throw error;
  }
});

/**
 * @swagger
 * /api/v1/users/search:
 *   get:
 *     tags: [Users]
 *     summary: Buscar usuários
 *     description: Busca usuários por critérios específicos
 *     security:
 *       - bearerAuth: []
 *     parameters:
 *       - in: query
 *         name: email
 *         schema:
 *           type: string
 *         description: Email do usuário
 *       - in: query
 *         name: segment
 *         schema:
 *           type: string
 *           enum: [high_value, medium_value, low_value, new_user, inactive]
 *         description: Segmento do usuário
 *       - in: query
 *         name: cluster_id
 *         schema:
 *           type: integer
 *         description: ID do cluster
 *       - in: query
 *         name: page
 *         schema:
 *           type: integer
 *           minimum: 1
 *           default: 1
 *         description: Página
 *       - in: query
 *         name: limit
 *         schema:
 *           type: integer
 *           minimum: 1
 *           maximum: 100
 *           default: 20
 *         description: Itens por página
 *     responses:
 *       200:
 *         description: Lista de usuários
 *         content:
 *           application/json:
 *             schema:
 *               type: object
 *               properties:
 *                 users:
 *                   type: array
 *                   items:
 *                     $ref: '#/components/schemas/User'
 *                 pagination:
 *                   type: object
 *                   properties:
 *                     page:
 *                       type: integer
 *                     limit:
 *                       type: integer
 *                     total:
 *                       type: integer
 *                     pages:
 *                       type: integer
 */
const searchUsers = asyncHandler(async (req, res) => {
  const { email, segment, cluster_id, page = 1, limit = 20 } = req.query;
  
  try {
    const searchCriteria = {
      email,
      segment,
      cluster_id: cluster_id ? parseInt(cluster_id) : undefined,
      page: parseInt(page),
      limit: parseInt(limit)
    };
    
    const result = await userService.searchUsers(searchCriteria);
    
    logger.debug('User search performed', {
      criteria: searchCriteria,
      resultCount: result.users.length,
      totalCount: result.pagination.total,
      requesterId: req.user.id
    });
    
    res.status(200).json(result);
    
  } catch (error) {
    logger.error('Error searching users:', {
      searchCriteria: req.query,
      error: error.message,
      requesterId: req.user.id
    });
    throw error;
  }
});

/**
 * @swagger
 * /api/v1/users/{id}/recommendations:
 *   get:
 *     tags: [Users]
 *     summary: Recomendações para usuário
 *     description: Retorna recomendações personalizadas baseadas no perfil do usuário
 *     security:
 *       - bearerAuth: []
 *     parameters:
 *       - in: path
 *         name: id
 *         required: true
 *         schema:
 *           type: integer
 *         description: ID do usuário
 *     responses:
 *       200:
 *         description: Recomendações personalizadas
 *         content:
 *           application/json:
 *             schema:
 *               type: object
 *               properties:
 *                 user_id:
 *                   type: integer
 *                 recommendations:
 *                   type: array
 *                   items:
 *                     type: object
 *                     properties:
 *                       id:
 *                         type: string
 *                       type:
 *                         type: string
 *                         enum: [campaign, bonus, game, promotion]
 *                       title:
 *                         type: string
 *                       description:
 *                         type: string
 *                       priority:
 *                         type: string
 *                         enum: [high, medium, low]
 *                       confidence:
 *                         type: number
 *                         description: Confiança na recomendação (0-1)
 *                       expected_impact:
 *                         type: object
 *                         properties:
 *                           engagement_increase:
 *                             type: number
 *                           revenue_potential:
 *                             type: number
 *                       valid_until:
 *                         type: string
 *                         format: date-time
 */
const getUserRecommendations = asyncHandler(async (req, res) => {
  const userId = parseInt(req.params.id);
  const cacheKey = `user_recommendations:${userId}`;
  
  try {
    // Verificar cache primeiro (1 hora)
    let recommendations = await cache.get(cacheKey);
    
    if (!recommendations) {
      recommendations = await userService.getUserRecommendations(userId);
      
      if (!recommendations) {
        throw new NotFoundError('User');
      }
      
      await cache.set(cacheKey, recommendations, 3600);
    }
    
    logger.business('User recommendations retrieved', {
      userId,
      recommendationCount: recommendations.recommendations.length,
      requesterId: req.user.id
    });
    
    res.status(200).json(recommendations);
    
  } catch (error) {
    logger.error('Error getting user recommendations:', {
      userId,
      error: error.message,
      requesterId: req.user.id
    });
    throw error;
  }
});

module.exports = {
  getUserSegment,
  getSegmentStats,
  getUserBehavior,
  searchUsers,
  getUserRecommendations
};