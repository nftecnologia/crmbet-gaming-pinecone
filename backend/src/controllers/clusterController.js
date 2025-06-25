/**
 * Cluster Controller
 * 
 * Controller para gestão de clusters de Machine Learning
 * com analytics avançados e otimização de performance
 * 
 * @author CRM Team
 */

const clusterService = require('../services/clusterService');
const { cache } = require('../config/redis');
const logger = require('../utils/logger');
const { asyncHandler, NotFoundError, ValidationError } = require('../middleware/errorHandler');

/**
 * @swagger
 * /api/v1/clusters:
 *   get:
 *     tags: [Clusters]
 *     summary: Listar clusters ML
 *     description: Retorna lista de clusters de Machine Learning com estatísticas
 *     security:
 *       - bearerAuth: []
 *     parameters:
 *       - in: query
 *         name: algorithm
 *         schema:
 *           type: string
 *           enum: [kmeans, dbscan, hierarchical]
 *         description: Filtrar por algoritmo
 *       - in: query
 *         name: active_only
 *         schema:
 *           type: boolean
 *           default: true
 *         description: Mostrar apenas clusters ativos
 *       - in: query
 *         name: min_users
 *         schema:
 *           type: integer
 *           minimum: 1
 *         description: Mínimo de usuários no cluster
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
 *         description: Lista de clusters
 *         content:
 *           application/json:
 *             schema:
 *               type: object
 *               properties:
 *                 clusters:
 *                   type: array
 *                   items:
 *                     allOf:
 *                       - $ref: '#/components/schemas/Cluster'
 *                       - type: object
 *                         properties:
 *                           statistics:
 *                             type: object
 *                             properties:
 *                               avg_ltv:
 *                                 type: number
 *                                 description: LTV médio do cluster
 *                               avg_deposits:
 *                                 type: number
 *                                 description: Depósitos médios
 *                               avg_bets:
 *                                 type: number
 *                                 description: Apostas médias
 *                               churn_rate:
 *                                 type: number
 *                                 description: Taxa de churn
 *                               engagement_score:
 *                                 type: number
 *                                 description: Score de engajamento
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
 *                 summary:
 *                   type: object
 *                   properties:
 *                     total_clusters:
 *                       type: integer
 *                     total_users_clustered:
 *                       type: integer
 *                     algorithms_used:
 *                       type: array
 *                       items:
 *                         type: string
 *       401:
 *         $ref: '#/components/responses/Unauthorized'
 *       500:
 *         $ref: '#/components/responses/ServerError'
 */
const getClusters = asyncHandler(async (req, res) => {
  const {
    algorithm,
    active_only = true,
    min_users,
    page = 1,
    limit = 20
  } = req.query;
  
  const cacheKey = `clusters:${JSON.stringify(req.query)}`;
  
  try {
    // Verificar cache primeiro (15 minutos)
    let result = await cache.get(cacheKey);
    
    if (!result) {
      const filters = {
        algorithm,
        activeOnly: active_only === 'true',
        minUsers: min_users ? parseInt(min_users) : undefined,
        page: parseInt(page),
        limit: parseInt(limit)
      };
      
      result = await clusterService.getClusters(filters);
      await cache.set(cacheKey, result, 900); // 15 minutos
    }
    
    logger.debug('Clusters retrieved', {
      filters: req.query,
      resultCount: result.clusters.length,
      totalCount: result.pagination.total,
      requesterId: req.user.id
    });
    
    res.status(200).json(result);
    
  } catch (error) {
    logger.error('Error getting clusters:', {
      filters: req.query,
      error: error.message,
      requesterId: req.user.id
    });
    throw error;
  }
});

/**
 * @swagger
 * /api/v1/clusters/{id}:
 *   get:
 *     tags: [Clusters]
 *     summary: Obter cluster específico
 *     description: Retorna detalhes completos de um cluster específico
 *     security:
 *       - bearerAuth: []
 *     parameters:
 *       - in: path
 *         name: id
 *         required: true
 *         schema:
 *           type: integer
 *         description: ID do cluster
 *     responses:
 *       200:
 *         description: Detalhes do cluster
 *         content:
 *           application/json:
 *             schema:
 *               allOf:
 *                 - $ref: '#/components/schemas/Cluster'
 *                 - type: object
 *                   properties:
 *                     detailed_statistics:
 *                       type: object
 *                       properties:
 *                         demographics:
 *                           type: object
 *                           properties:
 *                             age_distribution:
 *                               type: object
 *                             country_distribution:
 *                               type: object
 *                             registration_period:
 *                               type: object
 *                         behavior_patterns:
 *                           type: object
 *                           properties:
 *                             peak_activity_hours:
 *                               type: array
 *                               items:
 *                                 type: integer
 *                             preferred_games:
 *                               type: array
 *                               items:
 *                                 type: string
 *                             session_patterns:
 *                               type: object
 *                         financial_metrics:
 *                           type: object
 *                           properties:
 *                             deposit_patterns:
 *                               type: object
 *                             bet_size_distribution:
 *                               type: object
 *                             ltv_percentiles:
 *                               type: object
 *                     feature_importance:
 *                       type: object
 *                       description: Importância das features usadas na clusterização
 *                     similar_clusters:
 *                       type: array
 *                       items:
 *                         type: object
 *                         properties:
 *                           cluster_id:
 *                             type: integer
 *                           similarity_score:
 *                             type: number
 *                           name:
 *                             type: string
 *       404:
 *         $ref: '#/components/responses/NotFound'
 *       401:
 *         $ref: '#/components/responses/Unauthorized'
 */
const getCluster = asyncHandler(async (req, res) => {
  const clusterId = parseInt(req.params.id);
  const cacheKey = `cluster_details:${clusterId}`;
  
  try {
    // Verificar cache primeiro (30 minutos)
    let clusterData = await cache.get(cacheKey);
    
    if (!clusterData) {
      clusterData = await clusterService.getClusterDetails(clusterId);
      
      if (!clusterData) {
        throw new NotFoundError('Cluster');
      }
      
      await cache.set(cacheKey, clusterData, 1800); // 30 minutos
    }
    
    logger.debug('Cluster details retrieved', {
      clusterId,
      userCount: clusterData.user_count,
      requesterId: req.user.id
    });
    
    res.status(200).json(clusterData);
    
  } catch (error) {
    logger.error('Error getting cluster details:', {
      clusterId,
      error: error.message,
      requesterId: req.user.id
    });
    throw error;
  }
});

/**
 * @swagger
 * /api/v1/clusters/{id}/users:
 *   get:
 *     tags: [Clusters]
 *     summary: Listar usuários do cluster
 *     description: Retorna lista de usuários pertencentes ao cluster
 *     security:
 *       - bearerAuth: []
 *     parameters:
 *       - in: path
 *         name: id
 *         required: true
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
 *           default: 50
 *         description: Itens por página
 *       - in: query
 *         name: sort_by
 *         schema:
 *           type: string
 *           enum: [ltv, deposits, bets, activity, registration_date]
 *           default: ltv
 *         description: Campo para ordenação
 *       - in: query
 *         name: sort_order
 *         schema:
 *           type: string
 *           enum: [asc, desc]
 *           default: desc
 *         description: Ordem da classificação
 *       - in: query
 *         name: active_only
 *         schema:
 *           type: boolean
 *           default: true
 *         description: Mostrar apenas usuários ativos
 *     responses:
 *       200:
 *         description: Lista de usuários do cluster
 *         content:
 *           application/json:
 *             schema:
 *               type: object
 *               properties:
 *                 cluster_id:
 *                   type: integer
 *                 cluster_name:
 *                   type: string
 *                 users:
 *                   type: array
 *                   items:
 *                     allOf:
 *                       - $ref: '#/components/schemas/User'
 *                       - type: object
 *                         properties:
 *                           cluster_distance:
 *                             type: number
 *                             description: Distância do usuário ao centroide do cluster
 *                           ltv:
 *                             type: number
 *                             description: Lifetime Value calculado
 *                           activity_score:
 *                             type: number
 *                             description: Score de atividade recente
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
 *                 cluster_statistics:
 *                   type: object
 *                   properties:
 *                     avg_ltv:
 *                       type: number
 *                     median_ltv:
 *                       type: number
 *                     total_deposits:
 *                       type: number
 *                     total_bets:
 *                       type: number
 *       404:
 *         $ref: '#/components/responses/NotFound'
 *       401:
 *         $ref: '#/components/responses/Unauthorized'
 */
const getClusterUsers = asyncHandler(async (req, res) => {
  const clusterId = parseInt(req.params.id);
  const {
    page = 1,
    limit = 50,
    sort_by = 'ltv',
    sort_order = 'desc',
    active_only = true
  } = req.query;
  
  const cacheKey = `cluster_users:${clusterId}:${JSON.stringify(req.query)}`;
  
  try {
    // Verificar cache primeiro (10 minutos)
    let result = await cache.get(cacheKey);
    
    if (!result) {
      const options = {
        page: parseInt(page),
        limit: parseInt(limit),
        sortBy: sort_by,
        sortOrder: sort_order,
        activeOnly: active_only === 'true'
      };
      
      result = await clusterService.getClusterUsers(clusterId, options);
      
      if (!result || result.users.length === 0) {
        // Verificar se o cluster existe
        const clusterExists = await clusterService.clusterExists(clusterId);
        if (!clusterExists) {
          throw new NotFoundError('Cluster');
        }
      }
      
      await cache.set(cacheKey, result, 600); // 10 minutos
    }
    
    logger.debug('Cluster users retrieved', {
      clusterId,
      options: req.query,
      userCount: result.users.length,
      totalUsers: result.pagination.total,
      requesterId: req.user.id
    });
    
    res.status(200).json(result);
    
  } catch (error) {
    logger.error('Error getting cluster users:', {
      clusterId,
      options: req.query,
      error: error.message,
      requesterId: req.user.id
    });
    throw error;
  }
});

/**
 * @swagger
 * /api/v1/clusters/{id}/analytics:
 *   get:
 *     tags: [Clusters]
 *     summary: Analytics do cluster
 *     description: Retorna analytics avançados e insights do cluster
 *     security:
 *       - bearerAuth: []
 *     parameters:
 *       - in: path
 *         name: id
 *         required: true
 *         schema:
 *           type: integer
 *         description: ID do cluster
 *       - in: query
 *         name: period
 *         schema:
 *           type: string
 *           enum: [7d, 30d, 90d, 1y]
 *           default: 30d
 *         description: Período de análise
 *     responses:
 *       200:
 *         description: Analytics do cluster
 *         content:
 *           application/json:
 *             schema:
 *               type: object
 *               properties:
 *                 cluster_id:
 *                   type: integer
 *                 period:
 *                   type: string
 *                 performance_metrics:
 *                   type: object
 *                   properties:
 *                     revenue_contribution:
 *                       type: number
 *                       description: Contribuição percentual para receita total
 *                     avg_session_duration:
 *                       type: number
 *                       description: Duração média de sessão (minutos)
 *                     conversion_rate:
 *                       type: number
 *                       description: Taxa de conversão
 *                     retention_rate:
 *                       type: number
 *                       description: Taxa de retenção
 *                 trends:
 *                   type: object
 *                   properties:
 *                     user_growth:
 *                       type: array
 *                       items:
 *                         type: object
 *                         properties:
 *                           date:
 *                             type: string
 *                             format: date
 *                           count:
 *                             type: integer
 *                     revenue_trend:
 *                       type: array
 *                       items:
 *                         type: object
 *                         properties:
 *                           date:
 *                             type: string
 *                             format: date
 *                           revenue:
 *                             type: number
 *                     activity_trend:
 *                       type: array
 *                       items:
 *                         type: object
 *                         properties:
 *                           date:
 *                             type: string
 *                             format: date
 *                           activity_score:
 *                             type: number
 *                 insights:
 *                   type: array
 *                   items:
 *                     type: object
 *                     properties:
 *                       type:
 *                         type: string
 *                         enum: [opportunity, risk, trend]
 *                       title:
 *                         type: string
 *                       description:
 *                         type: string
 *                       impact:
 *                         type: string
 *                         enum: [high, medium, low]
 *                       confidence:
 *                         type: number
 *                         description: Confiança no insight (0-1)
 *       404:
 *         $ref: '#/components/responses/NotFound'
 */
const getClusterAnalytics = asyncHandler(async (req, res) => {
  const clusterId = parseInt(req.params.id);
  const period = req.query.period || '30d';
  const cacheKey = `cluster_analytics:${clusterId}:${period}`;
  
  try {
    // Verificar cache primeiro (1 hora)
    let analytics = await cache.get(cacheKey);
    
    if (!analytics) {
      analytics = await clusterService.getClusterAnalytics(clusterId, period);
      
      if (!analytics) {
        throw new NotFoundError('Cluster');
      }
      
      await cache.set(cacheKey, analytics, 3600); // 1 hora
    }
    
    logger.business('Cluster analytics retrieved', {
      clusterId,
      period,
      revenueContribution: analytics.performance_metrics?.revenue_contribution,
      insightCount: analytics.insights?.length || 0,
      requesterId: req.user.id
    });
    
    res.status(200).json(analytics);
    
  } catch (error) {
    logger.error('Error getting cluster analytics:', {
      clusterId,
      period,
      error: error.message,
      requesterId: req.user.id
    });
    throw error;
  }
});

/**
 * @swagger
 * /api/v1/clusters/comparison:
 *   post:
 *     tags: [Clusters]
 *     summary: Comparar clusters
 *     description: Compara métricas entre múltiplos clusters
 *     security:
 *       - bearerAuth: []
 *     requestBody:
 *       required: true
 *       content:
 *         application/json:
 *           schema:
 *             type: object
 *             required:
 *               - cluster_ids
 *             properties:
 *               cluster_ids:
 *                 type: array
 *                 items:
 *                   type: integer
 *                 minItems: 2
 *                 maxItems: 5
 *                 description: IDs dos clusters para comparar
 *               metrics:
 *                 type: array
 *                 items:
 *                   type: string
 *                   enum: [user_count, avg_ltv, revenue, activity, retention]
 *                 default: [user_count, avg_ltv, revenue, activity, retention]
 *                 description: Métricas para comparar
 *               period:
 *                 type: string
 *                 enum: [7d, 30d, 90d, 1y]
 *                 default: 30d
 *                 description: Período de análise
 *     responses:
 *       200:
 *         description: Comparação entre clusters
 *         content:
 *           application/json:
 *             schema:
 *               type: object
 *               properties:
 *                 comparison:
 *                   type: array
 *                   items:
 *                     type: object
 *                     properties:
 *                       cluster_id:
 *                         type: integer
 *                       cluster_name:
 *                         type: string
 *                       metrics:
 *                         type: object
 *                 insights:
 *                   type: array
 *                   items:
 *                     type: object
 *                     properties:
 *                       type:
 *                         type: string
 *                       description:
 *                         type: string
 *                       clusters_involved:
 *                         type: array
 *                         items:
 *                           type: integer
 */
const compareClusters = asyncHandler(async (req, res) => {
  const { cluster_ids, metrics = ['user_count', 'avg_ltv', 'revenue', 'activity', 'retention'], period = '30d' } = req.body;
  
  if (!cluster_ids || cluster_ids.length < 2 || cluster_ids.length > 5) {
    throw new ValidationError('Must provide between 2 and 5 cluster IDs for comparison');
  }
  
  const cacheKey = `cluster_comparison:${cluster_ids.sort().join(',')}:${metrics.join(',')}:${period}`;
  
  try {
    // Verificar cache primeiro (30 minutos)
    let comparison = await cache.get(cacheKey);
    
    if (!comparison) {
      comparison = await clusterService.compareClusters(cluster_ids, metrics, period);
      await cache.set(cacheKey, comparison, 1800); // 30 minutos
    }
    
    logger.business('Cluster comparison performed', {
      clusterIds: cluster_ids,
      metrics,
      period,
      requesterId: req.user.id
    });
    
    res.status(200).json(comparison);
    
  } catch (error) {
    logger.error('Error comparing clusters:', {
      clusterIds: cluster_ids,
      metrics,
      period,
      error: error.message,
      requesterId: req.user.id
    });
    throw error;
  }
});

module.exports = {
  getClusters,
  getCluster,
  getClusterUsers,
  getClusterAnalytics,
  compareClusters
};