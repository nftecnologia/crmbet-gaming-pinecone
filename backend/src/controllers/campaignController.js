/**
 * Campaign Controller
 * 
 * Controller para gestão de campanhas de marketing
 * com integração Smartico e analytics avançados
 * 
 * @author CRM Team
 */

const campaignService = require('../services/campaignService');
const smarticoService = require('../services/smarticoService');
const { cache } = require('../config/redis');
const { publisher } = require('../config/rabbitmq');
const logger = require('../utils/logger');
const { asyncHandler, NotFoundError, ValidationError, ConflictError } = require('../middleware/errorHandler');

/**
 * @swagger
 * /api/v1/campaigns:
 *   post:
 *     tags: [Campaigns]
 *     summary: Criar nova campanha
 *     description: Cria uma nova campanha de marketing com targeting específico
 *     security:
 *       - bearerAuth: []
 *     requestBody:
 *       required: true
 *       content:
 *         application/json:
 *           schema:
 *             $ref: '#/components/schemas/CampaignCreateRequest'
 *     responses:
 *       201:
 *         description: Campanha criada com sucesso
 *         content:
 *           application/json:
 *             schema:
 *               type: object
 *               properties:
 *                 campaign:
 *                   $ref: '#/components/schemas/Campaign'
 *                 estimated_reach:
 *                   type: integer
 *                   description: Número estimado de usuários que receberão a campanha
 *                 targeting_preview:
 *                   type: object
 *                   properties:
 *                     segments:
 *                       type: object
 *                       description: Distribuição por segmentos
 *                     clusters:
 *                       type: object
 *                       description: Distribuição por clusters
 *       400:
 *         $ref: '#/components/responses/ValidationError'
 *       401:
 *         $ref: '#/components/responses/Unauthorized'
 *       429:
 *         $ref: '#/components/responses/RateLimit'
 *       500:
 *         $ref: '#/components/responses/ServerError'
 */
const createCampaign = asyncHandler(async (req, res) => {
  const campaignData = {
    ...req.body,
    created_by: req.user.id
  };
  
  try {
    // Validar targeting (pelo menos um critério deve ser fornecido)
    if (!campaignData.target_cluster_id && !campaignData.target_segment && 
        (!campaignData.target_criteria || Object.keys(campaignData.target_criteria).length === 0)) {
      throw new ValidationError('At least one targeting criteria must be provided');
    }
    
    // Criar campanha
    const result = await campaignService.createCampaign(campaignData);
    
    // Publicar evento para processamento assíncrono
    await publisher.publishCampaign('create', {
      campaignId: result.campaign.id,
      userId: req.user.id,
      timestamp: new Date().toISOString()
    });
    
    // Log da criação
    logger.business('Campaign created', {
      campaignId: result.campaign.id,
      name: result.campaign.name,
      type: result.campaign.type,
      estimatedReach: result.estimated_reach,
      createdBy: req.user.id,
      targetCluster: result.campaign.target_cluster_id,
      targetSegment: result.campaign.target_segment
    });
    
    res.status(201).json(result);
    
  } catch (error) {
    logger.error('Error creating campaign:', {
      campaignData: {
        name: campaignData.name,
        type: campaignData.type,
        targetCluster: campaignData.target_cluster_id,
        targetSegment: campaignData.target_segment
      },
      error: error.message,
      createdBy: req.user.id
    });
    throw error;
  }
});

/**
 * @swagger
 * /api/v1/campaigns:
 *   get:
 *     tags: [Campaigns]
 *     summary: Listar campanhas
 *     description: Retorna lista de campanhas com filtros
 *     security:
 *       - bearerAuth: []
 *     parameters:
 *       - in: query
 *         name: status
 *         schema:
 *           type: string
 *           enum: [draft, scheduled, running, completed, cancelled]
 *         description: Filtrar por status
 *       - in: query
 *         name: type
 *         schema:
 *           type: string
 *           enum: [email, sms, push, in_app]
 *         description: Filtrar por tipo
 *       - in: query
 *         name: created_by
 *         schema:
 *           type: integer
 *         description: Filtrar por criador
 *       - in: query
 *         name: page
 *         schema:
 *           type: integer
 *           minimum: 1
 *           default: 1
 *       - in: query
 *         name: limit
 *         schema:
 *           type: integer
 *           minimum: 1
 *           maximum: 100
 *           default: 20
 *       - in: query
 *         name: sort_by
 *         schema:
 *           type: string
 *           enum: [created_at, updated_at, name, total_sent]
 *           default: created_at
 *       - in: query
 *         name: sort_order
 *         schema:
 *           type: string
 *           enum: [asc, desc]
 *           default: desc
 *     responses:
 *       200:
 *         description: Lista de campanhas
 *         content:
 *           application/json:
 *             schema:
 *               type: object
 *               properties:
 *                 campaigns:
 *                   type: array
 *                   items:
 *                     allOf:
 *                       - $ref: '#/components/schemas/Campaign'
 *                       - type: object
 *                         properties:
 *                           performance_summary:
 *                             type: object
 *                             properties:
 *                               open_rate:
 *                                 type: number
 *                               click_rate:
 *                                 type: number
 *                               conversion_rate:
 *                                 type: number
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
const getCampaigns = asyncHandler(async (req, res) => {
  const {
    status,
    type,
    created_by,
    page = 1,
    limit = 20,
    sort_by = 'created_at',
    sort_order = 'desc'
  } = req.query;
  
  const cacheKey = `campaigns:${JSON.stringify(req.query)}:${req.user.id}`;
  
  try {
    // Verificar cache primeiro (5 minutos)
    let result = await cache.get(cacheKey);
    
    if (!result) {
      const filters = {
        status,
        type,
        createdBy: created_by ? parseInt(created_by) : undefined,
        page: parseInt(page),
        limit: parseInt(limit),
        sortBy: sort_by,
        sortOrder: sort_order,
        requesterId: req.user.id
      };
      
      result = await campaignService.getCampaigns(filters);
      await cache.set(cacheKey, result, 300); // 5 minutos
    }
    
    logger.debug('Campaigns retrieved', {
      filters: req.query,
      resultCount: result.campaigns.length,
      totalCount: result.pagination.total,
      requesterId: req.user.id
    });
    
    res.status(200).json(result);
    
  } catch (error) {
    logger.error('Error getting campaigns:', {
      filters: req.query,
      error: error.message,
      requesterId: req.user.id
    });
    throw error;
  }
});

/**
 * @swagger
 * /api/v1/campaigns/{id}:
 *   get:
 *     tags: [Campaigns]
 *     summary: Obter campanha específica
 *     description: Retorna detalhes completos de uma campanha
 *     security:
 *       - bearerAuth: []
 *     parameters:
 *       - in: path
 *         name: id
 *         required: true
 *         schema:
 *           type: integer
 *         description: ID da campanha
 *     responses:
 *       200:
 *         description: Detalhes da campanha
 *         content:
 *           application/json:
 *             schema:
 *               allOf:
 *                 - $ref: '#/components/schemas/Campaign'
 *                 - type: object
 *                   properties:
 *                     targeting_analysis:
 *                       type: object
 *                       properties:
 *                         actual_reach:
 *                           type: integer
 *                         segment_distribution:
 *                           type: object
 *                         cluster_distribution:
 *                           type: object
 *                     performance_metrics:
 *                       type: object
 *                       properties:
 *                         open_rate:
 *                           type: number
 *                         click_rate:
 *                           type: number
 *                         conversion_rate:
 *                           type: number
 *                         revenue_generated:
 *                           type: number
 *                         cost_per_conversion:
 *                           type: number
 *       404:
 *         $ref: '#/components/responses/NotFound'
 */
const getCampaign = asyncHandler(async (req, res) => {
  const campaignId = parseInt(req.params.id);
  const cacheKey = `campaign_details:${campaignId}`;
  
  try {
    // Verificar cache primeiro (10 minutos)
    let campaign = await cache.get(cacheKey);
    
    if (!campaign) {
      campaign = await campaignService.getCampaignDetails(campaignId);
      
      if (!campaign) {
        throw new NotFoundError('Campaign');
      }
      
      await cache.set(cacheKey, campaign, 600); // 10 minutos
    }
    
    logger.debug('Campaign details retrieved', {
      campaignId,
      status: campaign.status,
      totalSent: campaign.total_sent,
      requesterId: req.user.id
    });
    
    res.status(200).json(campaign);
    
  } catch (error) {
    logger.error('Error getting campaign details:', {
      campaignId,
      error: error.message,
      requesterId: req.user.id
    });
    throw error;
  }
});

/**
 * @swagger
 * /api/v1/campaigns/{id}:
 *   put:
 *     tags: [Campaigns]
 *     summary: Atualizar campanha
 *     description: Atualiza dados de uma campanha (apenas se status permitir)
 *     security:
 *       - bearerAuth: []
 *     parameters:
 *       - in: path
 *         name: id
 *         required: true
 *         schema:
 *           type: integer
 *         description: ID da campanha
 *     requestBody:
 *       required: true
 *       content:
 *         application/json:
 *           schema:
 *             type: object
 *             properties:
 *               name:
 *                 type: string
 *                 minLength: 3
 *                 maxLength: 255
 *               description:
 *                 type: string
 *                 maxLength: 1000
 *               status:
 *                 type: string
 *                 enum: [draft, scheduled, cancelled]
 *               schedule_at:
 *                 type: string
 *                 format: date-time
 *     responses:
 *       200:
 *         description: Campanha atualizada
 *         content:
 *           application/json:
 *             schema:
 *               $ref: '#/components/schemas/Campaign'
 *       400:
 *         $ref: '#/components/responses/ValidationError'
 *       404:
 *         $ref: '#/components/responses/NotFound'
 *       409:
 *         description: Conflito - campanha não pode ser editada no status atual
 */
const updateCampaign = asyncHandler(async (req, res) => {
  const campaignId = parseInt(req.params.id);
  const updateData = req.body;
  
  try {
    const updatedCampaign = await campaignService.updateCampaign(campaignId, updateData, req.user.id);
    
    // Invalidar cache
    await cache.del(`campaign_details:${campaignId}`);
    await cache.clear('campaigns:*');
    
    // Log da atualização
    logger.business('Campaign updated', {
      campaignId,
      updates: Object.keys(updateData),
      updatedBy: req.user.id,
      newStatus: updatedCampaign.status
    });
    
    res.status(200).json(updatedCampaign);
    
  } catch (error) {
    logger.error('Error updating campaign:', {
      campaignId,
      updateData,
      error: error.message,
      updatedBy: req.user.id
    });
    throw error;
  }
});

/**
 * @swagger
 * /api/v1/campaigns/{id}:
 *   delete:
 *     tags: [Campaigns]
 *     summary: Deletar campanha
 *     description: Deleta uma campanha (apenas se status permitir)
 *     security:
 *       - bearerAuth: []
 *     parameters:
 *       - in: path
 *         name: id
 *         required: true
 *         schema:
 *           type: integer
 *         description: ID da campanha
 *     responses:
 *       204:
 *         description: Campanha deletada com sucesso
 *       404:
 *         $ref: '#/components/responses/NotFound'
 *       409:
 *         description: Conflito - campanha não pode ser deletada no status atual
 */
const deleteCampaign = asyncHandler(async (req, res) => {
  const campaignId = parseInt(req.params.id);
  
  try {
    await campaignService.deleteCampaign(campaignId, req.user.id);
    
    // Invalidar cache
    await cache.del(`campaign_details:${campaignId}`);
    await cache.clear('campaigns:*');
    
    // Log da deleção
    logger.business('Campaign deleted', {
      campaignId,
      deletedBy: req.user.id
    });
    
    res.status(204).send();
    
  } catch (error) {
    logger.error('Error deleting campaign:', {
      campaignId,
      error: error.message,
      deletedBy: req.user.id
    });
    throw error;
  }
});

/**
 * @swagger
 * /api/v1/campaigns/{id}/results:
 *   get:
 *     tags: [Campaigns]
 *     summary: Obter resultados da campanha
 *     description: Retorna resultados detalhados e analytics da campanha
 *     security:
 *       - bearerAuth: []
 *     parameters:
 *       - in: path
 *         name: id
 *         required: true
 *         schema:
 *           type: integer
 *         description: ID da campanha
 *       - in: query
 *         name: include_users
 *         schema:
 *           type: boolean
 *           default: false
 *         description: Incluir dados individuais dos usuários
 *       - in: query
 *         name: status_filter
 *         schema:
 *           type: string
 *           enum: [sent, opened, clicked, converted, failed]
 *         description: Filtrar resultados por status
 *       - in: query
 *         name: page
 *         schema:
 *           type: integer
 *           minimum: 1
 *           default: 1
 *       - in: query
 *         name: limit
 *         schema:
 *           type: integer
 *           minimum: 1
 *           maximum: 100
 *           default: 50
 *     responses:
 *       200:
 *         description: Resultados da campanha
 *         content:
 *           application/json:
 *             schema:
 *               $ref: '#/components/schemas/CampaignResults'
 *       404:
 *         $ref: '#/components/responses/NotFound'
 */
const getCampaignResults = asyncHandler(async (req, res) => {
  const campaignId = parseInt(req.params.id);
  const {
    include_users = false,
    status_filter,
    page = 1,
    limit = 50
  } = req.query;
  
  const cacheKey = `campaign_results:${campaignId}:${JSON.stringify(req.query)}`;
  
  try {
    // Verificar cache primeiro (5 minutos para dados agregados, 1 minuto para dados individuais)
    const cacheTTL = include_users === 'true' ? 60 : 300;
    let results = await cache.get(cacheKey);
    
    if (!results) {
      const options = {
        includeUsers: include_users === 'true',
        statusFilter: status_filter,
        page: parseInt(page),
        limit: parseInt(limit)
      };
      
      results = await campaignService.getCampaignResults(campaignId, options);
      
      if (!results) {
        throw new NotFoundError('Campaign');
      }
      
      await cache.set(cacheKey, results, cacheTTL);
    }
    
    logger.debug('Campaign results retrieved', {
      campaignId,
      options: req.query,
      totalSent: results.total_sent,
      conversionRate: results.conversion_rate,
      requesterId: req.user.id
    });
    
    res.status(200).json(results);
    
  } catch (error) {
    logger.error('Error getting campaign results:', {
      campaignId,
      options: req.query,
      error: error.message,
      requesterId: req.user.id
    });
    throw error;
  }
});

/**
 * @swagger
 * /api/v1/campaigns/{id}/launch:
 *   post:
 *     tags: [Campaigns]
 *     summary: Lançar campanha
 *     description: Lança uma campanha imediatamente ou agenda para execução
 *     security:
 *       - bearerAuth: []
 *     parameters:
 *       - in: path
 *         name: id
 *         required: true
 *         schema:
 *           type: integer
 *         description: ID da campanha
 *     requestBody:
 *       content:
 *         application/json:
 *           schema:
 *             type: object
 *             properties:
 *               schedule_at:
 *                 type: string
 *                 format: date-time
 *                 description: Data/hora para agendamento (omitir para lançamento imediato)
 *               confirm_targeting:
 *                 type: boolean
 *                 default: true
 *                 description: Confirmar que o targeting está correto
 *     responses:
 *       200:
 *         description: Campanha lançada ou agendada
 *         content:
 *           application/json:
 *             schema:
 *               type: object
 *               properties:
 *                 message:
 *                   type: string
 *                 campaign_id:
 *                   type: integer
 *                 status:
 *                   type: string
 *                 launched_at:
 *                   type: string
 *                   format: date-time
 *                 estimated_completion:
 *                   type: string
 *                   format: date-time
 *                 target_count:
 *                   type: integer
 *       400:
 *         description: Campanha não pode ser lançada no status atual
 *       404:
 *         $ref: '#/components/responses/NotFound'
 */
const launchCampaign = asyncHandler(async (req, res) => {
  const campaignId = parseInt(req.params.id);
  const { schedule_at, confirm_targeting = true } = req.body;
  
  try {
    if (!confirm_targeting) {
      throw new ValidationError('Targeting confirmation is required to launch campaign');
    }
    
    const launchResult = await campaignService.launchCampaign(campaignId, {
      scheduleAt: schedule_at,
      launchedBy: req.user.id
    });
    
    // Integração com Smartico
    if (launchResult.status === 'running') {
      await smarticoService.createCampaign(campaignId);
    }
    
    // Publicar evento para processamento
    await publisher.publishCampaign('launch', {
      campaignId,
      launchedBy: req.user.id,
      scheduledAt: schedule_at,
      timestamp: new Date().toISOString()
    });
    
    // Invalidar cache
    await cache.del(`campaign_details:${campaignId}`);
    
    logger.business('Campaign launched', {
      campaignId,
      status: launchResult.status,
      targetCount: launchResult.target_count,
      launchedBy: req.user.id,
      scheduledAt: schedule_at
    });
    
    res.status(200).json(launchResult);
    
  } catch (error) {
    logger.error('Error launching campaign:', {
      campaignId,
      scheduleAt: schedule_at,
      error: error.message,
      launchedBy: req.user.id
    });
    throw error;
  }
});

/**
 * @swagger
 * /api/v1/campaigns/analytics/overview:
 *   get:
 *     tags: [Campaigns]
 *     summary: Overview de analytics de campanhas
 *     description: Retorna visão geral das métricas de todas as campanhas
 *     security:
 *       - bearerAuth: []
 *     parameters:
 *       - in: query
 *         name: period
 *         schema:
 *           type: string
 *           enum: [7d, 30d, 90d, 1y]
 *           default: 30d
 *         description: Período de análise
 *     responses:
 *       200:
 *         description: Overview de analytics
 *         content:
 *           application/json:
 *             schema:
 *               type: object
 *               properties:
 *                 period:
 *                   type: string
 *                 summary:
 *                   type: object
 *                   properties:
 *                     total_campaigns:
 *                       type: integer
 *                     active_campaigns:
 *                       type: integer
 *                     total_sent:
 *                       type: integer
 *                     total_revenue:
 *                       type: number
 *                     avg_open_rate:
 *                       type: number
 *                     avg_click_rate:
 *                       type: number
 *                     avg_conversion_rate:
 *                       type: number
 *                 performance_by_type:
 *                   type: object
 *                   properties:
 *                     email:
 *                       type: object
 *                     sms:
 *                       type: object
 *                     push:
 *                       type: object
 *                     in_app:
 *                       type: object
 *                 trends:
 *                   type: array
 *                   items:
 *                     type: object
 *                     properties:
 *                       date:
 *                         type: string
 *                         format: date
 *                       campaigns_sent:
 *                         type: integer
 *                       revenue:
 *                         type: number
 */
const getCampaignAnalytics = asyncHandler(async (req, res) => {
  const period = req.query.period || '30d';
  const cacheKey = `campaign_analytics:${period}:${req.user.id}`;
  
  try {
    // Verificar cache primeiro (15 minutos)
    let analytics = await cache.get(cacheKey);
    
    if (!analytics) {
      analytics = await campaignService.getCampaignAnalytics(period, req.user.id);
      await cache.set(cacheKey, analytics, 900); // 15 minutos
    }
    
    logger.debug('Campaign analytics retrieved', {
      period,
      totalCampaigns: analytics.summary?.total_campaigns,
      totalRevenue: analytics.summary?.total_revenue,
      requesterId: req.user.id
    });
    
    res.status(200).json(analytics);
    
  } catch (error) {
    logger.error('Error getting campaign analytics:', {
      period,
      error: error.message,
      requesterId: req.user.id
    });
    throw error;
  }
});

module.exports = {
  createCampaign,
  getCampaigns,
  getCampaign,
  updateCampaign,
  deleteCampaign,
  getCampaignResults,
  launchCampaign,
  getCampaignAnalytics
};