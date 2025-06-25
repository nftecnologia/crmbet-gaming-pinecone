/**
 * Smartico Service
 * 
 * Serviço para integração completa com Smartico CRM
 * com webhook handling, retry logic e fallback
 * 
 * @author CRM Team
 */

const axios = require('axios');
const crypto = require('crypto');
const { cache } = require('../config/redis');
const { publisher } = require('../config/rabbitmq');
const { query, transaction } = require('../config/database');
const logger = require('../utils/logger');
const { ExternalServiceError, ValidationError } = require('../middleware/errorHandler');

/**
 * Configuração da API Smartico
 */
const smarticoConfig = {
  baseURL: process.env.SMARTICO_API_URL || 'https://api.smartico.ai',
  apiKey: process.env.SMARTICO_API_KEY,
  secretKey: process.env.SMARTICO_SECRET_KEY,
  timeout: parseInt(process.env.SMARTICO_TIMEOUT) || 30000,
  retryAttempts: parseInt(process.env.SMARTICO_RETRY_ATTEMPTS) || 3,
  retryDelay: parseInt(process.env.SMARTICO_RETRY_DELAY) || 1000
};

/**
 * Cliente HTTP configurado para Smartico
 */
const smarticoClient = axios.create({
  baseURL: smarticoConfig.baseURL,
  timeout: smarticoConfig.timeout,
  headers: {
    'Authorization': `Bearer ${smarticoConfig.apiKey}`,
    'Content-Type': 'application/json',
    'User-Agent': 'CRMBet-Backend/1.0.0'
  }
});

// Interceptors para logging e retry
smarticoClient.interceptors.request.use(
  (config) => {
    logger.debug('Smartico API request', {
      method: config.method,
      url: config.url,
      data: config.data ? 'redacted' : null
    });
    return config;
  },
  (error) => {
    logger.error('Smartico request interceptor error:', error);
    return Promise.reject(error);
  }
);

smarticoClient.interceptors.response.use(
  (response) => {
    logger.debug('Smartico API response', {
      status: response.status,
      url: response.config.url
    });
    return response;
  },
  (error) => {
    logger.error('Smartico API error', {
      status: error.response?.status,
      url: error.config?.url,
      message: error.message,
      data: error.response?.data
    });
    return Promise.reject(error);
  }
);

/**
 * Classe principal do serviço Smartico
 */
class SmarticoService {
  constructor() {
    this.initialized = false;
    this.healthStatus = 'unknown';
    this.lastSync = null;
  }

  /**
   * Inicializar serviço
   */
  async initialize() {
    try {
      // Verificar conectividade
      await this.healthCheck();
      
      // Sincronizar configurações
      await this.syncConfiguration();
      
      this.initialized = true;
      logger.info('Smartico service initialized successfully');
      
    } catch (error) {
      logger.error('Failed to initialize Smartico service:', error);
      throw new ExternalServiceError('Smartico', 'Initialization failed', error);
    }
  }

  /**
   * Health check da API Smartico
   */
  async healthCheck() {
    try {
      const response = await smarticoClient.get('/health');
      this.healthStatus = 'healthy';
      return {
        status: 'healthy',
        version: response.data.version,
        latency: response.headers['x-response-time']
      };
    } catch (error) {
      this.healthStatus = 'unhealthy';
      throw new ExternalServiceError('Smartico', 'Health check failed', error);
    }
  }

  /**
   * Sincronizar configurações com Smartico
   */
  async syncConfiguration() {
    try {
      const response = await smarticoClient.get('/api/v1/config');
      const config = response.data;
      
      // Cachear configurações por 1 hora
      await cache.set('smartico:config', config, 3600);
      
      logger.info('Smartico configuration synced', {
        features: config.features,
        limits: config.limits
      });
      
      return config;
    } catch (error) {
      logger.error('Failed to sync Smartico configuration:', error);
      throw error;
    }
  }

  /**
   * Criar campanha no Smartico
   */
  async createCampaign(campaignId) {
    try {
      // Buscar dados da campanha local
      const campaignResult = await query(
        'SELECT * FROM campaigns WHERE id = $1',
        [campaignId]
      );
      
      if (campaignResult.rows.length === 0) {
        throw new ValidationError('Campaign not found');
      }
      
      const campaign = campaignResult.rows[0];
      
      // Mapear dados para formato Smartico
      const smarticoCampaign = await this.mapCampaignToSmartico(campaign);
      
      // Retry logic para criação
      const smarticoResponse = await this.executeWithRetry(async () => {
        return await smarticoClient.post('/api/v1/campaigns', smarticoCampaign);
      });
      
      const smarticoCampaignId = smarticoResponse.data.campaign_id;
      
      // Atualizar campanha local com ID do Smartico
      await query(
        'UPDATE campaigns SET smartico_campaign_id = $1, updated_at = CURRENT_TIMESTAMP WHERE id = $2',
        [smarticoCampaignId, campaignId]
      );
      
      // Cachear mapeamento
      await cache.set(`smartico:campaign:${campaignId}`, smarticoCampaignId, 86400);
      
      logger.integration('Smartico', 'Campaign created', true, {
        campaignId,
        smarticoCampaignId,
        type: campaign.type
      });
      
      return {
        campaignId,
        smarticoCampaignId,
        status: 'created'
      };
      
    } catch (error) {
      logger.integration('Smartico', 'Campaign creation failed', false, {
        campaignId,
        error: error.message
      });
      throw new ExternalServiceError('Smartico', 'Campaign creation failed', error);
    }
  }

  /**
   * Sincronizar dados de usuários
   */
  async syncUserData(userId = null) {
    try {
      const query_text = userId 
        ? 'SELECT * FROM users WHERE id = $1'
        : 'SELECT * FROM users WHERE updated_at > $1 ORDER BY updated_at LIMIT 1000';
      
      const params = userId 
        ? [userId]
        : [this.lastSync || new Date(Date.now() - 24 * 60 * 60 * 1000)]; // Últimas 24h
      
      const result = await query(query_text, params);
      const users = result.rows;
      
      if (users.length === 0) {
        logger.debug('No users to sync with Smartico');
        return { synced: 0 };
      }
      
      // Processar em lotes de 50
      const batchSize = 50;
      let totalSynced = 0;
      
      for (let i = 0; i < users.length; i += batchSize) {
        const batch = users.slice(i, i + batchSize);
        const syncedCount = await this.syncUserBatch(batch);
        totalSynced += syncedCount;
      }
      
      this.lastSync = new Date();
      await cache.set('smartico:last_sync', this.lastSync, 86400);
      
      logger.integration('Smartico', 'User data sync completed', true, {
        totalUsers: users.length,
        synced: totalSynced
      });
      
      return { synced: totalSynced };
      
    } catch (error) {
      logger.integration('Smartico', 'User sync failed', false, {
        error: error.message
      });
      throw error;
    }
  }

  /**
   * Sincronizar lote de usuários
   */
  async syncUserBatch(users) {
    try {
      const smarticoUsers = users.map(user => this.mapUserToSmartico(user));
      
      const response = await this.executeWithRetry(async () => {
        return await smarticoClient.post('/api/v1/users/batch', {
          users: smarticoUsers
        });
      });
      
      return response.data.processed || users.length;
      
    } catch (error) {
      logger.error('Failed to sync user batch:', error);
      return 0;
    }
  }

  /**
   * Processar webhook do Smartico
   */
  async handleWebhook(req, res) {
    try {
      // Verificar assinatura
      const isValid = this.verifyWebhookSignature(req);
      
      if (!isValid) {
        logger.security('Invalid Smartico webhook signature', {
          ip: req.ip,
          headers: req.headers
        });
        return res.status(401).json({ error: 'Invalid signature' });
      }
      
      const { event_type, user_id, data, timestamp } = req.body;
      
      // Processar evento de forma assíncrona
      await publisher.publishSmarticoEvent('webhook', {
        eventType: event_type,
        userId: user_id,
        data,
        timestamp,
        receivedAt: new Date().toISOString()
      });
      
      // Resposta rápida para Smartico
      res.status(200).json({ 
        status: 'received',
        event_id: data.event_id || null
      });
      
      // Processar evento
      await this.processWebhookEvent(event_type, user_id, data);
      
      logger.integration('Smartico', 'Webhook processed', true, {
        eventType: event_type,
        userId: user_id
      });
      
    } catch (error) {
      logger.integration('Smartico', 'Webhook processing failed', false, {
        error: error.message,
        body: req.body
      });
      
      res.status(500).json({ 
        error: 'Processing failed',
        message: error.message
      });
    }
  }

  /**
   * Processar evento do webhook
   */
  async processWebhookEvent(eventType, userId, data) {
    try {
      switch (eventType) {
        case 'user.updated':
          await this.handleUserUpdated(userId, data);
          break;
          
        case 'campaign.delivered':
          await this.handleCampaignDelivered(data);
          break;
          
        case 'campaign.opened':
          await this.handleCampaignOpened(data);
          break;
          
        case 'campaign.clicked':
          await this.handleCampaignClicked(data);
          break;
          
        case 'user.converted':
          await this.handleUserConverted(userId, data);
          break;
          
        default:
          logger.warn('Unknown Smartico webhook event type:', eventType);
      }
    } catch (error) {
      logger.error('Failed to process webhook event:', {
        eventType,
        userId,
        error: error.message
      });
      throw error;
    }
  }

  /**
   * Verificar assinatura do webhook
   */
  verifyWebhookSignature(req) {
    const signature = req.headers['x-smartico-signature'];
    const timestamp = req.headers['x-smartico-timestamp'];
    const body = JSON.stringify(req.body);
    
    if (!signature || !timestamp) {
      return false;
    }
    
    // Verificar se timestamp não é muito antigo (5 minutos)
    const now = Math.floor(Date.now() / 1000);
    if (Math.abs(now - parseInt(timestamp)) > 300) {
      return false;
    }
    
    // Calcular assinatura esperada
    const payload = `${timestamp}.${body}`;
    const expectedSignature = crypto
      .createHmac('sha256', smarticoConfig.secretKey)
      .update(payload)
      .digest('hex');
    
    return crypto.timingSafeEqual(
      Buffer.from(signature),
      Buffer.from(`sha256=${expectedSignature}`)
    );
  }

  /**
   * Executar com retry logic
   */
  async executeWithRetry(operation, attempts = smarticoConfig.retryAttempts) {
    for (let i = 0; i < attempts; i++) {
      try {
        return await operation();
      } catch (error) {
        if (i === attempts - 1) {
          throw error;
        }
        
        // Backoff exponencial
        const delay = smarticoConfig.retryDelay * Math.pow(2, i);
        await new Promise(resolve => setTimeout(resolve, delay));
        
        logger.warn(`Smartico operation retry ${i + 1}/${attempts}`, {
          error: error.message,
          delay
        });
      }
    }
  }

  /**
   * Mapear campanha para formato Smartico
   */
  async mapCampaignToSmartico(campaign) {
    return {
      name: campaign.name,
      description: campaign.description,
      type: campaign.type,
      content: {
        subject: campaign.content.subject,
        message: campaign.content.message,
        template_id: campaign.content.template_id
      },
      targeting: {
        segment: campaign.target_segment,
        cluster_id: campaign.target_cluster_id,
        criteria: campaign.target_criteria
      },
      schedule_at: campaign.schedule_at,
      metadata: {
        source: 'crmbet',
        campaign_id: campaign.id
      }
    };
  }

  /**
   * Mapear usuário para formato Smartico
   */
  mapUserToSmartico(user) {
    return {
      external_id: user.external_id,
      email: user.email,
      name: user.name,
      segment: user.segment,
      cluster_id: user.cluster_id,
      attributes: {
        total_deposits: user.total_deposits,
        total_bets: user.total_bets,
        win_rate: user.win_rate,
        registration_date: user.registration_date,
        last_activity: user.last_activity
      },
      metadata: user.metadata
    };
  }

  // ===== HANDLERS DE EVENTOS =====

  async handleUserUpdated(userId, data) {
    await transaction(async (client) => {
      await client.query(
        `UPDATE users SET 
         metadata = metadata || $1,
         updated_at = CURRENT_TIMESTAMP 
         WHERE external_id = $2`,
        [JSON.stringify({ smartico_data: data }), userId]
      );
    });
  }

  async handleCampaignDelivered(data) {
    await query(
      `UPDATE campaign_results SET 
       status = 'sent',
       sent_at = $1,
       metadata = metadata || $2
       WHERE campaign_id = (
         SELECT id FROM campaigns WHERE smartico_campaign_id = $3
       ) AND user_id = (
         SELECT id FROM users WHERE external_id = $4
       )`,
      [
        new Date(data.delivered_at),
        JSON.stringify({ smartico_message_id: data.message_id }),
        data.campaign_id,
        data.user_id
      ]
    );
  }

  async handleCampaignOpened(data) {
    await query(
      `UPDATE campaign_results SET 
       status = 'opened',
       opened_at = $1
       WHERE campaign_id = (
         SELECT id FROM campaigns WHERE smartico_campaign_id = $2
       ) AND user_id = (
         SELECT id FROM users WHERE external_id = $3
       )`,
      [
        new Date(data.opened_at),
        data.campaign_id,
        data.user_id
      ]
    );
  }

  async handleCampaignClicked(data) {
    await query(
      `UPDATE campaign_results SET 
       status = 'clicked',
       clicked_at = $1
       WHERE campaign_id = (
         SELECT id FROM campaigns WHERE smartico_campaign_id = $2
       ) AND user_id = (
         SELECT id FROM users WHERE external_id = $3
       )`,
      [
        new Date(data.clicked_at),
        data.campaign_id,
        data.user_id
      ]
    );
  }

  async handleUserConverted(userId, data) {
    await query(
      `UPDATE campaign_results SET 
       status = 'converted',
       converted_at = $1,
       conversion_value = $2
       WHERE campaign_id = (
         SELECT id FROM campaigns WHERE smartico_campaign_id = $3
       ) AND user_id = (
         SELECT id FROM users WHERE external_id = $4
       )`,
      [
        new Date(data.converted_at),
        data.conversion_value || 0,
        data.campaign_id,
        userId
      ]
    );
  }
}

// Instância global do serviço
const smarticoService = new SmarticoService();

/**
 * Funções exportadas para uso nos controllers
 */
module.exports = {
  initialize: () => smarticoService.initialize(),
  healthCheck: () => smarticoService.healthCheck(),
  createCampaign: (campaignId) => smarticoService.createCampaign(campaignId),
  syncData: () => smarticoService.syncUserData(),
  syncUserData: (userId) => smarticoService.syncUserData(userId),
  handleWebhook: (req, res) => smarticoService.handleWebhook(req, res),
  getHealthStatus: () => smarticoService.healthStatus,
  
  // Para testes
  _service: smarticoService
};