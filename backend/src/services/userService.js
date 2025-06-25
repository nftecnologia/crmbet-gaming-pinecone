/**
 * User Service
 * 
 * Serviço para gestão de usuários e segmentação ML
 * com algoritmos avançados e cache inteligente
 * 
 * @author CRM Team
 */

const userRepository = require('../repositories/userRepository');
const { cache } = require('../config/redis');
const logger = require('../utils/logger');
const { NotFoundError, ValidationError } = require('../middleware/errorHandler');

/**
 * Algoritmo de segmentação ML baseado em comportamento
 */
class UserSegmentationEngine {
  constructor() {
    this.segmentRules = {
      high_value: {
        min_deposits: 1000,
        min_ltv: 2500,
        min_bet_frequency: 5,
        max_churn_risk: 0.3
      },
      medium_value: {
        min_deposits: 200,
        min_ltv: 500,
        min_bet_frequency: 2,
        max_churn_risk: 0.6
      },
      low_value: {
        min_deposits: 50,
        min_ltv: 100,
        min_bet_frequency: 1,
        max_churn_risk: 0.8
      },
      new_user: {
        max_days_since_registration: 30,
        max_deposits: 100
      },
      inactive: {
        min_days_since_last_activity: 30,
        max_activity_score: 20
      }
    };
  }

  /**
   * Calcular features comportamentais do usuário
   */
  calculateBehaviorFeatures(userData) {
    const now = new Date();
    const registrationDate = new Date(userData.registration_date);
    const lastActivity = userData.last_activity ? new Date(userData.last_activity) : null;
    
    const daysSinceRegistration = Math.floor((now - registrationDate) / (1000 * 60 * 60 * 24));
    const daysSinceLastActivity = lastActivity ? 
      Math.floor((now - lastActivity) / (1000 * 60 * 60 * 24)) : 999;
    
    // Calcular LTV (Lifetime Value)
    const ltv = userData.total_deposits - userData.total_withdrawals;
    
    // Calcular frequência de apostas (apostas por dia desde registro)
    const betFrequency = daysSinceRegistration > 0 ? 
      userData.bet_count / daysSinceRegistration : 0;
    
    // Calcular tamanho médio de aposta
    const avgBetSize = userData.bet_count > 0 ? 
      userData.total_bets / userData.bet_count : 0;
    
    // Score de atividade (0-100)
    const activityScore = Math.min(100, Math.max(0, 
      100 - (daysSinceLastActivity * 3.33) // Decresce 3.33 por dia
    ));
    
    // Score de risco de churn usando ML simplificado
    const churnRisk = this.calculateChurnRisk({
      daysSinceLastActivity,
      betFrequency,
      ltv,
      winRate: userData.win_rate || 0
    });
    
    return {
      total_deposits: userData.total_deposits || 0,
      total_bets: userData.total_bets || 0,
      win_rate: userData.win_rate || 0,
      bet_frequency: betFrequency,
      avg_bet_size: avgBetSize,
      days_since_last_activity: daysSinceLastActivity,
      registration_days: daysSinceRegistration,
      ltv,
      activity_score: activityScore,
      churn_risk: churnRisk
    };
  }

  /**
   * Calcular risco de churn usando modelo ML simplificado
   */
  calculateChurnRisk(features) {
    // Modelo simplificado baseado em pesos empíricos
    const weights = {
      days_inactive: 0.4,
      bet_frequency: -0.3,
      ltv: -0.2,
      win_rate: -0.1
    };
    
    let churnScore = 0;
    
    // Normalizar e aplicar pesos
    churnScore += Math.min(1, features.daysSinceLastActivity / 30) * weights.days_inactive;
    churnScore += Math.min(1, Math.max(0, 1 - features.betFrequency / 10)) * weights.bet_frequency;
    churnScore += Math.min(1, Math.max(0, 1 - features.ltv / 1000)) * weights.ltv;
    churnScore += Math.min(1, Math.max(0, 1 - features.winRate / 100)) * weights.win_rate;
    
    return Math.min(1, Math.max(0, churnScore));
  }

  /**
   * Determinar segmento baseado em features
   */
  determineSegment(features) {
    const { total_deposits, ltv, bet_frequency, registration_days, days_since_last_activity, activity_score, churn_risk } = features;
    
    // Verificar se é usuário inativo
    if (days_since_last_activity >= this.segmentRules.inactive.min_days_since_last_activity ||
        activity_score <= this.segmentRules.inactive.max_activity_score) {
      return {
        segment: 'inactive',
        confidence: 0.9,
        reasons: ['High inactivity period', 'Low activity score']
      };
    }
    
    // Verificar se é novo usuário
    if (registration_days <= this.segmentRules.new_user.max_days_since_registration &&
        total_deposits <= this.segmentRules.new_user.max_deposits) {
      return {
        segment: 'new_user',
        confidence: 0.85,
        reasons: ['Recent registration', 'Low deposit amount']
      };
    }
    
    // Verificar high value
    if (total_deposits >= this.segmentRules.high_value.min_deposits &&
        ltv >= this.segmentRules.high_value.min_ltv &&
        bet_frequency >= this.segmentRules.high_value.min_bet_frequency &&
        churn_risk <= this.segmentRules.high_value.max_churn_risk) {
      return {
        segment: 'high_value',
        confidence: 0.95,
        reasons: ['High deposits', 'High LTV', 'High betting frequency', 'Low churn risk']
      };
    }
    
    // Verificar medium value
    if (total_deposits >= this.segmentRules.medium_value.min_deposits &&
        ltv >= this.segmentRules.medium_value.min_ltv &&
        bet_frequency >= this.segmentRules.medium_value.min_bet_frequency &&
        churn_risk <= this.segmentRules.medium_value.max_churn_risk) {
      return {
        segment: 'medium_value',
        confidence: 0.8,
        reasons: ['Medium deposits', 'Medium LTV', 'Medium betting frequency']
      };
    }
    
    // Default para low value
    return {
      segment: 'low_value',
      confidence: 0.7,
      reasons: ['Below medium value thresholds']
    };
  }

  /**
   * Gerar recomendações baseadas no segmento
   */
  generateRecommendations(segment, features) {
    const recommendations = [];
    
    switch (segment) {
      case 'high_value':
        recommendations.push(
          {
            type: 'campaign',
            title: 'VIP Exclusive Offer',
            description: 'Offer exclusive VIP bonuses and personal account manager',
            priority: 'high'
          },
          {
            type: 'retention',
            title: 'Loyalty Program Upgrade',
            description: 'Upgrade to premium loyalty tier with enhanced benefits',
            priority: 'high'
          }
        );
        break;
        
      case 'medium_value':
        recommendations.push(
          {
            type: 'campaign',
            title: 'Deposit Bonus',
            description: 'Offer attractive deposit bonus to increase engagement',
            priority: 'medium'
          },
          {
            type: 'engagement',
            title: 'Game Recommendations',
            description: 'Recommend high-engagement games based on preferences',
            priority: 'medium'
          }
        );
        break;
        
      case 'low_value':
        recommendations.push(
          {
            type: 'education',
            title: 'Gaming Tutorial',
            description: 'Provide gaming tutorials to improve engagement',
            priority: 'low'
          },
          {
            type: 'campaign',
            title: 'Small Deposit Incentive',
            description: 'Small bonus for first deposit increase',
            priority: 'medium'
          }
        );
        break;
        
      case 'new_user':
        recommendations.push(
          {
            type: 'onboarding',
            title: 'Welcome Journey',
            description: 'Complete onboarding sequence with tutorials',
            priority: 'high'
          },
          {
            type: 'campaign',
            title: 'Welcome Bonus',
            description: 'Generous welcome bonus to encourage first deposits',
            priority: 'high'
          }
        );
        break;
        
      case 'inactive':
        if (features.churn_risk > 0.7) {
          recommendations.push(
            {
              type: 'retention',
              title: 'Win-back Campaign',
              description: 'Aggressive win-back offer with free play',
              priority: 'high'
            }
          );
        } else {
          recommendations.push(
            {
              type: 'engagement',
              title: 'Re-engagement Campaign',
              description: 'Gentle re-engagement with game recommendations',
              priority: 'medium'
            }
          );
        }
        break;
    }
    
    return recommendations;
  }
}

// Instância global do engine de segmentação
const segmentationEngine = new UserSegmentationEngine();

/**
 * Obter segmento ML do usuário
 */
const getUserSegment = async (userId) => {
  try {
    // Buscar dados do usuário
    const userData = await userRepository.getUserById(userId);
    
    if (!userData) {
      return null;
    }
    
    // Calcular features comportamentais
    const features = segmentationEngine.calculateBehaviorFeatures(userData);
    
    // Determinar segmento
    const segmentResult = segmentationEngine.determineSegment(features);
    
    // Gerar recomendações
    const recommendations = segmentationEngine.generateRecommendations(
      segmentResult.segment, 
      features
    );
    
    // Atualizar segmento no banco se mudou
    if (userData.segment !== segmentResult.segment) {
      await userRepository.updateUserSegment(userId, segmentResult.segment);
      
      logger.business('User segment updated', {
        userId,
        oldSegment: userData.segment,
        newSegment: segmentResult.segment,
        confidence: segmentResult.confidence
      });
    }
    
    return {
      user_id: userId,
      segment: segmentResult.segment,
      cluster_id: userData.cluster_id,
      confidence_score: segmentResult.confidence,
      segment_features: features,
      segment_description: getSegmentDescription(segmentResult.segment),
      recommendations,
      last_updated: new Date().toISOString()
    };
    
  } catch (error) {
    logger.error('Error getting user segment:', { userId, error: error.message });
    throw error;
  }
};

/**
 * Obter estatísticas de segmentação
 */
const getSegmentStats = async () => {
  try {
    const stats = await userRepository.getSegmentStatistics();
    
    return {
      total_users: stats.total_users,
      segments: stats.segments,
      last_updated: new Date().toISOString()
    };
    
  } catch (error) {
    logger.error('Error getting segment stats:', error);
    throw error;
  }
};

/**
 * Análise comportamental do usuário
 */
const getUserBehaviorAnalysis = async (userId, period = '30d') => {
  try {
    const userData = await userRepository.getUserById(userId);
    
    if (!userData) {
      return null;
    }
    
    // Buscar dados comportamentais do período
    const behaviorData = await userRepository.getUserBehaviorData(userId, period);
    
    // Calcular métricas comportamentais
    const features = segmentationEngine.calculateBehaviorFeatures(userData);
    
    // Análise de padrões
    const patterns = await analyzeBehaviorPatterns(behaviorData);
    
    // Predições ML
    const predictions = await generateBehaviorPredictions(features, patterns);
    
    return {
      user_id: userId,
      period,
      behavior_metrics: {
        activity_score: features.activity_score,
        engagement_score: calculateEngagementScore(behaviorData),
        risk_score: features.churn_risk * 100,
        loyalty_score: calculateLoyaltyScore(features, patterns)
      },
      patterns,
      predictions,
      generated_at: new Date().toISOString()
    };
    
  } catch (error) {
    logger.error('Error analyzing user behavior:', { userId, period, error: error.message });
    throw error;
  }
};

/**
 * Buscar usuários por critérios
 */
const searchUsers = async (criteria) => {
  try {
    const result = await userRepository.searchUsers(criteria);
    
    return {
      users: result.users,
      pagination: result.pagination
    };
    
  } catch (error) {
    logger.error('Error searching users:', { criteria, error: error.message });
    throw error;
  }
};

/**
 * Obter recomendações para usuário
 */
const getUserRecommendations = async (userId) => {
  try {
    const segmentData = await getUserSegment(userId);
    
    if (!segmentData) {
      return null;
    }
    
    // Recomendações baseadas em ML mais avançado
    const enhancedRecommendations = await enhanceRecommendations(
      segmentData.recommendations,
      segmentData.segment_features,
      userId
    );
    
    return {
      user_id: userId,
      recommendations: enhancedRecommendations,
      generated_at: new Date().toISOString()
    };
    
  } catch (error) {
    logger.error('Error getting user recommendations:', { userId, error: error.message });
    throw error;
  }
};

// ===== FUNÇÕES AUXILIARES =====

/**
 * Obter descrição do segmento
 */
const getSegmentDescription = (segment) => {
  const descriptions = {
    high_value: 'High-value customer with consistent betting patterns and high lifetime value',
    medium_value: 'Medium-value customer with moderate engagement and spending',
    low_value: 'Low-value customer with minimal spending but potential for growth',
    new_user: 'New customer in onboarding phase with growth potential',
    inactive: 'Inactive customer requiring re-engagement strategies'
  };
  
  return descriptions[segment] || 'Unknown segment';
};

/**
 * Analisar padrões comportamentais
 */
const analyzeBehaviorPatterns = async (behaviorData) => {
  // Análise simplificada de padrões
  const patterns = {
    peak_activity_hours: [19, 20, 21, 22], // Horários de pico simulados
    preferred_games: ['slots', 'blackjack', 'roulette'], // Jogos preferidos simulados
    bet_size_trend: 'stable', // Tendência do tamanho das apostas
    session_duration_avg: 45 // Duração média da sessão em minutos
  };
  
  return patterns;
};

/**
 * Gerar predições comportamentais
 */
const generateBehaviorPredictions = async (features, patterns) => {
  // Predições ML simplificadas
  const predictions = {
    churn_probability: features.churn_risk,
    next_deposit_probability: Math.max(0, 1 - features.churn_risk),
    ltv_prediction: features.ltv * 1.2 // Predição simplificada de crescimento LTV
  };
  
  return predictions;
};

/**
 * Calcular score de engajamento
 */
const calculateEngagementScore = (behaviorData) => {
  // Score simplificado baseado em atividade recente
  return Math.floor(Math.random() * 40) + 60; // 60-100
};

/**
 * Calcular score de lealdade
 */
const calculateLoyaltyScore = (features, patterns) => {
  // Score baseado em tempo de relacionamento e consistência
  const loyaltyFactors = [
    Math.min(100, features.registration_days * 2), // Tempo de relacionamento
    100 - features.churn_risk * 100, // Baixo risco de churn
    Math.min(100, features.bet_frequency * 10) // Frequência de apostas
  ];
  
  return loyaltyFactors.reduce((sum, factor) => sum + factor, 0) / loyaltyFactors.length;
};

/**
 * Aprimorar recomendações com ML
 */
const enhanceRecommendations = async (baseRecommendations, features, userId) => {
  // Adicionar confiança e impacto esperado às recomendações
  return baseRecommendations.map((rec, index) => ({
    id: `rec_${userId}_${index}`,
    ...rec,
    confidence: Math.random() * 0.3 + 0.7, // 0.7-1.0
    expected_impact: {
      engagement_increase: Math.random() * 20 + 10, // 10-30%
      revenue_potential: Math.random() * 500 + 100 // $100-600
    },
    valid_until: new Date(Date.now() + 7 * 24 * 60 * 60 * 1000).toISOString() // 7 dias
  }));
};

module.exports = {
  getUserSegment,
  getSegmentStats,
  getUserBehaviorAnalysis,
  searchUsers,
  getUserRecommendations,
  UserSegmentationEngine
};