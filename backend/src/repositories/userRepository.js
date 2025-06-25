/**
 * User Repository
 * 
 * Repositório para acesso a dados de usuários
 * com queries otimizadas e cache inteligente
 * 
 * @author CRM Team
 */

const { query, transaction } = require('../config/database');
const logger = require('../utils/logger');

/**
 * Obter usuário por ID
 */
const getUserById = async (userId) => {
  try {
    const result = await query(
      `SELECT 
        id, external_id, email, name, segment, cluster_id,
        registration_date, last_activity, total_deposits, 
        total_withdrawals, total_bets, bet_count, win_rate,
        metadata, created_at, updated_at
       FROM users 
       WHERE id = $1 AND active = true`,
      [userId]
    );
    
    return result.rows[0] || null;
  } catch (error) {
    logger.error('Error getting user by ID:', { userId, error: error.message });
    throw error;
  }
};

/**
 * Obter usuário por ID externo
 */
const getUserByExternalId = async (externalId) => {
  try {
    const result = await query(
      `SELECT 
        id, external_id, email, name, segment, cluster_id,
        registration_date, last_activity, total_deposits, 
        total_withdrawals, total_bets, bet_count, win_rate,
        metadata, created_at, updated_at
       FROM users 
       WHERE external_id = $1 AND active = true`,
      [externalId]
    );
    
    return result.rows[0] || null;
  } catch (error) {
    logger.error('Error getting user by external ID:', { externalId, error: error.message });
    throw error;
  }
};

/**
 * Atualizar segmento do usuário
 */
const updateUserSegment = async (userId, segment) => {
  try {
    const result = await query(
      `UPDATE users 
       SET segment = $1, updated_at = CURRENT_TIMESTAMP 
       WHERE id = $2 AND active = true
       RETURNING id, segment`,
      [segment, userId]
    );
    
    return result.rows[0] || null;
  } catch (error) {
    logger.error('Error updating user segment:', { userId, segment, error: error.message });
    throw error;
  }
};

/**
 * Obter estatísticas de segmentação
 */
const getSegmentStatistics = async () => {
  try {
    const result = await query(`
      SELECT 
        COUNT(*) as total_users,
        COUNT(CASE WHEN segment = 'high_value' THEN 1 END) as high_value_count,
        COUNT(CASE WHEN segment = 'medium_value' THEN 1 END) as medium_value_count,
        COUNT(CASE WHEN segment = 'low_value' THEN 1 END) as low_value_count,
        COUNT(CASE WHEN segment = 'new_user' THEN 1 END) as new_user_count,
        COUNT(CASE WHEN segment = 'inactive' THEN 1 END) as inactive_count,
        AVG(CASE WHEN segment = 'high_value' THEN total_deposits - total_withdrawals END) as high_value_avg_ltv,
        AVG(CASE WHEN segment = 'medium_value' THEN total_deposits - total_withdrawals END) as medium_value_avg_ltv,
        AVG(CASE WHEN segment = 'low_value' THEN total_deposits - total_withdrawals END) as low_value_avg_ltv,
        AVG(CASE WHEN segment = 'new_user' THEN total_deposits - total_withdrawals END) as new_user_avg_ltv,
        AVG(CASE WHEN segment = 'inactive' THEN total_deposits - total_withdrawals END) as inactive_avg_ltv
      FROM users 
      WHERE active = true
    `);
    
    const row = result.rows[0];
    const totalUsers = parseInt(row.total_users);
    
    return {
      total_users: totalUsers,
      segments: {
        high_value: {
          count: parseInt(row.high_value_count),
          percentage: totalUsers > 0 ? (parseInt(row.high_value_count) / totalUsers * 100) : 0,
          avg_ltv: parseFloat(row.high_value_avg_ltv) || 0
        },
        medium_value: {
          count: parseInt(row.medium_value_count),
          percentage: totalUsers > 0 ? (parseInt(row.medium_value_count) / totalUsers * 100) : 0,
          avg_ltv: parseFloat(row.medium_value_avg_ltv) || 0
        },
        low_value: {
          count: parseInt(row.low_value_count),
          percentage: totalUsers > 0 ? (parseInt(row.low_value_count) / totalUsers * 100) : 0,
          avg_ltv: parseFloat(row.low_value_avg_ltv) || 0
        },
        new_user: {
          count: parseInt(row.new_user_count),
          percentage: totalUsers > 0 ? (parseInt(row.new_user_count) / totalUsers * 100) : 0,
          avg_ltv: parseFloat(row.new_user_avg_ltv) || 0
        },
        inactive: {
          count: parseInt(row.inactive_count),
          percentage: totalUsers > 0 ? (parseInt(row.inactive_count) / totalUsers * 100) : 0,
          avg_ltv: parseFloat(row.inactive_avg_ltv) || 0
        }
      }
    };
  } catch (error) {
    logger.error('Error getting segment statistics:', error);
    throw error;
  }
};

/**
 * Obter dados comportamentais do usuário
 */
const getUserBehaviorData = async (userId, period = '30d') => {
  try {
    // Converter período para dias
    const days = period === '7d' ? 7 : period === '30d' ? 30 : period === '90d' ? 90 : 365;
    
    const result = await query(`
      SELECT 
        u.id, u.total_deposits, u.total_bets, u.bet_count,
        u.win_rate, u.last_activity, u.registration_date,
        -- Dados simulados de comportamento (em um sistema real viriam de outras tabelas)
        CASE 
          WHEN u.last_activity > CURRENT_DATE - INTERVAL '7 days' THEN 80
          WHEN u.last_activity > CURRENT_DATE - INTERVAL '30 days' THEN 60
          ELSE 30
        END as activity_level,
        CASE 
          WHEN u.bet_count > 100 THEN 'high'
          WHEN u.bet_count > 20 THEN 'medium'
          ELSE 'low'
        END as engagement_level
      FROM users u
      WHERE u.id = $1 AND u.active = true
    `, [userId]);
    
    return result.rows[0] || null;
  } catch (error) {
    logger.error('Error getting user behavior data:', { userId, period, error: error.message });
    throw error;
  }
};

/**
 * Buscar usuários por critérios
 */
const searchUsers = async (criteria) => {
  try {
    const { email, segment, cluster_id, page = 1, limit = 20 } = criteria;
    const offset = (page - 1) * limit;
    
    let whereConditions = ['active = true'];
    let params = [];
    let paramIndex = 1;
    
    if (email) {
      whereConditions.push(`email ILIKE $${paramIndex}`);
      params.push(`%${email}%`);
      paramIndex++;
    }
    
    if (segment) {
      whereConditions.push(`segment = $${paramIndex}`);
      params.push(segment);
      paramIndex++;
    }
    
    if (cluster_id) {
      whereConditions.push(`cluster_id = $${paramIndex}`);
      params.push(cluster_id);
      paramIndex++;
    }
    
    const whereClause = whereConditions.join(' AND ');
    
    // Query para contar total
    const countResult = await query(
      `SELECT COUNT(*) as total FROM users WHERE ${whereClause}`,
      params
    );
    
    const total = parseInt(countResult.rows[0].total);
    
    // Query para buscar usuários
    const usersResult = await query(
      `SELECT 
        id, external_id, email, name, segment, cluster_id,
        registration_date, last_activity, total_deposits, 
        total_withdrawals, total_bets, bet_count, win_rate,
        created_at, updated_at
       FROM users 
       WHERE ${whereClause}
       ORDER BY created_at DESC
       LIMIT $${paramIndex} OFFSET $${paramIndex + 1}`,
      [...params, limit, offset]
    );
    
    return {
      users: usersResult.rows,
      pagination: {
        page,
        limit,
        total,
        pages: Math.ceil(total / limit)
      }
    };
  } catch (error) {
    logger.error('Error searching users:', { criteria, error: error.message });
    throw error;
  }
};

/**
 * Criar novo usuário
 */
const createUser = async (userData) => {
  try {
    const result = await query(
      `INSERT INTO users (
        external_id, email, name, segment, cluster_id,
        total_deposits, total_withdrawals, total_bets, 
        bet_count, win_rate, metadata
       ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)
       RETURNING *`,
      [
        userData.external_id,
        userData.email,
        userData.name,
        userData.segment || 'new_user',
        userData.cluster_id || null,
        userData.total_deposits || 0,
        userData.total_withdrawals || 0,
        userData.total_bets || 0,
        userData.bet_count || 0,
        userData.win_rate || 0,
        userData.metadata || {}
      ]
    );
    
    return result.rows[0];
  } catch (error) {
    logger.error('Error creating user:', { userData, error: error.message });
    throw error;
  }
};

/**
 * Atualizar dados do usuário
 */
const updateUser = async (userId, updateData) => {
  try {
    const setClause = [];
    const params = [];
    let paramIndex = 1;
    
    // Campos atualizáveis
    const updatableFields = [
      'email', 'name', 'segment', 'cluster_id', 'total_deposits',
      'total_withdrawals', 'total_bets', 'bet_count', 'win_rate', 
      'last_activity', 'metadata'
    ];
    
    for (const field of updatableFields) {
      if (updateData[field] !== undefined) {
        setClause.push(`${field} = $${paramIndex}`);
        params.push(updateData[field]);
        paramIndex++;
      }
    }
    
    if (setClause.length === 0) {
      throw new Error('No fields to update');
    }
    
    setClause.push('updated_at = CURRENT_TIMESTAMP');
    params.push(userId);
    
    const result = await query(
      `UPDATE users 
       SET ${setClause.join(', ')}
       WHERE id = $${paramIndex} AND active = true
       RETURNING *`,
      params
    );
    
    return result.rows[0] || null;
  } catch (error) {
    logger.error('Error updating user:', { userId, updateData, error: error.message });
    throw error;
  }
};

/**
 * Soft delete do usuário
 */
const deleteUser = async (userId) => {
  try {
    const result = await query(
      `UPDATE users 
       SET active = false, updated_at = CURRENT_TIMESTAMP 
       WHERE id = $1
       RETURNING id`,
      [userId]
    );
    
    return result.rows[0] || null;
  } catch (error) {
    logger.error('Error deleting user:', { userId, error: error.message });
    throw error;
  }
};

/**
 * Obter usuários por cluster
 */
const getUsersByCluster = async (clusterId, options = {}) => {
  try {
    const { page = 1, limit = 50, sortBy = 'total_deposits', sortOrder = 'desc', activeOnly = true } = options;
    const offset = (page - 1) * limit;
    
    let whereConditions = [`cluster_id = $1`];
    let params = [clusterId];
    let paramIndex = 2;
    
    if (activeOnly) {
      whereConditions.push('active = true');
    }
    
    const whereClause = whereConditions.join(' AND ');
    const orderClause = `${sortBy} ${sortOrder.toUpperCase()}`;
    
    // Query para contar total
    const countResult = await query(
      `SELECT COUNT(*) as total FROM users WHERE ${whereClause}`,
      params
    );
    
    const total = parseInt(countResult.rows[0].total);
    
    // Query para buscar usuários
    const usersResult = await query(
      `SELECT 
        id, external_id, email, name, segment, cluster_id,
        registration_date, last_activity, total_deposits, 
        total_withdrawals, total_bets, bet_count, win_rate,
        (total_deposits - total_withdrawals) as ltv,
        CASE 
          WHEN last_activity > CURRENT_DATE - INTERVAL '7 days' THEN 100
          WHEN last_activity > CURRENT_DATE - INTERVAL '30 days' THEN 70
          WHEN last_activity > CURRENT_DATE - INTERVAL '90 days' THEN 40
          ELSE 10
        END as activity_score,
        created_at, updated_at
       FROM users 
       WHERE ${whereClause}
       ORDER BY ${orderClause}
       LIMIT $${paramIndex} OFFSET $${paramIndex + 1}`,
      [...params, limit, offset]
    );
    
    return {
      users: usersResult.rows,
      pagination: {
        page,
        limit,
        total,
        pages: Math.ceil(total / limit)
      }
    };
  } catch (error) {
    logger.error('Error getting users by cluster:', { clusterId, options, error: error.message });
    throw error;
  }
};

/**
 * Atualizar atividade do usuário
 */
const updateUserActivity = async (userId, activityData) => {
  try {
    await transaction(async (client) => {
      // Atualizar última atividade
      await client.query(
        `UPDATE users 
         SET last_activity = CURRENT_TIMESTAMP,
             updated_at = CURRENT_TIMESTAMP
         WHERE id = $1`,
        [userId]
      );
      
      // Se há dados de aposta, atualizar estatísticas
      if (activityData.bet_amount) {
        await client.query(
          `UPDATE users 
           SET total_bets = total_bets + $1,
               bet_count = bet_count + 1,
               updated_at = CURRENT_TIMESTAMP
           WHERE id = $2`,
          [activityData.bet_amount, userId]
        );
      }
      
      // Se há dados de depósito, atualizar
      if (activityData.deposit_amount) {
        await client.query(
          `UPDATE users 
           SET total_deposits = total_deposits + $1,
               updated_at = CURRENT_TIMESTAMP
           WHERE id = $2`,
          [activityData.deposit_amount, userId]
        );
      }
      
      // Se há dados de saque, atualizar
      if (activityData.withdrawal_amount) {
        await client.query(
          `UPDATE users 
           SET total_withdrawals = total_withdrawals + $1,
               updated_at = CURRENT_TIMESTAMP
           WHERE id = $2`,
          [activityData.withdrawal_amount, userId]
        );
      }
    });
    
    return true;
  } catch (error) {
    logger.error('Error updating user activity:', { userId, activityData, error: error.message });
    throw error;
  }
};

module.exports = {
  getUserById,
  getUserByExternalId,
  updateUserSegment,
  getSegmentStatistics,
  getUserBehaviorData,
  searchUsers,
  createUser,
  updateUser,
  deleteUser,
  getUsersByCluster,
  updateUserActivity
};