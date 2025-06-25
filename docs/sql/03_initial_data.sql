-- =====================================================
-- CRM INTELIGENTE - INITIAL DATA & CONFIGURATION
-- Default configurations and sample data for system bootstrap
-- =====================================================

-- =====================================================
-- ML CONFIGURATION DEFAULTS
-- System parameters for machine learning pipelines
-- =====================================================

INSERT INTO ml_configurations (config_key, config_value, description, is_active) VALUES

-- Clustering algorithm parameters
('clustering.algorithm', '"kmeans"', 'Default clustering algorithm (kmeans, dbscan, hierarchical)', true),
('clustering.n_clusters', '8', 'Default number of clusters for K-means', true),
('clustering.min_samples', '50', 'Minimum samples per cluster for DBSCAN', true),
('clustering.eps', '0.5', 'Epsilon parameter for DBSCAN clustering', true),
('clustering.max_iter', '300', 'Maximum iterations for clustering algorithms', true),
('clustering.random_state', '42', 'Random state for reproducible clustering', true),

-- Feature engineering parameters
('features.time_windows', '[7, 30, 90, 180]', 'Time windows in days for feature aggregation', true),
('features.transaction_types', '["bet", "deposit", "withdrawal", "bonus"]', 'Transaction types to include in features', true),
('features.min_transactions', '10', 'Minimum transactions required for clustering', true),
('features.normalization', '"standard"', 'Feature normalization method (standard, minmax, robust)', true),

-- Model retraining schedule
('model.retrain_frequency_days', '7', 'How often to retrain clustering models (days)', true),
('model.min_data_points', '1000', 'Minimum data points required for model training', true),
('model.validation_split', '0.2', 'Validation split ratio for model evaluation', true),
('model.confidence_threshold', '0.7', 'Minimum confidence threshold for cluster assignment', true),

-- Campaign targeting parameters
('campaign.default_send_timezone', '"America/Sao_Paulo"', 'Default timezone for campaign sending', true),
('campaign.max_daily_sends_per_user', '3', 'Maximum campaigns per user per day', true),
('campaign.attribution_window_days', '7', 'Attribution window for campaign conversions', true),
('campaign.min_cluster_size', '100', 'Minimum cluster size for targeting', true),

-- Performance and monitoring
('system.batch_size_clustering', '5000', 'Batch size for clustering operations', true),
('system.batch_size_features', '10000', 'Batch size for feature extraction', true),
('system.cache_ttl_minutes', '60', 'Cache TTL for ML predictions in minutes', true),
('system.monitoring_enabled', 'true', 'Enable system monitoring and alerts', true),

-- Risk and compliance
('risk.high_risk_threshold', '75.0', 'Risk score threshold for high-risk classification', true),
('risk.suspicious_transaction_limit', '10000.0', 'Transaction amount threshold for suspicion', true),
('compliance.kyc_required_limit', '2000.0', 'Transaction limit requiring KYC verification', true),
('compliance.aml_check_enabled', 'true', 'Enable Anti-Money Laundering checks', true);

-- =====================================================
-- DEFAULT USER SEGMENTS AND CLUSTER DEFINITIONS
-- Predefined segments for initial system operation
-- =====================================================

-- Insert sample cluster configurations for initial setup
-- These will be replaced by actual ML-generated clusters
INSERT INTO user_clusters (
    user_id, cluster_id, cluster_name, cluster_description,
    model_version, algorithm_used, features, confidence,
    cluster_characteristics, value_segment, behavior_pattern,
    churn_risk, valid_from, is_current
) VALUES
-- Placeholder entries will be created when first users register
-- This table will be populated by the ML pipeline

-- For now, we'll create the system user for administrative purposes
((SELECT uuid_generate_v4()), 0, 'System Default', 'Default cluster for new users',
 '1.0.0', 'rule_based', '{"new_user": true}', 1.0000,
 '{"type": "new_user", "characteristics": ["unclassified", "pending_data"]}',
 'new_user', 'unclassified', 0.0000, NOW(), true)
ON CONFLICT DO NOTHING;

-- =====================================================
-- SYSTEM USERS AND ROLES
-- Administrative users for system operations
-- =====================================================

-- Create system admin user for internal operations
INSERT INTO users (
    id, name, email, phone, country_code, language,
    age_range, gender, status, email_verified, phone_verified, kyc_status,
    created_at, updated_at
) VALUES (
    '00000000-0000-0000-0000-000000000001',
    'System Administrator',
    'admin@crmbet.com',
    NULL,
    'BRA',
    'pt-BR',
    NULL,
    NULL,
    'active',
    true,
    false,
    'approved',
    NOW(),
    NOW()
) ON CONFLICT (id) DO NOTHING;

-- Create ML service user for automated operations
INSERT INTO users (
    id, name, email, phone, country_code, language,
    status, email_verified, kyc_status,
    created_at, updated_at
) VALUES (
    '00000000-0000-0000-0000-000000000002',
    'ML Service',
    'ml-service@crmbet.com',
    NULL,
    'BRA',
    'pt-BR',
    'active',
    true,
    'approved',
    NOW(),
    NOW()
) ON CONFLICT (id) DO NOTHING;

-- =====================================================
-- SAMPLE CAMPAIGN TEMPLATES
-- Pre-built campaign templates for common use cases
-- =====================================================

-- Welcome campaign for new users
INSERT INTO campaigns (
    id, name, description, campaign_type, user_segments,
    subject_line, message, cta_text, cta_url,
    template_id, personalization_fields,
    status, priority, requires_approval,
    created_by, updated_by
) VALUES (
    uuid_generate_v4(),
    'Welcome New Users',
    'Welcome campaign for newly registered users',
    'email',
    ARRAY['new_user'],
    'Bem-vindo ao CRMBet! 🎉',
    'Olá {{name}}, seja bem-vindo(a) ao CRMBet! Estamos felizes em tê-lo(a) conosco. Aproveite nossa oferta especial de boas-vindas e comece a ganhar hoje mesmo!',
    'Começar Agora',
    'https://crmbet.com/welcome-bonus',
    'welcome_template_v1',
    '{"name": "user.name", "bonus_amount": "campaign.bonus_amount"}',
    'draft',
    1,
    false,
    '00000000-0000-0000-0000-000000000001',
    '00000000-0000-0000-0000-000000000001'
),

-- Re-engagement campaign for inactive users
(
    uuid_generate_v4(),
    'Win-Back Inactive Users',
    'Re-engagement campaign for users who haven\'t been active in 30+ days',
    'email',
    ARRAY['at_risk'],
    'Sentimos sua falta! Volta aqui! 💫',
    'Olá {{name}}, notamos que você não tem visitado a plataforma ultimamente. Que tal voltar e aproveitar nossa oferta especial de retorno?',
    'Voltar e Jogar',
    'https://crmbet.com/comeback-bonus',
    'winback_template_v1',
    '{"name": "user.name", "last_login": "user.last_login_at", "comeback_bonus": "campaign.bonus_amount"}',
    'draft',
    3,
    true,
    '00000000-0000-0000-0000-000000000001',
    '00000000-0000-0000-0000-000000000001'
),

-- VIP promotion campaign
(
    uuid_generate_v4(),
    'VIP User Promotion',
    'Exclusive offers for high-value users',
    'email',
    ARRAY['high_value'],
    'Oferta VIP Exclusiva Apenas Para Você! 👑',
    'Caro {{name}}, como um de nossos jogadores VIP, você tem acesso a ofertas exclusivas. Aproveite esta promoção especial disponível apenas para membros VIP.',
    'Ver Oferta VIP',
    'https://crmbet.com/vip-offers',
    'vip_template_v1',
    '{"name": "user.name", "vip_level": "user.vip_level", "exclusive_bonus": "campaign.vip_bonus"}',
    'draft',
    2,
    true,
    '00000000-0000-0000-0000-000000000001',
    '00000000-0000-0000-0000-000000000001'
);

-- =====================================================
-- DATABASE MAINTENANCE CONFIGURATIONS
-- =====================================================

-- Configure automatic vacuum and analyze
ALTER TABLE users SET (
    autovacuum_vacuum_scale_factor = 0.1,
    autovacuum_analyze_scale_factor = 0.05
);

ALTER TABLE user_transactions SET (
    autovacuum_vacuum_scale_factor = 0.05,
    autovacuum_analyze_scale_factor = 0.02
);

ALTER TABLE campaign_results SET (
    autovacuum_vacuum_scale_factor = 0.05,
    autovacuum_analyze_scale_factor = 0.02
);

-- =====================================================
-- VIEWS FOR COMMON QUERIES
-- Pre-built views for frequently accessed data
-- =====================================================

-- Active users with latest cluster information
CREATE VIEW v_active_users_with_clusters AS
SELECT 
    u.id,
    u.name,
    u.email,
    u.status,
    u.lifetime_value,
    u.last_activity_at,
    u.created_at,
    uc.cluster_id,
    uc.cluster_name,
    uc.value_segment,
    uc.behavior_pattern,
    uc.churn_risk,
    uc.confidence
FROM users u
LEFT JOIN user_clusters uc ON u.id = uc.user_id AND uc.is_current = true
WHERE u.status = 'active';

-- User transaction summary for the last 30 days
CREATE VIEW v_user_transaction_summary_30d AS
SELECT 
    ut.user_id,
    u.name as user_name,
    u.email,
    COUNT(*) as transaction_count,
    SUM(CASE WHEN ut.transaction_type = 'bet' THEN ut.amount ELSE 0 END) as total_bets,
    SUM(CASE WHEN ut.transaction_type = 'deposit' THEN ut.amount ELSE 0 END) as total_deposits,
    SUM(CASE WHEN ut.transaction_type = 'withdrawal' THEN ut.amount ELSE 0 END) as total_withdrawals,
    AVG(ut.amount) as avg_transaction_amount,
    COUNT(DISTINCT ut.game_type) as unique_games_played,
    MAX(ut.timestamp) as last_transaction_at
FROM user_transactions ut
JOIN users u ON ut.user_id = u.id
WHERE ut.timestamp >= (NOW() - INTERVAL '30 days')
    AND ut.status = 'completed'
GROUP BY ut.user_id, u.name, u.email;

-- Campaign performance summary
CREATE VIEW v_campaign_performance AS
SELECT 
    c.id as campaign_id,
    c.name,
    c.campaign_type,
    c.status,
    c.created_at,
    COUNT(cr.id) as total_sent,
    COUNT(cr.opened_at) as total_opened,
    COUNT(cr.clicked_at) as total_clicked,
    COUNT(cr.converted_at) as total_converted,
    ROUND(
        CASE WHEN COUNT(cr.id) > 0 
        THEN (COUNT(cr.opened_at)::decimal / COUNT(cr.id) * 100) 
        ELSE 0 END, 2
    ) as open_rate_percent,
    ROUND(
        CASE WHEN COUNT(cr.opened_at) > 0 
        THEN (COUNT(cr.clicked_at)::decimal / COUNT(cr.opened_at) * 100) 
        ELSE 0 END, 2
    ) as click_rate_percent,
    ROUND(
        CASE WHEN COUNT(cr.clicked_at) > 0 
        THEN (COUNT(cr.converted_at)::decimal / COUNT(cr.clicked_at) * 100) 
        ELSE 0 END, 2
    ) as conversion_rate_percent,
    SUM(cr.conversion_value) as total_revenue,
    SUM(cr.attributed_revenue) as attributed_revenue
FROM campaigns c
LEFT JOIN campaign_results cr ON c.id = cr.campaign_id
GROUP BY c.id, c.name, c.campaign_type, c.status, c.created_at;

-- High-risk users requiring attention
CREATE VIEW v_high_risk_users AS
SELECT 
    u.id,
    u.name,
    u.email,
    u.status,
    uc.churn_risk,
    uc.cluster_name,
    u.last_activity_at,
    u.lifetime_value,
    (NOW() - u.last_activity_at) as days_inactive
FROM users u
JOIN user_clusters uc ON u.id = uc.user_id AND uc.is_current = true
WHERE (uc.churn_risk > 0.7 OR u.last_activity_at < (NOW() - INTERVAL '14 days'))
    AND u.status = 'active'
    AND u.lifetime_value > 100
ORDER BY uc.churn_risk DESC, u.last_activity_at ASC;

-- =====================================================
-- SECURITY AND PERMISSIONS
-- Row-level security and access controls
-- =====================================================

-- Enable row-level security on sensitive tables
-- ALTER TABLE users ENABLE ROW LEVEL SECURITY;
-- ALTER TABLE user_transactions ENABLE ROW LEVEL SECURITY;
-- ALTER TABLE user_clusters ENABLE ROW LEVEL SECURITY;

-- Note: Actual RLS policies should be implemented based on your application's
-- user roles and security requirements

-- =====================================================
-- HEALTH CHECK FUNCTIONS
-- Functions for monitoring database health
-- =====================================================

CREATE OR REPLACE FUNCTION get_database_health()
RETURNS TABLE(
    metric_name VARCHAR(50),
    metric_value TEXT,
    status VARCHAR(20)
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        'Total Users'::VARCHAR(50),
        COUNT(*)::TEXT,
        CASE WHEN COUNT(*) > 0 THEN 'OK' ELSE 'WARNING' END::VARCHAR(20)
    FROM users
    
    UNION ALL
    
    SELECT 
        'Active Users'::VARCHAR(50),
        COUNT(*)::TEXT,
        CASE WHEN COUNT(*) > 0 THEN 'OK' ELSE 'WARNING' END::VARCHAR(20)
    FROM users WHERE status = 'active'
    
    UNION ALL
    
    SELECT 
        'Transactions Today'::VARCHAR(50),
        COUNT(*)::TEXT,
        'OK'::VARCHAR(20)
    FROM user_transactions 
    WHERE timestamp >= CURRENT_DATE
    
    UNION ALL
    
    SELECT 
        'Current Clusters'::VARCHAR(50),
        COUNT(DISTINCT cluster_id)::TEXT,
        CASE WHEN COUNT(DISTINCT cluster_id) > 0 THEN 'OK' ELSE 'WARNING' END::VARCHAR(20)
    FROM user_clusters WHERE is_current = true;
END;
$$ LANGUAGE plpgsql;

-- =====================================================
-- COMPLETION MESSAGE
-- =====================================================

SELECT 'Initial data and configurations loaded successfully!' as status;
SELECT 'ML configurations: ' || COUNT(*) as ml_config_count FROM ml_configurations WHERE is_active = true;
SELECT 'System users created: ' || COUNT(*) as system_users FROM users WHERE email LIKE '%@crmbet.com';
SELECT 'Sample campaigns: ' || COUNT(*) as sample_campaigns FROM campaigns WHERE created_by = '00000000-0000-0000-0000-000000000001';