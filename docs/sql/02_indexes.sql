-- =====================================================
-- CRM INTELIGENTE - PERFORMANCE INDEXES
-- Optimized for high-volume gaming/betting workloads
-- =====================================================

-- =====================================================
-- USERS TABLE INDEXES
-- =====================================================

-- Primary lookup indexes
CREATE INDEX CONCURRENTLY idx_users_email ON users(email);
CREATE INDEX CONCURRENTLY idx_users_phone ON users(phone) WHERE phone IS NOT NULL;
CREATE INDEX CONCURRENTLY idx_users_status ON users(status) WHERE status != 'active';

-- Performance indexes for common queries
CREATE INDEX CONCURRENTLY idx_users_created_at ON users(created_at);
CREATE INDEX CONCURRENTLY idx_users_last_activity ON users(last_activity_at DESC) WHERE last_activity_at IS NOT NULL;
CREATE INDEX CONCURRENTLY idx_users_lifetime_value ON users(lifetime_value DESC) WHERE lifetime_value > 0;

-- ML and segmentation indexes
CREATE INDEX CONCURRENTLY idx_users_country_lang ON users(country_code, language);
CREATE INDEX CONCURRENTLY idx_users_age_gender ON users(age_range, gender) WHERE age_range IS NOT NULL AND gender IS NOT NULL;
CREATE INDEX CONCURRENTLY idx_users_risk_tolerance ON users(risk_tolerance);

-- Full-text search on user names (for admin searches)
CREATE INDEX CONCURRENTLY idx_users_name_trgm ON users USING gin(name gin_trgm_ops);

-- Composite index for active users with recent activity
CREATE INDEX CONCURRENTLY idx_users_active_recent ON users(status, last_activity_at DESC) 
    WHERE status = 'active' AND last_activity_at > (NOW() - INTERVAL '30 days');

-- =====================================================
-- USER_TRANSACTIONS TABLE INDEXES
-- Critical for real-time analytics and ML pipelines
-- =====================================================

-- Foreign key optimization
CREATE INDEX CONCURRENTLY idx_transactions_user_id ON user_transactions(user_id);

-- Time-series optimized indexes (partitioning-ready)
CREATE INDEX CONCURRENTLY idx_transactions_timestamp ON user_transactions(timestamp DESC);
CREATE INDEX CONCURRENTLY idx_transactions_user_time ON user_transactions(user_id, timestamp DESC);
CREATE INDEX CONCURRENTLY idx_transactions_processed_at ON user_transactions(processed_at DESC);

-- Game and transaction type analytics
CREATE INDEX CONCURRENTLY idx_transactions_game_type ON user_transactions(game_type) WHERE game_type IS NOT NULL;
CREATE INDEX CONCURRENTLY idx_transactions_type_status ON user_transactions(transaction_type, status);
CREATE INDEX CONCURRENTLY idx_transactions_game_provider ON user_transactions(game_provider) WHERE game_provider IS NOT NULL;

-- Financial analytics (high-performance for reports)
CREATE INDEX CONCURRENTLY idx_transactions_amount ON user_transactions(amount DESC) WHERE amount > 0;
CREATE INDEX CONCURRENTLY idx_transactions_currency_amount ON user_transactions(currency, amount DESC);

-- Behavioral analysis indexes
CREATE INDEX CONCURRENTLY idx_transactions_session ON user_transactions(session_id) WHERE session_id IS NOT NULL;
CREATE INDEX CONCURRENTLY idx_transactions_channel ON user_transactions(channel);
CREATE INDEX CONCURRENTLY idx_transactions_device ON user_transactions(device_type);

-- Risk and compliance indexes
CREATE INDEX CONCURRENTLY idx_transactions_suspicious ON user_transactions(is_suspicious) WHERE is_suspicious = TRUE;
CREATE INDEX CONCURRENTLY idx_transactions_risk_score ON user_transactions(risk_score DESC) WHERE risk_score > 50.0;

-- ML feature engineering indexes (optimized for time windows)
CREATE INDEX CONCURRENTLY idx_transactions_ml_features ON user_transactions(user_id, game_type, amount, timestamp) 
    WHERE status = 'completed';

-- Recent activity index (last 24 hours) - frequently accessed
CREATE INDEX CONCURRENTLY idx_transactions_recent_24h ON user_transactions(user_id, timestamp) 
    WHERE timestamp > (NOW() - INTERVAL '24 hours');

-- Weekly aggregation index
CREATE INDEX CONCURRENTLY idx_transactions_weekly ON user_transactions(user_id, date_trunc('week', timestamp));

-- External reference lookups
CREATE INDEX CONCURRENTLY idx_transactions_external_id ON user_transactions(external_transaction_id) 
    WHERE external_transaction_id IS NOT NULL;

-- =====================================================
-- USER_CLUSTERS TABLE INDEXES
-- Optimized for ML model serving and analytics
-- =====================================================

-- Primary cluster lookups
CREATE INDEX CONCURRENTLY idx_clusters_user_current ON user_clusters(user_id) WHERE is_current = TRUE;
CREATE INDEX CONCURRENTLY idx_clusters_cluster_id ON user_clusters(cluster_id);
CREATE INDEX CONCURRENTLY idx_clusters_cluster_name ON user_clusters(cluster_name);

-- Model versioning and temporal queries
CREATE INDEX CONCURRENTLY idx_clusters_model_version ON user_clusters(model_version, created_at DESC);
CREATE INDEX CONCURRENTLY idx_clusters_valid_period ON user_clusters(valid_from, valid_to);
CREATE INDEX CONCURRENTLY idx_clusters_current_active ON user_clusters(is_current, created_at DESC) WHERE is_current = TRUE;

-- Business intelligence indexes
CREATE INDEX CONCURRENTLY idx_clusters_value_segment ON user_clusters(value_segment) WHERE is_current = TRUE;
CREATE INDEX CONCURRENTLY idx_clusters_behavior_pattern ON user_clusters(behavior_pattern) WHERE is_current = TRUE;
CREATE INDEX CONCURRENTLY idx_clusters_churn_risk ON user_clusters(churn_risk DESC) WHERE is_current = TRUE AND churn_risk > 0.5;

-- Confidence and quality indexes
CREATE INDEX CONCURRENTLY idx_clusters_confidence ON user_clusters(confidence DESC) WHERE is_current = TRUE;
CREATE INDEX CONCURRENTLY idx_clusters_low_confidence ON user_clusters(confidence ASC) 
    WHERE is_current = TRUE AND confidence < 0.7;

-- Feature similarity searches (GIN index for JSONB)
CREATE INDEX CONCURRENTLY idx_clusters_features ON user_clusters USING gin(features) WHERE is_current = TRUE;

-- Cluster analytics
CREATE INDEX CONCURRENTLY idx_clusters_size_analysis ON user_clusters(cluster_id, cluster_size, created_at) 
    WHERE is_current = TRUE;

-- =====================================================
-- CAMPAIGNS TABLE INDEXES
-- Optimized for campaign management and targeting
-- =====================================================

-- Campaign management indexes
CREATE INDEX CONCURRENTLY idx_campaigns_status ON campaigns(status, created_at DESC);
CREATE INDEX CONCURRENTLY idx_campaigns_created_by ON campaigns(created_by, created_at DESC);
CREATE INDEX CONCURRENTLY idx_campaigns_campaign_type ON campaigns(campaign_type);

-- Scheduling and execution indexes
CREATE INDEX CONCURRENTLY idx_campaigns_scheduled ON campaigns(scheduled_at) WHERE scheduled_at IS NOT NULL;
CREATE INDEX CONCURRENTLY idx_campaigns_active ON campaigns(status, scheduled_at) 
    WHERE status IN ('scheduled', 'running');

-- Targeting optimization
CREATE INDEX CONCURRENTLY idx_campaigns_cluster_target ON campaigns USING gin(cluster_target);
CREATE INDEX CONCURRENTLY idx_campaigns_user_segments ON campaigns USING gin(user_segments);
CREATE INDEX CONCURRENTLY idx_campaigns_geo_target ON campaigns USING gin(geo_target);

-- Budget and performance tracking
CREATE INDEX CONCURRENTLY idx_campaigns_budget ON campaigns(budget_total DESC, budget_spent);
CREATE INDEX CONCURRENTLY idx_campaigns_priority ON campaigns(priority, status) WHERE status IN ('scheduled', 'running');

-- A/B testing indexes
CREATE INDEX CONCURRENTLY idx_campaigns_ab_test ON campaigns(ab_test_variant, campaign_type) 
    WHERE ab_test_variant IS NOT NULL;

-- Approval workflow
CREATE INDEX CONCURRENTLY idx_campaigns_approval ON campaigns(requires_approval, approved_at) 
    WHERE requires_approval = TRUE;

-- =====================================================
-- CAMPAIGN_RESULTS TABLE INDEXES
-- High-performance for real-time campaign analytics
-- =====================================================

-- Primary relationship indexes
CREATE INDEX CONCURRENTLY idx_results_campaign_id ON campaign_results(campaign_id);
CREATE INDEX CONCURRENTLY idx_results_user_id ON campaign_results(user_id);

-- Funnel analysis indexes (critical for campaign optimization)
CREATE INDEX CONCURRENTLY idx_results_sent_at ON campaign_results(sent_at) WHERE sent_at IS NOT NULL;
CREATE INDEX CONCURRENTLY idx_results_opened_at ON campaign_results(opened_at) WHERE opened_at IS NOT NULL;
CREATE INDEX CONCURRENTLY idx_results_clicked_at ON campaign_results(clicked_at) WHERE clicked_at IS NOT NULL;
CREATE INDEX CONCURRENTLY idx_results_converted_at ON campaign_results(converted_at) WHERE converted_at IS NOT NULL;

-- Campaign performance analytics
CREATE INDEX CONCURRENTLY idx_results_campaign_performance ON campaign_results(
    campaign_id, 
    sent_at, 
    opened_at, 
    clicked_at, 
    converted_at
);

-- Conversion tracking and revenue attribution
CREATE INDEX CONCURRENTLY idx_results_conversion_value ON campaign_results(conversion_value DESC) 
    WHERE conversion_value > 0;
CREATE INDEX CONCURRENTLY idx_results_attributed_revenue ON campaign_results(attributed_revenue DESC) 
    WHERE attributed_revenue > 0;
CREATE INDEX CONCURRENTLY idx_results_conversion_type ON campaign_results(conversion_type) 
    WHERE conversion_type IS NOT NULL;

-- Real-time performance monitoring
CREATE INDEX CONCURRENTLY idx_results_recent_activity ON campaign_results(created_at DESC) 
    WHERE created_at > (NOW() - INTERVAL '1 hour');

-- Engagement analysis
CREATE INDEX CONCURRENTLY idx_results_engagement_score ON campaign_results(engagement_score DESC) 
    WHERE engagement_score IS NOT NULL;
CREATE INDEX CONCURRENTLY idx_results_click_count ON campaign_results(click_count DESC) WHERE click_count > 0;

-- Device and channel analysis
CREATE INDEX CONCURRENTLY idx_results_device_type ON campaign_results(device_type) WHERE device_type IS NOT NULL;
CREATE INDEX CONCURRENTLY idx_results_channel ON campaign_results(channel) WHERE channel IS NOT NULL;

-- A/B testing analysis
CREATE INDEX CONCURRENTLY idx_results_variant_performance ON campaign_results(variant_shown, conversion_value) 
    WHERE variant_shown IS NOT NULL;

-- =====================================================
-- USER_SESSIONS TABLE INDEXES
-- Behavioral analysis and fraud detection
-- =====================================================

-- Primary session tracking
CREATE INDEX CONCURRENTLY idx_sessions_user_id ON user_sessions(user_id);
CREATE INDEX CONCURRENTLY idx_sessions_session_id ON user_sessions(session_id);

-- Time-based analysis
CREATE INDEX CONCURRENTLY idx_sessions_started_at ON user_sessions(started_at DESC);
CREATE INDEX CONCURRENTLY idx_sessions_user_time ON user_sessions(user_id, started_at DESC);
CREATE INDEX CONCURRENTLY idx_sessions_duration ON user_sessions(duration_minutes DESC) WHERE duration_minutes IS NOT NULL;

-- Geographic and device analysis
CREATE INDEX CONCURRENTLY idx_sessions_country ON user_sessions(country_code) WHERE country_code IS NOT NULL;
CREATE INDEX CONCURRENTLY idx_sessions_device ON user_sessions(device_type, is_mobile);
CREATE INDEX CONCURRENTLY idx_sessions_ip_address ON user_sessions(ip_address);

-- Behavioral metrics
CREATE INDEX CONCURRENTLY idx_sessions_activity ON user_sessions(games_played DESC, total_bet_amount DESC);
CREATE INDEX CONCURRENTLY idx_sessions_suspicious ON user_sessions(is_suspicious) WHERE is_suspicious = TRUE;

-- Real-time monitoring (last 2 hours)
CREATE INDEX CONCURRENTLY idx_sessions_realtime ON user_sessions(started_at DESC) 
    WHERE started_at > (NOW() - INTERVAL '2 hours');

-- =====================================================
-- ML_CONFIGURATIONS TABLE INDEXES
-- =====================================================

CREATE INDEX CONCURRENTLY idx_ml_config_key ON ml_configurations(config_key) WHERE is_active = TRUE;
CREATE INDEX CONCURRENTLY idx_ml_config_active ON ml_configurations(is_active, updated_at DESC);

-- =====================================================
-- PARTIAL INDEXES FOR COMMON FILTERING
-- These indexes are smaller and faster for specific use cases
-- =====================================================

-- Active users only
CREATE INDEX CONCURRENTLY idx_users_active_only ON users(id, email, created_at) WHERE status = 'active';

-- Completed transactions only (for analytics)
CREATE INDEX CONCURRENTLY idx_transactions_completed ON user_transactions(user_id, timestamp, amount) 
    WHERE status = 'completed';

-- High-value transactions (for VIP analysis)
CREATE INDEX CONCURRENTLY idx_transactions_high_value ON user_transactions(user_id, amount, timestamp) 
    WHERE amount >= 1000.00 AND status = 'completed';

-- Recent successful campaigns
CREATE INDEX CONCURRENTLY idx_campaigns_recent_success ON campaigns(id, name, created_at) 
    WHERE status = 'completed' AND created_at > (NOW() - INTERVAL '90 days');

-- =====================================================
-- COMPOSITE INDEXES FOR COMPLEX QUERIES
-- =====================================================

-- User transaction summary (for dashboard queries)
CREATE INDEX CONCURRENTLY idx_user_transaction_summary ON user_transactions(
    user_id, 
    transaction_type, 
    timestamp DESC, 
    amount
) WHERE status = 'completed';

-- Campaign targeting effectiveness
CREATE INDEX CONCURRENTLY idx_campaign_targeting ON campaign_results(
    campaign_id,
    sent_at,
    opened_at,
    clicked_at,
    conversion_value
) WHERE sent_at IS NOT NULL;

-- ML feature extraction optimization
CREATE INDEX CONCURRENTLY idx_ml_feature_extraction ON user_transactions(
    user_id,
    game_type,
    amount,
    timestamp,
    channel
) WHERE status = 'completed' AND timestamp > (NOW() - INTERVAL '180 days');

-- =====================================================
-- MAINTENANCE COMMANDS
-- =====================================================

-- Update table statistics for query planner optimization
ANALYZE users;
ANALYZE user_transactions;
ANALYZE user_clusters;
ANALYZE campaigns;
ANALYZE campaign_results;
ANALYZE user_sessions;
ANALYZE ml_configurations;

-- Enable parallel index builds for future maintenance
SET maintenance_work_mem = '2GB';
SET max_parallel_maintenance_workers = 4;

SELECT 'Performance indexes created successfully!' as status;
SELECT 'Total indexes created: ' || count(*) as index_count 
FROM pg_indexes 
WHERE schemaname = 'public';