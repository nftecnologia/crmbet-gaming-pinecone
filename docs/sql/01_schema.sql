-- =====================================================
-- CRM INTELIGENTE - DATABASE SCHEMA
-- PostgreSQL 15+ Compatible
-- =====================================================

-- Enable required extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_trgm";
CREATE EXTENSION IF NOT EXISTS "btree_gin";

-- =====================================================
-- USERS TABLE
-- Core user information and profile data
-- =====================================================
CREATE TABLE users (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name VARCHAR(255) NOT NULL,
    email VARCHAR(320) UNIQUE NOT NULL,
    phone VARCHAR(20),
    country_code VARCHAR(3),
    language VARCHAR(5) DEFAULT 'pt-BR',
    timezone VARCHAR(50) DEFAULT 'America/Sao_Paulo',
    
    -- Profile enrichment fields
    age_range VARCHAR(20), -- '18-25', '26-35', '36-45', '46-55', '55+'
    gender VARCHAR(10),
    location_city VARCHAR(100),
    location_state VARCHAR(100),
    
    -- Gaming profile
    preferred_games TEXT[], -- Array of game types
    risk_tolerance VARCHAR(20) DEFAULT 'medium', -- 'low', 'medium', 'high'
    lifetime_value DECIMAL(12,2) DEFAULT 0.00,
    
    -- Account status
    status VARCHAR(20) DEFAULT 'active' CHECK (status IN ('active', 'inactive', 'suspended', 'banned')),
    email_verified BOOLEAN DEFAULT FALSE,
    phone_verified BOOLEAN DEFAULT FALSE,
    kyc_status VARCHAR(20) DEFAULT 'pending' CHECK (kyc_status IN ('pending', 'approved', 'rejected')),
    
    -- Metadata
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    last_login_at TIMESTAMP WITH TIME ZONE,
    last_activity_at TIMESTAMP WITH TIME ZONE,
    
    -- Audit fields
    created_by UUID,
    updated_by UUID
);

-- =====================================================
-- USER_TRANSACTIONS TABLE
-- Comprehensive transaction tracking for ML features
-- =====================================================
CREATE TABLE user_transactions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    
    -- Transaction details
    transaction_type VARCHAR(50) NOT NULL, -- 'bet', 'deposit', 'withdrawal', 'bonus', 'commission'
    game_type VARCHAR(100), -- 'slots', 'live_casino', 'sports_betting', 'poker', etc.
    game_provider VARCHAR(100),
    game_name VARCHAR(200),
    
    -- Financial data
    amount DECIMAL(12,2) NOT NULL,
    currency VARCHAR(3) DEFAULT 'BRL',
    balance_before DECIMAL(12,2),
    balance_after DECIMAL(12,2),
    
    -- Transaction metadata
    channel VARCHAR(50), -- 'web', 'mobile_app', 'mobile_web', 'api'
    device_type VARCHAR(20), -- 'desktop', 'mobile', 'tablet'
    payment_method VARCHAR(50), -- for deposits/withdrawals
    
    -- Behavioral data for ML
    session_id VARCHAR(100),
    session_duration_minutes INTEGER,
    ip_address INET,
    user_agent TEXT,
    
    -- Risk and compliance
    risk_score DECIMAL(5,2), -- 0.00 to 100.00
    is_suspicious BOOLEAN DEFAULT FALSE,
    compliance_flags TEXT[], -- Array of compliance flags
    
    -- Status and processing
    status VARCHAR(20) DEFAULT 'completed' CHECK (status IN ('pending', 'completed', 'failed', 'cancelled')),
    processed_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    
    -- Timestamps
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    
    -- External references
    external_transaction_id VARCHAR(200),
    provider_transaction_id VARCHAR(200)
);

-- =====================================================
-- USER_CLUSTERS TABLE
-- ML clustering results and user segmentation
-- =====================================================
CREATE TABLE user_clusters (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    
    -- Cluster identification
    cluster_id INTEGER NOT NULL,
    cluster_name VARCHAR(100) NOT NULL,
    cluster_description TEXT,
    
    -- ML model information
    model_version VARCHAR(50) NOT NULL,
    algorithm_used VARCHAR(50), -- 'kmeans', 'dbscan', 'hierarchical', etc.
    
    -- Feature vectors and confidence
    features JSONB NOT NULL, -- Store the feature vector used for clustering
    confidence DECIMAL(5,4) NOT NULL, -- 0.0000 to 1.0000
    distance_to_centroid DECIMAL(10,6),
    
    -- Cluster characteristics
    cluster_size INTEGER, -- Number of users in this cluster
    cluster_characteristics JSONB, -- Key traits of this cluster
    
    -- Business insights
    value_segment VARCHAR(50), -- 'high_value', 'medium_value', 'low_value', 'at_risk', 'new_user'
    behavior_pattern VARCHAR(100), -- 'casual_player', 'high_roller', 'bonus_hunter', etc.
    churn_risk DECIMAL(5,4), -- 0.0000 to 1.0000
    
    -- Temporal data
    valid_from TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    valid_to TIMESTAMP WITH TIME ZONE,
    is_current BOOLEAN DEFAULT TRUE,
    
    -- Metadata
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    
    -- Ensure one current cluster per user
    UNIQUE(user_id, is_current) WHERE is_current = TRUE
);

-- =====================================================
-- CAMPAIGNS TABLE
-- Marketing campaign management and targeting
-- =====================================================
CREATE TABLE campaigns (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    
    -- Campaign identification
    name VARCHAR(255) NOT NULL,
    description TEXT,
    campaign_type VARCHAR(50) NOT NULL, -- 'email', 'sms', 'push', 'in_app', 'whatsapp'
    
    -- Targeting configuration
    cluster_target INTEGER[], -- Array of cluster IDs to target
    user_segments VARCHAR(100)[], -- Array of value segments to target
    geo_target VARCHAR(100)[], -- Array of countries/regions
    
    -- Campaign content
    subject_line VARCHAR(200),
    message TEXT NOT NULL,
    cta_text VARCHAR(100), -- Call-to-action text
    cta_url TEXT, -- Call-to-action URL
    
    -- Creative assets
    template_id VARCHAR(100),
    image_urls TEXT[],
    video_url TEXT,
    
    -- Personalization
    personalization_fields JSONB, -- Fields to personalize per user
    ab_test_variant VARCHAR(20), -- 'A', 'B', 'C', etc.
    
    -- Scheduling
    scheduled_at TIMESTAMP WITH TIME ZONE,
    send_timezone VARCHAR(50) DEFAULT 'America/Sao_Paulo',
    
    -- Budget and limits
    budget_total DECIMAL(12,2),
    budget_spent DECIMAL(12,2) DEFAULT 0.00,
    max_sends INTEGER,
    
    -- Campaign status and control
    status VARCHAR(20) DEFAULT 'draft' CHECK (status IN ('draft', 'scheduled', 'running', 'paused', 'completed', 'cancelled')),
    priority INTEGER DEFAULT 5, -- 1 (highest) to 10 (lowest)
    
    -- Approval workflow
    requires_approval BOOLEAN DEFAULT TRUE,
    approved_by UUID,
    approved_at TIMESTAMP WITH TIME ZONE,
    
    -- Performance tracking
    target_audience_size INTEGER,
    estimated_reach INTEGER,
    
    -- Metadata
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    created_by UUID NOT NULL,
    updated_by UUID
);

-- =====================================================
-- CAMPAIGN_RESULTS TABLE
-- Detailed campaign performance and user interactions
-- =====================================================
CREATE TABLE campaign_results (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    campaign_id UUID NOT NULL REFERENCES campaigns(id) ON DELETE CASCADE,
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    
    -- Delivery tracking
    sent_at TIMESTAMP WITH TIME ZONE,
    delivery_status VARCHAR(20), -- 'sent', 'delivered', 'bounced', 'failed'
    delivery_error TEXT,
    
    -- Engagement tracking
    opened_at TIMESTAMP WITH TIME ZONE,
    clicked_at TIMESTAMP WITH TIME ZONE,
    unsubscribed_at TIMESTAMP WITH TIME ZONE,
    
    -- Conversion tracking
    converted_at TIMESTAMP WITH TIME ZONE,
    conversion_type VARCHAR(50), -- 'deposit', 'bet', 'registration', 'reactivation'
    conversion_value DECIMAL(12,2),
    conversion_currency VARCHAR(3) DEFAULT 'BRL',
    
    -- Interaction details
    click_count INTEGER DEFAULT 0,
    time_to_open_minutes INTEGER, -- Minutes from sent to opened
    time_to_click_minutes INTEGER, -- Minutes from opened to clicked
    time_to_convert_minutes INTEGER, -- Minutes from clicked to converted
    
    -- Device and context
    device_type VARCHAR(20),
    channel VARCHAR(50),
    ip_address INET,
    user_agent TEXT,
    
    -- A/B testing
    variant_shown VARCHAR(20),
    
    -- Revenue attribution
    attributed_revenue DECIMAL(12,2) DEFAULT 0.00,
    attribution_window_days INTEGER DEFAULT 7,
    
    -- Quality scores
    engagement_score DECIMAL(5,2), -- Calculated engagement metric
    
    -- Metadata
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    
    -- Prevent duplicate results per user per campaign
    UNIQUE(campaign_id, user_id)
);

-- =====================================================
-- ADDITIONAL SUPPORTING TABLES
-- =====================================================

-- User sessions for behavioral analysis
CREATE TABLE user_sessions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    session_id VARCHAR(100) NOT NULL,
    
    -- Session details
    started_at TIMESTAMP WITH TIME ZONE NOT NULL,
    ended_at TIMESTAMP WITH TIME ZONE,
    duration_minutes INTEGER,
    
    -- Device and location
    device_type VARCHAR(20),
    device_fingerprint VARCHAR(200),
    ip_address INET,
    country_code VARCHAR(3),
    city VARCHAR(100),
    
    -- Activity metrics
    page_views INTEGER DEFAULT 0,
    games_played INTEGER DEFAULT 0,
    total_bets INTEGER DEFAULT 0,
    total_bet_amount DECIMAL(12,2) DEFAULT 0.00,
    
    -- Behavioral flags
    is_mobile BOOLEAN DEFAULT FALSE,
    is_suspicious BOOLEAN DEFAULT FALSE,
    
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- System configuration for ML parameters
CREATE TABLE ml_configurations (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    config_key VARCHAR(100) UNIQUE NOT NULL,
    config_value JSONB NOT NULL,
    description TEXT,
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- =====================================================
-- TRIGGERS FOR AUTOMATIC UPDATES
-- =====================================================

-- Function to update updated_at timestamp
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Apply triggers to all tables with updated_at
CREATE TRIGGER update_users_updated_at BEFORE UPDATE ON users FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
CREATE TRIGGER update_user_clusters_updated_at BEFORE UPDATE ON user_clusters FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
CREATE TRIGGER update_campaigns_updated_at BEFORE UPDATE ON campaigns FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
CREATE TRIGGER update_campaign_results_updated_at BEFORE UPDATE ON campaign_results FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
CREATE TRIGGER update_ml_configurations_updated_at BEFORE UPDATE ON ml_configurations FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- Update last_activity_at when user has transactions
CREATE OR REPLACE FUNCTION update_user_last_activity()
RETURNS TRIGGER AS $$
BEGIN
    UPDATE users 
    SET last_activity_at = NEW.timestamp,
        updated_at = NOW()
    WHERE id = NEW.user_id;
    RETURN NEW;
END;
$$ language 'plpgsql';

CREATE TRIGGER update_user_activity_on_transaction 
    AFTER INSERT ON user_transactions 
    FOR EACH ROW EXECUTE FUNCTION update_user_last_activity();

-- =====================================================
-- COMMENTS FOR DOCUMENTATION
-- =====================================================

COMMENT ON TABLE users IS 'Core user profiles with gaming preferences and status tracking';
COMMENT ON TABLE user_transactions IS 'Comprehensive transaction log for ML feature engineering';
COMMENT ON TABLE user_clusters IS 'ML clustering results with temporal versioning';
COMMENT ON TABLE campaigns IS 'Marketing campaign definitions and targeting rules';
COMMENT ON TABLE campaign_results IS 'Detailed campaign performance and conversion tracking';
COMMENT ON TABLE user_sessions IS 'Session-level behavioral data for analysis';
COMMENT ON TABLE ml_configurations IS 'System configuration for ML pipelines and parameters';

-- Schema creation completed successfully
SELECT 'CRM Schema created successfully!' as status;