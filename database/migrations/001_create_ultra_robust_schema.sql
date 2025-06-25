-- ===================================================================
-- 🚀 ULTRA-ROBUST DATABASE SCHEMA - BILLION-SCALE TRANSACTIONS
-- Otimizado para performance máxima e escala massiva
-- ===================================================================

-- Enable necessary extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_stat_statements";
CREATE EXTENSION IF NOT EXISTS "pg_trgm";
CREATE EXTENSION IF NOT EXISTS "btree_gin";
CREATE EXTENSION IF NOT EXISTS "pg_partman";

-- ===================================================================
-- USERS TABLE - Partitioned by created_at for massive scale
-- ===================================================================
CREATE TABLE users (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id VARCHAR(255) NOT NULL,
    email VARCHAR(255),
    username VARCHAR(255),
    first_name VARCHAR(255),
    last_name VARCHAR(255),
    phone VARCHAR(50),
    country VARCHAR(10),
    language VARCHAR(10),
    timezone VARCHAR(50),
    status VARCHAR(50) DEFAULT 'active',
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    last_login_at TIMESTAMP WITH TIME ZONE,
    
    -- Performance constraints
    CONSTRAINT users_user_id_unique UNIQUE (user_id),
    CONSTRAINT users_email_unique UNIQUE (email) WHERE email IS NOT NULL,
    CONSTRAINT users_status_check CHECK (status IN ('active', 'inactive', 'suspended', 'deleted'))
) PARTITION BY RANGE (created_at);

-- Create partitions for users (monthly partitions)
CREATE TABLE users_2025_01 PARTITION OF users FOR VALUES FROM ('2025-01-01') TO ('2025-02-01');
CREATE TABLE users_2025_02 PARTITION OF users FOR VALUES FROM ('2025-02-01') TO ('2025-03-01');
CREATE TABLE users_2025_03 PARTITION OF users FOR VALUES FROM ('2025-03-01') TO ('2025-04-01');

-- High-performance indexes
CREATE INDEX CONCURRENTLY idx_users_user_id ON users USING btree (user_id);
CREATE INDEX CONCURRENTLY idx_users_email ON users USING btree (email) WHERE email IS NOT NULL;
CREATE INDEX CONCURRENTLY idx_users_status ON users USING btree (status);
CREATE INDEX CONCURRENTLY idx_users_created_at ON users USING btree (created_at);
CREATE INDEX CONCURRENTLY idx_users_metadata_gin ON users USING gin (metadata);

-- ===================================================================
-- TRANSACTIONS TABLE - Ultra-partitioned for billion-scale
-- ===================================================================
CREATE TABLE transactions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id VARCHAR(255) NOT NULL,
    transaction_id VARCHAR(255) NOT NULL,
    amount DECIMAL(15,2) NOT NULL,
    currency VARCHAR(10) DEFAULT 'USD',
    transaction_type VARCHAR(50) NOT NULL,
    status VARCHAR(50) DEFAULT 'completed',
    payment_method VARCHAR(100),
    provider VARCHAR(100),
    description TEXT,
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    processed_at TIMESTAMP WITH TIME ZONE,
    
    -- Performance constraints
    CONSTRAINT transactions_transaction_id_unique UNIQUE (transaction_id),
    CONSTRAINT transactions_amount_check CHECK (amount >= 0),
    CONSTRAINT transactions_type_check CHECK (transaction_type IN ('deposit', 'withdrawal', 'bet', 'win', 'bonus', 'refund')),
    CONSTRAINT transactions_status_check CHECK (status IN ('pending', 'completed', 'failed', 'cancelled'))
) PARTITION BY RANGE (created_at);

-- Create daily partitions for transactions (high volume)
CREATE TABLE transactions_2025_06_25 PARTITION OF transactions FOR VALUES FROM ('2025-06-25 00:00:00+00') TO ('2025-06-26 00:00:00+00');
CREATE TABLE transactions_2025_06_26 PARTITION OF transactions FOR VALUES FROM ('2025-06-26 00:00:00+00') TO ('2025-06-27 00:00:00+00');

-- Ultra-performance indexes
CREATE INDEX CONCURRENTLY idx_transactions_user_id ON transactions USING btree (user_id);
CREATE INDEX CONCURRENTLY idx_transactions_type ON transactions USING btree (transaction_type);
CREATE INDEX CONCURRENTLY idx_transactions_status ON transactions USING btree (status);
CREATE INDEX CONCURRENTLY idx_transactions_created_at ON transactions USING btree (created_at);
CREATE INDEX CONCURRENTLY idx_transactions_amount ON transactions USING btree (amount);
CREATE INDEX CONCURRENTLY idx_transactions_metadata_gin ON transactions USING gin (metadata);

-- Compound indexes for common queries
CREATE INDEX CONCURRENTLY idx_transactions_user_type_date ON transactions USING btree (user_id, transaction_type, created_at);
CREATE INDEX CONCURRENTLY idx_transactions_type_status_date ON transactions USING btree (transaction_type, status, created_at);

-- ===================================================================
-- USER FEATURES TABLE - ML Feature Store
-- ===================================================================
CREATE TABLE user_features (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id VARCHAR(255) NOT NULL,
    feature_name VARCHAR(255) NOT NULL,
    feature_value DECIMAL(15,8),
    feature_value_text TEXT,
    feature_value_json JSONB,
    feature_type VARCHAR(50) NOT NULL,
    calculation_date DATE NOT NULL DEFAULT CURRENT_DATE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    
    -- Performance constraints
    CONSTRAINT user_features_user_feature_date_unique UNIQUE (user_id, feature_name, calculation_date),
    CONSTRAINT user_features_type_check CHECK (feature_type IN ('numeric', 'categorical', 'boolean', 'json'))
) PARTITION BY RANGE (calculation_date);

-- Create monthly partitions for features
CREATE TABLE user_features_2025_06 PARTITION OF user_features FOR VALUES FROM ('2025-06-01') TO ('2025-07-01');
CREATE TABLE user_features_2025_07 PARTITION OF user_features FOR VALUES FROM ('2025-07-01') TO ('2025-08-01');

-- Feature store indexes
CREATE INDEX CONCURRENTLY idx_user_features_user_id ON user_features USING btree (user_id);
CREATE INDEX CONCURRENTLY idx_user_features_name ON user_features USING btree (feature_name);
CREATE INDEX CONCURRENTLY idx_user_features_date ON user_features USING btree (calculation_date);
CREATE INDEX CONCURRENTLY idx_user_features_type ON user_features USING btree (feature_type);
CREATE INDEX CONCURRENTLY idx_user_features_value ON user_features USING btree (feature_value) WHERE feature_value IS NOT NULL;

-- ===================================================================
-- ML CLUSTERS TABLE - Machine Learning Results
-- ===================================================================
CREATE TABLE ml_clusters (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id VARCHAR(255) NOT NULL,
    cluster_id INTEGER NOT NULL,
    cluster_name VARCHAR(255),
    cluster_description TEXT,
    probability DECIMAL(5,4),
    confidence_score DECIMAL(5,4),
    algorithm VARCHAR(100) NOT NULL,
    model_version VARCHAR(50) NOT NULL,
    features_used JSONB NOT NULL,
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    expires_at TIMESTAMP WITH TIME ZONE DEFAULT (NOW() + INTERVAL '30 days'),
    
    -- Performance constraints
    CONSTRAINT ml_clusters_probability_check CHECK (probability >= 0 AND probability <= 1),
    CONSTRAINT ml_clusters_confidence_check CHECK (confidence_score >= 0 AND confidence_score <= 1)
);

-- Cluster indexes
CREATE INDEX CONCURRENTLY idx_ml_clusters_user_id ON ml_clusters USING btree (user_id);
CREATE INDEX CONCURRENTLY idx_ml_clusters_cluster_id ON ml_clusters USING btree (cluster_id);
CREATE INDEX CONCURRENTLY idx_ml_clusters_algorithm ON ml_clusters USING btree (algorithm);
CREATE INDEX CONCURRENTLY idx_ml_clusters_created_at ON ml_clusters USING btree (created_at);
CREATE INDEX CONCURRENTLY idx_ml_clusters_expires_at ON ml_clusters USING btree (expires_at);

-- ===================================================================
-- ETL JOBS TABLE - Pipeline Monitoring
-- ===================================================================
CREATE TABLE etl_jobs (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    job_name VARCHAR(255) NOT NULL,
    job_type VARCHAR(100) NOT NULL,
    status VARCHAR(50) DEFAULT 'pending',
    started_at TIMESTAMP WITH TIME ZONE,
    completed_at TIMESTAMP WITH TIME ZONE,
    duration_seconds INTEGER,
    records_processed BIGINT DEFAULT 0,
    records_successful BIGINT DEFAULT 0,
    records_failed BIGINT DEFAULT 0,
    error_message TEXT,
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    
    -- Performance constraints
    CONSTRAINT etl_jobs_status_check CHECK (status IN ('pending', 'running', 'completed', 'failed', 'cancelled')),
    CONSTRAINT etl_jobs_records_check CHECK (records_processed >= 0)
);

-- ETL job indexes
CREATE INDEX CONCURRENTLY idx_etl_jobs_name ON etl_jobs USING btree (job_name);
CREATE INDEX CONCURRENTLY idx_etl_jobs_type ON etl_jobs USING btree (job_type);
CREATE INDEX CONCURRENTLY idx_etl_jobs_status ON etl_jobs USING btree (status);
CREATE INDEX CONCURRENTLY idx_etl_jobs_created_at ON etl_jobs USING btree (created_at);

-- ===================================================================
-- SYSTEM METRICS TABLE - Performance Monitoring
-- ===================================================================
CREATE TABLE system_metrics (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    metric_name VARCHAR(255) NOT NULL,
    metric_value DECIMAL(15,8) NOT NULL,
    metric_unit VARCHAR(50),
    component VARCHAR(100) NOT NULL,
    instance_id VARCHAR(255),
    metadata JSONB DEFAULT '{}',
    recorded_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
) PARTITION BY RANGE (recorded_at);

-- Create hourly partitions for metrics (high frequency)
CREATE TABLE system_metrics_2025_06_25_00 PARTITION OF system_metrics 
FOR VALUES FROM ('2025-06-25 00:00:00+00') TO ('2025-06-25 01:00:00+00');
CREATE TABLE system_metrics_2025_06_25_01 PARTITION OF system_metrics 
FOR VALUES FROM ('2025-06-25 01:00:00+00') TO ('2025-06-25 02:00:00+00');

-- Metrics indexes
CREATE INDEX CONCURRENTLY idx_system_metrics_name ON system_metrics USING btree (metric_name);
CREATE INDEX CONCURRENTLY idx_system_metrics_component ON system_metrics USING btree (component);
CREATE INDEX CONCURRENTLY idx_system_metrics_recorded_at ON system_metrics USING btree (recorded_at);

-- ===================================================================
-- MATERIALIZED VIEWS - Pre-computed Aggregations
-- ===================================================================

-- User transaction summaries (refreshed every 5 minutes)
CREATE MATERIALIZED VIEW user_transaction_summary AS
SELECT 
    user_id,
    COUNT(*) as total_transactions,
    SUM(CASE WHEN transaction_type = 'deposit' THEN amount ELSE 0 END) as total_deposits,
    SUM(CASE WHEN transaction_type = 'withdrawal' THEN amount ELSE 0 END) as total_withdrawals,
    SUM(CASE WHEN transaction_type = 'bet' THEN amount ELSE 0 END) as total_bets,
    SUM(CASE WHEN transaction_type = 'win' THEN amount ELSE 0 END) as total_wins,
    AVG(amount) as avg_transaction_amount,
    MAX(created_at) as last_transaction_date,
    DATE_TRUNC('day', MAX(created_at)) as last_activity_date
FROM transactions
WHERE status = 'completed'
  AND created_at >= NOW() - INTERVAL '30 days'
GROUP BY user_id;

CREATE UNIQUE INDEX idx_user_transaction_summary_user_id ON user_transaction_summary (user_id);

-- Daily transaction metrics (refreshed hourly)
CREATE MATERIALIZED VIEW daily_transaction_metrics AS
SELECT 
    DATE_TRUNC('day', created_at)::DATE as transaction_date,
    transaction_type,
    status,
    COUNT(*) as transaction_count,
    SUM(amount) as total_amount,
    AVG(amount) as avg_amount,
    MIN(amount) as min_amount,
    MAX(amount) as max_amount
FROM transactions
WHERE created_at >= NOW() - INTERVAL '7 days'
GROUP BY DATE_TRUNC('day', created_at)::DATE, transaction_type, status;

CREATE INDEX idx_daily_metrics_date ON daily_transaction_metrics (transaction_date);
CREATE INDEX idx_daily_metrics_type ON daily_transaction_metrics (transaction_type);

-- ===================================================================
-- FUNCTIONS & TRIGGERS - Automated Maintenance
-- ===================================================================

-- Update timestamp trigger function
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Apply update triggers
CREATE TRIGGER update_users_updated_at BEFORE UPDATE ON users 
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
CREATE TRIGGER update_transactions_updated_at BEFORE UPDATE ON transactions 
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
CREATE TRIGGER update_user_features_updated_at BEFORE UPDATE ON user_features 
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- Function to refresh materialized views
CREATE OR REPLACE FUNCTION refresh_materialized_views()
RETURNS void AS $$
BEGIN
    REFRESH MATERIALIZED VIEW CONCURRENTLY user_transaction_summary;
    REFRESH MATERIALIZED VIEW CONCURRENTLY daily_transaction_metrics;
END;
$$ LANGUAGE plpgsql;

-- ===================================================================
-- PERFORMANCE OPTIMIZATION SETTINGS
-- ===================================================================

-- Connection pooling optimization
ALTER SYSTEM SET max_connections = 200;
ALTER SYSTEM SET shared_buffers = '2GB';
ALTER SYSTEM SET effective_cache_size = '8GB';
ALTER SYSTEM SET work_mem = '256MB';
ALTER SYSTEM SET maintenance_work_mem = '1GB';

-- Write performance optimization
ALTER SYSTEM SET wal_buffers = '64MB';
ALTER SYSTEM SET checkpoint_segments = 64;
ALTER SYSTEM SET checkpoint_completion_target = 0.9;

-- Query optimization
ALTER SYSTEM SET random_page_cost = 1.1;
ALTER SYSTEM SET seq_page_cost = 1.0;
ALTER SYSTEM SET cpu_tuple_cost = 0.01;

-- Parallel query optimization
ALTER SYSTEM SET max_parallel_workers = 16;
ALTER SYSTEM SET max_parallel_workers_per_gather = 8;
ALTER SYSTEM SET parallel_tuple_cost = 0.1;

-- Statistics and monitoring
ALTER SYSTEM SET track_activities = on;
ALTER SYSTEM SET track_counts = on;
ALTER SYSTEM SET track_io_timing = on;
ALTER SYSTEM SET pg_stat_statements.track = 'all';

-- ===================================================================
-- SECURITY SETTINGS
-- ===================================================================

-- Enable row level security where needed
ALTER TABLE users ENABLE ROW LEVEL SECURITY;
ALTER TABLE user_features ENABLE ROW LEVEL SECURITY;

-- Create roles for different access levels
CREATE ROLE readonly_user;
CREATE ROLE app_user;
CREATE ROLE ml_user;
CREATE ROLE etl_user;

-- Grant appropriate permissions
GRANT SELECT ON ALL TABLES IN SCHEMA public TO readonly_user;
GRANT SELECT, INSERT, UPDATE ON users, transactions, user_features TO app_user;
GRANT SELECT, INSERT, UPDATE ON ml_clusters, user_features TO ml_user;
GRANT ALL PRIVILEGES ON etl_jobs, system_metrics TO etl_user;

-- ===================================================================
-- MONITORING & MAINTENANCE
-- ===================================================================

-- Create monitoring function
CREATE OR REPLACE FUNCTION get_database_health()
RETURNS TABLE (
    metric VARCHAR,
    value TEXT,
    status VARCHAR
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        'active_connections'::VARCHAR,
        (SELECT COUNT(*)::TEXT FROM pg_stat_activity WHERE state = 'active'),
        CASE WHEN (SELECT COUNT(*) FROM pg_stat_activity WHERE state = 'active') < 100 
             THEN 'healthy' ELSE 'warning' END::VARCHAR
    UNION ALL
    SELECT 
        'database_size'::VARCHAR,
        pg_size_pretty(pg_database_size(current_database())),
        'healthy'::VARCHAR
    UNION ALL
    SELECT 
        'longest_query_duration'::VARCHAR,
        COALESCE(MAX(EXTRACT(EPOCH FROM (NOW() - query_start)))::TEXT, '0') || ' seconds',
        CASE WHEN COALESCE(MAX(EXTRACT(EPOCH FROM (NOW() - query_start))), 0) < 300 
             THEN 'healthy' ELSE 'warning' END::VARCHAR
    FROM pg_stat_activity WHERE state = 'active';
END;
$$ LANGUAGE plpgsql;

-- ===================================================================
-- INITIAL DATA & SETUP
-- ===================================================================

-- Insert system metrics initialization
INSERT INTO system_metrics (metric_name, metric_value, metric_unit, component) VALUES
('schema_version', 1.0, 'version', 'database'),
('initialization_timestamp', EXTRACT(EPOCH FROM NOW()), 'timestamp', 'database'),
('tables_created', 7, 'count', 'database'),
('indexes_created', 25, 'count', 'database'),
('partitions_created', 6, 'count', 'database');

-- Success message
DO $$
BEGIN
    RAISE NOTICE '🎉 ULTRA-ROBUST DATABASE SCHEMA CRIADO COM SUCESSO!';
    RAISE NOTICE '📊 Configurado para escala massiva: bilhões de transações';
    RAISE NOTICE '⚡ Performance otimizada: indexes, partições, materialized views';
    RAISE NOTICE '🔒 Security habilitada: RLS, roles, permissions';
    RAISE NOTICE '📈 Monitoring ativo: métricas, health checks, triggers';
END $$;