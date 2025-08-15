-- Carbon Tracker Database Initialization
-- This script sets up the PostgreSQL database schema for the carbon footprint tracker

-- Create extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_stat_statements";

-- Create schemas
CREATE SCHEMA IF NOT EXISTS carbon_data;
CREATE SCHEMA IF NOT EXISTS ml_models;
CREATE SCHEMA IF NOT EXISTS user_management;

-- Set search path
SET search_path TO carbon_data, ml_models, user_management, public;

-- =============================================
-- CARBON DATA SCHEMA
-- =============================================

-- Device registry table
CREATE TABLE carbon_data.devices (
    device_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    device_name VARCHAR(255) NOT NULL,
    device_type VARCHAR(100) NOT NULL, -- 'sensor', 'meter', 'vehicle', 'hvac'
    location VARCHAR(255),
    installation_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_seen TIMESTAMP,
    is_active BOOLEAN DEFAULT true,
    metadata JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Carbon emission factors table
CREATE TABLE carbon_data.emission_factors (
    factor_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    source_type VARCHAR(100) NOT NULL, -- 'electricity', 'gas', 'fuel', 'transport'
    region VARCHAR(100) NOT NULL,
    factor_value DECIMAL(10, 6) NOT NULL, -- kg CO2e per unit
    unit VARCHAR(50) NOT NULL, -- 'kWh', 'liter', 'km', etc.
    valid_from DATE NOT NULL,
    valid_to DATE,
    source VARCHAR(255),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Real-time carbon readings (summary table for quick queries)
CREATE TABLE carbon_data.carbon_readings_summary (
    reading_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    device_id UUID REFERENCES carbon_data.devices(device_id),
    timestamp TIMESTAMP NOT NULL,
    carbon_emission DECIMAL(12, 6) NOT NULL, -- kg CO2e
    energy_consumption DECIMAL(12, 6), -- kWh or other units
    emission_factor DECIMAL(10, 6),
    data_source VARCHAR(100),
    quality_score DECIMAL(3, 2), -- 0.00 to 1.00
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Aggregated carbon data (hourly, daily, monthly)
CREATE TABLE carbon_data.carbon_aggregates (
    aggregate_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    device_id UUID REFERENCES carbon_data.devices(device_id),
    period_start TIMESTAMP NOT NULL,
    period_end TIMESTAMP NOT NULL,
    aggregation_level VARCHAR(20) NOT NULL, -- 'hour', 'day', 'week', 'month'
    total_emissions DECIMAL(15, 6) NOT NULL,
    avg_emissions DECIMAL(12, 6) NOT NULL,
    max_emissions DECIMAL(12, 6) NOT NULL,
    min_emissions DECIMAL(12, 6) NOT NULL,
    total_energy DECIMAL(15, 6),
    data_points_count INTEGER,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Alerts and anomalies
CREATE TABLE carbon_data.alerts (
    alert_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    device_id UUID REFERENCES carbon_data.devices(device_id),
    alert_type VARCHAR(100) NOT NULL, -- 'high_emission', 'anomaly', 'device_offline'
    severity VARCHAR(20) NOT NULL, -- 'low', 'medium', 'high', 'critical'
    message TEXT NOT NULL,
    threshold_value DECIMAL(12, 6),
    actual_value DECIMAL(12, 6),
    triggered_at TIMESTAMP NOT NULL,
    resolved_at TIMESTAMP,
    is_resolved BOOLEAN DEFAULT false,
    metadata JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- =============================================
-- ML MODELS SCHEMA
-- =============================================

-- Model registry
CREATE TABLE ml_models.model_registry (
    model_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    model_name VARCHAR(255) NOT NULL,
    model_version VARCHAR(50) NOT NULL,
    model_type VARCHAR(100) NOT NULL, -- 'prediction', 'anomaly_detection', 'optimization'
    algorithm VARCHAR(100),
    training_data_period DATERANGE,
    accuracy_metrics JSONB,
    model_path VARCHAR(500),
    is_active BOOLEAN DEFAULT false,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    deployed_at TIMESTAMP,
    UNIQUE(model_name, model_version)
);

-- Model predictions
CREATE TABLE ml_models.predictions (
    prediction_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    model_id UUID REFERENCES ml_models.model_registry(model_id),
    device_id UUID REFERENCES carbon_data.devices(device_id),
    prediction_timestamp TIMESTAMP NOT NULL,
    predicted_value DECIMAL(12, 6) NOT NULL,
    confidence_score DECIMAL(3, 2),
    actual_value DECIMAL(12, 6), -- filled when actual data becomes available
    prediction_error DECIMAL(12, 6), -- calculated when actual value is known
    features JSONB, -- input features used for prediction
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Model training jobs
CREATE TABLE ml_models.training_jobs (
    job_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    model_name VARCHAR(255) NOT NULL,
    job_status VARCHAR(50) NOT NULL, -- 'running', 'completed', 'failed'
    training_config JSONB,
    start_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    end_time TIMESTAMP,
    metrics JSONB,
    error_message TEXT,
    created_by VARCHAR(255)
);

-- =============================================
-- USER MANAGEMENT SCHEMA
-- =============================================

-- Users table
CREATE TABLE user_management.users (
    user_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    username VARCHAR(100) UNIQUE NOT NULL,
    email VARCHAR(255) UNIQUE NOT NULL,
    password_hash VARCHAR(255) NOT NULL,
    full_name VARCHAR(255),
    role VARCHAR(50) DEFAULT 'viewer', -- 'admin', 'analyst', 'viewer'
    is_active BOOLEAN DEFAULT true,
    last_login TIMESTAMP,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- User sessions
CREATE TABLE user_management.user_sessions (
    session_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID REFERENCES user_management.users(user_id),
    session_token VARCHAR(500) NOT NULL,
    expires_at TIMESTAMP NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_accessed TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- User preferences
CREATE TABLE user_management.user_preferences (
    preference_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID REFERENCES user_management.users(user_id),
    preference_key VARCHAR(100) NOT NULL,
    preference_value JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(user_id, preference_key)
);

-- =============================================
-- INDEXES FOR PERFORMANCE
-- =============================================

-- Carbon data indexes
CREATE INDEX idx_devices_type_active ON carbon_data.devices(device_type, is_active);
CREATE INDEX idx_devices_location ON carbon_data.devices(location);

CREATE INDEX idx_emission_factors_source_region ON carbon_data.emission_factors(source_type, region);
CREATE INDEX idx_emission_factors_valid_period ON carbon_data.emission_factors(valid_from, valid_to);

CREATE INDEX idx_carbon_readings_device_timestamp ON carbon_data.carbon_readings_summary(device_id, timestamp DESC);
CREATE INDEX idx_carbon_readings_timestamp ON carbon_data.carbon_readings_summary(timestamp DESC);

CREATE INDEX idx_carbon_aggregates_device_period ON carbon_data.carbon_aggregates(device_id, period_start, period_end);
CREATE INDEX idx_carbon_aggregates_level_period ON carbon_data.carbon_aggregates(aggregation_level, period_start);

CREATE INDEX idx_alerts_device_triggered ON carbon_data.alerts(device_id, triggered_at DESC);
CREATE INDEX idx_alerts_type_severity ON carbon_data.alerts(alert_type, severity);
CREATE INDEX idx_alerts_unresolved ON carbon_data.alerts(is_resolved, triggered_at DESC) WHERE is_resolved = false;

-- ML models indexes
CREATE INDEX idx_model_registry_active ON ml_models.model_registry(is_active, model_type);
CREATE INDEX idx_predictions_model_timestamp ON ml_models.predictions(model_id, prediction_timestamp DESC);
CREATE INDEX idx_predictions_device_timestamp ON ml_models.predictions(device_id, prediction_timestamp DESC);

-- User management indexes
CREATE INDEX idx_users_email ON user_management.users(email);
CREATE INDEX idx_users_active ON user_management.users(is_active);
CREATE INDEX idx_user_sessions_token ON user_management.user_sessions(session_token);
CREATE INDEX idx_user_sessions_expires ON user_management.user_sessions(expires_at);

-- =============================================
-- FUNCTIONS AND TRIGGERS
-- =============================================

-- Function to update updated_at timestamp
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Triggers for updated_at
CREATE TRIGGER update_devices_updated_at BEFORE UPDATE ON carbon_data.devices
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_users_updated_at BEFORE UPDATE ON user_management.users
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_user_preferences_updated_at BEFORE UPDATE ON user_management.user_preferences
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- Function to calculate prediction error
CREATE OR REPLACE FUNCTION calculate_prediction_error()
RETURNS TRIGGER AS $$
BEGIN
    IF NEW.actual_value IS NOT NULL AND OLD.actual_value IS NULL THEN
        NEW.prediction_error = ABS(NEW.predicted_value - NEW.actual_value);
    END IF;
    RETURN NEW;
END;
$$ language 'plpgsql';

CREATE TRIGGER calculate_prediction_error_trigger BEFORE UPDATE ON ml_models.predictions
    FOR EACH ROW EXECUTE FUNCTION calculate_prediction_error();

-- =============================================
-- INITIAL DATA
-- =============================================

-- Insert default emission factors (example data)
INSERT INTO carbon_data.emission_factors (source_type, region, factor_value, unit, valid_from, source) VALUES
('electricity', 'US-AVERAGE', 0.4, 'kWh', '2024-01-01', 'EPA eGRID 2024'),
('electricity', 'EU-AVERAGE', 0.3, 'kWh', '2024-01-01', 'EEA 2024'),
('natural_gas', 'GLOBAL', 0.2, 'm3', '2024-01-01', 'IPCC 2024'),
('gasoline', 'GLOBAL', 2.3, 'liter', '2024-01-01', 'IPCC 2024'),
('diesel', 'GLOBAL', 2.7, 'liter', '2024-01-01', 'IPCC 2024');

-- Insert default admin user (password: admin123)
INSERT INTO user_management.users (username, email, password_hash, full_name, role) VALUES
('admin', 'admin@carbontracker.com', '$2b$12$LQv3c1yqBWVHxkd0LHAkCOYz6TtxMQJqhN8/LewdBPj3bp.Gm.F5e', 'System Administrator', 'admin');

-- Create partitioning for large tables (optional, for high-volume deployments)
-- This would be implemented based on expected data volume

-- Grant permissions
GRANT USAGE ON SCHEMA carbon_data TO carbon_user;
GRANT USAGE ON SCHEMA ml_models TO carbon_user;
GRANT USAGE ON SCHEMA user_management TO carbon_user;

GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA carbon_data TO carbon_user;
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA ml_models TO carbon_user;
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA user_management TO carbon_user;

GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA carbon_data TO carbon_user;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA ml_models TO carbon_user;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA user_management TO carbon_user;

-- Enable row level security (optional, for multi-tenant deployments)
-- ALTER TABLE carbon_data.devices ENABLE ROW LEVEL SECURITY;
-- CREATE POLICY device_access_policy ON carbon_data.devices FOR ALL TO carbon_user USING (true);

COMMIT;