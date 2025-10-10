-- DATABASE: pv_forecast_meta_architecture

-- Table 1: Historical weather data
CREATE TABLE historical_weather (
    id SERIAL PRIMARY KEY,
    location_id INT NOT NULL,
    datetime TIMESTAMP NOT NULL,
    temperature FLOAT,
    humidity FLOAT,
    wind_speed FLOAT,
    cloud_cover FLOAT,
    solar_radiation FLOAT,
    raw_api_source VARCHAR(50),
    created_at TIMESTAMP DEFAULT NOW()
);

-- Table 2: Historical PV output data
CREATE TABLE historical_pv_output (
    id SERIAL PRIMARY KEY,
    location_id INT NOT NULL,
    datetime TIMESTAMP NOT NULL,
    pv_output_kw FLOAT,
    created_at TIMESTAMP DEFAULT NOW()
);

-- Table 3: Feature store (preprocessed inputs & encoded outputs)
CREATE TABLE feature_store (
    id SERIAL PRIMARY KEY,
    location_id INT NOT NULL,
    datetime TIMESTAMP NOT NULL,
    lstm_encoded_signal JSONB,       -- encoded learner signal
    pvlib_baseline_output FLOAT,     -- baseline/moderator
    forecast_api_covariates JSONB,   -- future covariates
    additional_features JSONB,       -- any extra features (e.g. calendar, lag variables)
    preprocessing_metadata JSONB,
    created_at TIMESTAMP DEFAULT NOW()
);

-- Table 4: Model store (weights, params)
CREATE TABLE model_store (
    id SERIAL PRIMARY KEY,
    model_name VARCHAR(50),
    model_version VARCHAR(20),
    parameters JSONB,
    metrics JSONB,
    created_at TIMESTAMP DEFAULT NOW()
);

-- Table 5: TFT forecast output
CREATE TABLE tft_output (
    id SERIAL PRIMARY KEY,
    location_id INT NOT NULL,
    forecast_horizon_days INT NOT NULL,
    forecast_datetime TIMESTAMP NOT NULL,
    forecast_values JSONB,             -- array of forecasts
    confidence_intervals JSONB,
    created_at TIMESTAMP DEFAULT NOW()
);

-- Table 6: Error metrics (tracking model performance)
CREATE TABLE error_metrics (
    id SERIAL PRIMARY KEY,
    model_name VARCHAR(50),
    location_id INT NOT NULL,
    datetime TIMESTAMP NOT NULL,
    rmse FLOAT,
    mae FLOAT,
    r2 FLOAT,
    drift_metrics JSONB,
    created_at TIMESTAMP DEFAULT NOW()
);

-- Table 7: Pipeline log (execution logs & retraining events)
CREATE TABLE pipeline_log (
    id SERIAL PRIMARY KEY,
    pipeline_step VARCHAR(50),
    status VARCHAR(20),
    details JSONB,
    executed_at TIMESTAMP DEFAULT NOW()
);

-- Table 8: RL state-actions (meta controller history)
CREATE TABLE rl_state_actions (
    id SERIAL PRIMARY KEY,
    datetime TIMESTAMP NOT NULL,
    state JSONB,        -- encapsulates all RL state variables
    action JSONB,       -- chosen action
    reward FLOAT,
    created_at TIMESTAMP DEFAULT NOW()
);

-- Table 9: API metrics (optional — for meta controller decisions)
CREATE TABLE api_metrics (
    id SERIAL PRIMARY KEY,
    api_name VARCHAR(50),
    request_datetime TIMESTAMP NOT NULL,
    response_time_ms FLOAT,
    success BOOLEAN,
    forecast_accuracy FLOAT,
    details JSONB,
    created_at TIMESTAMP DEFAULT NOW()
);

-- Table 10: Metadata store (pipeline & model state tracking)
CREATE TABLE metadata_store (
    id SERIAL PRIMARY KEY,
    meta_key VARCHAR(50),
    meta_value JSONB,
    created_at TIMESTAMP DEFAULT NOW()
);
