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
    additional_vars JSONB,
    created_at TIMESTAMP DEFAULT NOW()
);

-- Table 2: Historical PV output
CREATE TABLE historical_pv_output (
    id SERIAL PRIMARY KEY,
    location_id INT NOT NULL,
    datetime TIMESTAMP NOT NULL,
    pv_output FLOAT,
    additional_vars JSONB,
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
    additional_features JSONB,       -- extra features like calendar, lag variables
    created_at TIMESTAMP DEFAULT NOW()
);

-- Table 4: Model store (saved weights & parameters for reproducibility)
CREATE TABLE model_store (
    id SERIAL PRIMARY KEY,
    model_name TEXT NOT NULL,
    version TEXT NOT NULL,
    weights BYTEA,
    parameters JSONB,
    created_at TIMESTAMP DEFAULT NOW()
);

-- Table 5: TFT output (final forecast + confidence intervals)
CREATE TABLE tft_output (
    id SERIAL PRIMARY KEY,
    location_id INT NOT NULL,
    forecast_horizon INTERVAL NOT NULL,
    datetime TIMESTAMP NOT NULL,
    forecast_values JSONB,
    confidence_intervals JSONB,
    created_at TIMESTAMP DEFAULT NOW()
);

-- Table 6: Error metrics (for all models + RL feedback loop)
CREATE TABLE error_metrics (
    id SERIAL PRIMARY KEY,
    location_id INT NOT NULL,
    datetime TIMESTAMP NOT NULL,
    model_name TEXT NOT NULL,
    rmse FLOAT,
    mae FLOAT,
    r2 FLOAT,
    drift_metrics JSONB,
    created_at TIMESTAMP DEFAULT NOW()
);

-- Table 7: Pipeline log (execution, retraining, API calls, errors)
CREATE TABLE pipeline_log (
    id SERIAL PRIMARY KEY,
    run_id UUID NOT NULL,
    step TEXT NOT NULL,
    status TEXT NOT NULL,
    message TEXT,
    timestamp TIMESTAMP DEFAULT NOW()
);

-- Table 8: RL state actions (for meta controller decisions)
CREATE TABLE rl_state_actions (
    id SERIAL PRIMARY KEY,
    state JSONB NOT NULL,
    action TEXT NOT NULL,
    reward FLOAT,
    timestamp TIMESTAMP DEFAULT NOW()
);

-- Table 9: API metrics (optional — for RL & debugging)
CREATE TABLE api_metrics (
    id SERIAL PRIMARY KEY,
    api_name TEXT NOT NULL,
    location_id INT,
    datetime TIMESTAMP NOT NULL,
    response_time FLOAT,
    success_rate FLOAT,
    data_quality JSONB,
    created_at TIMESTAMP DEFAULT NOW()
);

-- Table 10: Metadata table (pipeline + preprocessing metadata)
CREATE TABLE metadata (
    id SERIAL PRIMARY KEY,
    key TEXT NOT NULL,
    value JSONB,
    description TEXT,
    created_at TIMESTAMP DEFAULT NOW()
);
