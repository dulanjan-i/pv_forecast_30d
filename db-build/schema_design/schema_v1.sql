-- =========================
-- Table 1: Historical Weather
-- =========================
CREATE TABLE historical_weather (
    id SERIAL PRIMARY KEY,
    location_id INT NOT NULL,
    datetime TIMESTAMP NOT NULL,
    uv_index FLOAT,
    uv_index_clear_sky FLOAT,
    ghi FLOAT,
    global_tilted_irradiance FLOAT,
    dhi FLOAT,
    dni FLOAT,
    solar_zenith_angle FLOAT,
    solar_azimuth FLOAT,
    albedo FLOAT,
    cloud_cover FLOAT,
    clear_sky_index FLOAT,
    air_temperature FLOAT,
    wind_speed FLOAT,
    wind_direction FLOAT,
    relative_humidity FLOAT,
    precipitation_total FLOAT,
    precipitation_probability FLOAT,
    rain FLOAT,
    shower FLOAT,
    snowfall FLOAT,
    time TIME,
    max_temp FLOAT,
    min_temp FLOAT,
    sunlight_duration FLOAT,
    sunrise TIME,
    sunset TIME,
    created_at TIMESTAMP DEFAULT NOW(),
    data_tag VARCHAR(10) DEFAULT 'train'  -- 'train', 'test', or 'inference'
);

-- =========================
-- Table 2: Historical PV Output
-- =========================
CREATE TABLE historical_pv_output (
    id SERIAL PRIMARY KEY,
    location_id INT NOT NULL,
    datetime TIMESTAMP NOT NULL,
    pv_output FLOAT,
    created_at TIMESTAMP DEFAULT NOW(),
    data_tag VARCHAR(10) DEFAULT 'train'
);

-- =========================
-- Table 3: Feature Store (preprocessed inputs & encoded outputs)
-- =========================
CREATE TABLE feature_store (
    id SERIAL PRIMARY KEY,
    location_id INT NOT NULL,
    datetime TIMESTAMP NOT NULL,
    lstm_encoded_signal JSONB,       -- encoded learner signal
    pvlib_baseline_output FLOAT,     -- baseline/moderator
    forecast_api_covariates JSONB,   -- future covariates
    additional_features JSONB,       -- e.g. calendar, lag variables
    created_at TIMESTAMP DEFAULT NOW()
);

-- =========================
-- Table 4: Model Store
-- =========================
CREATE TABLE model_store (
    id SERIAL PRIMARY KEY,
    model_name VARCHAR(50) NOT NULL,
    model_type VARCHAR(50),           -- LSTM, TFT, PVLib
    version VARCHAR(20),
    serialized_weights BYTEA,         -- binary blob
    training_params JSONB,
    created_at TIMESTAMP DEFAULT NOW()
);

-- =========================
-- Table 5: TFT Output
-- =========================
CREATE TABLE tft_output (
    id SERIAL PRIMARY KEY,
    location_id INT NOT NULL,
    datetime TIMESTAMP NOT NULL,
    forecast_horizon INT NOT NULL,   -- e.g., 15-min step number
    predicted_pv FLOAT,
    confidence_lower FLOAT,
    confidence_upper FLOAT,
    created_at TIMESTAMP DEFAULT NOW()
);

-- =========================
-- Table 6: Error Metrics
-- =========================
CREATE TABLE error_metrics (
    id SERIAL PRIMARY KEY,
    location_id INT NOT NULL,
    datetime TIMESTAMP NOT NULL,
    model_name VARCHAR(50),
    rmse FLOAT,
    mae FLOAT,
    mape FLOAT,
    drift_metric FLOAT,
    created_at TIMESTAMP DEFAULT NOW()
);

-- =========================
-- Table 7: Pipeline Log
-- =========================
CREATE TABLE pipeline_log (
    id SERIAL PRIMARY KEY,
    step_name VARCHAR(50),
    status VARCHAR(20),
    message TEXT,
    timestamp TIMESTAMP DEFAULT NOW()
);

-- =========================
-- Table 8: RL State & Actions
-- =========================
CREATE TABLE rl_state_actions (
    id SERIAL PRIMARY KEY,
    location_id INT NOT NULL,
    datetime TIMESTAMP NOT NULL,
    state JSONB,
    action JSONB,
    reward FLOAT,
    created_at TIMESTAMP DEFAULT NOW()
);

-- =========================
-- Table 9: API Metrics (Optional)
-- =========================
CREATE TABLE api_metrics (
    id SERIAL PRIMARY KEY,
    api_name VARCHAR(50),
    location_id INT,
    datetime TIMESTAMP NOT NULL,
    response_time_ms FLOAT,
    status_code INT,
    success BOOLEAN,
    created_at TIMESTAMP DEFAULT NOW()
);

-- =========================
-- Table 10: Metadata (Pipeline / Model / Feature info)
-- =========================
CREATE TABLE metadata (
    id SERIAL PRIMARY KEY,
    table_name VARCHAR(50),
    last_updated TIMESTAMP,
    description TEXT,
    created_at TIMESTAMP DEFAULT NOW()
);

-- =========================
-- Table 11: Model explainability (Pipeline / Model / Feature info)
-- =========================
CREATE TABLE model_explainability (
   id SERIAL PRIMARY KEY,
   model VARCHAR,
   model_version VARCHAR,
   timestamp TIMESTAMP,
   location_id INT,
   horizon INT,
   attention_map JSONB,
   attention_summary JSONB
);
