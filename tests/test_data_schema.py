"""
tests/test_data_schema.py — Schema and contract tests for PV forecast data.

Tests that synthetic DataFrames representing the expected pipeline schema
are structurally sound. No real data files are read — all inputs are generated
in-memory. These tests act as living documentation of the data contract.
"""
from __future__ import annotations

import pytest
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Expected schema for the 15-min PV forecast pipeline
# (mirrors docs/final_data_cols.md)
# ---------------------------------------------------------------------------

REQUIRED_FEATURE_COLS = [
    "temperature_2m",
    "shortwave_radiation",
    "direct_normal_irradiance",
    "diffuse_radiation",
    "windspeed_10m",
    "precipitation",
    "cloudcover",
]

REQUIRED_TARGET_COLS = ["power_kw"]

REQUIRED_INDEX_FREQ = "15min"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_forecast_df(
    n_steps: int = 96 * 30,
    freq: str = "15min",
    include_target: bool = True,
    add_extra_cols: bool = False,
) -> pd.DataFrame:
    """Generate a synthetic 30-day, 15-min DataFrame matching the pipeline schema."""
    rng = np.random.default_rng(0)
    idx = pd.date_range("2026-01-01", periods=n_steps, freq=freq, tz="UTC")
    data = {col: rng.uniform(0.0, 1.0, n_steps) for col in REQUIRED_FEATURE_COLS}
    if include_target:
        data["power_kw"] = np.abs(rng.normal(50.0, 10.0, n_steps))
    if add_extra_cols:
        data["extra_col"] = 0.0
    return pd.DataFrame(data, index=idx)


def validate_schema(df: pd.DataFrame) -> list[str]:
    """Return list of schema violations (empty = valid)."""
    errors = []
    for col in REQUIRED_FEATURE_COLS + REQUIRED_TARGET_COLS:
        if col not in df.columns:
            errors.append(f"Missing column: {col}")
    if not isinstance(df.index, pd.DatetimeIndex):
        errors.append("Index is not DatetimeIndex")
    elif df.index.tz is None:
        errors.append("Index has no timezone (expected UTC)")
    if df.isnull().any().any():
        null_cols = df.columns[df.isnull().any()].tolist()
        errors.append(f"NaN values in: {null_cols}")
    present_targets = [c for c in REQUIRED_TARGET_COLS if c in df.columns]
    if present_targets and (df[present_targets] < 0).any().any():
        errors.append("Negative values in target column(s)")
    return errors


# ---------------------------------------------------------------------------
# Index and time-series structure tests
# ---------------------------------------------------------------------------

class TestIndexStructure:
    def test_datetimeindex_type(self):
        df = make_forecast_df()
        assert isinstance(df.index, pd.DatetimeIndex)

    def test_index_is_utc(self):
        df = make_forecast_df()
        assert df.index.tz is not None
        assert str(df.index.tz) == "UTC"

    def test_index_monotonic_increasing(self):
        df = make_forecast_df()
        assert df.index.is_monotonic_increasing

    def test_15min_frequency(self):
        df = make_forecast_df()
        diffs = df.index.to_series().diff().dropna().unique()
        assert len(diffs) == 1
        assert diffs[0] == pd.Timedelta("15min")

    def test_30_day_horizon_has_correct_rows(self):
        df = make_forecast_df(n_steps=96 * 30)
        assert len(df) == 96 * 30

    def test_no_duplicate_timestamps(self):
        df = make_forecast_df()
        assert df.index.is_unique


# ---------------------------------------------------------------------------
# Column presence and type tests
# ---------------------------------------------------------------------------

class TestColumnSchema:
    def test_all_feature_columns_present(self):
        df = make_forecast_df()
        for col in REQUIRED_FEATURE_COLS:
            assert col in df.columns, f"Missing: {col}"

    def test_target_column_present(self):
        df = make_forecast_df()
        assert "power_kw" in df.columns

    def test_all_columns_numeric(self):
        df = make_forecast_df()
        for col in df.columns:
            assert pd.api.types.is_numeric_dtype(df[col]), f"{col} is not numeric"

    def test_extra_columns_allowed(self):
        """Schema validation should not reject DataFrames with extra columns."""
        df = make_forecast_df(add_extra_cols=True)
        errors = validate_schema(df)
        assert errors == []


# ---------------------------------------------------------------------------
# Data quality / contract tests
# ---------------------------------------------------------------------------

class TestDataQuality:
    def test_no_nulls_in_valid_frame(self):
        df = make_forecast_df()
        assert not df.isnull().any().any()

    def test_power_kw_non_negative(self):
        df = make_forecast_df()
        assert (df["power_kw"] >= 0).all()

    def test_schema_validator_catches_missing_column(self):
        df = make_forecast_df()
        df = df.drop(columns=["power_kw"])
        errors = validate_schema(df)
        assert any("power_kw" in e for e in errors)

    def test_schema_validator_catches_nulls(self):
        df = make_forecast_df()
        df.iloc[0, 0] = np.nan
        errors = validate_schema(df)
        assert any("NaN" in e for e in errors)

    def test_schema_validator_catches_no_timezone(self):
        df = make_forecast_df()
        df.index = df.index.tz_localize(None)
        errors = validate_schema(df)
        assert any("timezone" in e for e in errors)

    def test_schema_validator_catches_negative_target(self):
        df = make_forecast_df()
        df.loc[df.index[0], "power_kw"] = -1.0
        errors = validate_schema(df)
        assert any("Negative" in e for e in errors)

    def test_valid_frame_has_no_errors(self):
        df = make_forecast_df()
        assert validate_schema(df) == []


# ---------------------------------------------------------------------------
# Daylight / night masking contract
# ---------------------------------------------------------------------------

class TestDaylightMask:
    def test_shortwave_zero_at_night(self):
        """Night-time shortwave radiation should be 0 (physics constraint)."""
        rng = np.random.default_rng(1)
        idx = pd.date_range("2026-01-01", periods=96, freq="15min", tz="UTC")
        sw = np.zeros(96)
        # Simulate daytime 06:00–18:00 UTC
        daytime = (idx.hour >= 6) & (idx.hour < 18)
        sw[daytime] = rng.uniform(0.0, 800.0, daytime.sum())
        df = pd.DataFrame({"shortwave_radiation": sw}, index=idx)

        night_mask = ~daytime
        assert (df.loc[night_mask, "shortwave_radiation"] == 0.0).all()

    def test_power_zero_when_no_irradiance(self):
        """power_kw should be 0 wherever shortwave_radiation is 0 (night clamp)."""
        idx = pd.date_range("2026-01-01", periods=10, freq="15min", tz="UTC")
        df = pd.DataFrame({
            "shortwave_radiation": [0.0] * 10,
            "power_kw": [0.0] * 10,
        }, index=idx)
        night_mask = df["shortwave_radiation"] == 0.0
        assert (df.loc[night_mask, "power_kw"] == 0.0).all()
