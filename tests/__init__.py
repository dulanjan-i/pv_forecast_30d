"""
MiRACLE Test Suite

Integration and unit tests for the MiRACLE forecasting system.

Test organization:
- test_rl_integration.py: RL meta-controller integration tests
- test_action_executors.py: Action executor unit tests
- test_tft_integration.py: TFT model integration
- test_full_pipeline_real_tft.py: End-to-end pipeline tests
- test_hierarchical_pipeline.py: Hierarchical forecasting tests
- test_orchestration.py: System orchestration tests
- test_live_weather_forecast.py: Live weather API tests

Run all tests from repo root:
    python -m pytest tests/

Run specific test:
    python tests/test_rl_integration.py
"""
