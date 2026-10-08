# Weather API Decision for MiRACLE PV Forecasting
**Date**: 2026-01-03  
**Status**: FINAL - Keep Open-Meteo (No Change)

## Executive Summary
After testing ECMWF Direct API, ECMWF Open Data, and Open-Meteo APIs, we determined that **Open-Meteo is optimal** for 30-day PV forecasting in Germany. No changes needed to existing WeatherClient.

## APIs Tested

### 1. ECMWF Direct API (via ecmwf-api-client)
- **Status**: ❌ Not accessible
- **Issue**: Package import failed, would require MARS system integration
- **Complexity**: High (GRIB2 format, asynchronous retrieval, minutes delay)
- **Benefit**: Extended range up to 46 days (sub-seasonal)
- **Verdict**: Not worth complexity for marginal benefit

### 2. ECMWF Open Data API (Free)
- **Status**: ⚠️ Accessible but problematic
- **Pros**: 
  - Free (no credentials)
  - IFS HRES: 10 days @ 0.25° resolution
  - IFS ENS: 15 days @ 0.5° (51 ensemble members)
- **Cons**:
  - GRIB2 format (requires cfgrib, xarray parsing)
  - SSL certificate issues behind corporate proxy
  - Solar radiation accumulated (needs post-processing)
  - No tilt/azimuth support (manual calculation needed)
  - 500 simultaneous connection limit
- **Verdict**: Too much overhead for PV use case

### 3. Open-Meteo APIs (Current Setup) ✅
- **Status**: ✅ Working perfectly
- **Performance**:
  - Forecast API: 15 days, <0.01s latency, quality=8
  - ECMWF proxy: 15 days @ 0.25°, quality=7 (**uses ECMWF IFS underneath!**)
  - GFS: 16 days @ 50km, 0.18s latency, quality=5
- **Pros**:
  - JSON API (trivial parsing)
  - Pre-computed solar irradiance (GHI, DNI, DHI, GTI)
  - Panel tilt/azimuth built-in
  - Fast response (<0.2s for all APIs)
  - Generous rate limits (10,000/day)
- **Verdict**: **OPTIMAL FOR PV FORECASTING**

## Final Routing Strategy (No Change)

```python
# WeatherClient.select_best_api() - UNCHANGED
if days <= 7:
    return "Forecast", self.forecast_url      # High accuracy, fast
elif days <= 15:
    return "ECMWF", self.ecmwf_url           # ECMWF IFS via Open-Meteo
else:
    return "GFS", self.gfs_url               # Long-range (16+ days)
```

**For 30-day PV forecasting:**
- Days 0-7: Open-Meteo Forecast API (best short-term)
- Days 8-15: Open-Meteo ECMWF proxy (**already ECMWF IFS data!**)
- Days 16-30: Open-Meteo GFS or repeat/climatology

## Key Insight
**Open-Meteo's ECMWF endpoint IS ECMWF data** - it's just accessed via a convenient JSON API instead of complex GRIB2/MARS. We get ECMWF quality without complexity.

## Test Results Summary

| API Source | Horizon | Latency | Quality | Solar Vars | Status |
|-----------|---------|---------|---------|------------|--------|
| OM Forecast | 15d | <0.01s | 8 | ✅ GHI/DNI/DHI/GTI | ✅ OK |
| OM ECMWF | 15d | <0.01s | 7 | ✅ Pre-computed | ✅ OK |
| OM GFS | 15d | 0.18s | 5 | ✅ Pre-computed | ✅ OK |
| ECMWF Direct | N/A | N/A | N/A | ❌ Need processing | ❌ FAIL |
| ECMWF Open Data | 10d | >2s | 10 | ⚠️ Accumulated | ⚠️ SSL Issues |

## Why Open-Meteo Wins for PV Forecasting

1. **Solar-Ready**: GHI, DNI, DHI, GTI pre-computed with tilt/azimuth
2. **Fast**: JSON API, <0.2s response time
3. **ECMWF Quality**: ECMWF endpoint uses IFS data (same source)
4. **Works Now**: No SSL issues, no parsing complexity
5. **Proven**: Already integrated and tested in production

## Action Items
- ✅ Keep existing WeatherClient unchanged
- ✅ Use Open-Meteo for all RL data collection
- ✅ Document decision for future reference
- 🚀 Proceed with RL training data generation

## Related Files
- `src/inference/weather_client.py` - Current implementation (KEEP AS-IS)
- `tests/test_weather_api_comparison.py` - API benchmark results
- `tests/test_ecmwf_opendata.py` - ECMWF Open Data test
- `.ecmwf_credentials.json` - Unused (for future if needed)

## References
- [ECMWF Open Data](https://www.ecmwf.int/en/forecasts/datasets/open-data)
- [Open-Meteo API Docs](https://open-meteo.com/en/docs)
- [ECMWF Public Datasets](https://confluence.ecmwf.int/display/WEBAPI/Access+ECMWF+Public+Datasets)
