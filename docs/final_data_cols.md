TEST.PARQUET COLUMNS (33 total):

METADATA & TARGET
1. timestamp_utc
2. plant_id
3. power_norm

PVLIB BASELINE INPUTS (Metadata & Irradiance)
4. latitude
5. longitude
6. tilt_deg
7. azimuth_deg
8. installed_capacity_kw
9. timezone
10. ghi
11. dni
12. dhi

DEEP LEARNING WEATHER & RAW IRRADIANCE (TFT Inputs)
13. cloud_cover
14. diffuse_radiation_instant
15. direct_normal_irradiance_instant_raw
16. direct_radiation_instant
17. global_tilted_irradiance_instant_raw
18. precipitation
19. relative_humidity_2m
20. shortwave_radiation_instant_raw
21. surface_pressure
22. temperature_2m
23. weather_code
24. wind_direction_10m
25. wind_speed_10m

PVLIB DERIVED PHYSICS FEATURES (TFT Inputs)
26. pvlib_ac_kw
27. pvlib_dc_kw
28. pvlib_poa_diffuse
29. pvlib_poa_direct
30. pvlib_poa_global
31. pvlib_poa_ground_diffuse
32. pvlib_solar_azimuth
33. pvlib_solar_zenith
