# Code to call Open-Meteo Historical Weather API and process the response

import openmeteo_requests

import pandas as pd
import requests_cache
from retry_requests import retry

# Setup the Open-Meteo API client with cache and retry on error
cache_session = requests_cache.CachedSession('.cache', expire_after = -1)
retry_session = retry(cache_session, retries = 5, backoff_factor = 0.2)
openmeteo = openmeteo_requests.Client(session = retry_session)

# Make sure all required weather variables are listed here
# The order of variables in hourly or daily is important to assign them correctly below
url = "https://archive-api.open-meteo.com/v1/archive"
params = {
	"latitude": 38.996306,
	"longitude": -122.134111,
	"start_date": "2017-10-31",
	"end_date": "2024-11-02",
	"daily": ["sunrise", "sunset", "daylight_duration", "sunshine_duration", "precipitation_sum", "rain_sum", "snowfall_sum", "precipitation_hours", "temperature_2m_max", "temperature_2m_min", "apparent_temperature_max"],
	"hourly": ["temperature_2m", "relative_humidity_2m", "precipitation", "weather_code", "cloud_cover", "wind_speed_10m", "wind_direction_10m", "shortwave_radiation_instant", "direct_radiation_instant", "diffuse_radiation_instant", "direct_normal_irradiance_instant", "global_tilted_irradiance_instant", "surface_pressure"],
	"models": "era5_seamless",
	"timezone": "America/Los_Angeles",
	"tilt": 25,
	"azimuth": 180,
}
responses = openmeteo.weather_api(url, params=params)

# Process first location. Add a for-loop for multiple locations or weather models
response = responses[0]
print(f"Coordinates: {response.Latitude()}°N {response.Longitude()}°E")
print(f"Elevation: {response.Elevation()} m asl")
print(f"Timezone: {response.Timezone()}{response.TimezoneAbbreviation()}")
print(f"Timezone difference to GMT+0: {response.UtcOffsetSeconds()}s")

# Process hourly data. The order of variables needs to be the same as requested.
hourly = response.Hourly()
hourly_temperature_2m = hourly.Variables(0).ValuesAsNumpy()
hourly_relative_humidity_2m = hourly.Variables(1).ValuesAsNumpy()
hourly_precipitation = hourly.Variables(2).ValuesAsNumpy()
hourly_weather_code = hourly.Variables(3).ValuesAsNumpy()
hourly_cloud_cover = hourly.Variables(4).ValuesAsNumpy()
hourly_wind_speed_10m = hourly.Variables(5).ValuesAsNumpy()
hourly_wind_direction_10m = hourly.Variables(6).ValuesAsNumpy()
hourly_shortwave_radiation_instant = hourly.Variables(7).ValuesAsNumpy()
hourly_direct_radiation_instant = hourly.Variables(8).ValuesAsNumpy()
hourly_diffuse_radiation_instant = hourly.Variables(9).ValuesAsNumpy()
hourly_direct_normal_irradiance_instant = hourly.Variables(10).ValuesAsNumpy()
hourly_global_tilted_irradiance_instant = hourly.Variables(11).ValuesAsNumpy()
hourly_surface_pressure = hourly.Variables(12).ValuesAsNumpy()

hourly_data = {"date": pd.date_range(
	start = pd.to_datetime(hourly.Time(), unit = "s", utc = True).tz_convert("America/Los_Angeles").tz_localize(None),
	end =  pd.to_datetime(hourly.TimeEnd(), unit = "s", utc = True).tz_convert("America/Los_Angeles").tz_localize(None),
	freq = pd.Timedelta(seconds = hourly.Interval()),
	inclusive = "left"
)}

hourly_data["temperature_2m"] = hourly_temperature_2m
hourly_data["relative_humidity_2m"] = hourly_relative_humidity_2m
hourly_data["precipitation"] = hourly_precipitation
hourly_data["weather_code"] = hourly_weather_code
hourly_data["cloud_cover"] = hourly_cloud_cover
hourly_data["wind_speed_10m"] = hourly_wind_speed_10m
hourly_data["wind_direction_10m"] = hourly_wind_direction_10m
hourly_data["shortwave_radiation_instant"] = hourly_shortwave_radiation_instant
hourly_data["direct_radiation_instant"] = hourly_direct_radiation_instant
hourly_data["diffuse_radiation_instant"] = hourly_diffuse_radiation_instant
hourly_data["direct_normal_irradiance_instant"] = hourly_direct_normal_irradiance_instant
hourly_data["global_tilted_irradiance_instant"] = hourly_global_tilted_irradiance_instant
hourly_data["surface_pressure"] = hourly_surface_pressure

hourly_dataframe = pd.DataFrame(data = hourly_data)
print("\nHourly data\n", hourly_dataframe)

# Process daily data. The order of variables needs to be the same as requested.
daily = response.Daily()
daily_sunrise = daily.Variables(0).ValuesInt64AsNumpy()
daily_sunset = daily.Variables(1).ValuesInt64AsNumpy()
daily_daylight_duration = daily.Variables(2).ValuesAsNumpy()
daily_sunshine_duration = daily.Variables(3).ValuesAsNumpy()
daily_precipitation_sum = daily.Variables(4).ValuesAsNumpy()
daily_rain_sum = daily.Variables(5).ValuesAsNumpy()
daily_snowfall_sum = daily.Variables(6).ValuesAsNumpy()
daily_precipitation_hours = daily.Variables(7).ValuesAsNumpy()
daily_temperature_2m_max = daily.Variables(8).ValuesAsNumpy()
daily_temperature_2m_min = daily.Variables(9).ValuesAsNumpy()
daily_apparent_temperature_max = daily.Variables(10).ValuesAsNumpy()

daily_data = {"date": pd.date_range(
	start = pd.to_datetime(daily.Time(), unit = "s", utc = True).tz_convert("America/Los_Angeles").tz_localize(None),
	end =  pd.to_datetime(daily.TimeEnd(), unit = "s", utc = True).tz_convert("America/Los_Angeles").tz_localize(None),
	freq = pd.Timedelta(seconds = daily.Interval()),
	inclusive = "left"
)}

daily_data["sunrise"] = daily_sunrise
daily_data["sunset"] = daily_sunset
daily_data["daylight_duration"] = daily_daylight_duration
daily_data["sunshine_duration"] = daily_sunshine_duration
daily_data["precipitation_sum"] = daily_precipitation_sum
daily_data["rain_sum"] = daily_rain_sum
daily_data["snowfall_sum"] = daily_snowfall_sum
daily_data["precipitation_hours"] = daily_precipitation_hours
daily_data["temperature_2m_max"] = daily_temperature_2m_max
daily_data["temperature_2m_min"] = daily_temperature_2m_min
daily_data["apparent_temperature_max"] = daily_apparent_temperature_max

daily_dataframe = pd.DataFrame(data = daily_data)
print("\nDaily data\n", daily_dataframe)

# Create output directories
import os
raw_dir = "data/raw/farm_2107"
interim_dir = "data/interim/farm_2107"
os.makedirs(raw_dir, exist_ok=True)
os.makedirs(interim_dir, exist_ok=True)

# Save hourly data as CSV (raw)
hourly_csv_path = os.path.join(raw_dir, "historical_weather_hourly.csv")
hourly_dataframe.to_csv(hourly_csv_path, index=False)
print(f"\n✓ Hourly CSV saved to: {hourly_csv_path}")

# Save daily data as CSV (raw)
daily_csv_path = os.path.join(raw_dir, "historical_weather_daily.csv")
daily_dataframe.to_csv(daily_csv_path, index=False)
print(f"✓ Daily CSV saved to: {daily_csv_path}")

# Save hourly data as Parquet (interim)
hourly_parquet_path = os.path.join(interim_dir, "historical_weather_hourly.parquet")
hourly_dataframe.to_parquet(hourly_parquet_path, index=False, engine='pyarrow', compression='snappy')
print(f"\n✓ Hourly Parquet saved to: {hourly_parquet_path}")
print(f"  Rows: {len(hourly_dataframe):,}, Columns: {len(hourly_dataframe.columns)}")

# Save daily data as Parquet (interim)
daily_parquet_path = os.path.join(interim_dir, "historical_weather_daily.parquet")
daily_dataframe.to_parquet(daily_parquet_path, index=False, engine='pyarrow', compression='snappy')
print(f"✓ Daily Parquet saved to: {daily_parquet_path}")
print(f"  Rows: {len(daily_dataframe):,}, Columns: {len(daily_dataframe.columns)}")
