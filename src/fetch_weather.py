import pandas as pd
import numpy as np
import requests
import requests_cache
from retry_requests import retry
import time
import os

# ── Setup cached session (avoids re-fetching same data) ────────
cache_session = requests_cache.CachedSession('.cache', expire_after=-1)
retry_session = retry(cache_session, retries=5, backoff_factor=0.2)

# ── Load race & circuit data ───────────────────────────────────
races    = pd.read_csv('../data/raw/races.csv')
circuits = pd.read_csv('../data/raw/circuits.csv')

# Only races from 1980+ (older weather data is unreliable)
races = races[races['year'] >= 1980].copy()
races['date'] = pd.to_datetime(races['date'], errors='coerce')
races = races.dropna(subset=['date'])

# Merge circuit coordinates
races = races.merge(
    circuits[['circuitId', 'lat', 'lng']],
    on='circuitId', how='left'
)
races['lat'] = pd.to_numeric(races['lat'], errors='coerce')
races['lng'] = pd.to_numeric(races['lng'], errors='coerce')
races = races.dropna(subset=['lat', 'lng'])

print(f"✅ {len(races)} races to fetch weather for")

# ── Fetch weather for each race ────────────────────────────────
results = []
failed  = []

for i, row in races.iterrows():
    race_date = row['date'].strftime('%Y-%m-%d')
    lat       = round(float(row['lat']), 4)
    lng       = round(float(row['lng']), 4)
    race_id   = row['raceId']

    try:
        url    = "https://archive-api.open-meteo.com/v1/archive"
        params = {
            "latitude":        lat,
            "longitude":       lng,
            "start_date":      race_date,
            "end_date":        race_date,
            "daily":           ["precipitation_sum", "temperature_2m_max", "windspeed_10m_max"],
            "timezone":        "auto"
        }
        response = retry_session.get(url, params=params, timeout=15)
        data     = response.json()

        if 'daily' not in data:
            failed.append(race_id)
            continue

        daily        = data['daily']
        precip       = daily.get('precipitation_sum', [None])[0]
        temp         = daily.get('temperature_2m_max', [None])[0]
        wind         = daily.get('windspeed_10m_max',  [None])[0]

        results.append({
            'raceId':            race_id,
            'precipitation_mm':  float(precip) if precip is not None else 0.0,
            'temp_max':          float(temp)   if temp   is not None else np.nan,
            'wind_max':          float(wind)   if wind   is not None else np.nan,
        })

        # Progress update every 50 races
        if len(results) % 50 == 0:
            print(f"   fetched {len(results)}/{len(races)} races...")

        time.sleep(0.1)  # be polite to the API

    except Exception as e:
        print(f"   ⚠️ Failed raceId {race_id}: {e}")
        failed.append(race_id)
        continue

# ── Derive wet/dry flag ────────────────────────────────────────
weather_df = pd.DataFrame(results)

# Wet race = more than 1mm of rainfall on race day
weather_df['is_wet_race'] = (weather_df['precipitation_mm'] > 1.0).astype(int)

print(f"\n✅ Successfully fetched: {len(weather_df)} races")
print(f"⚠️  Failed:              {len(failed)} races")
print(f"\n🌧️  Wet races:  {weather_df['is_wet_race'].sum()}")
print(f"☀️   Dry races:  {(weather_df['is_wet_race'] == 0).sum()}")

# ── Save ───────────────────────────────────────────────────────
os.makedirs('../data/processed', exist_ok=True)
weather_df.to_csv('../data/processed/weather.csv', index=False)
print("\n✅ Saved to data/processed/weather.csv")