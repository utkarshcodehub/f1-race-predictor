import os
import subprocess
import pandas as pd
import numpy as np

print("🏎️  F1 Race Predictor — Data Setup")
print("="*45)

# ── Step 1: Download Kaggle dataset ───────────────────────────
print("\n📦 Downloading F1 dataset from Kaggle...")
os.makedirs('data/raw', exist_ok=True)
os.makedirs('data/processed', exist_ok=True)
os.makedirs('models', exist_ok=True)

subprocess.run([
    'kaggle', 'datasets', 'download',
    '-d', 'rohanrao/formula-1-world-championship-1950-2020',
    '-p', 'data/raw', '--unzip', '--force'
], check=True)
print("✅ Dataset downloaded!")

# ── Step 2: Feature engineering ───────────────────────────────
print("\n🔧 Running feature engineering...")

results      = pd.read_csv('data/raw/results.csv').replace('\\N', np.nan)
drivers      = pd.read_csv('data/raw/drivers.csv')
constructors = pd.read_csv('data/raw/constructors.csv')
races        = pd.read_csv('data/raw/races.csv')
qualifying   = pd.read_csv('data/raw/qualifying.csv').replace('\\N', np.nan)
pit_stops    = pd.read_csv('data/raw/pit_stops.csv').replace('\\N', np.nan)
standings    = pd.read_csv('data/raw/driver_standings.csv').replace('\\N', np.nan)

for df in [results, qualifying, pit_stops, standings]:
    df.replace('\\N', np.nan, inplace=True)

# Base dataframe
df = results[['raceId','driverId','constructorId','grid',
              'positionOrder','points','statusId']].copy()
df = df.merge(races[['raceId','year','round','circuitId']], on='raceId', how='left')
df = df.merge(drivers[['driverId','nationality']], on='driverId', how='left')
df = df.merge(
    constructors[['constructorId','nationality']].rename(
        columns={'nationality':'constructor_nationality'}),
    on='constructorId', how='left'
)

df['grid']          = pd.to_numeric(df['grid'],          errors='coerce')
df['positionOrder'] = pd.to_numeric(df['positionOrder'], errors='coerce')
df['points']        = pd.to_numeric(df['points'],        errors='coerce')

# Qualifying
qualifying['position'] = pd.to_numeric(qualifying['position'], errors='coerce')
quali_clean = qualifying[['raceId','driverId','position']].rename(
    columns={'position':'quali_position'})
df = df.merge(quali_clean, on=['raceId','driverId'], how='left')

# Rolling features
df = df.sort_values(['driverId','year','round']).reset_index(drop=True)
df['driver_rolling_avg_finish'] = (
    df.groupby('driverId')['positionOrder']
    .transform(lambda x: x.shift(1).rolling(window=5, min_periods=1).mean())
)
df['constructor_rolling_avg_finish'] = (
    df.groupby('constructorId')['positionOrder']
    .transform(lambda x: x.shift(1).rolling(window=5, min_periods=1).mean())
)
df = df.sort_values(['driverId','circuitId','year']).reset_index(drop=True)
df['driver_circuit_avg_finish'] = (
    df.groupby(['driverId','circuitId'])['positionOrder']
    .transform(lambda x: x.shift(1).expanding().mean())
)

# Standings
standings['position'] = pd.to_numeric(standings['position'], errors='coerce')
standings['points']   = pd.to_numeric(standings['points'],   errors='coerce')
standings_sorted = standings.sort_values(['driverId','raceId'])
standings_sorted['prev_standing'] = (
    standings_sorted.groupby('driverId')['position'].shift(1)
)
df = df.merge(
    standings_sorted[['raceId','driverId','prev_standing']],
    on=['raceId','driverId'], how='left'
)

# Pit stops
pit_counts = pit_stops.groupby(['raceId','driverId']).size().reset_index(name='pit_stop_count')
df = df.merge(pit_counts, on=['raceId','driverId'], how='left')
df['pit_stop_count'] = df['pit_stop_count'].fillna(0)

# Weather
if os.path.exists('data/processed/weather.csv'):
    weather = pd.read_csv('data/processed/weather.csv')
    df = df.merge(weather[['raceId','is_wet_race','precipitation_mm']],
                  on='raceId', how='left')
    df['is_wet_race']      = df['is_wet_race'].fillna(0).astype(int)
    df['precipitation_mm'] = df['precipitation_mm'].fillna(0.0)
else:
    df['is_wet_race']      = 0
    df['precipitation_mm'] = 0.0
    print("⚠️  No weather data found — defaulting to dry for all races")

# Final features
feature_cols = [
    'grid','quali_position',
    'driver_rolling_avg_finish','constructor_rolling_avg_finish',
    'driver_circuit_avg_finish','prev_standing','pit_stop_count',
    'is_wet_race','precipitation_mm',
    'year','round','driverId','constructorId','circuitId',
    'positionOrder'
]
df_model = df[feature_cols].copy()
df_model = df_model.dropna(subset=['positionOrder','grid','driver_rolling_avg_finish'])
df_model.to_csv('data/processed/f1_features.csv', index=False)
print(f"✅ Features saved! Shape: {df_model.shape}")

# ── Step 3: Train model ────────────────────────────────────────
print("\n🤖 Training XGBoost model...")
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error
from xgboost import XGBRegressor
import joblib

le = LabelEncoder()
for col in ['driverId','constructorId','circuitId']:
    df_model[col] = le.fit_transform(df_model[col].astype(str))

fill_cols = ['quali_position','driver_circuit_avg_finish',
             'prev_standing','pit_stop_count']
for col in fill_cols:
    df_model[col] = df_model[col].fillna(df_model[col].median())

df_model = df_model.replace([np.inf,-np.inf], np.nan)
df_model = df_model.fillna(df_model.median(numeric_only=True))

X = df_model.drop(columns=['positionOrder'])
y = df_model['positionOrder']

train_mask = df_model['year'] <= 2022
X_train, y_train = X[train_mask], y[train_mask]
X_test,  y_test  = X[~train_mask], y[~train_mask]

xgb = XGBRegressor(
    n_estimators=300, max_depth=6, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8,
    random_state=42, n_jobs=-1
)
xgb.fit(X_train, y_train)

mae = mean_absolute_error(y_test, xgb.predict(X_test))
print(f"✅ Model trained! MAE: {mae:.3f} positions")

joblib.dump(xgb, 'models/xgboost.pkl')
print("✅ Model saved to models/xgboost.pkl")
print("\n🏁 Setup complete! You can now run the app.")
