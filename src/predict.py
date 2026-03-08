import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder

# ── Load model & data ──────────────────────────────────────────
xgb        = joblib.load('../models/xgboost.pkl')
df         = pd.read_csv('../data/processed/f1_features.csv')
df_history = pd.read_csv('../data/raw/results.csv').replace('\\N', np.nan)
df_history['positionOrder'] = pd.to_numeric(df_history['positionOrder'], errors='coerce')
raw_races   = pd.read_csv('../data/raw/races.csv')

# ── Rebuild encoders (must match training) ─────────────────────
le_driver      = LabelEncoder().fit(df['driverId'].astype(str))
le_constructor = LabelEncoder().fit(df['constructorId'].astype(str))
le_circuit     = LabelEncoder().fit(df['circuitId'].astype(str))

# ── Helper: show available options ────────────────────────────
def show_options():
    raw_results      = pd.read_csv('../data/raw/results.csv')
    raw_drivers      = pd.read_csv('../data/raw/drivers.csv')
    raw_constructors = pd.read_csv('../data/raw/constructors.csv')
    raw_circuits     = pd.read_csv('../data/raw/circuits.csv')

    # Recent drivers (last 3 years)
    recent_races = raw_races[raw_races['year'] >= raw_races['year'].max() - 3]['raceId']
    recent_driver_ids = raw_results[
        raw_results['raceId'].isin(recent_races)
    ]['driverId'].unique()
    recent_drivers = raw_drivers[
        raw_drivers['driverId'].isin(recent_driver_ids)
    ][['driverId', 'forename', 'surname']]
    recent_drivers['name'] = recent_drivers['forename'] + ' ' + recent_drivers['surname']

    print("\n🏎️  RECENT DRIVERS:")
    for _, row in recent_drivers.iterrows():
        print(f"   ID: {row['driverId']:>4}  →  {row['name']}")

    # Recent constructors
    recent_con_ids = raw_results[
        raw_results['raceId'].isin(recent_races)
    ]['constructorId'].unique()
    recent_cons = raw_constructors[
        raw_constructors['constructorId'].isin(recent_con_ids)
    ][['constructorId', 'name']]

    print("\n🏭  RECENT CONSTRUCTORS:")
    for _, row in recent_cons.iterrows():
        print(f"   ID: {row['constructorId']:>4}  →  {row['name']}")

    # All circuits
    raw_circuits_all = pd.read_csv('../data/raw/circuits.csv')
    print("\n🏁  CIRCUITS:")
    for _, row in raw_circuits_all[['circuitId', 'name', 'country']].iterrows():
        print(f"   ID: {row['circuitId']:>4}  →  {row['name']} ({row['country']})")

# ── Build feature row for prediction ──────────────────────────
def build_features(driver_id, constructor_id, circuit_id, grid, quali_pos, year, round_num):
    driver_id      = str(driver_id)
    constructor_id = str(constructor_id)
    circuit_id     = str(circuit_id)

    # Check IDs are known
    if driver_id not in le_driver.classes_:
        print(f"⚠️  Driver ID {driver_id} not in training data.")
        return None
    if constructor_id not in le_constructor.classes_:
        print(f"⚠️  Constructor ID {constructor_id} not in training data.")
        return None
    if circuit_id not in le_circuit.classes_:
        print(f"⚠️  Circuit ID {circuit_id} not in training data.")
        return None

    # Historical averages from raw results
    d_hist = df_history[
        df_history['driverId'] == int(driver_id)
    ]['positionOrder']

    c_hist = df_history[
        df_history['constructorId'] == int(constructor_id)
    ]['positionOrder']

    # Circuit history: map circuitId → raceIds → filter results
    circuit_races = raw_races[
        raw_races['circuitId'] == int(circuit_id)
    ]['raceId']
    dc_hist = df_history[
        (df_history['driverId'] == int(driver_id)) &
        (df_history['raceId'].isin(circuit_races))
    ]['positionOrder']

    stand_hist = df[df['driverId'] == int(driver_id)]['prev_standing']

    features = {
        'grid':                           grid,
        'quali_position':                 quali_pos,
        'driver_rolling_avg_finish':      d_hist.tail(5).mean()               if len(d_hist)  > 0 else 10,
        'constructor_rolling_avg_finish': c_hist.tail(5).mean()               if len(c_hist)  > 0 else 10,
        'driver_circuit_avg_finish':      dc_hist.mean()                      if len(dc_hist) > 0 else 10,
        'prev_standing':                  stand_hist.dropna().iloc[-1]        if len(stand_hist.dropna()) > 0 else 10,
        'pit_stop_count':                 2,
        'year':                           year,
        'round':                          round_num,
        'driverId':                       le_driver.transform([driver_id])[0],
        'constructorId':                  le_constructor.transform([constructor_id])[0],
        'circuitId':                      le_circuit.transform([circuit_id])[0],
    }

    return pd.DataFrame([features])

# ── Predict ────────────────────────────────────────────────────
def predict(driver_id, constructor_id, circuit_id, grid, quali_pos, year, round_num):
    row = build_features(driver_id, constructor_id, circuit_id, grid, quali_pos, year, round_num)
    if row is None:
        return

    prob    = xgb.predict_proba(row)[0][1]
    verdict = "🏆 PODIUM" if prob >= 0.5 else "❌ NO PODIUM"

    print("\n" + "="*45)
    print(f"  Prediction:   {verdict}")
    print(f"  Podium Prob:  {prob*100:.1f}%")
    print("="*45)

    if prob >= 0.75:
        print("  💪 Very likely on the podium!")
    elif prob >= 0.5:
        print("  🤞 Could go either way, slight podium edge")
    elif prob >= 0.25:
        print("  😬 Unlikely but not impossible")
    else:
        print("  💨 Very unlikely to podium")

# ── Main interactive loop ──────────────────────────────────────
if __name__ == '__main__':
    print("🏎️  F1 Podium Predictor")
    print("="*45)

    while True:
        print("\nOptions:")
        print("  1 → Make a prediction")
        print("  2 → Show available driver/constructor/circuit IDs")
        print("  3 → Quit")

        choice = input("\nYour choice (1/2/3): ").strip()

        if choice == '2':
            show_options()

        elif choice == '1':
            try:
                driver_id      = int(input("\n  Driver ID:      "))
                constructor_id = int(input("  Constructor ID: "))
                circuit_id     = int(input("  Circuit ID:     "))
                grid           = int(input("  Grid position:  "))
                quali_pos      = int(input("  Quali position: "))
                year           = int(input("  Year:           "))
                round_num      = int(input("  Round number:   "))

                predict(driver_id, constructor_id, circuit_id,
                        grid, quali_pos, year, round_num)

            except ValueError:
                print("⚠️  Please enter valid numbers only.")

        elif choice == '3':
            print("\n👋 Goodbye!")
            break

        else:
            print("⚠️  Please enter 1, 2, or 3.")