import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import LabelEncoder

st.set_page_config(page_title="Prediction", page_icon="🎯", layout="wide")

import os
css_path = os.path.join(os.path.dirname(__file__), '..', 'style.css')
with open(css_path) as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

# ── Load data & model ──────────────────────────────────────────
@st.cache_resource
def load_all():
    xgb        = joblib.load('../models/xgboost.pkl')
    df         = pd.read_csv('../data/processed/f1_features.csv')
    df_history = pd.read_csv('../data/raw/results.csv').replace('\\N', np.nan)
    df_history['positionOrder'] = pd.to_numeric(df_history['positionOrder'], errors='coerce')
    raw_races        = pd.read_csv('../data/raw/races.csv')
    raw_drivers      = pd.read_csv('../data/raw/drivers.csv')
    raw_constructors = pd.read_csv('../data/raw/constructors.csv')
    raw_circuits     = pd.read_csv('../data/raw/circuits.csv')

    le_driver      = LabelEncoder().fit(df['driverId'].astype(str))
    le_constructor = LabelEncoder().fit(df['constructorId'].astype(str))
    le_circuit     = LabelEncoder().fit(df['circuitId'].astype(str))

    return (xgb, df, df_history, raw_races, raw_drivers,
            raw_constructors, raw_circuits,
            le_driver, le_constructor, le_circuit)

(xgb, df, df_history, raw_races, raw_drivers,
 raw_constructors, raw_circuits,
 le_driver, le_constructor, le_circuit) = load_all()

# ── Helpers ────────────────────────────────────────────────────
def get_driver_features(driver_id, constructor_id, circuit_id,
                         grid, quali_pos, year, round_num,
                         is_wet_race=0, precipitation_mm=0.0):
    d_str  = str(driver_id)
    c_str  = str(constructor_id)
    ci_str = str(circuit_id)

    if d_str not in le_driver.classes_:   return None
    if c_str not in le_constructor.classes_: return None
    if ci_str not in le_circuit.classes_: return None

    d_hist  = df_history[df_history['driverId'] == driver_id]['positionOrder']
    c_hist  = df_history[df_history['constructorId'] == constructor_id]['positionOrder']
    c_races = raw_races[raw_races['circuitId'] == circuit_id]['raceId']
    dc_hist = df_history[
        (df_history['driverId'] == driver_id) &
        (df_history['raceId'].isin(c_races))
    ]['positionOrder']
    stand_hist = df[df['driverId'] == driver_id]['prev_standing']

    return {
        'grid':                           grid,
        'quali_position':                 quali_pos,
        'driver_rolling_avg_finish':      d_hist.tail(5).mean()        if len(d_hist)  > 0 else 10,
        'constructor_rolling_avg_finish': c_hist.tail(5).mean()        if len(c_hist)  > 0 else 10,
        'driver_circuit_avg_finish':      dc_hist.mean()               if len(dc_hist) > 0 else 10,
        'prev_standing':                  stand_hist.dropna().iloc[-1] if len(stand_hist.dropna()) > 0 else 10,
        'pit_stop_count':                 2,
        'is_wet_race':                    is_wet_race,
        'precipitation_mm':               precipitation_mm,
        'year':                           year,
        'round':                          round_num,
        'driverId':                       le_driver.transform([d_str])[0],
        'constructorId':                  le_constructor.transform([c_str])[0],
        'circuitId':                      le_circuit.transform([ci_str])[0],
    }

# ── Dropdowns ──────────────────────────────────────────────────
recent_race_ids   = raw_races[raw_races['year'] >= raw_races['year'].max() - 3]['raceId']
recent_results    = pd.read_csv('../data/raw/results.csv').replace('\\N', np.nan)
recent_driver_ids = recent_results[recent_results['raceId'].isin(recent_race_ids)]['driverId'].unique()
recent_con_ids    = recent_results[recent_results['raceId'].isin(recent_race_ids)]['constructorId'].unique()

recent_drivers = raw_drivers[raw_drivers['driverId'].isin(recent_driver_ids)].copy()
recent_drivers['label'] = recent_drivers['forename'] + ' ' + recent_drivers['surname']

recent_cons = raw_constructors[raw_constructors['constructorId'].isin(recent_con_ids)].copy()

circuit_options = raw_circuits[['circuitId', 'name', 'country']].copy()
circuit_options['label'] = circuit_options['name'] + ' (' + circuit_options['country'] + ')'

# ── UI ─────────────────────────────────────────────────────────
st.title("🎯 RACE LEADERBOARD PREDICTOR")
st.markdown("---")
st.markdown("Build your race grid below — add up to 20 drivers and predict the full finishing order.")

# Race settings
st.subheader("🏁 Race Settings")
col1, col2, col3, col4 = st.columns(4)
with col1:
    circuit_label = st.selectbox("Circuit", sorted(circuit_options['label'].unique()))
with col2:
    year      = st.number_input("Year",         min_value=2000, max_value=2025, value=2024)
with col3:
    round_num = st.number_input("Round Number", min_value=1,    max_value=24,   value=1)
with col4:
    weather_label  = st.selectbox("🌦️ Conditions", ["☀️ Dry", "🌧️ Wet"])
    is_wet_race    = 1 if weather_label == "🌧️ Wet" else 0
    precipitation  = st.number_input("Rainfall (mm)", min_value=0.0, 
                                      max_value=50.0, value=0.0 if is_wet_race == 0 else 5.0,
                                      step=0.5)

circuit_id = int(circuit_options[circuit_options['label'] == circuit_label]['circuitId'].iloc[0])

# Grid builder
st.subheader("🏎️ Build Your Grid")
st.markdown("Add each driver to the grid with their starting position and constructor.")

if 'grid_entries' not in st.session_state:
    st.session_state.grid_entries = []

with st.expander("➕ Add a Driver to the Grid", expanded=True):
    a1, a2, a3, a4 = st.columns(4)
    with a1:
        sel_driver = st.selectbox("Driver", sorted(recent_drivers['label'].unique()), key='sel_driver')
    with a2:
        sel_constructor = st.selectbox("Constructor", sorted(recent_cons['name'].unique()), key='sel_con')
    with a3:
        sel_grid  = st.number_input("Grid Pos",  min_value=1, max_value=20, value=len(st.session_state.grid_entries)+1, key='sel_grid')
    with a4:
        sel_quali = st.number_input("Quali Pos", min_value=1, max_value=20, value=len(st.session_state.grid_entries)+1, key='sel_quali')

    b1, b2 = st.columns([1, 5])
    with b1:
        if st.button("➕ Add Driver"):
            driver_row = recent_drivers[recent_drivers['label'] == sel_driver].iloc[0]
            con_row    = recent_cons[recent_cons['name'] == sel_constructor].iloc[0]
            st.session_state.grid_entries.append({
                'driver_label':      sel_driver,
                'driver_id':         int(driver_row['driverId']),
                'constructor_label': sel_constructor,
                'constructor_id':    int(con_row['constructorId']),
                'grid':              sel_grid,
                'quali_pos':         sel_quali,
            })
            st.success(f"✅ Added {sel_driver} (P{sel_grid})")
    with b2:
        if st.button("🗑️ Clear Grid"):
            st.session_state.grid_entries = []
            st.info("Grid cleared.")

# Show current grid
if st.session_state.grid_entries:
    st.markdown("**Current Grid:**")
    grid_df = pd.DataFrame(st.session_state.grid_entries)[
        ['grid', 'driver_label', 'constructor_label', 'quali_pos']
    ].rename(columns={
        'grid':              'Grid Pos',
        'driver_label':      'Driver',
        'constructor_label': 'Constructor',
        'quali_pos':         'Quali Pos'
    }).sort_values('Grid Pos')
    st.dataframe(grid_df, use_container_width=True, hide_index=True)

st.markdown("---")

# Predict button
if st.button("🏁 PREDICT RACE RESULT", disabled=len(st.session_state.grid_entries) < 2):
    rows = []
    skipped = []

    for entry in st.session_state.grid_entries:
        feats = get_driver_features(
            entry['driver_id'], entry['constructor_id'], circuit_id,
            entry['grid'], entry['quali_pos'], year, round_num,
            is_wet_race=is_wet_race, precipitation_mm=precipitation
        )
        if feats is None:
            skipped.append(entry['driver_label'])
            continue
        feats['driver_label']      = entry['driver_label']
        feats['constructor_label'] = entry['constructor_label']
        rows.append(feats)

    if skipped:
        st.warning(f"⚠️ Skipped (not in training data): {', '.join(skipped)}")

    if len(rows) < 2:
        st.error("Not enough valid drivers to predict. Please add more.")
    else:
        pred_df   = pd.DataFrame(rows)
        label_cols = ['driver_label', 'constructor_label']
        feat_cols  = [c for c in pred_df.columns if c not in label_cols]

        preds = xgb.predict(pred_df[feat_cols])
        pred_df['predicted_score'] = preds

        # Rank by predicted score (lower = better finish)
        pred_df = pred_df.sort_values('predicted_score').reset_index(drop=True)
        pred_df['Predicted Position'] = range(1, len(pred_df) + 1)

        # Medal emojis for top 3
        def medal(pos):
            return {1: '🥇', 2: '🥈', 3: '🥉'}.get(pos, f'P{pos}')

        pred_df['Pos'] = pred_df['Predicted Position'].apply(medal)

        # ── Leaderboard table ──────────────────────────────────
        st.subheader("🏆 Predicted Race Leaderboard")
        leaderboard = pred_df[['Pos', 'driver_label', 'constructor_label', 'grid']].rename(columns={
            'driver_label':      'Driver',
            'constructor_label': 'Constructor',
            'grid':              'Starting Grid'
        })
        st.dataframe(leaderboard, use_container_width=True, hide_index=True)

        st.markdown("---")

        # ── Bar chart ──────────────────────────────────────────
        st.subheader("📊 Predicted Finishing Order")
        chart_df = pred_df.copy()
        chart_df['color'] = chart_df['Predicted Position'].apply(
            lambda x: '#e10600' if x <= 3 else '#444444'
        )
        fig = go.Figure(go.Bar(
            x=chart_df['driver_label'],
            y=chart_df['Predicted Position'],
            marker_color=chart_df['color'],
            text=chart_df['Pos'],
            textposition='outside'
        ))
        fig.update_yaxes(autorange='reversed', title='Predicted Position', dtick=1)
        fig.update_xaxes(title='Driver')
        fig.update_layout(
            template='plotly_dark',
            paper_bgcolor='#0f0f0f',
            plot_bgcolor='#1a1a1a',
            showlegend=False
        )
        st.plotly_chart(fig, use_container_width=True)

        # ── Podium spotlight ───────────────────────────────────
        st.markdown("---")
        st.subheader("🏆 Podium")
        top3 = pred_df.head(3)
        num_podium = len(top3)

        p1, p2, p3 = st.columns(3)
        if num_podium >= 1:
            p1.success(f"🥇 1st\n\n**{top3.iloc[0]['driver_label']}**\n\n{top3.iloc[0]['constructor_label']}")
        if num_podium >= 2:
            p2.info(   f"🥈 2nd\n\n**{top3.iloc[1]['driver_label']}**\n\n{top3.iloc[1]['constructor_label']}")
        if num_podium >= 3:
            p3.warning(f"🥉 3rd\n\n**{top3.iloc[2]['driver_label']}**\n\n{top3.iloc[2]['constructor_label']}")
        if num_podium < 3:
            st.info(f"ℹ️ Add at least 3 drivers to see the full podium.")

else:
    if len(st.session_state.grid_entries) < 2:
        st.info("👆 Add at least 2 drivers to the grid to enable prediction.")