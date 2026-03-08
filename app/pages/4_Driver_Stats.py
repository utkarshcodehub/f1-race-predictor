import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

st.set_page_config(page_title="Driver Stats", page_icon="👤", layout="wide")

import os
css_path = os.path.join(os.path.dirname(__file__), '..', 'style.css')
with open(css_path) as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

@st.cache_data
def load_data():
    ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    results      = pd.read_csv(os.path.join(ROOT, 'data', 'raw', 'results.csv')).replace('\\N', np.nan)
    drivers      = pd.read_csv(os.path.join(ROOT, 'data', 'raw', 'drivers.csv'))
    constructors = pd.read_csv(os.path.join(ROOT, 'data', 'raw', 'constructors.csv'))
    races        = pd.read_csv(os.path.join(ROOT, 'data', 'raw', 'races.csv'))
    circuits     = pd.read_csv(os.path.join(ROOT, 'data', 'raw', 'circuits.csv'))
    results['positionOrder'] = pd.to_numeric(results['positionOrder'], errors='coerce')
    results['points']        = pd.to_numeric(results['points'],        errors='coerce')
    results['grid']          = pd.to_numeric(results['grid'],          errors='coerce')
    drivers['label'] = drivers['forename'] + ' ' + drivers['surname']
    return results, drivers, constructors, races, circuits

results, drivers, constructors, races, circuits = load_data()

st.title("👤 DRIVER & CONSTRUCTOR STATS")
st.markdown("---")

tab1, tab2 = st.tabs(["🏎️ Driver Lookup", "🏭 Constructor Lookup"])

# ── Driver Lookup ──────────────────────────────────────────────
with tab1:
    driver_label = st.selectbox("Select a Driver", sorted(drivers['label'].unique()))
    driver_row   = drivers[drivers['label'] == driver_label].iloc[0]
    driver_id    = driver_row['driverId']

    driver_results = results[results['driverId'] == driver_id].merge(
        races[['raceId', 'year', 'round', 'name', 'circuitId']], on='raceId'
    )

    if driver_results.empty:
        st.warning("No data found for this driver.")
    else:
        total_races  = len(driver_results)
        total_wins   = (driver_results['positionOrder'] == 1).sum()
        total_podiums= (driver_results['positionOrder'] <= 3).sum()
        total_points = driver_results['points'].sum()
        avg_finish   = driver_results['positionOrder'].mean()

        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("🏁 Races",        total_races)
        c2.metric("🏆 Wins",         total_wins)
        c3.metric("🥇 Podiums",      total_podiums)
        c4.metric("⭐ Points",       f"{total_points:.0f}")
        c5.metric("📊 Avg Finish",   f"{avg_finish:.1f}")

        st.markdown("---")

        # Points per season
        st.subheader(f"{driver_label} — Points Per Season")
        season_pts = driver_results.groupby('year')['points'].sum().reset_index()
        fig = px.bar(season_pts, x='year', y='points',
                     template='plotly_dark', color='points',
                     color_continuous_scale='reds',
                     labels={'year': 'Season', 'points': 'Points'})
        fig.update_layout(paper_bgcolor='#0f0f0f', plot_bgcolor='#1a1a1a',
                          showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

        # Finish position history
        st.subheader("Finishing Position Over Career")
        fig2 = px.line(driver_results.sort_values(['year', 'round']),
                       x='round', y='positionOrder', color='year',
                       template='plotly_dark',
                       labels={'round': 'Round', 'positionOrder': 'Finish Position'})
        fig2.update_yaxes(autorange='reversed')
        fig2.update_layout(paper_bgcolor='#0f0f0f', plot_bgcolor='#1a1a1a')
        st.plotly_chart(fig2, use_container_width=True)

# ── Constructor Lookup ─────────────────────────────────────────
with tab2:
    con_name = st.selectbox("Select a Constructor",
                            sorted(constructors['name'].unique()))
    con_id   = constructors[constructors['name'] == con_name].iloc[0]['constructorId']

    con_results = results[results['constructorId'] == con_id].merge(
        races[['raceId', 'year', 'round']], on='raceId'
    )

    if con_results.empty:
        st.warning("No data found for this constructor.")
    else:
        total_races   = len(con_results)
        total_wins    = (con_results['positionOrder'] == 1).sum()
        total_podiums = (con_results['positionOrder'] <= 3).sum()
        total_points  = con_results['points'].sum()

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("🏁 Race Entries", total_races)
        c2.metric("🏆 Wins",         total_wins)
        c3.metric("🥇 Podiums",      total_podiums)
        c4.metric("⭐ Total Points", f"{total_points:.0f}")

        st.markdown("---")

        st.subheader(f"{con_name} — Wins Per Season")
        season_wins = con_results[con_results['positionOrder'] == 1]\
            .groupby('year').size().reset_index(name='wins')
        fig3 = px.bar(season_wins, x='year', y='wins',
                      template='plotly_dark', color='wins',
                      color_continuous_scale='reds',
                      labels={'year': 'Season', 'wins': 'Wins'})
        fig3.update_layout(paper_bgcolor='#0f0f0f', plot_bgcolor='#1a1a1a',
                           showlegend=False)
        st.plotly_chart(fig3, use_container_width=True)