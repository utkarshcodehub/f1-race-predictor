import streamlit as st
import pandas as pd
import numpy as np
import os
import plotly.express as px

st.set_page_config(page_title="EDA", page_icon="📊", layout="wide")

css_path = os.path.join(os.path.dirname(__file__), '..', 'style.css')
with open(css_path) as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

@st.cache_data
def load_data():
    ROOT         = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    results      = pd.read_csv(os.path.join(ROOT, 'data', 'raw', 'results.csv')).replace('\\N', np.nan)
    drivers      = pd.read_csv(os.path.join(ROOT, 'data', 'raw', 'drivers.csv'))
    constructors = pd.read_csv(os.path.join(ROOT, 'data', 'raw', 'constructors.csv'))
    races        = pd.read_csv(os.path.join(ROOT, 'data', 'raw', 'races.csv'))
    results['positionOrder'] = pd.to_numeric(results['positionOrder'], errors='coerce')
    results['grid']          = pd.to_numeric(results['grid'],          errors='coerce')
    return results, drivers, constructors, races

results, drivers, constructors, races = load_data()

st.title("📊 F1 DATA EXPLORER")
st.markdown("---")

tab1, tab2, tab3 = st.tabs(["🏆 Top Winners", "🏁 Grid vs Finish", "🏭 Constructor Dominance"])

with tab1:
    st.subheader("Top Drivers by Race Wins")
    n = st.slider("Show top N drivers", 5, 25, 10)
    wins = results[results['positionOrder'] == 1].merge(
        drivers[['driverId', 'forename', 'surname']], on='driverId'
    )
    wins['driver'] = wins['forename'] + ' ' + wins['surname']
    top = wins['driver'].value_counts().head(n).reset_index()
    top.columns = ['Driver', 'Wins']
    fig = px.bar(top, x='Wins', y='Driver', orientation='h',
                 color='Wins', color_continuous_scale='reds',
                 template='plotly_dark')
    fig.update_layout(paper_bgcolor='#0f0f0f', plot_bgcolor='#0f0f0f',
                      showlegend=False, yaxis={'categoryorder': 'total ascending'})
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.subheader("Grid Position vs Finishing Position")
    st.markdown("Does starting position determine the result?")
    sample = results[results['grid'] > 0].dropna(
        subset=['grid', 'positionOrder']
    ).sample(3000, random_state=42)
    fig2 = px.scatter(sample, x='grid', y='positionOrder',
                      opacity=0.3, template='plotly_dark',
                      labels={'grid': 'Grid Position', 'positionOrder': 'Finishing Position'},
                      trendline='ols', trendline_color_override='#e10600')
    fig2.update_traces(marker=dict(color='#e10600'))
    fig2.update_layout(paper_bgcolor='#0f0f0f', plot_bgcolor='#1a1a1a')
    st.plotly_chart(fig2, use_container_width=True)

with tab3:
    st.subheader("Constructor Wins Per Year")
    wins_con  = results[results['positionOrder'] == 1].merge(
        races[['raceId', 'year']], on='raceId'
    ).merge(constructors[['constructorId', 'name']], on='constructorId')
    top_teams = wins_con['name'].value_counts().head(8).index
    data_plot = wins_con[wins_con['name'].isin(top_teams)]
    grouped   = data_plot.groupby(['year', 'name']).size().reset_index(name='wins')
    fig3 = px.area(grouped, x='year', y='wins', color='name',
                   template='plotly_dark',
                   labels={'year': 'Year', 'wins': 'Wins', 'name': 'Constructor'})
    fig3.update_layout(paper_bgcolor='#0f0f0f', plot_bgcolor='#1a1a1a')
    st.plotly_chart(fig3, use_container_width=True)