import streamlit as st
import os
import sys

# Ensure project root is in path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

st.set_page_config(
    page_title="F1 Race Predictor",
    page_icon="🏎️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load CSS
css_path = os.path.join(os.path.dirname(__file__), 'style.css')
with open(css_path) as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

st.title("🏎️ F1 RACE PREDICTOR")
st.markdown("---")

st.markdown("""
Welcome to the **F1 Race Predictor** — a machine learning app built on historical
Formula 1 data from 1950 to 2024.

### 📌 Navigation
Use the sidebar to explore:

| Page | Description |
|------|-------------|
| 🎯 Prediction | Predict the full race leaderboard for any grid |
| 📊 EDA | Explore historical F1 data with interactive charts |
| 🤖 Model Comparison | Compare ML model performance |
| 👤 Driver Stats | Look up any driver or constructor's stats |
""")

st.markdown("---")
col1, col2, col3 = st.columns(3)
col1.metric("🏁 Total Races in Dataset", "1,100+")
col2.metric("🏎️ Drivers",               "857+")
col3.metric("🏭 Constructors",           "210+")