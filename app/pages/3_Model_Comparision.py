import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

st.set_page_config(page_title="Model Comparison", page_icon="🤖", layout="wide")

import os
css_path = os.path.join(os.path.dirname(__file__), '..', 'style.css')
with open(css_path) as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

@st.cache_resource
def load_models_and_data():
   ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    lr  = joblib.load(os.path.join(ROOT, 'models', 'linear_regression.pkl'))
    rf  = joblib.load(os.path.join(ROOT, 'models', 'random_forest.pkl'))
    xgb = joblib.load(os.path.join(ROOT, 'models', 'xgboost.pkl'))
    df  = pd.read_csv(os.path.join(ROOT, 'data', 'processed', 'f1_features.csv'))

    for col in ['driverId', 'constructorId', 'circuitId']:
        df[col] = LabelEncoder().fit_transform(df[col].astype(str))

    fill_cols = ['quali_position', 'driver_circuit_avg_finish',
                 'prev_standing', 'pit_stop_count']
    for col in fill_cols:
        df[col] = df[col].fillna(df[col].median())

    df = df.replace([np.inf, -np.inf], np.nan).fillna(df.median(numeric_only=True))

    X = df.drop(columns=['positionOrder'])
    y = df['positionOrder']
    test_mask = df['year'] > 2021
    return lr, rf, xgb, X[test_mask], y[test_mask]

lr, rf, xgb, X_test, y_test = load_models_and_data()

st.title("🤖 MODEL COMPARISON")
st.markdown("---")

models = {'Linear Regression': lr, 'Random Forest': rf, 'XGBoost': xgb}
colors = {'Linear Regression': '#636EFA', 'Random Forest': '#00CC96', 'XGBoost': '#e10600'}

results = {}
for name, model in models.items():
    preds = model.predict(X_test)
    results[name] = {
        'MAE':  mean_absolute_error(y_test, preds),
        'RMSE': np.sqrt(mean_squared_error(y_test, preds)),
        'R2':   r2_score(y_test, preds),
        'preds': preds
    }

# ── Metrics table ──────────────────────────────────────────────
st.subheader("📊 Metrics Summary")
summary = pd.DataFrame({
    name: {'MAE': v['MAE'], 'RMSE': v['RMSE'], 'R²': v['R2']}
    for name, v in results.items()
}).T.round(3)
st.dataframe(summary, use_container_width=True)
st.caption("MAE = avg positions off | Lower MAE/RMSE is better | Higher R² is better")

st.markdown("---")

# ── MAE bar chart ──────────────────────────────────────────────
st.subheader("Mean Absolute Error (lower = better)")
fig_mae = px.bar(
    x=list(results.keys()),
    y=[v['MAE'] for v in results.values()],
    color=list(results.keys()),
    color_discrete_map=colors,
    template='plotly_dark',
    labels={'x': 'Model', 'y': 'MAE (positions)'}
)
fig_mae.update_layout(paper_bgcolor='#0f0f0f', plot_bgcolor='#1a1a1a', showlegend=False)
st.plotly_chart(fig_mae, use_container_width=True)

# ── Predicted vs Actual ────────────────────────────────────────
st.markdown("---")
st.subheader("Predicted vs Actual Position (XGBoost)")
fig_scatter = px.scatter(
    x=y_test, y=results['XGBoost']['preds'],
    opacity=0.3, template='plotly_dark',
    labels={'x': 'Actual Position', 'y': 'Predicted Position'}
)
fig_scatter.add_trace(go.Scatter(
    x=[1, 20], y=[1, 20], mode='lines',
    line=dict(color='#e10600', dash='dash'),
    name='Perfect prediction'
))
fig_scatter.update_traces(marker=dict(color='#e10600'), selector=dict(mode='markers'))
fig_scatter.update_layout(paper_bgcolor='#0f0f0f', plot_bgcolor='#1a1a1a')
st.plotly_chart(fig_scatter, use_container_width=True)

# ── Feature importance ─────────────────────────────────────────
st.markdown("---")
st.subheader("XGBoost — Feature Importance")
feat_imp = pd.Series(xgb.feature_importances_, index=X_test.columns).sort_values(ascending=True)
fig_fi = px.bar(feat_imp, orientation='h', template='plotly_dark',
                color=feat_imp.values, color_continuous_scale='reds',
                labels={'index': 'Feature', 'value': 'Importance'})
fig_fi.update_layout(paper_bgcolor='#0f0f0f', plot_bgcolor='#1a1a1a', showlegend=False)
st.plotly_chart(fig_fi, use_container_width=True)