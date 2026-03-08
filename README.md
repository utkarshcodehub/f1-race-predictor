# 🏎️ F1 Race Predictor

Predict full Formula 1 race leaderboards using 70+ years of historical data and machine learning.

🌐 **Live app → [f1-race-predictoru.streamlit.app](https://f1-race-predictoru.streamlit.app)**

---

## What it does

- Build a custom race grid with any drivers and constructors
- Set circuit, year, and weather conditions (wet or dry)
- Get a full predicted finishing order with podium spotlight
- Explore historical F1 stats, model performance, and driver career data

---

## How the model works

Trained an **XGBoost Regressor** on F1 race data from 1950–2024 to predict exact finishing positions.

Key features used:
- Grid & qualifying position
- Driver's rolling average finish (last 5 races)
- Constructor's rolling average finish
- Driver's historical average at the circuit
- Championship standing before the race
- Wet/dry race flag + rainfall amount

---

## Setup

**1. Clone the repo**
```bash
git clone https://github.com/utkarshcodehub/f1-race-predictor.git
cd f1-race-predictor
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Add your Kaggle API token**

Place `kaggle.json` in `C:\Users\<You>\.kaggle\` (Windows) or `~/.kaggle/` (Mac/Linux).

**4. Download data & train the model**
```bash
python src/setup_data.py
python src/fetch_weather.py
```

**5. Run the app**
```bash
cd app
streamlit run main.py
```

---

## Stack

Python · XGBoost · Scikit-learn · Streamlit · Plotly · Pandas · Open-Meteo API · Kaggle

---

*Lights out and away we go! 🏁*