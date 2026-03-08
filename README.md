f1-race-predictor/
├── data/
│   ├── raw/          ← Kaggle CSV files go here
│   └── processed/    ← cleaned/merged files
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_feature_engineering.ipynb
│   └── 03_modeling.ipynb
├── src/
│   ├── features.py   ← feature engineering functions
│   └── predict.py    ← prediction script
├── models/           ← saved .pkl model files
├── requirements.txt
└── README.md

f1-race-predictor/
├── app/
│   ├── main.py              ← app entry point
│   ├── pages/
│   │   ├── 1_Prediction.py
│   │   ├── 2_EDA.py
│   │   ├── 3_Model_Comparison.py
│   │   └── 4_Driver_Stats.py
│   └── style.css            ← custom F1 styling