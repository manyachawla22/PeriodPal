import os
import pandas as pd
from sklearn.ensemble import IsolationForest
from functools import lru_cache

DATA_PATH = os.path.join('data', 'FedCycleData071012.csv')

@lru_cache(maxsize=1)
def get_anomaly_detector():
    df = pd.read_csv(DATA_PATH)
    X = df[['LengthofCycle', 'Age', 'BMI']].apply(pd.to_numeric, errors='coerce').dropna()

    model = IsolationForest(
        n_estimators=200,
        contamination=0.03,
        max_samples='auto',
        random_state=42
    )
    model.fit(X)
    return model
