import pandas as pd
from sklearn.ensemble import IsolationForest
import os

def get_anomaly_detector():
    data_path = os.path.join('data', 'FedCycleData071012 (2).csv')
    df = pd.read_csv(data_path)
    
    X = df[['LengthofCycle']].dropna()
    
    model = IsolationForest(contamination=0.05, random_state=42)
    model.fit(X)
    
    return model