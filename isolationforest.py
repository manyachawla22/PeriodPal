import pandas as pd
from sklearn.ensemble import IsolationForest
import os

def get_anomaly_detector():
    data_path = os.path.join('data', 'FedCycleData071012 (2).csv')
    df = pd.read_csv(data_path)
    
    X = df[['LengthofCycle', 'Age', 'BMI']].dropna()
    
    model = IsolationForest(
        n_estimators=200,      
        contamination=0.03,    
        max_samples='auto',    
        random_state=42
    )
    
    model.fit(X)
    
    return model