import pandas as pd
from sklearn.ensemble import IsolationForest
import os

def detect_anomalies():
    data_path = os.path.join('data', 'FedCycleData071012 (2).csv')
    df = pd.read_csv(data_path)
    
    X = df[['LengthofCycle']].dropna()
    
    model = IsolationForest(contamination=0.05, random_state=42)
    X['anomaly'] = model.fit_predict(X)
    
    anomalies = X[X['anomaly'] == -1]
    
    print(f"--- Anomaly Detection Results ---")
    print(f"Total Cycles Analyzed: {len(X)}")
    print(f"Number of Anomalies Found: {len(anomalies)}")
    print("\nExample of 'Unusual' Cycle Lengths detected:")
    print(anomalies['LengthofCycle'].unique()[:10])
    
    return anomalies

if __name__ == "__main__":
    detect_anomalies()