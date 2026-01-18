import pandas as pd
import os

def load_kagggle_data():
    data_path = os.path.join('data', 'FedCycleData071012 (2).csv')
    
    if not os.path.exists(data_path):
        print(f"Error: Could not find file at {data_path}")
        return None

    df = pd.read_csv(data_path)
    
    important_cols = [
        'ClientID', 
        'CycleNumber', 
        'LengthofCycle', 
        'EstimatedDayofOvulation', 
        'LengthofLutealPhase'
    ]
    
    clean_df = df[important_cols].copy()
    
    clean_df = clean_df.dropna(subset=['LengthofCycle'])
    
    return clean_df

if __name__ == "__main__":
    data = load_kagggle_data()
    if data is not None:
        print("Dataset loaded successfully!")
        print(f"Total records: {len(data)}")
        print(data.head())
        