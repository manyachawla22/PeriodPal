import pandas as pd
from prophet import Prophet
import os
from datetime import datetime, timedelta

def run_cycle_prediction():
    data_path = os.path.join('data', 'FedCycleData071012 (2).csv')
    
    if not os.path.exists(data_path):
        print(f"Error: Could not find {data_path}")
        return
    
    raw_df = pd.read_csv(data_path)
    
    df = raw_df[['LengthofCycle']].copy()
    df = df.dropna()
    
    start_date = datetime(2023, 1, 1)
    dates = []
    current_date = start_date
    
    for length in df['LengthofCycle']:
        dates.append(current_date)
        current_date += timedelta(days=int(length))
        
    df['ds'] = dates
    df = df.rename(columns={'LengthofCycle': 'y'})
    
    model = Prophet(yearly_seasonality=True, daily_seasonality=False, weekly_seasonality=False)
    model.fit(df)
    
    future = model.make_future_dataframe(periods=1, freq='ME')
    forecast = model.predict(future)
    
    prediction = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(1)
    
    print("\n--- Next Cycle Prediction ---")
    print(f"Predicted Date: {prediction['ds'].iloc[0].date()}")
    print(f"Predicted Length: {round(prediction['yhat'].iloc[0], 1)} days")
    print(f"Range: {round(prediction['yhat_lower'].iloc[0], 1)} to {round(prediction['yhat_upper'].iloc[0], 1)} days")
    
    return model, forecast

if __name__ == "__main__":
    run_cycle_prediction()