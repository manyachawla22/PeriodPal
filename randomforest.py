import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import os

def predict_cycle_length(user_age, user_bmi, user_menses_len):
    data_path = os.path.join('data', 'FedCycleData071012 (2).csv')
    df = pd.read_csv(data_path)

    features = ['Age', 'BMI', 'LengthofMenses']
    target = 'LengthofCycle'

    cols_needed = features + [target, 'LengthofLutealPhase']
    df_clean = df[cols_needed].apply(pd.to_numeric, errors='coerce').dropna()

    # biologically correct ovulation day
    df_clean['OvulationDay'] = (
        df_clean['LengthofCycle'] - df_clean['LengthofLutealPhase']
    )

    # physiological filtering
    df_clean = df_clean[
        (df_clean[target] >= 24) & (df_clean[target] <= 35) &
        (df_clean['OvulationDay'] >= 12) & (df_clean['OvulationDay'] <= 18)
    ]

    X = df_clean[features]
    y = df_clean[target]

    model = RandomForestRegressor(
        n_estimators=300,
        max_depth=10,
        random_state=42
    )
    model.fit(X, y)

    user_input = pd.DataFrame(
        [[user_age, user_bmi, user_menses_len]],
        columns=features
    )

    prediction = model.predict(user_input)
    return float(prediction[0])
