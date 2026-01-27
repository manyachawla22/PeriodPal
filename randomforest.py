import os
import pandas as pd
from functools import lru_cache

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


DATA_DIR = "data"
# Use a consistent filename in README + code:
DATA_FILE = "FedCycleData071012.csv"   # <-- recommended name
DATA_PATH = os.path.join(DATA_DIR, DATA_FILE)

FEATURES = ["Age", "BMI", "LengthofMenses"]
TARGET = "LengthofCycle"


def _load_and_clean_df() -> pd.DataFrame:
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(
            f"Dataset not found at {DATA_PATH}. "
            f"Please place '{DATA_FILE}' inside the 'data/' folder (see README Dataset Setup)."
        )

    df = pd.read_csv(DATA_PATH)

    cols_needed = FEATURES + [TARGET, "LengthofLutealPhase"]
    df = df[cols_needed].apply(pd.to_numeric, errors="coerce").dropna()

    # biologically correct ovulation day
    df["OvulationDay"] = df["LengthofCycle"] - df["LengthofLutealPhase"]

    # physiological filtering (same logic you already had)
    df = df[
        (df[TARGET] >= 24) & (df[TARGET] <= 35) &
        (df["OvulationDay"] >= 12) & (df["OvulationDay"] <= 18)
    ]

    return df
import math

def _metrics(y_true, y_pred) -> dict:
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = math.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    return {"MAE": float(mae), "RMSE": float(rmse), "R2": float(r2)}


@lru_cache(maxsize=1)
def get_model_and_report(test_size: float = 0.2, val_size: float = 0.2, seed: int = 42):
    """
    Splits: Train / Val / Test
    - test_size = 0.2 means 20% test.
    - val_size is applied on the remaining 80%. So val becomes 16% overall if val_size=0.2.
    """
    df = _load_and_clean_df()

    X = df[FEATURES]
    y = df[TARGET]

    # 1) Train+Val vs Test
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed
    )

    # 2) Train vs Val (split from trainval)
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval, test_size=val_size, random_state=seed
    )

    model = RandomForestRegressor(
        n_estimators=300,
        max_depth=10,
        random_state=seed
    )
    model.fit(X_train, y_train)

    val_pred = model.predict(X_val)
    test_pred = model.predict(X_test)

    report = {
        "rows_total": int(len(df)),
        "rows_train": int(len(X_train)),
        "rows_val": int(len(X_val)),
        "rows_test": int(len(X_test)),
        "val_metrics": _metrics(y_val, val_pred),
        "test_metrics": _metrics(y_test, test_pred),
        "feature_importance": dict(zip(FEATURES, map(float, model.feature_importances_))),
    }
    return model, report


def predict_cycle_length(user_age: float, user_bmi: float, user_menses_len: float) -> float:
    model, _ = get_model_and_report()
    user_input = pd.DataFrame([[user_age, user_bmi, user_menses_len]], columns=FEATURES)
    return float(model.predict(user_input)[0])
