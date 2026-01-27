import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
import os

def cluster_symptom_patterns(data_path=None):
    if data_path is None:
        data_path = os.path.join("data", "FedCycleData071012.csv")

    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset not found at {data_path}")

    df = pd.read_csv(data_path)

    features = [
        "LengthofCycle",
        "EstimatedDayofOvulation",
        "LengthofLutealPhase"
    ]

    subset = df[features].copy()

    subset = subset.replace(r"^\s*$", np.nan, regex=True)
    subset = subset.apply(pd.to_numeric, errors="coerce")
    subset = subset.dropna()

    if subset.empty:
        raise ValueError("No data left after cleaning. Check column names or dataset.")

    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    subset["Cluster"] = kmeans.fit_predict(subset)

    return subset
