import numpy as np
import pandas as pd

from sklearn.cluster import KMeans

import os



def cluster_symptom_patterns():

    data_path = os.path.join('data', 'FedCycleData071012 (2).csv')

   

    if not os.path.exists(data_path):

        print("File not found.")

        return



    df = pd.read_csv(data_path)

   

    features = ['LengthofCycle', 'EstimatedDayofOvulation', 'LengthofLutealPhase']

   

    subset = df[features].copy()

   

    subset = subset.replace(r'^\s*$', np.nan, regex=True)

   

    subset = subset.apply(pd.to_numeric, errors='coerce')

   

    subset = subset.dropna()

   

    if subset.empty:

        print("Error: No data left after cleaning. Check column names.")

        return



    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)

    subset['Cluster'] = kmeans.fit_predict(subset)

   

    print("--- Symptom & Cycle Clustering ---")

    for i in range(3):

        cluster_group = subset[subset['Cluster'] == i]

        print(f"\nCluster {i} (Group Size: {len(cluster_group)})")

        print(f"Avg Cycle Length: {round(cluster_group['LengthofCycle'].mean(), 1)} days")

        print(f"Avg Ovulation Day: {round(cluster_group['EstimatedDayofOvulation'].mean(), 1)}")

   

    return subset



if __name__ == "__main__":

    cluster_symptom_patterns()