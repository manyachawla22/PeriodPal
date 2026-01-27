import pandas as pd
import numpy as np
from phasecalculation import get_current_phase
from clustering import cluster_symptom_patterns
from randomforest import predict_cycle_length
from isolationforest import get_anomaly_detector

def run_period_pal_engine(user_day, user_age, user_bmi, user_menses_len):
    predicted_len = predict_cycle_length(user_age, user_bmi, user_menses_len)
    phase = get_current_phase(user_day, predicted_len)
    
    clusters_df = cluster_symptom_patterns()
    cluster_means = clusters_df.groupby('Cluster')['LengthofCycle'].mean()
    user_cluster_id = (cluster_means - predicted_len).abs().idxmin()

    personas = {
        0: {"name": "The Consistent Core", "desc": "Your patterns match the most common population group with stable, textbook cycle lengths."},
        1: {"name": "The High-Metabolic Profile", "desc": "Your data aligns with a cluster that typically shows higher efficiency in hormonal transitions."},
        2: {"name": "The Variable-Sensitive Group", "desc": "You match a group that often experiences slight fluctuations; tracking is key for you."}
    }
    
    user_persona = personas.get(user_cluster_id, {"name": f"Cluster {user_cluster_id}", "desc": "Comparing your data to similar biological profiles."})

    iso_model = get_anomaly_detector()
    user_data = pd.DataFrame(
    [[predicted_len, user_age, user_bmi]],
    columns=['LengthofCycle', 'Age', 'BMI'])

    
    anomaly_score = iso_model.decision_function(user_data)[0]
    is_anomaly = iso_model.predict(user_data)[0]

    if is_anomaly == -1 or anomaly_score < 0.05:
        status_label = "⚠️ UNUSUAL PATTERN"
        status_color = "red"
    else:
        status_label = "✅ TYPICAL PATTERN"
        status_color = "green"

    ml_context = f"""
    ### 🧬 Clinical Biological Profile
    * **Predicted Cycle Length:** {round(predicted_len, 1)} days
    * **Current Phase:** **{phase}**
    * **Health Integrity Check:** {status_label} (Score: {round(anomaly_score, 3)})
    
    ---
    ### 👥 Population Match: {user_persona['name']}
    > {user_persona['desc']}
    
    *Internal Data: Age {user_age}, BMI {round(user_bmi, 1)}*
    """
    
    return ml_context, phase