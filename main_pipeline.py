import pandas as pd
from phasecalculation import get_current_phase
from clustering import cluster_symptom_patterns
from randomforest import predict_cycle_length
from isolationforest import get_anomaly_detector

def run_period_pal_engine(user_day, user_age, user_bmi, user_menses_len):
    predicted_len = predict_cycle_length(user_age, user_bmi, user_menses_len)
    
    phase = get_current_phase(user_day, predicted_len)
    
    clusters_df = cluster_symptom_patterns()
    cluster_means = clusters_df.groupby('Cluster')['LengthofCycle'].mean()
    user_cluster = (cluster_means - predicted_len).abs().idxmin()

    iso_model = get_anomaly_detector()
    user_data = pd.DataFrame([[predicted_len]], columns=['LengthofCycle'])
    is_anomaly = iso_model.predict(user_data)[0]
    
    status_label = "⚠️ UNUSUAL PATTERN" if is_anomaly == -1 else "✅ TYPICAL PATTERN"

    ml_context = f"""
    ### 🧬 Clinical Biological Profile
    * **Predicted Cycle Length:** {round(predicted_len, 1)} days
    * **Current Phase:** **{phase}**
    * **Population Match:** Cluster {user_cluster}
    * **Health Integrity Check:** {status_label}
    
    *Metrics: Age {user_age}, BMI {round(user_bmi, 1)}*
    """
    
    return ml_context, phase