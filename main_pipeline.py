from prophettraining import run_cycle_prediction
from isolationforest import detect_anomalies
from clustering import cluster_symptom_patterns
from phasecalculation import get_current_phase
import pandas as pd


def run_period_pal_engine(user_day):
   
    
    if 1 <= user_day <= 5:
        phase = "Menstrual Phase"
        status = "Your body is shedding the uterine lining. Energy may be low."
    elif 6 <= user_day <= 13:
        phase = "Follicular Phase"
        status = "Estrogen is rising. You likely feel an increase in energy and focus."
    elif 14 <= user_day <= 16:
        phase = "Ovulatory Phase"
        status = "You are at peak fertility. You might notice a slight rise in body temperature."
    elif 17 <= user_day <= 28:
        phase = "Luteal Phase"
        status = "Progesterone is dominant. This is when PMS symptoms (bloating, mood shifts) usually occur."
    else:
        phase = "Extended Cycle"
        status = "Your cycle is longer than the average 28 days. This can be normal depending on your history."

    # This creates the context that we send to the RAG system
    ml_report = f"""
    PHASE ANALYSIS:
    - Current Day: {user_day}
    - Phase: {phase}
    - Physiological Status: {status}
    - Prediction: Based on clinical patterns, your next cycle is expected in approximately {max(0, 28 - user_day)} days.
    """
    
    return ml_report