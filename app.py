import streamlit as st
from main_pipeline import run_period_pal_engine
from rag import setup_rag, get_ai_response

st.set_page_config(page_title="PeriodPal AI", page_icon="🌸", layout="centered")

st.markdown("""
    <style>
    .main { background-color: #fff5f7; }
    .stButton>button { width: 100%; border-radius: 20px; background-color: #ff4b6b; color: white; }
    </style>
    """, unsafe_allow_html=True)

st.title("🌸 PeriodPal: AI Clinical Consultant")

if 'collection' not in st.session_state:
    with st.spinner("Loading Clinical Knowledge Base..."):
        st.session_state.collection = setup_rag()

with st.sidebar:
    st.header("📋 Biological Profile")
    user_age = st.number_input("Age", min_value=13, max_value=55, value=25)
    
    c1, c2 = st.columns(2)
    weight = c1.number_input("Weight (kg)", 30.0, 150.0, 60.0)
    height = c2.number_input("Height (cm)", 100.0, 220.0, 165.0)
    
    user_bmi = weight / ((height/100)**2)
    st.info(f"Calculated BMI: {round(user_bmi, 2)}")
    
    user_menses = st.slider("Typical Menses Length (Days)", 1, 10, 5)
    
    st.divider()
    st.header("🔄 Current Status")
    user_day = st.number_input("Current Day of Cycle", 1, 45, 1)
    
    analyze_btn = st.button("Generate Health Profile")

st.subheader("🔍 Ask a Clinical Question")
user_query = st.text_input("Example: What causes irregular periods?", placeholder="Type your question here...")

if analyze_btn:
    with st.spinner("Processing Clinical Models..."):
        ml_context, phase = run_period_pal_engine(user_day, user_age, user_bmi, user_menses)
        
        if user_query:
            final_report = get_ai_response(user_query, ml_context, st.session_state.collection, phase)
            st.session_state.report = final_report
        else:
            st.session_state.report = ml_context + "\n\n*Enter a question above to see related clinical guidance.*"

if 'report' in st.session_state:
    st.markdown(st.session_state.report)

from randomforest import get_model_and_report

_, report = get_model_and_report()

with st.expander("📊 Model Evaluation (Train/Val/Test)"):
    st.write("Rows:", {
        "total": report["rows_total"],
        "train": report["rows_train"],
        "val": report["rows_val"],
        "test": report["rows_test"],
    })
    st.write("Validation metrics:", report["val_metrics"])
    st.write("Test metrics:", report["test_metrics"])
    st.write("Feature importance:", report["feature_importance"])