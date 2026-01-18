import streamlit as st
from main_pipeline import run_period_pal_engine
from rag import setup_rag, get_ai_response

st.set_page_config(page_title="PeriodPal AI", page_icon="🌸", layout="centered")

st.title("🌸 PeriodPal: AI Health Assistant")

if 'collection' not in st.session_state:
    with st.spinner("Initializing Clinical Intelligence..."):
        st.session_state.collection = setup_rag()

with st.sidebar:
    st.header("Daily Check-in")
    user_day = st.number_input("What day of your cycle are you on?", min_value=1, max_value=45, value=1)
    user_symptoms = st.text_area("Describe your symptoms or problems:", placeholder="e.g., I have sharp cramps and I'm feeling very tired.")
    
    analyze_btn = st.button("Generate Health Report")

if analyze_btn:
    if user_symptoms:
        with st.spinner("Analyzing your data..."):
            ml_context = run_period_pal_engine(user_day)
            
            final_report = get_ai_response(
                user_symptoms, 
                ml_context, 
                st.session_state.collection, 
                user_day
            )
            st.session_state.report = final_report
    else:
        st.error("Please describe your symptoms so the AI can help!")

if 'report' in st.session_state:
    st.markdown(st.session_state.report)