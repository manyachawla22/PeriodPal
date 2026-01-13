import streamlit as st
from rag import RAGRetrievalGenerator

st.set_page_config(page_title="Period Pal — RAG Chatbot", layout="centered")

st.title("Period Pal — (Demo RAG Chatbot)")
st.markdown("""
This demo shows a simple Retrieval-Augmented Generation pipeline for menstrual health Q&A.
It retrieves relevant passages from a small local knowledge base and uses a generator model to produce an answer.
""")


st.sidebar.header("Settings")
TOP_K = st.sidebar.slider("Number of documents to retrieve (k)", 1, 5, 3)
use_generator = st.sidebar.checkbox("Use generator model (may be slow on CPU)", value=True)
st.sidebar.write("Note: first run may download model weights (~100-500MB).")

# Load or build index (cached inside RAG object)
@st.cache_resource
def get_rag():
    rag = RAGRetrievalGenerator(data_dir="data", top_k=TOP_K, use_generator=use_generator)
    return rag

rag = get_rag()

st.write("---")
query = st.text_input("Ask Period Pal a question", value="Why do I get cramps before my period?")

if st.button("Ask"):
    if not query.strip():
        st.warning("Please enter a question.")
    else:
        with st.spinner("Retrieving relevant documents..."):
            docs = rag.retrieve(query, k=TOP_K)
        st.subheader("Retrieved passages")
        for i, d in enumerate(docs):
            st.markdown(f"**Doc {i+1} (score {d['score']:.3f})** — {d['source']}")
            st.write(d['text'])

        if use_generator:
            with st.spinner("Generating answer..."):
                answer = rag.generate_answer(query, docs)
            st.subheader("Period Pal's answer")
            st.write(answer)
        else:
            st.info("Generator disabled — showing retrieved passages only. Turn on 'Use generator model' in the sidebar to synthesize an answer.")
