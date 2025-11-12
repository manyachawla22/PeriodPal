
# Period Pal — Minimal RAG Chatbot (Student Project)

This is a lightweight Retrieval-Augmented Generation (RAG) demo for menstrual health Q&A.
It is intended as a **GitHub-ready** student project you can present on your resume and demo in interviews.

## What's included
- `main.py` — Streamlit app to ask questions and see retrieved passages + generated answer
- `rag.py` — simple RAG backend (SentenceTransformers + FAISS + HuggingFace generator)
- `data/docs.txt` — small sample knowledge base (you should expand this)
- `requirements.txt` — Python dependencies

## How to run (recommended)
1. Create a Python environment (Python 3.8+).
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the Streamlit app:
   ```bash
   streamlit run main.py
   ```

The app will build embeddings for the documents and may download model weights (embedding & generator models) the first time it runs.

## Notes for interviews
- This is a **demo**: it uses small, local models (e.g. `gpt2`) for generation. For production you would use a stronger LLM (or hosted API).
- Add more domain-specific documents to `data/` to improve the quality of answers.
- You can explain the pipeline in interviews as: *embed user query → retrieve similar passages → condition LLM on retrieved passages → return grounded answer*.

---
Generated at 2025-11-12T16:50:49.241277 UTC
