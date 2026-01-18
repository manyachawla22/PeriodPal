import os
import chromadb
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('all-MiniLM-L6-v2')

def setup_rag():
    client = chromadb.Client()
    collection = client.get_or_create_collection(name="period_pal_final")
    
    master_file = os.path.join('data', 'master_knowledge.txt')
    
    if os.path.exists(master_file):
        with open(master_file, 'r', encoding='utf-8') as f:
            content = [block.strip() for block in f.read().split("---") if block.strip()]
            if content:
                collection.add(
                    documents=content,
                    ids=[f"id_{i}" for i in range(len(content))]
                )
    return collection

def get_ai_response(user_query, ml_context, collection, user_day):
    results = collection.query(query_texts=[user_query], n_results=1)
    
    if not results['documents'][0]:
        clinical_data = "No specific clinical match found, but following general wellness guidelines."
    else:
        clinical_data = results['documents'][0][0]

    clean_insight = clinical_data.replace("Scenario:", "**Condition:**").replace("Advice:", "**Expert Guidance:**")

    report = f"""
    {ml_context}
    
    ---
    ### 🩺 Clinical Insight: "{user_query}"
    {clean_insight}
    
    ---
    *Disclaimer: This analysis is based on clinical datasets and is for informational purposes only.*
    """
    return report