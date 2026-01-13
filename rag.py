import os
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
from transformers import pipeline, AutoModelForSeq2SeqLM, AutoTokenizer

class RAGRetrievalGenerator:
    def __init__(self, data_dir="data", embed_model_name="all-MiniLM-L6-v2", top_k=3, use_generator=True):
        self.data_dir = data_dir
        self.top_k = top_k
        self.embed_model_name = embed_model_name
        self.use_generator = use_generator

        self.passages = self._load_passages()
        self.embedder = SentenceTransformer(self.embed_model_name)
        
        self.d_emb = self.embedder.get_sentence_embedding_dimension()
        self.index = faiss.IndexFlatIP(self.d_emb)
        self._build_index()

        self.generator = None
        if self.use_generator:
            model_name = "google/flan-t5-base"
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
            self.generator = pipeline(
                "text2text-generation", 
                model=self.model, 
                tokenizer=self.tokenizer, 
                device=-1
            )

    def _load_passages(self):
        texts = []
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)
            
        for fname in os.listdir(self.data_dir):
            if fname.endswith(".txt"):
                with open(os.path.join(self.data_dir, fname), "r", encoding="utf-8") as f:
                    content = f.read().strip()
                    paras = [p.strip() for p in content.split("\n\n") if p.strip()]
                    for p in paras:
                        texts.append({"source": fname, "text": p})
        return texts

    def _build_index(self):
        texts = [p["text"] for p in self.passages]
        embeddings = self.embedder.encode(texts, convert_to_numpy=True, show_progress_bar=False)
        faiss.normalize_L2(embeddings)
        self.index.add(embeddings)

    def retrieve(self, query, k=None, min_score=0.2):
        if k is None:
            k = self.top_k
        q_emb = self.embedder.encode([query], convert_to_numpy=True)
        faiss.normalize_L2(q_emb)
        D, I = self.index.search(q_emb, k)

        results = []
        for score, idx in zip(D[0], I[0]):
            if score < min_score or idx == -1:
                continue
            results.append({
                "score": float(score),
                "text": self.passages[int(idx)]["text"],
                "source": self.passages[int(idx)]["source"]
            })
        return results

    def generate_answer(self, query, docs):
        if not docs:
            return "I'm sorry, I don't have enough information in my database to answer that fully."

        context_text = "\n".join([f"Context: {d['text']}" for d in docs])
        
        prompt = (
            f"Please provide a detailed and supportive explanation for the following question "
            f"based only on the context provided. Use multiple sentences.\n\n"
            f"{context_text}\n\n"
            f"Question: {query}\n\n"
            f"Detailed Answer:"
        )

        output = self.generator(
            prompt,
            max_length=512,
            min_length=50,
            do_sample=True,
            temperature=0.7,
            top_k=50,
            top_p=0.9,
            repetition_penalty=2.5
        )
        
        
        return output[0]["generated_text"]