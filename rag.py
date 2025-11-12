import os
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer

class RAGRetrievalGenerator:
    def __init__(self, data_dir="data", embed_model_name="all-MiniLM-L6-v2", top_k=3, use_generator=True):
        self.data_dir = data_dir
        self.top_k = top_k
        self.embed_model_name = embed_model_name
        self.use_generator = use_generator

        # load passages
        self.passages = self._load_passages()
        # load embedder
        self.embedder = SentenceTransformer(self.embed_model_name)
        # build or load faiss index
        self.d_emb = self.embedder.get_sentence_embedding_dimension()
        self.index = faiss.IndexFlatIP(self.d_emb)
        self._build_index()

        # generator (optional)
        self.generator = None
        if self.use_generator:
            try:
                # small model for demo (can be replaced by a larger model)
                model_name = "gpt2"
                self.tokenizer = AutoTokenizer.from_pretrained(model_name)
                self.model = AutoModelForCausalLM.from_pretrained(model_name)
                self.generator = pipeline("text-generation", model=self.model, tokenizer=self.tokenizer, device=-1)
            except Exception as e:
                print("Warning: failed to load generator model locally:", e)
                self.generator = None

    def _load_passages(self):
        # combine all .txt files in data/ into passages (split by double newline)
        texts = []
        for fname in os.listdir(self.data_dir):
            if fname.endswith(".txt"):
                with open(os.path.join(self.data_dir, fname), "r", encoding="utf-8") as f:
                    content = f.read().strip()
                paras = [p.strip() for p in content.split("\n\n") if p.strip()]
                for i,p in enumerate(paras):
                    texts.append({"source": fname, "text": p})
        if not texts:
            raise ValueError("No .txt docs found in data/ — add docs to data/ and retry.")
        return texts

    def _build_index(self):
        # embed all passages and add to faiss
        texts = [p["text"] for p in self.passages]
        embeddings = self.embedder.encode(texts, convert_to_numpy=True, show_progress_bar=False)
        # normalize for cosine similarity with inner product
        faiss.normalize_L2(embeddings)
        self.index.add(embeddings)
        self.embeddings = embeddings

    def retrieve(self, query, k=None):
        if k is None:
            k = self.top_k
        q_emb = self.embedder.encode([query], convert_to_numpy=True)
        faiss.normalize_L2(q_emb)
        D, I = self.index.search(q_emb, k)
        results = []
        for score, idx in zip(D[0], I[0]):
            results.append({
                "score": float(score),
                "text": self.passages[int(idx)]["text"],
                "source": self.passages[int(idx)]["source"]
            })
        return results

    def generate_answer(self, query, docs):
        # build a prompt combining retrieved docs and the question
        context = "\n\n".join([f"Source ({d['source']}): {d['text']}" for d in docs])
        prompt = f"""You are an assistant specialized in menstrual health. Use the provided sources to answer the user's question accurately and in a friendly tone.

{context}

User question: {query}

Answer:"""
        if self.generator is None:
            # fallback: return the concatenated context as a simple answer
            return "I couldn't load a local generator model. Here are the retrieved passages:\n\n" + context
        gen = self.generator(prompt, max_length=200, do_sample=True, temperature=0.7, top_p=0.95, num_return_sequences=1)
        return gen[0]["generated_text"][len(prompt):].strip()
