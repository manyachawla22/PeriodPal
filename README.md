# 🌸 PeriodPal: AI Clinical Consultant

**PeriodPal** is an AI-powered, clinical-grade *educational* health assistant that combines **Multivariate Machine Learning** with **Retrieval-Augmented Generation (RAG)** to deliver personalized, biologically grounded menstrual-cycle insights.

> ⚠️ **Educational Use Only** — PeriodPal does **not** replace professional medical care.

---

## 🚀 Key Features

### 🧮 Multivariate Cycle Prediction

Predicts menstrual cycle length using a **Random Forest Regressor** trained on:

* Age
* Body Mass Index (BMI)
* Menses duration

### 🧬 Population Clustering (Biological Archetypes)

Identifies a user’s *biological archetype* using **K-Means clustering** over population-level cycle features.

### 🚨 Anomaly Detection

Detects irregular or atypical cycle patterns with **Isolation Forest**, flagging potential deviations from population norms.

### 📚 Intelligent FAQ Retrieval (RAG)

A custom **RAG pipeline** powered by:

* **Sentence Transformers** for semantic embeddings
* **ChromaDB** for vector similarity search

Provides context-aware, evidence-grounded answers to menstrual health FAQs.

---

## 🛠️ Tech Stack

**Frontend**

* Streamlit

**Machine Learning**

  * Scikit-learn
  * Random Forest Regressor
  * K-Means Clustering
  * Isolation Forest

**Retrieval & NLP**

* ChromaDB (Vector Database)
* Sentence-Transformers (`all-MiniLM-L6-v2`)

**Data Handling**

* Pandas
* NumPy

---

## 📂 Project Structure

```text
├── app.py                 # Streamlit UI
├── main_pipeline.py       # Orchestrates ML + RAG pipeline
├── randomforest.py        # Cycle length prediction model
├── clustering.py          # Biological archetype clustering
├── isolationforest.py     # Anomaly detection
├── phasecalculation.py    # Cycle phase & ovulation logic
├── rag.py                 # RAG pipeline (ChromaDB + embeddings)
└── data/
    ├── FedCycleData071012.csv   # Population cycle dataset
    └── master_knowledge.txt # Clinical FAQ knowledge base
```

---

## ⚙️ Installation & Usage

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/manyachawla22/PeriodPal.git
cd PeriodPal
```

### 2️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```
## 📦 Dataset Setup (Required)

This project uses the **FedCycleData071012** dataset (CSV). The dataset is not committed to this repo.

1) Download the dataset (commonly provided as `FedCycleData071012 (2).csv`) from a public dataset source (example: Kaggle notebooks reference this filename).  
2) Create a folder named `data/` in the project root (if not present).
3) Place the CSV inside `data/` and rename it exactly to:

`FedCycleData071012.csv`

So the path becomes:
`data/FedCycleData071012.csv`


### 3️⃣ Run the Application

```bash
streamlit run app.py
```

---

## ⚠️ Medical Disclaimer

**PeriodPal is an AI-driven educational tool and is NOT a substitute for professional medical advice, diagnosis, or treatment.**

* **Informational Purposes Only**
  All insights, cycle predictions, and FAQ responses are derived from population-level data (FedCycleData) and are intended solely for educational use.

* **Prediction Uncertainty**
  Machine learning models provide *probabilistic estimates*. Individual biological variation may differ substantially from model outputs.

* **Consult a Professional**
  Always seek advice from a qualified healthcare provider regarding medical conditions. Never disregard professional medical guidance because of information from this application.

---

## 📌 Dataset Attribution

This project uses the following publicly available datasets for **research and educational purposes only**:

* **FedCycleData** — A large-scale menstrual cycle dataset used for population-level cycle analysis and modeling.
* **Menstrual Health Awareness Dataset** — A curated dataset containing educational menstrual health FAQs and awareness-related content.

Both datasets are sourced from **Kaggle** and are used strictly for non-commercial, educational, and research objectives.

---

## 💡 Author

**Manya Chawla**
Engineering Student, Delhi Technological University (DTU)

---

⭐ If you find this project useful, consider giving it a star!
