# 🌸 PeriodPal: AI-Powered Menstrual Health Assistant

PeriodPal is an advanced health tracking application that combines **Machine Learning (Clustering & Prophet)** with **RAG (Retrieval-Augmented Generation)** to provide personalized cycle predictions and clinical symptom analysis.

## 🚀 Features
- **Cycle Prediction:** Uses Facebook Prophet for time-series forecasting of next period dates.
- **User Clustering:** Groups users into biological clusters using K-Means to identify health patterns.
- **Medical RAG:** A custom Retrieval-Augmented Generation engine using **ChromaDB** and **Sentence Transformers** to search clinical datasets (Training/Testing Data) for symptom advice.
- **Interactive Dashboard:** Built with Streamlit for a seamless user experience.

## 📂 Project Structure
- `app.py`: The main Streamlit interface.
- `main_pipeline.py`: The core engine connecting ML models.
- `rag.py`: Logic for medical document retrieval and AI responses.
- `clustering.py` & `prophettraining.py`: Machine Learning modules for cycle analysis.
- `convert_dataset.py`: Utility to process Kaggle CSV datasets into searchable vector data.
- `data/`: Contains clinical Q&A datasets.

## 🛠️ Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/manyachawla22/PeriodPal.git](https://github.com/manyachawla22/PeriodPal.git)
   cd PeriodPal
   Install Dependencies:

Bash

pip install streamlit pandas chromadb sentence-transformers prophet scikit-learn
Prepare the Knowledge Base: Run the converter to process the CSV data:

Bash

python convert_dataset.py
Run the App:

Bash

python -m streamlit run app.py
📊 Data Source
This project utilizes clinical Q&A datasets (Instruction/Output format) focused on menstrual health, hygiene, and reproductive disorders.

Disclaimer: PeriodPal is an AI educational tool and does not provide professional medical diagnoses.
