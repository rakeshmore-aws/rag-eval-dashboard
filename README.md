# 📊 RAG Evaluation Lab  

A Streamlit-based Proof of Concept (POC) to experiment with **Normal RAG**, **Self-RAG**, and **Agentic RAG**, and evaluate them using factual accuracy, retrieval metrics, hallucination rate, reasoning score, and latency.  

This project demonstrates:  
- **Normal RAG**: Retrieval-Augmented Generation with direct context injection.  
- **Self-RAG**: Adds self-reflection and answer refinement.  
- **Agentic RAG**: Uses reasoning agents with retrieval tools.  
- **Evaluation**: Automatic computation of  
  - Factual Accuracy (F1)  
  - Retrieval Precision / Recall  
  - Hallucination Rate  
  - Multi-hop Reasoning Score  
  - End-to-End Latency  

---

## 🚀 Features
- Upload your **knowledge base (TXT files)**.  
- Switch between **Normal RAG, Self-RAG, Agentic RAG**.  
- Upload an evaluation dataset (`rag_eval_dataset.csv`).  
- Auto-compute evaluation metrics for all queries.  
- Download results as CSV.  

---

## 📂 Repository Structure
```
├── RAG_POC.py                # Streamlit app (Normal RAG, Self-RAG, Agentic RAG + evaluation)
├── rag_eval_dataset.csv  # Sample dataset with 100 queries
├── requirements.txt      # Python dependencies
└── README.md             # Documentation
```

---

## ⚙️ Installation & Setup

### 1. Clone the repo
```bash
git clone https://github.com/<your-username>/rag-eval-lab.git
cd rag-eval-lab
```

### 2. Create virtual environment (optional but recommended)
```bash
python -m venv venv
source venv/bin/activate   # On Mac/Linux
venv\Scripts\activate      # On Windows
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Set up OpenAI API Key
Get your API key from [OpenAI Dashboard](https://platform.openai.com/account/api-keys) and set it as an environment variable:

```bash
export OPENAI_API_KEY="your_api_key_here"     # Mac/Linux
setx OPENAI_API_KEY "your_api_key_here"       # Windows
```

Alternatively, you can enter the key directly in the **Streamlit sidebar**.

---

## ▶️ Running the App

Start the Streamlit app:

```bash
streamlit run RAG_POC.py
```

Open the link shown in the terminal (default: http://localhost:8501).

---

## 📊 Evaluation Workflow

1. Upload **knowledge base files** (`.txt`).  
2. Choose a **RAG Mode** from sidebar:  
   - Normal RAG  
   - Self-RAG  
   - Agentic RAG  
3. Upload **dataset CSV** (`rag_eval_dataset.csv`) with the following format:  

```csv
query,ground_truth,relevant_docs
"Who wrote the novel 1984?","George Orwell","doc1"
"What company owns Instagram?","Meta","doc2"
...
```

4. View computed metrics in the app:  
   - Factual Accuracy (F1)  
   - Retrieval Precision (%)  
   - Retrieval Recall (%)  
   - Hallucination Rate (%)  
   - Multi-hop Reasoning Score  
   - End-to-End Latency (ms)  

5. Download the **results as CSV** for offline analysis.  

---

## 📦 Requirements

See `requirements.txt`:

```
streamlit
pandas
langchain
langchain-community
langchain-openai
faiss-cpu
openai
```

Install with:
```bash
pip install -r requirements.txt
```

---

## 🧪 Example Run

```bash
streamlit run RAG_POC.py
```

- Upload `rag_eval_dataset.csv`.  
- Select **Self-RAG**.  
- View the evaluation dashboard with all metrics.  

---

## 📌 Next Steps
- Improve hallucination detection (semantic similarity instead of keyword match).  
- Add multi-hop benchmark dataset.  
- Support more LLM backends (Anthropic, Llama, etc.).  
