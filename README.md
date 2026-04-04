# RAG-Based Sales Data Analysis

A Retrieval-Augmented Generation (RAG) system for analysing a real-world sales dataset. The app stores structured sales data in a vector database, retrieves the most relevant documents for each user query, and passes them to an LLM to answer analytical questions about sales trends, patterns, and insights.

## Architecture

```
CSV data
   │
   ▼
generate_documents.py   ← converts rows and aggregates into plain-text documents
   │
   ▼
build_index.py          ← embeds documents with sentence-transformers (all-MiniLM-L6-v2)
   │                       and stores them in a persistent ChromaDB collection
   ▼
ChromaDB (chroma_db/)
   │
   ▼
retrieval.py            ← similarity search with metadata filtering: finds the top-k
   │                       most relevant documents, optionally narrowed by metadata
   │                       fields (e.g. category, region, document type)
   │
   ▼
pipeline.py             ← builds a prompt (context + chat history) and calls the LLM
   │                       (Mistral via Ollama); keeps the last 3 Q&A turns
   ▼
ui.py                   ← Streamlit front-end; renders questions, answers and sources
```

**Key technology choices:**

| Component | Library / Tool |
|---|---|
| Vector store | ChromaDB (persistent, local) |
| Embeddings | `sentence-transformers` — `all-MiniLM-L6-v2` |
| LLM | Mistral (served locally via Ollama) |
| LLM integration | LangChain Ollama |
| Front-end | Streamlit |

## Dataset

Used dataset is [Superstore Dataset](https://www.kaggle.com/datasets/vivek468/superstore-dataset-final/data). It contains information related to sales, profits and other facts from a large retail superstore.

## Getting Started

### Prerequisites

- Python 3.12+
- [Ollama](https://ollama.com) installed and running with the Mistral model pulled:

  ```
  ollama pull mistral
  ```

### Installation

1. Clone the repository and navigate to it:

   ```bash
   git clone <repo-url>
   cd RAG
   ```

2. Create and activate a virtual environment:

   ```bash
   python -m venv .venv
   .venv\Scripts\activate   # Windows
   source .venv/bin/activate # macOS / Linux
   ```

3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

### Build the vector index

Run this once (or after updating the dataset) to embed the data and populate ChromaDB:

```bash
python -c "import pandas as pd; from src.build_index import build_index; build_index(pd.read_csv('data/cleaned_superstore.csv'))"
```

### Start the app

```bash
python -m streamlit run src/ui.py
```

The app opens in your browser at `http://localhost:8501`. Type a question in the text box to query the sales data.

### Run tests

```bash
python -m pytest tests/ -v
```
