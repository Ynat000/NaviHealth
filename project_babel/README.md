# NaviHealth POC - Project Babel

Multilingual healthcare triage assistant for British Columbia. Helps users determine whether to visit UPCC, Walk-in Clinic, or Emergency Department.

## Tech Stack

- **LLM**: Qwen QWQ via Groq; backup: Llama 3.3 70B
- **Embeddings**: paraphrase-multilingual-MiniLM-L12-v2
- **RAG Framework**: LlamaIndex
- **Vector Store**: In-memory for POC, pgvector for production

## Setup

Please use **Python 3.12.0** as some dependencies are not yet compatible with more recent versions.

```bash
# Create virtual environment
python -m venv venv
.\venv\Scripts\Activate.ps1     # Windows
source venv/bin/activate        # macOS/Linux

# Install dependencies
pip install -r requirements.txt
# or: pip install -r requirements-simple.txt if this doesn't work

# Set API key - register free account in https://console.groq.com/login and get API key
$env:GROQ_API_KEY = "<your-API-key>"  # Windows
export GROQ_API_KEY="<your-API-key>"  # macOS/Linux
```

## Project Structure

```
project_babel
├── data/                           # BC health policy documents, scraped from data pipeline
├── sample_data/                    # Sample BC health policy documents, used for POC
├── scripts/
│   ├── test_embedding.py           # Step 1: Verify multilingual embeddings
│   ├── test_retrieval.py           # Step 2: Verify cross-lingual retrieval
│   ├── test_retrieval_chunked.py   # Step 2 alt.: Verify cross-lingual retrieval
│   ├── test_llm.py                 # Step 3: Verify Groq API
│   └── test_rag_pipeline.py        # Step 4: Full RAG pipeline
├── requirements.txt
└── README.md
```

## Running Tests

```bash
# Step 1: Test embedding model
python scripts/test_embedding.py

# Step 2: Test retrieval
python scripts/test_retrieval.py
# Use `test_retrieval_chunked.py` for better results in Step 2, but Step 4 handles chunking by default)

# Step 3: Test LLM
python scripts/test_llm.py

# Step 4: Full pipeline
python scripts/test_rag_pipeline.py
```
