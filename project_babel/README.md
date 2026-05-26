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
pip install gradio python-dotenv

# Set API key - register free account in https://console.groq.com/login and get API key
# Create a .env file in the project_babel folder with:
# GROQ_API_KEY=your-api-key-here
# PYTHONIOENCODING=utf-8
```

## Running the App

Make sure you have a `.env` file set up (see Setup above), then:

```bash
python app.py
```

The app will take 1-2 minutes to load and index all documents, then open at `http://127.0.0.1:7860`.

### Sharing the App

The app is configured with `share=True`, so Gradio will also print a public URL like:

```
https://abc123.gradio.live
```

You can send this link to anyone — it works as long as your machine is running the app. The link is valid for 72 hours and changes each time you restart the app.

### Deploying for a Permanent URL

To have the app always available without running it on your machine, you need to host it on a server. Options:

| Option | Cost | Difficulty |
|--------|------|------------|
| Hugging Face Spaces | Free | Easy — connect GitHub repo, set API key as secret |
| Render | Free tier | Easy — connect GitHub repo |
| Railway | ~$5/month | Easy — connect GitHub repo |
| DigitalOcean / AWS | ~$6+/month | Medium — manage your own server |

For a custom domain (e.g. `navihealth.ca`), you additionally need to buy a domain (~$10-15/year from Namecheap or Google Domains) and point its DNS to your server's IP.

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
