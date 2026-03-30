import os
from pathlib import Path

# check API key
api_key = os.environ.get('GROQ_API_KEY')
if not api_key:
    print("ERROR: GROQ_API_KEY not set!")
    exit(1)

# DATA_DIR = "./sample_data"
DATA_DIR = "../hlbc_data_pipeline/artifacts/hlbc_files_md"  # true data from pipeline

print("Initializing RAG pipeline...")

from llama_index.core import (
    SimpleDirectoryReader,
    VectorStoreIndex,
    Settings,
    PromptTemplate,
)
from llama_index.llms.groq import Groq
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

# configure global configs
print("  Loading embedding model...")
Settings.embed_model = HuggingFaceEmbedding(
    model_name="paraphrase-multilingual-MiniLM-L12-v2"
)
print("  Embedding model embedded.\n")

print("  Configuring LLM...")
Settings.llm = Groq(
    model="qwen/qwen3-32b",  # switch models here if needed
    api_key=api_key
)
print("  LLM configured.\n")

# load documents
print("  Loading documents ...")
documents = SimpleDirectoryReader(DATA_DIR).load_data()
print(f"  Loaded {len(documents)} documents\n")

# create vector index
print("  Creating vector index...")
index = VectorStoreIndex.from_documents(documents)
print("  Index created.\n")

# design prompt template
QA_PROMPT_TMPL = """\
You are NaviHealth, a healthcare triage assistant for British Columbia. 
Please answer the user's questions based on the following policy documents.

Rules:
Respond in the same language used by the user.
(For example, answer in Chinese for Chinese questions, English for English, and French for French. 
However, please use English for your Chain of Thought for demoing purposes.

Answers must be based strictly on the provided policy documents; 
Do not hallucinate or invent information.

Clearly state the basis of your recommendation within the response.

If you are unable to determine the appropriate course of action, 
advise the user to call 811 for consultation.

In the event of an emergency (e.g., chest pain, difficulty breathing, severe bleeding), 
immediately advise the user to call 911.

Policy documents:
-----
{context_str}
-----

User Question: {query_str}

Answer:"""

qa_prompt = PromptTemplate(QA_PROMPT_TMPL)

# create query engine
query_engine = index.as_query_engine(
    text_qa_template=qa_prompt,
    similarity_top_k=3,  # query top three documents
)

print("\nRAG Pipeline ready!")
print("="*60)

# 测试用例
test_queries = [
    "我走路的时候腿很痛，应该去哪里看？",
    "I have a minor cut that might need stitches, where should I go?",
    "我胸口很痛，呼吸困难",
    "J'ai un peu de fièvre et je tousse, où dois-je aller?",
    "我不知道应该去哪里，感觉有点不舒服但不严重",
    "Pergelangan kaki saya terkilir",
    "Aku tidak bisa berhenti batuk"
]

for query in test_queries:
    print(f"\n🔍 Query: {query}")
    print("-" * 40)
    
    try:
        response = query_engine.query(query)
        print(f"Response:\n{response.response}")
        
        # show sources queried
        print(f"\nSources:")
        for node in response.source_nodes:
            # try to extract file name
            filename = node.metadata.get('file_name', 'unknown')
            score = node.score if hasattr(node, 'score') else 'N/A'
            print(f"  - {filename} (score: {score})")
            
    except Exception as e:
        print(f"ERROR: {e}")
    
    print("=" * 60)

# interactive mode
print("\n" + "="*60)
print("INTERACTIVE MODE")
print("Type your health question (or 'quit' to exit)")
print("="*60)

while True:
    try:
        query = input("\n> ").strip()
        if query.lower() in ['quit', 'exit', 'q']:
            break
        if not query:
            continue
            
        response = query_engine.query(query)
        print(f"\n{response.response}")
        
    except KeyboardInterrupt:
        break
    except Exception as e:
        print(f"Error: {e}")

print("\nWish you good health, goodbye!")
