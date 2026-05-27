import os
import sys
from pathlib import Path
from dotenv import load_dotenv

sys.path.append(str(Path(__file__).parent / "scripts"))
from call_logger import log_call
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, Settings, PromptTemplate
from llama_index.llms.groq import Groq
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
import gradio as gr

load_dotenv()
api_key = os.environ.get("GROQ_API_KEY")
if not api_key:
    raise RuntimeError("GROQ_API_KEY not set")

DATA_DIR = "../hlbc_data_pipeline/artifacts/hlbc_files_md"

QA_PROMPT_TMPL = """\
You are NaviHealth, a healthcare triage assistant for British Columbia.
Please answer the user's questions based on the following policy documents.

Rules:
Respond in the same language used by the user.
Answers must be based strictly on the provided policy documents; do not hallucinate or invent information.
Clearly state the basis of your recommendation within the response.
If you are unable to determine the appropriate course of action, advise the user to call 811 for consultation.
In the event of an emergency (e.g., chest pain, difficulty breathing, severe bleeding), immediately advise the user to call 911.

Policy documents:
-----
{context_str}
-----

User Question: {query_str}

Answer:"""


def build_index():
    print("Loading embedding model...")
    Settings.embed_model = HuggingFaceEmbedding(model_name="paraphrase-multilingual-MiniLM-L12-v2")
    Settings.llm = Groq(model="qwen/qwen3-32b", api_key=api_key)

    print("Loading documents...")
    documents = SimpleDirectoryReader(DATA_DIR).load_data()
    print(f"Loaded {len(documents)} documents")

    print("Creating vector index...")
    index = VectorStoreIndex.from_documents(documents)
    print("Index ready.")
    return index


index = build_index()
qa_prompt = PromptTemplate(QA_PROMPT_TMPL)
query_engine = index.as_query_engine(text_qa_template=qa_prompt, similarity_top_k=3)


def respond(message, history):
    response = query_engine.query(message)

    sources = [node.metadata.get("file_name", "unknown") for node in response.source_nodes]
    answer = response.response

    log_call(source="gradio_app", query=message, answer=answer, sources=sources)

    if sources:
        answer += "\n\n**Sources:**\n" + "\n".join(f"- {s}" for s in sources)

    return answer


demo = gr.ChatInterface(
    fn=respond,
    title="NaviHealth",
    description="Ask health questions about BC healthcare services. Answers are sourced from HealthLink BC.",
)

if __name__ == "__main__":
    demo.launch(share=True)
