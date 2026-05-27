import os
import json
from pathlib import Path
from datetime import datetime

from dotenv import load_dotenv

load_dotenv()

api_key = os.environ.get("GROQ_API_KEY")
if not api_key:
    raise RuntimeError("GROQ_API_KEY not set")

DATA_DIR = "../hlbc_data_pipeline/artifacts/hlbc_files_md"
RESULTS_DIR = Path("eval_results")
RESULTS_DIR.mkdir(exist_ok=True)

from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, Settings, PromptTemplate
from llama_index.llms.groq import Groq
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.evaluation import FaithfulnessEvaluator, RelevancyEvaluator

print("Setting up RAG pipeline...")
Settings.embed_model = HuggingFaceEmbedding(model_name="paraphrase-multilingual-MiniLM-L12-v2")
Settings.llm = Groq(model="qwen/qwen3-32b", api_key=api_key)

documents = SimpleDirectoryReader(DATA_DIR).load_data()
print(f"Loaded {len(documents)} documents")

index = VectorStoreIndex.from_documents(documents)

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

qa_prompt = PromptTemplate(QA_PROMPT_TMPL)
query_engine = index.as_query_engine(text_qa_template=qa_prompt, similarity_top_k=3)

groq_llm = Groq(model="qwen/qwen3-32b", api_key=api_key)
faithfulness_evaluator = FaithfulnessEvaluator(llm=groq_llm)
relevancy_evaluator = RelevancyEvaluator(llm=groq_llm)

# Test queries covering a range of topics and languages from the documents
test_queries = [
    # Emergency triage
    ("Emergency - chest pain", "I have severe chest pain and difficulty breathing, what should I do?"),
    ("Emergency - child seizure", "My child is having a seizure with a high fever, what do I do?"),

    # Condition-specific
    ("Concussion symptoms", "What are the signs of a concussion and when should I see a doctor?"),
    ("Tick bite", "I found a tick on my skin, what should I do?"),
    ("Measles symptoms", "What are the symptoms of measles and how does it spread?"),
    ("Head lice treatment", "My child has head lice, how do I treat it?"),
    ("E. coli infection", "What are the symptoms of E. coli infection and when is it serious?"),

    # Vaccination
    ("Flu vaccine seniors", "Why should seniors get the flu vaccine?"),
    ("Hepatitis B vaccine", "Who should get the hepatitis B vaccine?"),

    # Multilingual - Chinese
    ("Chinese - fever child", "我的孩子发烧了，我应该怎么办？"),
    ("Chinese - emergency", "我胸口很痛，呼吸困难，我应该怎么办？"),

    # Multilingual - French
    ("French - tick bite", "J'ai trouvé une tique sur ma peau, que dois-je faire?"),

    # Multilingual - Indonesian
    ("Indonesian - concussion", "Anak saya jatuh dan kepalanya terbentur, apa yang harus saya lakukan?"),

    # Out-of-scope (should redirect to 811)
    ("Out of scope - mental health", "I am feeling very depressed and anxious, where can I get help in BC?"),
    ("Out of scope - dental", "I have a bad toothache, should I go to emergency?"),
]


def evaluate_query(args):
    label, query = args
    try:
        response = query_engine.query(query)
        answer = response.response

        faith_result = faithfulness_evaluator.evaluate_response(response=response)
        relev_result = relevancy_evaluator.evaluate_response(query=query, response=response)

        sources = [node.metadata.get("file_name", "unknown") for node in response.source_nodes]

        return {
            "label": label,
            "query": query,
            "answer": answer,
            "faithfulness_pass": faith_result.passing,
            "faithfulness_score": faith_result.score,
            "relevancy_pass": relev_result.passing,
            "relevancy_score": relev_result.score,
            "sources": sources,
        }
    except Exception as e:
        return {"label": label, "query": query, "error": str(e)}


print(f"\nRunning {len(test_queries)} evaluation queries...\n")
print("=" * 70)

results = []

for q in test_queries:
    result = evaluate_query(q)
    results.append(result)

    label = result["label"]
    if "error" in result:
        print(f"[{label}] ERROR: {result['error']}")
    else:
        answer = result["answer"]
        print(f"[{label}]")
        print(f"Query: {result['query']}")
        print(f"Answer: {answer[:200]}{'...' if len(answer) > 200 else ''}")
        print(f"Faithfulness: {'PASS' if result['faithfulness_pass'] else 'FAIL'} (score: {result['faithfulness_score']})")
        print(f"Relevancy:    {'PASS' if result['relevancy_pass'] else 'FAIL'} (score: {result['relevancy_score']})")
        print(f"Sources: {', '.join(result['sources'])}")
    print("-" * 70)

# Summary
total = len([r for r in results if "error" not in r])
faith_pass = sum(1 for r in results if r.get("faithfulness_pass"))
relev_pass = sum(1 for r in results if r.get("relevancy_pass"))
faith_avg = sum(r["faithfulness_score"] for r in results if "faithfulness_score" in r) / max(total, 1)
relev_avg = sum(r["relevancy_score"] for r in results if "relevancy_score" in r) / max(total, 1)

print("\n" + "=" * 70)
print("EVALUATION SUMMARY")
print("=" * 70)
print(f"Total queries evaluated: {total}/{len(test_queries)}")
print(f"Faithfulness: {faith_pass}/{total} passed  (avg score: {faith_avg:.2f})")
print(f"Relevancy:    {relev_pass}/{total} passed  (avg score: {relev_avg:.2f})")

# Save results to JSON
output_path = RESULTS_DIR / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_basic_eval.json"
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

print(f"\nFull results saved to: {output_path}")