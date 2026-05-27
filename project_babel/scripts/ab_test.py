import os
import json
from pathlib import Path
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv
from scipy import stats

load_dotenv()

api_key = os.environ.get("GROQ_API_KEY")
if not api_key:
    raise RuntimeError("GROQ_API_KEY not set")

DATA_DIR = "../hlbc_data_pipeline/artifacts/hlbc_files_md"
RESULTS_DIR = Path("eval_results")
RESULTS_DIR.mkdir(exist_ok=True)

MODELS = [
    "qwen/qwen3-32b",
    "llama-3.3-70b-versatile",
]

TEST_QUERIES = [
    ("Emergency - chest pain",      "I have severe chest pain and difficulty breathing, what should I do?"),
    ("Emergency - child seizure",   "My child is having a seizure with a high fever, what do I do?"),
    ("Concussion symptoms",         "What are the signs of a concussion and when should I see a doctor?"),
    ("Tick bite",                   "I found a tick on my skin, what should I do?"),
    ("Measles symptoms",            "What are the symptoms of measles and how does it spread?"),
    ("Head lice treatment",         "My child has head lice, how do I treat it?"),
    ("E. coli infection",           "What are the symptoms of E. coli infection and when is it serious?"),
    ("Flu vaccine seniors",         "Why should seniors get the flu vaccine?"),
    ("Hepatitis B vaccine",         "Who should get the hepatitis B vaccine?"),
    ("Chinese - fever child",       "我的孩子发烧了，我应该怎么办？"),
    ("Chinese - emergency",         "我胸口很痛，呼吸困难，我应该怎么办？"),
    ("French - tick bite",          "J'ai trouvé une tique sur ma peau, que dois-je faire?"),
    ("Indonesian - concussion",     "Anak saya jatuh dan kepalanya terbentur, apa yang harus saya lakukan?"),
    ("Out of scope - mental health","I am feeling very depressed and anxious, where can I get help in BC?"),
    ("Out of scope - dental",       "I have a bad toothache, should I go to emergency?"),
]

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


def build_query_engine(model_name, index, prompt):
    from llama_index.llms.groq import Groq
    llm = Groq(model=model_name, api_key=api_key)
    return index.as_query_engine(llm=llm, text_qa_template=prompt, similarity_top_k=3)


def evaluate_query(query_engine, faithfulness_evaluator, relevancy_evaluator, label, query):
    try:
        response = query_engine.query(query)
        answer = response.response
        faith = faithfulness_evaluator.evaluate_response(response=response)
        relev = relevancy_evaluator.evaluate_response(query=query, response=response)
        sources = [node.metadata.get("file_name", "unknown") for node in response.source_nodes]
        return {
            "label": label,
            "query": query,
            "answer": answer,
            "faithfulness_pass": faith.passing,
            "faithfulness_score": faith.score,
            "relevancy_pass": relev.passing,
            "relevancy_score": relev.score,
            "sources": sources,
        }
    except Exception as e:
        return {"label": label, "query": query, "error": str(e)}


def run_model_eval(model_name, index, prompt, faithfulness_evaluator, relevancy_evaluator):
    print(f"\n  Running queries for: {model_name}")
    query_engine = build_query_engine(model_name, index, prompt)
    results = []

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = {
            executor.submit(evaluate_query, query_engine, faithfulness_evaluator, relevancy_evaluator, label, query): label
            for label, query in TEST_QUERIES
        }
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            label = result["label"]
            if "error" in result:
                print(f"    [{label}] ERROR: {result['error']}")
            else:
                f = result["faithfulness_score"]
                r = result["relevancy_score"]
                print(f"    [{label}] Faith: {f:.2f}  Relev: {r:.2f}")

    return results


def summarise(results):
    valid = [r for r in results if "error" not in r]
    if not valid:
        return {}
    faith_scores = [r["faithfulness_score"] for r in valid]
    relev_scores = [r["relevancy_score"] for r in valid]
    return {
        "total": len(TEST_QUERIES),
        "evaluated": len(valid),
        "faithfulness_pass": sum(1 for r in valid if r["faithfulness_pass"]),
        "faithfulness_mean": round(sum(faith_scores) / len(faith_scores), 4),
        "faithfulness_std": round(float(stats.tstd(faith_scores)), 4),
        "relevancy_pass": sum(1 for r in valid if r["relevancy_pass"]),
        "relevancy_mean": round(sum(relev_scores) / len(relev_scores), 4),
        "relevancy_std": round(float(stats.tstd(relev_scores)), 4),
    }


# ── Setup ──────────────────────────────────────────────────────────────────────

print("Setting up shared RAG index...")

from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, Settings, PromptTemplate
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.evaluation import FaithfulnessEvaluator, RelevancyEvaluator

from llama_index.llms.groq import Groq

Settings.embed_model = HuggingFaceEmbedding(model_name="paraphrase-multilingual-MiniLM-L12-v2")
judge_llm = Groq(model=MODELS[0], api_key=api_key)
Settings.llm = judge_llm

documents = SimpleDirectoryReader(DATA_DIR).load_data()
print(f"Loaded {len(documents)} documents")
index = VectorStoreIndex.from_documents(documents)

qa_prompt = PromptTemplate(QA_PROMPT_TMPL)
faithfulness_evaluator = FaithfulnessEvaluator(llm=judge_llm)
relevancy_evaluator = RelevancyEvaluator(llm=judge_llm)

# ── Run A/B test ───────────────────────────────────────────────────────────────

print(f"\nA/B Test: {MODELS[0]}  vs  {MODELS[1]}")
print("=" * 70)

model_results = {}
for model in MODELS:
    model_results[model] = run_model_eval(model, index, qa_prompt, faithfulness_evaluator, relevancy_evaluator)

# ── Statistical comparison ─────────────────────────────────────────────────────

summaries = {model: summarise(results) for model, results in model_results.items()}

# Align scores by label for paired t-test
labels_a = {r["label"]: r for r in model_results[MODELS[0]] if "error" not in r}
labels_b = {r["label"]: r for r in model_results[MODELS[1]] if "error" not in r}
shared_labels = sorted(set(labels_a) & set(labels_b))

faith_a = [labels_a[l]["faithfulness_score"] for l in shared_labels]
faith_b = [labels_b[l]["faithfulness_score"] for l in shared_labels]
relev_a = [labels_a[l]["relevancy_score"] for l in shared_labels]
relev_b = [labels_b[l]["relevancy_score"] for l in shared_labels]

faith_ttest = stats.ttest_rel(faith_a, faith_b)
relev_ttest = stats.ttest_rel(relev_a, relev_b)

statistical_test = {
    "paired_queries": len(shared_labels),
    "faithfulness": {
        "t_statistic": round(faith_ttest.statistic, 4),
        "p_value": round(faith_ttest.pvalue, 4),
        "significant": faith_ttest.pvalue < 0.05,
        "winner": MODELS[0] if summaries[MODELS[0]]["faithfulness_mean"] > summaries[MODELS[1]]["faithfulness_mean"] else MODELS[1],
    },
    "relevancy": {
        "t_statistic": round(relev_ttest.statistic, 4),
        "p_value": round(relev_ttest.pvalue, 4),
        "significant": relev_ttest.pvalue < 0.05,
        "winner": MODELS[0] if summaries[MODELS[0]]["relevancy_mean"] > summaries[MODELS[1]]["relevancy_mean"] else MODELS[1],
    },
}

# ── Print summary ──────────────────────────────────────────────────────────────

print("\n" + "=" * 70)
print("A/B TEST SUMMARY")
print("=" * 70)

for model in MODELS:
    s = summaries[model]
    print(f"\n{model}")
    print(f"  Faithfulness: {s['faithfulness_pass']}/{s['evaluated']} passed  mean={s['faithfulness_mean']}  std={s['faithfulness_std']}")
    print(f"  Relevancy:    {s['relevancy_pass']}/{s['evaluated']} passed  mean={s['relevancy_mean']}  std={s['relevancy_std']}")

print("\nSTATISTICAL TEST (paired t-test)")
ft = statistical_test["faithfulness"]
rt = statistical_test["relevancy"]
print(f"  Faithfulness: t={ft['t_statistic']}  p={ft['p_value']}  {'SIGNIFICANT' if ft['significant'] else 'not significant'}  winner={ft['winner']}")
print(f"  Relevancy:    t={rt['t_statistic']}  p={rt['p_value']}  {'SIGNIFICANT' if rt['significant'] else 'not significant'}  winner={rt['winner']}")

# ── Save results ───────────────────────────────────────────────────────────────

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
model_a_slug = MODELS[0].replace("/", "-")
model_b_slug = MODELS[1].replace("/", "-")
output_path = RESULTS_DIR / f"{timestamp}_ab_test_{model_a_slug}_vs_{model_b_slug}.json"

output = {
    "metadata": {
        "timestamp": timestamp,
        "test_type": "ab_test",
        "models": MODELS,
        "num_documents": len(documents),
        "num_queries": len(TEST_QUERIES),
    },
    "results": {model: model_results[model] for model in MODELS},
    "summaries": summaries,
    "statistical_test": statistical_test,
}

with open(output_path, "w", encoding="utf-8") as f:
    json.dump(output, f, indent=2, ensure_ascii=False)

print(f"\nFull results saved to: {output_path}")