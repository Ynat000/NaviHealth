import os
import json
import statistics
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv
from call_logger import log_call
from query_sampler import sample_queries

load_dotenv()

api_key = os.environ.get("GROQ_API_KEY")
if not api_key:
    raise RuntimeError("GROQ_API_KEY not set")

DATA_DIR = "../hlbc_data_pipeline/artifacts/hlbc_files_md"
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
RESULTS_DIR = Path("eval_results") / timestamp
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

N_RUNS = 3  # number of times each query is repeated
INSTABILITY_THRESHOLD = 1.5  # std dev above this = flagged as unstable

TEST_QUERIES = sample_queries(5)
print(f"Sampled {len(TEST_QUERIES)} queries from master query bank")

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

RUBRIC_PROMPT = """\
You are an expert evaluator for NaviHealth, a multilingual healthcare triage assistant for British Columbia.

Evaluate the following AI response on 5 criteria. For each criterion, give a score from 1 to 10 and a one-sentence reason.

---
RETRIEVED DOCUMENT:
{context}

USER QUESTION:
{query}

AI RESPONSE:
{answer}
---

Score each criterion from 1 (very poor) to 10 (excellent):

1. FAITHFULNESS (1-10): Is the answer grounded in the retrieved document? Penalise heavily for hallucinated facts not present in the document.
2. COMPLETENESS (1-10): Does the answer fully address all parts of the question?
3. EMERGENCY DETECTION (1-10): If the question describes an emergency, did the response correctly advise calling 911? If not an emergency, score 10.
4. LANGUAGE APPROPRIATENESS (1-10): Did the response reply in the same language as the question?
5. TRIAGE ACCURACY (1-10): Did the response give appropriate healthcare guidance, including 811 referrals for out-of-scope questions?

Respond ONLY in this exact JSON format with no extra text:
{{
  "faithfulness": {{"score": <1-10>, "reason": "<one sentence>"}},
  "completeness": {{"score": <1-10>, "reason": "<one sentence>"}},
  "emergency_detection": {{"score": <1-10>, "reason": "<one sentence>"}},
  "language_appropriateness": {{"score": <1-10>, "reason": "<one sentence>"}},
  "triage_accuracy": {{"score": <1-10>, "reason": "<one sentence>"}},
  "overall": <average of the 5 scores as a float>
}}"""

CRITERIA = ["faithfulness", "completeness", "emergency_detection", "language_appropriateness", "triage_accuracy"]

from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, Settings, PromptTemplate
from llama_index.llms.groq import Groq
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
import re

print("Setting up RAG pipeline...")
Settings.embed_model = HuggingFaceEmbedding(model_name="paraphrase-multilingual-MiniLM-L12-v2")
groq_llm = Groq(model="qwen/qwen3-32b", api_key=api_key)
Settings.llm = groq_llm

documents = SimpleDirectoryReader(DATA_DIR).load_data()
print(f"Loaded {len(documents)} documents")
index = VectorStoreIndex.from_documents(documents)
qa_prompt = PromptTemplate(QA_PROMPT_TMPL)
query_engine = index.as_query_engine(text_qa_template=qa_prompt, similarity_top_k=3)


def score_with_rubric(query, answer, context):
    rubric_input = RUBRIC_PROMPT.format(context=context, query=query, answer=answer)
    raw = groq_llm.complete(rubric_input).text.strip()
    raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    return json.loads(raw.strip())


print(f"\nConsistency Test: {len(TEST_QUERIES)} queries x {N_RUNS} runs each\n")
print("=" * 70)

all_results = {}

for label, query in TEST_QUERIES:
    print(f"\n[{label}]")
    runs = []
    for run_num in range(1, N_RUNS + 1):
        print(f"  Run {run_num}/{N_RUNS}...", end=" ")
        try:
            response = query_engine.query(query)
            answer = response.response
            context = "\n\n".join([node.node.get_content()[:500] for node in response.source_nodes])
            sources = [node.metadata.get("file_name", "unknown") for node in response.source_nodes]
            scores = score_with_rubric(query, answer, context)
            log_call(source="eval_consistency", query=query, answer=answer, sources=sources, scores=scores)
            runs.append({"run": run_num, "answer": answer, "sources": sources, "scores": scores})
            print(f"overall={scores.get('overall', '?')}/10")
        except Exception as e:
            print(f"ERROR: {e}")
            runs.append({"run": run_num, "error": str(e)})

    all_results[label] = {"query": query, "runs": runs}

print("\n" + "=" * 70)
print("CONSISTENCY SUMMARY")
print("=" * 70)

summary_rows = []
for label, data in all_results.items():
    valid_runs = [r for r in data["runs"] if "error" not in r]
    if len(valid_runs) < 2:
        print(f"[{label}] Not enough valid runs to measure consistency.")
        continue

    row = {"label": label, "query": data["query"], "valid_runs": len(valid_runs)}
    print(f"\n[{label}]")
    for criterion in CRITERIA + ["overall"]:
        if criterion == "overall":
            scores = [r["scores"]["overall"] for r in valid_runs]
        else:
            scores = [r["scores"][criterion]["score"] for r in valid_runs]
        mean = statistics.mean(scores)
        std = statistics.stdev(scores) if len(scores) > 1 else 0.0
        unstable = std > INSTABILITY_THRESHOLD
        row[criterion] = {"mean": round(mean, 2), "std": round(std, 2), "unstable": unstable}
        flag = " *** UNSTABLE ***" if unstable else ""
        print(f"  {criterion.replace('_',' ').title():<28} mean={mean:.2f}  std={std:.2f}{flag}")

    summary_rows.append(row)

# Save JSON
json_path = RESULTS_DIR / "consistency_eval.json"
with open(json_path, "w", encoding="utf-8") as f:
    json.dump({
        "metadata": {
            "timestamp": timestamp,
            "model": "qwen/qwen3-32b",
            "n_runs": N_RUNS,
            "instability_threshold": INSTABILITY_THRESHOLD,
            "num_queries": len(TEST_QUERIES),
        },
        "results": all_results,
        "summary": summary_rows,
    }, f, indent=2, ensure_ascii=False)

# Save markdown
md_lines = [
    "# Consistency Evaluation Results",
    "",
    f"**Date:** {timestamp[:4]}-{timestamp[4:6]}-{timestamp[6:8]}",
    f"**Model:** qwen/qwen3-32b",
    f"**Runs per query:** {N_RUNS}",
    f"**Instability threshold:** std dev > {INSTABILITY_THRESHOLD}",
    "",
    "---",
    "",
    "## Results",
    "",
    f"| Query | Criterion | Mean | Std Dev | Stable? |",
    f"|-------|-----------|------|---------|---------|",
]

for row in summary_rows:
    for criterion in CRITERIA + ["overall"]:
        c = row.get(criterion, {})
        stable = "YES" if not c.get("unstable") else "**NO - UNSTABLE**"
        md_lines.append(f"| {row['label']} | {criterion.replace('_',' ').title()} | {c.get('mean','?')} | {c.get('std','?')} | {stable} |")

md_path = RESULTS_DIR / "consistency_eval_summary.md"
with open(md_path, "w", encoding="utf-8") as f:
    f.write("\n".join(md_lines))

print(f"\nResults saved to: {json_path}")
print(f"Markdown saved to: {md_path}")
