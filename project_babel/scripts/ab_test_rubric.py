import os
import json
from pathlib import Path
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv
from scipy import stats
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

MODELS = [
    "qwen/qwen3-32b",
    "llama-3.3-70b-versatile",
]

JUDGE_MODEL = "qwen/qwen3-32b"

CRITERIA = [
    "faithfulness",
    "completeness",
    "emergency_detection",
    "language_appropriateness",
    "triage_accuracy",
]

TEST_QUERIES = sample_queries(15)

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

2. COMPLETENESS (1-10): Does the answer fully address all parts of the question? Penalise if key information from the document is missing.

3. EMERGENCY DETECTION (1-10): If the question describes an emergency (chest pain, difficulty breathing, severe bleeding, seizure, etc.), did the response correctly advise calling 911? If the question is NOT an emergency, score 10 by default.

4. LANGUAGE APPROPRIATENESS (1-10): Did the response reply in the same language as the question? If the question is in Chinese, the answer must be in Chinese. If English, English. Etc.

5. TRIAGE ACCURACY (1-10): Did the response give appropriate healthcare guidance? For out-of-scope questions (dental, mental health, etc.), did it redirect to 811? For valid health questions, did it give clear and appropriate advice?

Respond ONLY in this exact JSON format with no extra text:
{{
  "faithfulness": {{"score": <1-10>, "reason": "<one sentence>"}},
  "completeness": {{"score": <1-10>, "reason": "<one sentence>"}},
  "emergency_detection": {{"score": <1-10>, "reason": "<one sentence>"}},
  "language_appropriateness": {{"score": <1-10>, "reason": "<one sentence>"}},
  "triage_accuracy": {{"score": <1-10>, "reason": "<one sentence>"}},
  "overall": <average of the 5 scores as a float>
}}"""


def build_query_engine(model_name, index, prompt):
    from llama_index.llms.groq import Groq
    llm = Groq(model=model_name, api_key=api_key)
    return index.as_query_engine(llm=llm, text_qa_template=prompt, similarity_top_k=3)


def score_with_rubric(judge_llm, query, answer, context):
    rubric_input = RUBRIC_PROMPT.format(context=context, query=query, answer=answer)
    rubric_response = judge_llm.complete(rubric_input)
    raw = rubric_response.text.strip()

    # strip <think>...</think> blocks from qwen models
    import re
    raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()

    # strip markdown code fences
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]

    return json.loads(raw.strip())


def evaluate_query(query_engine, judge_llm, label, query, model_name="unknown"):
    try:
        response = query_engine.query(query)
        answer = response.response
        context = "\n\n".join([node.node.get_content()[:500] for node in response.source_nodes])
        sources = [node.metadata.get("file_name", "unknown") for node in response.source_nodes]
        scores = score_with_rubric(judge_llm, query, answer, context)
        log_call(source=f"ab_test_rubric:{model_name}", query=query, answer=answer, sources=sources, scores=scores)
        return {
            "label": label,
            "query": query,
            "answer": answer,
            "sources": sources,
            "scores": scores,
        }
    except Exception as e:
        return {"label": label, "query": query, "error": str(e)}


def run_model_eval(model_name, query_engine, judge_llm, checkpoint_path):
    print(f"\n  Evaluating: {model_name}")
    results = []
    for label, query in TEST_QUERIES:
        result = evaluate_query(query_engine, judge_llm, label, query, model_name)
        results.append(result)
        if "error" in result:
            print(f"    [{label}] ERROR: {result['error']}")
        else:
            print(f"    [{label}] overall={result['scores'].get('overall', '?')}/10")

        # save checkpoint after every single call
        with open(checkpoint_path, "a", encoding="utf-8") as f:
            f.write(json.dumps({"model": model_name, **result}, ensure_ascii=False) + "\n")

    return results


def summarise(results):
    valid = [r for r in results if "error" not in r]
    summary = {"evaluated": len(valid), "total": len(TEST_QUERIES)}
    for criterion in CRITERIA:
        scores = [r["scores"][criterion]["score"] for r in valid if criterion in r.get("scores", {})]
        if scores:
            summary[criterion] = {
                "mean": round(sum(scores) / len(scores), 4),
                "min": min(scores),
                "max": max(scores),
            }
    overall = [r["scores"]["overall"] for r in valid if "overall" in r.get("scores", {})]
    if overall:
        summary["overall_mean"] = round(sum(overall) / len(overall), 4)
    return summary


def paired_ttest(results_a, results_b, criterion):
    labels_a = {r["label"]: r for r in results_a if "error" not in r}
    labels_b = {r["label"]: r for r in results_b if "error" not in r}
    shared = sorted(set(labels_a) & set(labels_b))

    if criterion == "overall":
        scores_a = [labels_a[l]["scores"]["overall"] for l in shared]
        scores_b = [labels_b[l]["scores"]["overall"] for l in shared]
    else:
        scores_a = [labels_a[l]["scores"][criterion]["score"] for l in shared]
        scores_b = [labels_b[l]["scores"][criterion]["score"] for l in shared]

    t_stat, p_val = stats.ttest_rel(scores_a, scores_b)
    mean_a = sum(scores_a) / len(scores_a)
    mean_b = sum(scores_b) / len(scores_b)
    return {
        "paired_queries": len(shared),
        "t_statistic": round(float(t_stat), 4),
        "p_value": round(float(p_val), 4),
        "significant": bool(p_val < 0.05),
        "mean_a": round(mean_a, 4),
        "mean_b": round(mean_b, 4),
        "winner": MODELS[0] if mean_a >= mean_b else MODELS[1],
    }


# ── Setup ──────────────────────────────────────────────────────────────────────

print("Setting up shared RAG index...")

from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, Settings, PromptTemplate
from llama_index.llms.groq import Groq
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

Settings.embed_model = HuggingFaceEmbedding(model_name="paraphrase-multilingual-MiniLM-L12-v2")
judge_llm = Groq(model=JUDGE_MODEL, api_key=api_key)
Settings.llm = judge_llm

documents = SimpleDirectoryReader(DATA_DIR).load_data()
print(f"Loaded {len(documents)} documents")
index = VectorStoreIndex.from_documents(documents)
qa_prompt = PromptTemplate(QA_PROMPT_TMPL)

# ── Run A/B ────────────────────────────────────────────────────────────────────

checkpoint_path = RESULTS_DIR / "ab_rubric_checkpoint.jsonl"
print(f"\nCheckpoint file: {checkpoint_path}")

print(f"\nSampled {len(TEST_QUERIES)} queries from master query bank:")
for label, _ in TEST_QUERIES:
    print(f"  - {label}")

print(f"\nA/B Rubric Test: {MODELS[0]}  vs  {MODELS[1]}")
print("=" * 70)

model_results = {}
for model in MODELS:
    engine = build_query_engine(model, index, qa_prompt)
    model_results[model] = run_model_eval(model, engine, judge_llm, checkpoint_path)

summaries = {model: summarise(results) for model, results in model_results.items()}

# ── Statistical comparison ─────────────────────────────────────────────────────

stat_tests = {}
for criterion in CRITERIA + ["overall"]:
    stat_tests[criterion] = paired_ttest(model_results[MODELS[0]], model_results[MODELS[1]], criterion)

# ── Print summary ──────────────────────────────────────────────────────────────

print("\n" + "=" * 70)
print("A/B RUBRIC TEST SUMMARY")
print("=" * 70)

for model in MODELS:
    s = summaries[model]
    print(f"\n{model}  (evaluated {s['evaluated']}/{s['total']})")
    for criterion in CRITERIA:
        c = s.get(criterion, {})
        print(f"  {criterion.replace('_',' ').title():<28} mean={c.get('mean','?')}/10  min={c.get('min','?')}  max={c.get('max','?')}")
    print(f"  {'Overall':<28} mean={s.get('overall_mean','?')}/10")

print("\nSTATISTICAL TEST (paired t-test, p < 0.05 = significant)")
print(f"{'Criterion':<28} {'Model A mean':>12} {'Model B mean':>12} {'p-value':>10} {'Significant':>12} {'Winner'}")
print("-" * 90)
for criterion in CRITERIA + ["overall"]:
    t = stat_tests[criterion]
    label = criterion.replace("_", " ").title()
    sig = "YES" if t["significant"] else "no"
    winner = t["winner"].split("/")[-1]
    print(f"{label:<28} {t['mean_a']:>12.4f} {t['mean_b']:>12.4f} {t['p_value']:>10.4f} {sig:>12} {winner}")

# ── Save ───────────────────────────────────────────────────────────────────────

model_a_slug = MODELS[0].replace("/", "-")
model_b_slug = MODELS[1].replace("/", "-")
output_path = RESULTS_DIR / f"ab_rubric_{model_a_slug}_vs_{model_b_slug}.json"

with open(output_path, "w", encoding="utf-8") as f:
    json.dump({
        "metadata": {
            "timestamp": timestamp,
            "test_type": "ab_rubric",
            "models": MODELS,
            "judge_model": JUDGE_MODEL,
            "num_documents": len(documents),
            "num_queries": len(TEST_QUERIES),
            "criteria": CRITERIA,
        },
        "results": {model: model_results[model] for model in MODELS},
        "summaries": summaries,
        "statistical_tests": stat_tests,
    }, f, indent=2, ensure_ascii=False)

print(f"\nFull results saved to: {output_path}")

# ── Save markdown summary ──────────────────────────────────────────────────────

def format_p(p):
    return "NaN" if str(p) == "nan" else str(p)

md_lines = [
    f"# A/B Rubric Test Results",
    f"",
    f"**Date:** {timestamp[:4]}-{timestamp[4:6]}-{timestamp[6:8]}",
    f"**Test type:** A/B Rubric Evaluation",
    f"**Model A:** {MODELS[0]}",
    f"**Model B:** {MODELS[1]}",
    f"**Judge model:** {JUDGE_MODEL}",
    f"**Documents indexed:** {len(documents)}",
    f"**Queries evaluated:** {len(TEST_QUERIES)}",
    f"",
    f"---",
    f"",
    f"## Per-Model Summary",
    f"",
    f"| Criterion | {MODELS[0].split('/')[-1]} | {MODELS[1]} |",
    f"|-----------|{'---'*3}|{'---'*3}|",
]

for criterion in CRITERIA:
    sa = summaries[MODELS[0]].get(criterion, {})
    sb = summaries[MODELS[1]].get(criterion, {})
    label = criterion.replace("_", " ").title()
    md_lines.append(
        f"| {label} | {sa.get('mean','?')}/10 (min={sa.get('min','?')}, max={sa.get('max','?')}) "
        f"| {sb.get('mean','?')}/10 (min={sb.get('min','?')}, max={sb.get('max','?')}) |"
    )

md_lines += [
    f"| **Overall** | **{summaries[MODELS[0]].get('overall_mean','?')}/10** | **{summaries[MODELS[1]].get('overall_mean','?')}/10** |",
    f"",
    f"---",
    f"",
    f"## Statistical Test (Paired T-Test)",
    f"",
    f"> p < 0.05 = statistically significant difference between models",
    f"",
    f"| Criterion | {MODELS[0].split('/')[-1]} mean | {MODELS[1]} mean | p-value | Significant | Winner |",
    f"|-----------|{'---'*3}|{'---'*3}|---------|-------------|--------|",
]

for criterion in CRITERIA + ["overall"]:
    t = stat_tests[criterion]
    label = criterion.replace("_", " ").title()
    sig = "**YES**" if t["significant"] else "no"
    winner = t["winner"].split("/")[-1]
    p = format_p(t["p_value"])
    md_lines.append(f"| {label} | {t['mean_a']} | {t['mean_b']} | {p} | {sig} | {winner} |")

overall_winner = stat_tests["overall"]["winner"].split("/")[-1]
md_lines += [
    f"",
    f"---",
    f"",
    f"## Interpretation",
    f"",
    f"**{stat_tests['overall']['winner']} is the {'significantly ' if stat_tests['overall']['significant'] else ''}better model overall** (p={format_p(stat_tests['overall']['p_value'])}).",
    f"",
    f"Key findings:",
    f"",
]

for criterion in CRITERIA:
    t = stat_tests[criterion]
    label = criterion.replace("_", " ").title()
    if t["significant"]:
        winner = t["winner"].split("/")[-1]
        loser = [m for m in MODELS if m != t["winner"]][0].split("/")[-1]
        md_lines.append(f"- **{label}** (p={t['p_value']}): {winner} is significantly better. {loser} showed lower scores indicating potential issues.")
    else:
        if t["mean_a"] == t["mean_b"]:
            md_lines.append(f"- **{label}**: Both models tied — no difference detected.")
        else:
            md_lines.append(f"- **{label}** (p={format_p(t['p_value'])}): No significant difference between models.")

md_lines += [
    f"",
    f"---",
    f"",
    f"## Raw Results File",
    f"",
    f"`{output_path.name}`",
]

md_path = RESULTS_DIR / f"ab_rubric_{model_a_slug}_vs_{model_b_slug}_summary.md"
with open(md_path, "w", encoding="utf-8") as f:
    f.write("\n".join(md_lines))

print(f"Markdown summary saved to: {md_path}")
