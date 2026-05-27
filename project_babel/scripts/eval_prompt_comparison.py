import os
import json
import statistics
from pathlib import Path
from datetime import datetime
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

CRITERIA = ["faithfulness", "completeness", "emergency_detection", "language_appropriateness", "triage_accuracy"]

# ── 3 prompt variants ──────────────────────────────────────────────────────────

PROMPTS = {
    "strict": """\
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

Answer:""",

    "conversational": """\
You are NaviHealth, a friendly and caring healthcare assistant helping people in British Columbia navigate the healthcare system.

Use the policy documents provided to answer the user's question in a warm, easy-to-understand way.
Always respond in the same language the user writes in.
Only use information from the documents — don't guess or make things up.
If the documents don't cover the question, kindly suggest the user call 811 for advice.
If it sounds like an emergency (chest pain, trouble breathing, serious injury), tell them to call 911 right away.

Policy documents:
-----
{context_str}
-----

User Question: {query_str}

Answer:""",

    "minimal": """\
Answer the health question using only the documents below. Reply in the user's language.
Call 911 for emergencies. Call 811 if unsure.

Documents:
-----
{context_str}
-----

Question: {query_str}

Answer:""",
}

RUBRIC_PROMPT = """\
You are an expert evaluator for NaviHealth, a multilingual healthcare triage assistant for British Columbia.

Evaluate the following AI response on 5 criteria. Score from 1 to 10.

---
RETRIEVED DOCUMENT:
{context}

USER QUESTION:
{query}

AI RESPONSE:
{answer}
---

1. FAITHFULNESS (1-10): Is the answer grounded in the retrieved document?
2. COMPLETENESS (1-10): Does the answer fully address all parts of the question?
3. EMERGENCY DETECTION (1-10): If emergency, did it advise calling 911? If not emergency, score 10.
4. LANGUAGE APPROPRIATENESS (1-10): Did the response reply in the same language as the question?
5. TRIAGE ACCURACY (1-10): Did the response give appropriate guidance, including 811 for out-of-scope?

Respond ONLY in this exact JSON format:
{{
  "faithfulness": {{"score": <1-10>, "reason": "<one sentence>"}},
  "completeness": {{"score": <1-10>, "reason": "<one sentence>"}},
  "emergency_detection": {{"score": <1-10>, "reason": "<one sentence>"}},
  "language_appropriateness": {{"score": <1-10>, "reason": "<one sentence>"}},
  "triage_accuracy": {{"score": <1-10>, "reason": "<one sentence>"}},
  "overall": <average of the 5 scores as a float>
}}"""

TEST_QUERIES = sample_queries(10)
print(f"Sampled {len(TEST_QUERIES)} queries from master query bank")

import re
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, Settings, PromptTemplate
from llama_index.llms.groq import Groq
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

print("Setting up RAG index...")
Settings.embed_model = HuggingFaceEmbedding(model_name="paraphrase-multilingual-MiniLM-L12-v2")
groq_llm = Groq(model="qwen/qwen3-32b", api_key=api_key)
Settings.llm = groq_llm

documents = SimpleDirectoryReader(DATA_DIR).load_data()
print(f"Loaded {len(documents)} documents")
index = VectorStoreIndex.from_documents(documents)


def score_with_rubric(query, answer, context):
    raw = groq_llm.complete(RUBRIC_PROMPT.format(context=context, query=query, answer=answer)).text.strip()
    raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    return json.loads(raw.strip())


def run_prompt_eval(prompt_name, prompt_template):
    print(f"\n  Evaluating prompt: [{prompt_name}]")
    qa_prompt = PromptTemplate(prompt_template)
    query_engine = index.as_query_engine(text_qa_template=qa_prompt, similarity_top_k=3)
    results = []

    for label, query in TEST_QUERIES:
        try:
            response = query_engine.query(query)
            answer = response.response
            context = "\n\n".join([node.node.get_content()[:500] for node in response.source_nodes])
            sources = [node.metadata.get("file_name", "unknown") for node in response.source_nodes]
            scores = score_with_rubric(query, answer, context)
            log_call(source=f"eval_prompt:{prompt_name}", query=query, answer=answer, sources=sources, scores=scores)
            results.append({"label": label, "query": query, "answer": answer, "sources": sources, "scores": scores})
            print(f"    [{label}] overall={scores.get('overall','?')}/10")
        except Exception as e:
            results.append({"label": label, "query": query, "error": str(e)})
            print(f"    [{label}] ERROR: {e}")

    return results


def summarise(results):
    valid = [r for r in results if "error" not in r]
    summary = {"evaluated": len(valid), "total": len(TEST_QUERIES)}
    for criterion in CRITERIA:
        scores = [r["scores"][criterion]["score"] for r in valid]
        if scores:
            summary[criterion] = {"mean": round(statistics.mean(scores), 4), "min": min(scores), "max": max(scores)}
    overall = [r["scores"]["overall"] for r in valid]
    if overall:
        summary["overall_mean"] = round(statistics.mean(overall), 4)
    return summary


print(f"\nPrompt Comparison: {len(PROMPTS)} variants x {len(TEST_QUERIES)} queries")
print("=" * 70)

prompt_results = {}
for name, template in PROMPTS.items():
    prompt_results[name] = run_prompt_eval(name, template)

summaries = {name: summarise(results) for name, results in prompt_results.items()}

# Paired t-test: compare each variant vs "strict" (baseline)
baseline = "strict"
stat_tests = {}
for variant in [p for p in PROMPTS if p != baseline]:
    labels_base = {r["label"]: r for r in prompt_results[baseline] if "error" not in r}
    labels_var = {r["label"]: r for r in prompt_results[variant] if "error" not in r}
    shared = sorted(set(labels_base) & set(labels_var))
    stat_tests[variant] = {}
    for criterion in CRITERIA + ["overall"]:
        if criterion == "overall":
            a = [labels_base[l]["scores"]["overall"] for l in shared]
            b = [labels_var[l]["scores"]["overall"] for l in shared]
        else:
            a = [labels_base[l]["scores"][criterion]["score"] for l in shared]
            b = [labels_var[l]["scores"][criterion]["score"] for l in shared]
        t_stat, p_val = stats.ttest_rel(a, b)
        stat_tests[variant][criterion] = {
            "baseline_mean": round(float(statistics.mean(a)), 4),
            "variant_mean": round(float(statistics.mean(b)), 4),
            "t_statistic": round(float(t_stat), 4),
            "p_value": round(float(p_val), 4),
            "significant": bool(p_val < 0.05),
            "winner": baseline if statistics.mean(a) >= statistics.mean(b) else variant,
        }

# Print summary
print("\n" + "=" * 70)
print("PROMPT COMPARISON SUMMARY")
print("=" * 70)

for name in PROMPTS:
    s = summaries[name]
    print(f"\n[{name}]  evaluated {s['evaluated']}/{s['total']}")
    for criterion in CRITERIA:
        c = s.get(criterion, {})
        print(f"  {criterion.replace('_',' ').title():<28} mean={c.get('mean','?')}/10")
    print(f"  {'Overall':<28} mean={s.get('overall_mean','?')}/10")

for variant, tests in stat_tests.items():
    print(f"\nStatistical Test: {baseline} vs {variant} (paired t-test)")
    print(f"{'Criterion':<28} {'strict':>10} {variant:>16} {'p-value':>10} {'Sig':>6} {'Winner'}")
    print("-" * 80)
    for criterion in CRITERIA + ["overall"]:
        t = tests[criterion]
        label = criterion.replace("_", " ").title()
        sig = "YES" if t["significant"] else "no"
        print(f"{label:<28} {t['baseline_mean']:>10.4f} {t['variant_mean']:>16.4f} {t['p_value']:>10.4f} {sig:>6} {t['winner']}")

# Save
json_path = RESULTS_DIR / "prompt_comparison.json"
with open(json_path, "w", encoding="utf-8") as f:
    json.dump({
        "metadata": {"timestamp": timestamp, "model": "qwen/qwen3-32b", "prompts": list(PROMPTS.keys()), "baseline": baseline, "num_queries": len(TEST_QUERIES)},
        "results": prompt_results,
        "summaries": summaries,
        "statistical_tests": stat_tests,
    }, f, indent=2, ensure_ascii=False)

# Markdown
md_lines = [
    "# Prompt Comparison Results", "",
    f"**Date:** {timestamp[:4]}-{timestamp[4:6]}-{timestamp[6:8]}",
    f"**Model:** qwen/qwen3-32b",
    f"**Prompts tested:** {', '.join(PROMPTS.keys())}",
    f"**Baseline:** {baseline}",
    f"**Queries:** {len(TEST_QUERIES)}",
    "", "---", "", "## Summary", "",
    f"| Criterion | " + " | ".join(PROMPTS.keys()) + " |",
    f"|-----------|" + "|".join(["---"] * len(PROMPTS)) + "|",
]
for criterion in CRITERIA + ["overall"]:
    label = criterion.replace("_", " ").title() if criterion != "overall" else "**Overall**"
    row = f"| {label} |"
    for name in PROMPTS:
        val = summaries[name].get("overall_mean" if criterion == "overall" else criterion, {})
        mean = val if criterion == "overall" else val.get("mean", "N/A")
        row += f" {mean} |"
    md_lines.append(row)

md_lines += ["", "---", "", "## Statistical Tests vs Baseline (strict)", ""]
for variant, tests in stat_tests.items():
    md_lines += [
        f"### strict vs {variant}", "",
        f"| Criterion | strict mean | {variant} mean | p-value | Significant | Winner |",
        f"|-----------|------------|----------------|---------|-------------|--------|",
    ]
    for criterion in CRITERIA + ["overall"]:
        t = tests[criterion]
        label = criterion.replace("_", " ").title()
        sig = "**YES**" if t["significant"] else "no"
        md_lines.append(f"| {label} | {t['baseline_mean']} | {t['variant_mean']} | {t['p_value']} | {sig} | {t['winner']} |")
    md_lines.append("")

md_path = RESULTS_DIR / "prompt_comparison_summary.md"
with open(md_path, "w", encoding="utf-8") as f:
    f.write("\n".join(md_lines))

print(f"\nResults saved to: {json_path}")
print(f"Markdown saved to: {md_path}")
