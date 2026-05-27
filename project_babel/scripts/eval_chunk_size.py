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

CHUNK_SIZES = [256, 512, 1024, 2048]  # tokens
CHUNK_OVERLAP = 50
CRITERIA = ["faithfulness", "completeness", "emergency_detection", "language_appropriateness", "triage_accuracy"]

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

import re
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, Settings, PromptTemplate
from llama_index.core.node_parser import SentenceSplitter
from llama_index.llms.groq import Groq
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

print("Setting up...")
Settings.embed_model = HuggingFaceEmbedding(model_name="paraphrase-multilingual-MiniLM-L12-v2")
groq_llm = Groq(model="qwen/qwen3-32b", api_key=api_key)
Settings.llm = groq_llm

print("Loading documents...")
documents = SimpleDirectoryReader(DATA_DIR).load_data()
print(f"Loaded {len(documents)} documents")

qa_prompt = PromptTemplate(QA_PROMPT_TMPL)


def score_with_rubric(query, answer, context):
    raw = groq_llm.complete(RUBRIC_PROMPT.format(context=context, query=query, answer=answer)).text.strip()
    raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    return json.loads(raw.strip())


print(f"\nChunk Size Sensitivity: {len(CHUNK_SIZES)} sizes x {len(TEST_QUERIES)} queries")
print("=" * 70)

all_results = {}

for chunk_size in CHUNK_SIZES:
    print(f"\n[Chunk size: {chunk_size} tokens]")
    print(f"  Building index...", end=" ")

    splitter = SentenceSplitter(chunk_size=chunk_size, chunk_overlap=CHUNK_OVERLAP)
    nodes = splitter.get_nodes_from_documents(documents)
    index = VectorStoreIndex(nodes)
    query_engine = index.as_query_engine(text_qa_template=qa_prompt, similarity_top_k=3)
    print(f"done ({len(nodes)} chunks)")

    results = []
    for label, query in TEST_QUERIES:
        try:
            response = query_engine.query(query)
            answer = response.response
            context = "\n\n".join([node.node.get_content()[:500] for node in response.source_nodes])
            sources = [node.metadata.get("file_name", "unknown") for node in response.source_nodes]
            scores = score_with_rubric(query, answer, context)
            log_call(source=f"eval_chunk:{chunk_size}", query=query, answer=answer, sources=sources, scores=scores)
            results.append({"label": label, "query": query, "answer": answer, "sources": sources, "scores": scores})
            print(f"  [{label}] overall={scores.get('overall','?')}/10")
        except Exception as e:
            results.append({"label": label, "query": query, "error": str(e)})
            print(f"  [{label}] ERROR: {e}")

    all_results[str(chunk_size)] = {"num_chunks": len(nodes), "results": results}

# Summary
print("\n" + "=" * 70)
print("CHUNK SIZE SENSITIVITY SUMMARY")
print("=" * 70)

chunk_summaries = {}
for chunk_size in CHUNK_SIZES:
    results = all_results[str(chunk_size)]["results"]
    valid = [r for r in results if "error" not in r]
    summary = {"evaluated": len(valid), "num_chunks": all_results[str(chunk_size)]["num_chunks"]}
    for criterion in CRITERIA:
        scores = [r["scores"][criterion]["score"] for r in valid]
        summary[criterion] = round(statistics.mean(scores), 2) if scores else None
    overall = [r["scores"]["overall"] for r in valid]
    summary["overall_mean"] = round(statistics.mean(overall), 2) if overall else None
    chunk_summaries[str(chunk_size)] = summary

print(f"\n{'Criterion':<28} " + "  ".join(f"{c:>8}" for c in CHUNK_SIZES))
print("-" * 70)
for criterion in CRITERIA + ["overall"]:
    key = "overall_mean" if criterion == "overall" else criterion
    label = criterion.replace("_", " ").title()
    row = f"{label:<28} "
    row += "  ".join(f"{chunk_summaries[str(c)].get(key, 'N/A'):>8}" for c in CHUNK_SIZES)
    print(row)

best_chunk = max(CHUNK_SIZES, key=lambda c: chunk_summaries[str(c)].get("overall_mean") or 0)
print(f"\nBest overall chunk size: {best_chunk} tokens (mean={chunk_summaries[str(best_chunk)].get('overall_mean')}/10)")

# Save
json_path = RESULTS_DIR / "chunk_size_sensitivity.json"
with open(json_path, "w", encoding="utf-8") as f:
    json.dump({
        "metadata": {"timestamp": timestamp, "model": "qwen/qwen3-32b", "chunk_sizes": CHUNK_SIZES, "chunk_overlap": CHUNK_OVERLAP, "num_queries": len(TEST_QUERIES)},
        "results": all_results,
        "summaries": chunk_summaries,
    }, f, indent=2, ensure_ascii=False)

md_lines = [
    "# Chunk Size Sensitivity Results", "",
    f"**Date:** {timestamp[:4]}-{timestamp[4:6]}-{timestamp[6:8]}",
    f"**Model:** qwen/qwen3-32b",
    f"**Chunk sizes tested:** {', '.join(str(c) for c in CHUNK_SIZES)} tokens",
    f"**Chunk overlap:** {CHUNK_OVERLAP} tokens",
    f"**Queries:** {len(TEST_QUERIES)}",
    f"**Best chunk size:** {best_chunk} tokens",
    "", "---", "", "## Mean Rubric Scores by Chunk Size", "",
    f"| Criterion | " + " | ".join(f"{c} tokens" for c in CHUNK_SIZES) + " |",
    f"|-----------|" + "|".join(["---"] * len(CHUNK_SIZES)) + "|",
]
for criterion in CRITERIA + ["overall"]:
    key = "overall_mean" if criterion == "overall" else criterion
    label = criterion.replace("_", " ").title() if criterion != "overall" else "**Overall**"
    row = f"| {label} |"
    for c in CHUNK_SIZES:
        val = chunk_summaries[str(c)].get(key, "N/A")
        row += f" {val} |"
    md_lines.append(row)

md_lines += ["", "---", "", "## Index Size by Chunk Size", "",
    "| Chunk Size | Number of Chunks |",
    "|-----------|-----------------|",
]
for c in CHUNK_SIZES:
    md_lines.append(f"| {c} tokens | {chunk_summaries[str(c)]['num_chunks']} |")

md_path = RESULTS_DIR / "chunk_size_sensitivity_summary.md"
with open(md_path, "w", encoding="utf-8") as f:
    f.write("\n".join(md_lines))

print(f"\nResults saved to: {json_path}")
print(f"Markdown saved to: {md_path}")
