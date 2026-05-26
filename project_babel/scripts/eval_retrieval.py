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
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
RESULTS_DIR = Path("eval_results") / timestamp
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Queries with known correct source documents
# (query, expected filename)
RETRIEVAL_QUERIES = [
    ("What are the symptoms of measles and how does it spread?",         "14b_Measles_2023-12.md"),
    ("What are the signs of a concussion?",                              "122_Concussion_2024-10.md"),
    ("I found a tick on my skin, what should I do?",                     "01_Tick_bites_and_disease_2024-06.md"),
    ("My child has head lice, how do I treat it?",                       "06_Head_lice_2024-09.md"),
    ("What are the symptoms of E. coli infection?",                      "02_E._coli_infection_2022-02.md"),
    ("Why should seniors get the flu vaccine?",                          "12a_Why_seniors_should_get_the_inactivated_influenza_flu_vaccine_2025-09.md"),
    ("Who should get the hepatitis B vaccine?",                          "25a_Hepatitis_B_vaccine_2025-05.md"),
    ("What is a febrile seizure and what should I do?",                  "112_Febrile_seizures_fever_seizures_2025-08.md"),
    ("How is scabies treated?",                                          "09_Scabies_2024-08.md"),
    ("What are the symptoms of rabies and when should I get vaccinated?", "07a_Rabies_2025-12.md"),
]

TOP_K = 3

from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, Settings
from llama_index.llms.groq import Groq
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

print("Setting up retrieval index...")
Settings.embed_model = HuggingFaceEmbedding(model_name="paraphrase-multilingual-MiniLM-L12-v2")
Settings.llm = Groq(model="qwen/qwen3-32b", api_key=api_key)

documents = SimpleDirectoryReader(DATA_DIR).load_data()
print(f"Loaded {len(documents)} documents")
index = VectorStoreIndex.from_documents(documents)
retriever = index.as_retriever(similarity_top_k=TOP_K)

print(f"\nRetrieval Quality Test: {len(RETRIEVAL_QUERIES)} queries  top_k={TOP_K}\n")
print("=" * 70)

results = []
hit_count = 0
reciprocal_ranks = []

for query, expected_doc in RETRIEVAL_QUERIES:
    nodes = retriever.retrieve(query)
    retrieved_files = [node.metadata.get("file_name", "unknown") for node in nodes]

    hit = expected_doc in retrieved_files
    rank = retrieved_files.index(expected_doc) + 1 if hit else None
    rr = 1 / rank if rank else 0.0

    if hit:
        hit_count += 1
    reciprocal_ranks.append(rr)

    status = f"HIT (rank {rank})" if hit else "MISS"
    print(f"[{status}] {query[:60]}")
    print(f"  Expected: {expected_doc}")
    print(f"  Retrieved: {', '.join(retrieved_files)}")
    print()

    results.append({
        "query": query,
        "expected_doc": expected_doc,
        "retrieved_docs": retrieved_files,
        "hit": hit,
        "rank": rank,
        "reciprocal_rank": rr,
    })

hit_rate = hit_count / len(RETRIEVAL_QUERIES)
mrr = sum(reciprocal_ranks) / len(reciprocal_ranks)

print("=" * 70)
print("RETRIEVAL QUALITY SUMMARY")
print("=" * 70)
print(f"Hit Rate: {hit_count}/{len(RETRIEVAL_QUERIES)} = {hit_rate:.2%}")
print(f"MRR (Mean Reciprocal Rank): {mrr:.4f}")
print()
print("Interpretation:")
print(f"  Hit Rate {hit_rate:.2%} — the correct document appeared in top-{TOP_K} for {hit_rate:.0%} of queries")
print(f"  MRR {mrr:.4f} — on average the correct document was at rank {1/mrr:.1f}" if mrr > 0 else "  MRR 0 — correct document was never retrieved")

# Save JSON
json_path = RESULTS_DIR / "retrieval_quality.json"
with open(json_path, "w", encoding="utf-8") as f:
    json.dump({
        "metadata": {
            "timestamp": timestamp,
            "top_k": TOP_K,
            "num_queries": len(RETRIEVAL_QUERIES),
            "num_documents": len(documents),
        },
        "results": results,
        "summary": {
            "hit_rate": round(hit_rate, 4),
            "hit_count": hit_count,
            "mrr": round(mrr, 4),
        },
    }, f, indent=2, ensure_ascii=False)

# Save markdown
md_lines = [
    "# Retrieval Quality Evaluation Results", "",
    f"**Date:** {timestamp[:4]}-{timestamp[4:6]}-{timestamp[6:8]}",
    f"**Embedding model:** paraphrase-multilingual-MiniLM-L12-v2",
    f"**Top-K:** {TOP_K}",
    f"**Queries evaluated:** {len(RETRIEVAL_QUERIES)}",
    f"**Documents indexed:** {len(documents)}",
    "", "---", "", "## Summary", "",
    f"| Metric | Value | Interpretation |",
    f"|--------|-------|---------------|",
    f"| Hit Rate | {hit_rate:.2%} | Correct doc appeared in top-{TOP_K} for {hit_count}/{len(RETRIEVAL_QUERIES)} queries |",
    f"| MRR | {mrr:.4f} | Average rank of correct document: {1/mrr:.1f}" + (" |" if mrr > 0 else " (never retrieved) |"),
    "", "---", "", "## Per-Query Results", "",
    f"| Query | Expected Doc | Hit? | Rank |",
    f"|-------|-------------|------|------|",
]

for r in results:
    status = f"YES (rank {r['rank']})" if r["hit"] else "**NO**"
    md_lines.append(f"| {r['query'][:60]} | {r['expected_doc']} | {status} | {r['rank'] or '-'} |")

md_path = RESULTS_DIR / "retrieval_quality_summary.md"
with open(md_path, "w", encoding="utf-8") as f:
    f.write("\n".join(md_lines))

print(f"\nResults saved to: {json_path}")
print(f"Markdown saved to: {md_path}")
