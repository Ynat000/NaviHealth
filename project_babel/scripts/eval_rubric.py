import os
import json
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

from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, Settings, PromptTemplate
from llama_index.llms.groq import Groq
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

print("Setting up RAG pipeline...")
Settings.embed_model = HuggingFaceEmbedding(model_name="paraphrase-multilingual-MiniLM-L12-v2")
groq_llm = Groq(model="qwen/qwen3-32b", api_key=api_key)
Settings.llm = groq_llm

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

TEST_QUERIES = sample_queries(15)
print(f"Sampled {len(TEST_QUERIES)} queries from master query bank")

qa_prompt = PromptTemplate(QA_PROMPT_TMPL)
query_engine = index.as_query_engine(text_qa_template=qa_prompt, similarity_top_k=3)


def evaluate_query(label, query):
    try:
        response = query_engine.query(query)
        answer = response.response
        context = "\n\n".join([node.node.get_content()[:500] for node in response.source_nodes])
        sources = [node.metadata.get("file_name", "unknown") for node in response.source_nodes]

        rubric_input = RUBRIC_PROMPT.format(context=context, query=query, answer=answer)
        rubric_response = groq_llm.complete(rubric_input)

        import re
        raw = rubric_response.text.strip()
        raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        scores = json.loads(raw.strip())

        log_call(source="eval_rubric", query=query, answer=answer, sources=sources, scores=scores)
        return {
            "label": label,
            "query": query,
            "answer": answer,
            "sources": sources,
            "scores": scores,
        }
    except Exception as e:
        return {"label": label, "query": query, "error": str(e)}


print(f"\nRunning rubric evaluation on {len(TEST_QUERIES)} queries...\n")
print("=" * 70)

results = []

for label, query in TEST_QUERIES:
    result = evaluate_query(label, query)
    results.append(result)

    if "error" in result:
        print(f"[{result['label']}] ERROR: {result['error']}")
    else:
        s = result["scores"]
        print(f"[{result['label']}]")
        print(f"  Query:    {result['query']}")
        print(f"  Answer:   {result['answer'][:150]}{'...' if len(result['answer']) > 150 else ''}")
        print(f"  Faithfulness:          {s['faithfulness']['score']}/10  — {s['faithfulness']['reason']}")
        print(f"  Completeness:          {s['completeness']['score']}/10  — {s['completeness']['reason']}")
        print(f"  Emergency Detection:   {s['emergency_detection']['score']}/10  — {s['emergency_detection']['reason']}")
        print(f"  Language Appropriate:  {s['language_appropriateness']['score']}/10  — {s['language_appropriateness']['reason']}")
        print(f"  Triage Accuracy:       {s['triage_accuracy']['score']}/10  — {s['triage_accuracy']['reason']}")
        print(f"  OVERALL:               {s['overall']}/10")
        print("-" * 70)

# Summary
valid = [r for r in results if "error" not in r]
criteria = ["faithfulness", "completeness", "emergency_detection", "language_appropriateness", "triage_accuracy"]

print("\n" + "=" * 70)
print("RUBRIC EVALUATION SUMMARY")
print("=" * 70)
print(f"Evaluated: {len(valid)}/{len(TEST_QUERIES)} queries\n")

for criterion in criteria:
    scores = [r["scores"][criterion]["score"] for r in valid if criterion in r.get("scores", {})]
    if scores:
        avg = sum(scores) / len(scores)
        print(f"  {criterion.replace('_', ' ').title():<28} avg={avg:.2f}/10  min={min(scores)}  max={max(scores)}")

overall_scores = [r["scores"]["overall"] for r in valid if "overall" in r.get("scores", {})]
if overall_scores:
    print(f"\n  {'Overall':<28} avg={sum(overall_scores)/len(overall_scores):.2f}/10")

# Save
output_path = RESULTS_DIR / "rubric_eval.json"

with open(output_path, "w", encoding="utf-8") as f:
    json.dump({
        "metadata": {
            "timestamp": timestamp,
            "test_type": "rubric_eval",
            "model": "qwen/qwen3-32b",
            "num_documents": len(documents),
            "num_queries": len(TEST_QUERIES),
            "criteria": criteria,
        },
        "results": results,
    }, f, indent=2, ensure_ascii=False)

print(f"\nFull results saved to: {output_path}")
