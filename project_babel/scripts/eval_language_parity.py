import os
import json
import random
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv
from call_logger import log_call

load_dotenv()

api_key = os.environ.get("GROQ_API_KEY")
if not api_key:
    raise RuntimeError("GROQ_API_KEY not set")

DATA_DIR = "../hlbc_data_pipeline/artifacts/hlbc_files_md"
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
RESULTS_DIR = Path("eval_results") / timestamp
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

CRITERIA = ["faithfulness", "completeness", "emergency_detection", "language_appropriateness", "triage_accuracy"]

N_SCENARIOS = 5  # how many to sample each run

# Full pool of multilingual scenario groups — same situation in all 4 languages
_SCENARIO_POOL = [
    {
        "scenario": "Emergency - chest pain",
        "en": "I have severe chest pain and difficulty breathing, what should I do?",
        "zh": "我胸口很痛，呼吸困难，我应该怎么办？",
        "fr": "J'ai de fortes douleurs thoraciques et des difficultés à respirer, que dois-je faire?",
        "id": "Saya mengalami nyeri dada yang parah dan kesulitan bernapas, apa yang harus saya lakukan?",
    },
    {
        "scenario": "Child seizure with fever",
        "en": "My child is having a seizure with a high fever, what do I do?",
        "zh": "我的孩子发高烧抽搐了，我应该怎么做？",
        "fr": "Mon enfant a des convulsions avec une forte fièvre, que dois-je faire?",
        "id": "Anak saya kejang dengan demam tinggi, apa yang harus saya lakukan?",
    },
    {
        "scenario": "Fever in child",
        "en": "My child has a high fever, what should I do?",
        "zh": "我的孩子发烧了，我应该怎么办？",
        "fr": "Mon enfant a une forte fièvre, que dois-je faire?",
        "id": "Anak saya demam tinggi, apa yang harus saya lakukan?",
    },
    {
        "scenario": "Head injury",
        "en": "My child fell and hit their head, what should I do?",
        "zh": "我的孩子摔倒撞到了头，我应该怎么办？",
        "fr": "Mon enfant est tombé et s'est cogné la tête, que dois-je faire?",
        "id": "Anak saya jatuh dan kepalanya terbentur, apa yang harus saya lakukan?",
    },
    {
        "scenario": "Flu vaccine for seniors",
        "en": "Why should seniors get the flu vaccine?",
        "zh": "为什么老年人应该接种流感疫苗？",
        "fr": "Pourquoi les personnes âgées devraient-elles se faire vacciner contre la grippe?",
        "id": "Mengapa lansia harus mendapatkan vaksin flu?",
    },
    {
        "scenario": "Tick bite",
        "en": "I found a tick on my skin, what should I do?",
        "zh": "我发现我的皮肤上有一只蜱虫，我该怎么办？",
        "fr": "J'ai trouvé une tique sur ma peau, que dois-je faire?",
        "id": "Saya menemukan kutu pada kulit saya, apa yang harus saya lakukan?",
    },
    {
        "scenario": "Measles symptoms",
        "en": "What are the symptoms of measles and how does it spread?",
        "zh": "麻疹有哪些症状？会如何传播？",
        "fr": "Quels sont les symptômes de la rougeole et comment se propage-t-elle?",
        "id": "Apa gejala campak dan bagaimana penyebarannya?",
    },
    {
        "scenario": "Out of scope - dental",
        "en": "I have a bad toothache, should I go to emergency?",
        "zh": "我牙很痛，我应该去急诊吗？",
        "fr": "J'ai une forte douleur aux dents, dois-je aller aux urgences?",
        "id": "Gigi saya sangat sakit, apakah saya harus pergi ke UGD?",
    },
    {
        "scenario": "Out of scope - mental health",
        "en": "I am feeling very depressed and anxious, where can I get help in BC?",
        "zh": "我感到非常焦虑和抑郁，我应该去哪里寻求帮助？",
        "fr": "Je me sens très déprimé et anxieux, où puis-je obtenir de l'aide en Colombie-Britannique?",
        "id": "Saya merasa sangat cemas dan depresi, di mana saya bisa mendapatkan bantuan?",
    },
    {
        "scenario": "Hepatitis B vaccine",
        "en": "Who should get the hepatitis B vaccine?",
        "zh": "谁应该接种乙型肝炎疫苗？",
        "fr": "Qui devrait recevoir le vaccin contre l'hépatite B?",
        "id": "Siapa yang harus mendapatkan vaksin hepatitis B?",
    },
]

QUERY_SETS = random.sample(_SCENARIO_POOL, N_SCENARIOS)
print(f"Sampled {N_SCENARIOS} scenarios from pool of {len(_SCENARIO_POOL)}:"
      + "".join(f"\n  - {s['scenario']}" for s in QUERY_SETS))

LANGUAGES = {"en": "English", "zh": "Chinese", "fr": "French", "id": "Indonesian"}

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

1. FAITHFULNESS (1-10): Is the answer grounded in the retrieved document?
2. COMPLETENESS (1-10): Does the answer fully address all parts of the question?
3. EMERGENCY DETECTION (1-10): If emergency, did it advise calling 911? If not, score 10.
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
from llama_index.llms.groq import Groq
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

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
    raw = groq_llm.complete(RUBRIC_PROMPT.format(context=context, query=query, answer=answer)).text.strip()
    raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    return json.loads(raw.strip())


print(f"\nLanguage Parity Test: {len(QUERY_SETS)} scenarios x {len(LANGUAGES)} languages\n")
print("=" * 70)

results = []

for qs in QUERY_SETS:
    scenario = qs["scenario"]
    print(f"\n[{scenario}]")
    scenario_results = {"scenario": scenario, "languages": {}}

    for lang_code, lang_name in LANGUAGES.items():
        query = qs[lang_code]
        print(f"  {lang_name}...", end=" ")
        try:
            response = query_engine.query(query)
            answer = response.response
            context = "\n\n".join([node.node.get_content()[:500] for node in response.source_nodes])
            sources = [node.metadata.get("file_name", "unknown") for node in response.source_nodes]
            scores = score_with_rubric(query, answer, context)
            log_call(source="eval_language_parity", query=query, answer=answer, sources=sources, scores=scores)
            scenario_results["languages"][lang_code] = {"query": query, "answer": answer, "sources": sources, "scores": scores}
            print(f"overall={scores.get('overall','?')}/10")
        except Exception as e:
            print(f"ERROR: {e}")
            scenario_results["languages"][lang_code] = {"query": query, "error": str(e)}

    results.append(scenario_results)

# Summary - compute parity gap vs English
print("\n" + "=" * 70)
print("LANGUAGE PARITY SUMMARY")
print("=" * 70)

lang_scores = {lang: {c: [] for c in CRITERIA + ["overall"]} for lang in LANGUAGES}

for scenario_result in results:
    for lang_code in LANGUAGES:
        lang_data = scenario_result["languages"].get(lang_code, {})
        if "error" not in lang_data and "scores" in lang_data:
            for criterion in CRITERIA:
                lang_scores[lang_code][criterion].append(lang_data["scores"][criterion]["score"])
            lang_scores[lang_code]["overall"].append(lang_data["scores"]["overall"])

lang_means = {}
for lang_code in LANGUAGES:
    lang_means[lang_code] = {}
    for criterion in CRITERIA + ["overall"]:
        scores = lang_scores[lang_code][criterion]
        lang_means[lang_code][criterion] = round(sum(scores) / len(scores), 2) if scores else None

print(f"\n{'Criterion':<28} {'English':>10} {'Chinese':>10} {'French':>10} {'Indonesian':>12} {'Max Gap':>10}")
print("-" * 80)
for criterion in CRITERIA + ["overall"]:
    en_mean = lang_means["en"].get(criterion)
    row = f"{criterion.replace('_',' ').title():<28}"
    max_gap = 0
    for lang_code in LANGUAGES:
        mean = lang_means[lang_code].get(criterion)
        row += f" {str(mean) if mean is not None else 'N/A':>10}"
        if mean is not None and en_mean is not None:
            gap = abs(en_mean - mean)
            max_gap = max(max_gap, gap)
    row += f" {max_gap:>10.2f}"
    print(row)

# Save
json_path = RESULTS_DIR / "language_parity.json"
with open(json_path, "w", encoding="utf-8") as f:
    json.dump({
        "metadata": {"timestamp": timestamp, "model": "qwen/qwen3-32b", "scenarios_sampled": len(QUERY_SETS), "scenario_pool_size": len(_SCENARIO_POOL), "languages": list(LANGUAGES.values())},
        "results": results,
        "language_means": lang_means,
    }, f, indent=2, ensure_ascii=False)

md_lines = [
    "# Language Parity Evaluation Results", "",
    f"**Date:** {timestamp[:4]}-{timestamp[4:6]}-{timestamp[6:8]}",
    f"**Model:** qwen/qwen3-32b",
    f"**Scenarios:** {len(QUERY_SETS)} (sampled from pool of {len(_SCENARIO_POOL)})",
    f"**Languages:** English, Chinese, French, Indonesian",
    "", "---", "", "## Mean Scores by Language", "",
    f"| Criterion | English | Chinese | French | Indonesian | Max Gap |",
    f"|-----------|---------|---------|--------|------------|---------|",
]

for criterion in CRITERIA + ["overall"]:
    en = lang_means["en"].get(criterion, "N/A")
    zh = lang_means["zh"].get(criterion, "N/A")
    fr = lang_means["fr"].get(criterion, "N/A")
    id_ = lang_means["id"].get(criterion, "N/A")
    vals = [v for v in [en, zh, fr, id_] if isinstance(v, float)]
    gap = round(max(vals) - min(vals), 2) if vals else "N/A"
    label = criterion.replace("_", " ").title()
    md_lines.append(f"| {label} | {en} | {zh} | {fr} | {id_} | {gap} |")

md_path = RESULTS_DIR / "language_parity_summary.md"
with open(md_path, "w", encoding="utf-8") as f:
    f.write("\n".join(md_lines))

print(f"\nResults saved to: {json_path}")
print(f"Markdown saved to: {md_path}")
