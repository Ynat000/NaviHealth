# NaviHealth RAG Pipeline — Evaluation Strategy

This document outlines the evaluation and statistical testing approaches for the NaviHealth RAG pipeline, and the advantage each brings.

---

## 1. Basic Evaluation (Implemented)

**Script:** `scripts/eval_rag.py`

Runs a set of test queries through the RAG pipeline and scores each response using an LLM-as-judge approach (binary pass/fail).

| Metric | What it measures |
|--------|-----------------|
| **Faithfulness** | Is the answer grounded in the retrieved documents, or is the LLM hallucinating? |
| **Relevancy** | Does the answer actually address the question asked? |

**Limitation:** Returns binary 0/1 scores only — not useful for differentiating quality between strong models. Superseded by the rubric evaluator for detailed analysis.

**Advantage:** Catches the most common RAG failure modes — hallucination and off-topic answers — without needing a manually labelled ground truth dataset.

---

## 2. Rubric Evaluation (Implemented)

**Script:** `scripts/eval_rubric.py`

Scores each response on a **1-10 scale** across 5 criteria tailored to NaviHealth, using an LLM-as-judge approach. Replaces the binary pass/fail of basic evaluation with nuanced, actionable scores.

| Criterion | What it measures |
|-----------|-----------------|
| **Faithfulness** | Is the answer grounded in the retrieved document? Penalises hallucination. |
| **Completeness** | Does the answer fully address all parts of the question? |
| **Emergency Detection** | Did the response correctly advise calling 911 for emergencies? |
| **Language Appropriateness** | Did the response reply in the same language as the question? |
| **Triage Accuracy** | Did the response give appropriate guidance, including 811 referrals for out-of-scope questions? |

Each score includes a **one-sentence reason**, and an overall average is computed per query. Summary shows avg/min/max per criterion across all queries.

**Advantage:** Gives meaningful differentiation between responses — a model that scores 7/10 vs 9/10 on emergency detection is a real and important difference that binary scoring would miss. Criteria are specific to NaviHealth's use case.

---

## 3. A/B Model Comparison (Implemented)

**Scripts:**
- `scripts/ab_test_rubric.py` — A/B test using the 1-10 rubric (recommended)
- `scripts/ab_test.py` — A/B test using binary scores (superseded)

**Goal:** Compare two Groq models (`qwen/qwen3-32b` vs `meta-llama/llama-3.3-70b-versatile`) on the same set of queries and determine which performs significantly better.

**How it works:**
- Both models answer the same 15 queries using the same RAG pipeline
- A fixed **judge model** (`qwen/qwen3-32b`) scores both using the rubric — keeping evaluation consistent
- A **paired t-test** is applied per criterion to check if the difference is statistically significant (not just random variation)
- Results printed as a table: mean score per model, p-value, significance, and winner per criterion

**Metrics:**
- Mean score per criterion per model (1-10)
- p-value per criterion (< 0.05 = statistically significant)
- Overall winner per criterion

**Advantage:** Lets you make evidence-based model selection decisions with continuous scores and statistical rigour. The paired t-test accounts for query difficulty — harder queries are hard for both models, so only genuine model differences are surfaced.

### Understanding the Statistics

#### Mean Score
The average rubric score (1-10) across all queries for a given criterion. A higher mean = the model performs better on that criterion overall.

Example:
```
qwen3-32b       faithfulness mean = 8.6
llama3.3-70b    faithfulness mean = 7.9
```
At face value, qwen3-32b looks better — but is that difference real, or just noise from 15 queries?

#### Paired T-Test
A statistical test that answers: **"Is the difference between two models real, or could it have happened by chance?"**

"Paired" means we compare the two models on the **same query**, not on random separate queries. This matters because some queries are inherently harder than both models — pairing removes that shared difficulty from the comparison so only the model difference remains.

The test produces a **t-statistic** — the larger its absolute value, the more confident we are that a real difference exists.

Example:
```
Query: "I have chest pain"   → Model A scores 9,  Model B scores 7
Query: "Head lice treatment" → Model A scores 8,  Model B scores 8
Query: "Tick bite"           → Model A scores 10, Model B scores 9
...
```
The t-test looks at these pairs and calculates whether Model A consistently outperforms Model B, or whether the gaps are random.

#### P-Value
The probability that the observed score difference happened **by chance**, assuming the two models are actually identical.

| P-value | Interpretation |
|---------|---------------|
| p < 0.05 | Statistically significant — the difference is real with 95% confidence |
| p < 0.01 | Highly significant — 99% confidence |
| p ≥ 0.05 | Not significant — the difference could easily be random noise |

Example:
```
Faithfulness: p = 0.03 → SIGNIFICANT  → Model A is genuinely better at faithfulness
Completeness: p = 0.42 → not significant → The difference is likely just noise
```

#### Why This Matters for NaviHealth
Without a statistical test, you might look at two scores like 8.6 vs 7.9 and pick the higher one — but with only 15 queries, that gap could easily be random. The t-test tells you whether you can actually trust that difference before making a model decision that affects real users.

**Rule of thumb:** Only act on a difference if p < 0.05. Otherwise, either model is effectively equivalent on that criterion and you should choose based on cost, speed, or other factors.

---

## 4. Consistency Testing (Implemented)

**Script:** `scripts/eval_consistency.py`

**Goal:** Measure how stable the pipeline's answers are across repeated runs of the same query.

**Scoring approach: Rubric (1-10)**
Binary pass/fail is useless here — if a model always passes, the standard deviation is 0 and you learn nothing. Rubric scores like 7, 9, 6, 8 across repeated runs reveal real instability that binary scoring hides.

**How it works:**
- Run 5 representative queries × 3 runs each (15 total rubric calls)
- Score each run using the 5-criterion rubric
- Calculate **mean** and **standard deviation** per query per criterion
- Flag queries with std dev > 1.5 as "unstable"

**Metrics:**
- Mean rubric score per query across N runs
- Standard deviation per query (higher = more unstable)
- Queries with std dev > 1.5 flagged as unreliable

**Output:** Timestamped JSON + markdown summary saved to `eval_results/`

**Advantage:** LLMs are non-deterministic — a pipeline that scores well once but varies wildly between runs is unreliable in production. Low variance across rubric scores = consistent user experience you can trust.

---

## 5. Language Parity Testing (Implemented)

**Script:** `scripts/eval_language_parity.py`

**Goal:** Verify that the pipeline performs equally well across all supported languages, not just English.

**Supported languages:** English, Mandarin Chinese, French, Indonesian

**Scoring approach: Rubric (1-10)**
Binary scoring would hide a critical gap — a model might "pass" in Indonesian but score 5/10 vs 9/10 in English. The rubric surfaces exactly how large the parity gap is per criterion, not just whether it passes.

**How it works:**
- Run 5 health scenarios × 4 languages = 20 queries (same scenario, different language)
- Score each using the full rubric
- Compare mean scores per language per criterion
- Calculate max parity gap across all languages per criterion

**Metrics:**
- Mean rubric score per language per criterion
- Max gap across languages per criterion (target: < 1.0 points)
- Language with the largest drop identified per criterion

**Output:** Timestamped JSON + markdown summary saved to `eval_results/`

**Advantage:** NaviHealth specifically targets immigrant seniors in BC — if the pipeline performs well in English but poorly in Chinese or Indonesian, it fails its core users. The rubric parity gap quantifies exactly how large that failure is and on which criteria.

---

## 6. Retrieval Quality Testing (Implemented)

**Script:** `scripts/eval_retrieval.py`

**Goal:** Measure whether the vector search is retrieving the right documents before the LLM even sees them.

**Note:** This test makes **zero LLM API calls** — it is purely a vector search quality test and runs instantly.

**How it works:**
- 10 queries with known correct source documents are run through the retriever
- Check whether the expected document appears in the top-3 retrieved chunks
- Calculate **Hit Rate** and **Mean Reciprocal Rank (MRR)**

| Metric | Formula | What it means |
|--------|---------|---------------|
| **Hit Rate** | correct docs found / total queries | Did the right document appear at all in top-k? |
| **MRR** | average of 1/rank of first correct result | How high up was the right document? |

**Advantage:** Separates retrieval problems from LLM problems. If faithfulness scores are low, this tells you whether the issue is bad retrieval (wrong documents fetched) or bad generation (LLM ignoring the right documents).

---

## 7. Prompt Template Comparison (Implemented)

**Script:** `scripts/eval_prompt_comparison.py`

**Goal:** Test whether different system prompt wordings produce better or worse outputs.

**Scoring approach: Rubric (1-10)**
A prompt that shifts emergency detection from 7/10 to 9/10 is a critical improvement — binary scoring would show both as "pass" and miss it entirely.

**Prompt variants tested:**
- `strict` (baseline) — current production prompt with explicit rules
- `conversational` — warmer, friendlier tone, same rules
- `minimal` — very short prompt, bare minimum instructions

**How it works:**
- Run 10 queries through each of 3 prompt variants (60 total API calls)
- Score each using the full rubric
- Apply a paired t-test comparing each variant vs the `strict` baseline per criterion

**Metrics:**
- Mean rubric score per criterion per prompt variant
- p-value per criterion vs baseline (significant = worth switching)
- Criterion-level winner per comparison

**Output:** Timestamped JSON + markdown summary saved to `eval_results/`

**Advantage:** The prompt template has a large impact on output quality. A small wording change can meaningfully shift scores on specific criteria. The rubric + t-test tells you exactly which criteria improved and whether the change is statistically significant.

---

## 8. Chunk Size Sensitivity (Implemented)

**Script:** `scripts/eval_chunk_size.py`

**Goal:** Test how the document chunk size affects retrieval and answer quality.

**Scoring approach: Rubric (1-10)**
Chunk size affects answer quality gradually — binary scoring would show both sizes as "pass" and miss the difference.

**Chunk sizes tested:** 256, 512, 1024, 2048 tokens (with 50 token overlap)

**How it works:**
- Re-index the 253 documents at each chunk size using LlamaIndex `SentenceSplitter`
- Run 5 representative queries through each index (40 total API calls)
- Score each using the full rubric
- Compare mean scores per criterion per chunk size

**Metrics:**
- Mean rubric score per criterion per chunk size
- Number of chunks generated per size (smaller chunks = more chunks)
- Optimal chunk size = highest overall mean score

**Output:** Timestamped JSON + markdown summary saved to `eval_results/`

**Advantage:** Chunk size is one of the biggest levers in RAG quality. Too small = context lost, completeness drops. Too large = noise introduced, faithfulness drops. The rubric lets you see exactly which criteria degrade at each threshold.

---

## Summary Table

| Test | Script | Status | Impact | Priority |
|------|--------|--------|--------|----------|
| Basic evaluation | `eval_rag.py` | Done | Medium | Done — superseded by rubric |
| Rubric evaluation | `eval_rubric.py` | Done | High | Done |
| A/B model comparison (rubric) | `ab_test_rubric.py` | Done | High | Done |
| A/B model comparison (binary) | `ab_test.py` | Done | Low | Superseded by rubric version |
| Consistency testing | `eval_consistency.py` | Done | Medium | Done |
| Language parity | `eval_language_parity.py` | Done | Very High | Done |
| Retrieval quality | `eval_retrieval.py` | Done | High | Done |
| Prompt comparison | `eval_prompt_comparison.py` | Done | Medium | Done |
| Chunk size sensitivity | `eval_chunk_size.py` | Done | High | Done |