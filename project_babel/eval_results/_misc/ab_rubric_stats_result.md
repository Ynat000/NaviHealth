# A/B Rubric Test Results

**Date:** 2026-05-25  
**Test type:** A/B Rubric Evaluation  
**Model A:** qwen/qwen3-32b  
**Model B:** llama-3.3-70b-versatile  
**Judge model:** qwen/qwen3-32b  
**Documents indexed:** 253  
**Queries evaluated:** 15  

---

## Per-Model Summary

| Criterion | qwen3-32b | llama-3.3-70b | 
|-----------|-----------|----------------|
| Faithfulness | 9.80/10 (min=9, max=10) | 7.60/10 (min=3, max=10) |
| Completeness | 9.20/10 (min=7, max=10) | 7.73/10 (min=5, max=10) |
| Emergency Detection | 10.00/10 (min=10, max=10) | 10.00/10 (min=10, max=10) |
| Language Appropriateness | 10.00/10 (min=10, max=10) | 9.93/10 (min=9, max=10) |
| Triage Accuracy | 9.93/10 (min=9, max=10) | 9.53/10 (min=7, max=10) |
| **Overall** | **9.77/10** | **8.89/10** |

---

## Statistical Test (Paired T-Test)

> p < 0.05 = statistically significant difference between models

| Criterion | qwen3-32b mean | llama-3.3-70b mean | p-value | Significant | Winner |
|-----------|---------------|-------------------|---------|-------------|--------|
| Faithfulness | 9.80 | 7.60 | 0.0024 | **YES** | qwen3-32b |
| Completeness | 9.20 | 7.73 | 0.0129 | **YES** | qwen3-32b |
| Emergency Detection | 10.00 | 10.00 | NaN | no (tied) | — |
| Language Appropriateness | 10.00 | 9.93 | 0.3343 | no | qwen3-32b |
| Triage Accuracy | 9.93 | 9.53 | 0.1383 | no | qwen3-32b |
| **Overall** | **9.77** | **8.89** | **0.0064** | **YES** | **qwen3-32b** |

---

## Interpretation

**qwen/qwen3-32b is the significantly better model overall** (p=0.0064).

Key findings:

- **Faithfulness** (p=0.0024): qwen3-32b is significantly more grounded in the retrieved documents. llama scored as low as 3/10 on some queries, indicating hallucination.
- **Completeness** (p=0.0129): qwen3-32b answers more fully cover the question. llama dropped to 5/10 on some queries.
- **Emergency Detection**: Both models scored perfectly (10/10) on every emergency query — both correctly advise calling 911.
- **Language Appropriateness**: No significant difference — both models respond correctly in the user's language.
- **Triage Accuracy**: No significant difference, though qwen3-32b is consistently higher.

**Recommendation:** Use **qwen/qwen3-32b** for NaviHealth. The difference in faithfulness is the most critical finding — a model that hallucinates healthcare information is a safety risk.

---

## Raw Results File

`20260525_172612_ab_rubric_qwen-qwen3-32b_vs_llama-3.3-70b-versatile.json`
