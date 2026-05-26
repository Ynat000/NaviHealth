# Consistency Evaluation Results

**Date:** 2026-05-25
**Model:** qwen/qwen3-32b
**Runs per query:** 3
**Instability threshold:** std dev > 1.5

---

## Results

| Query | Criterion | Mean | Std Dev | Stable? |
|-------|-----------|------|---------|---------|
| ID - out of scope mental health | Faithfulness | 7.67 | 4.04 | **NO - UNSTABLE** |
| ID - out of scope mental health | Completeness | 8.67 | 2.31 | **NO - UNSTABLE** |
| ID - out of scope mental health | Emergency Detection | 10 | 0.0 | YES |
| ID - out of scope mental health | Language Appropriateness | 10 | 0.0 | YES |
| ID - out of scope mental health | Triage Accuracy | 9.33 | 1.15 | YES |
| ID - out of scope mental health | Overall | 8.87 | 1.96 | **NO - UNSTABLE** |
| Well water safety | Faithfulness | 8.33 | 1.53 | **NO - UNSTABLE** |
| Well water safety | Completeness | 9.67 | 0.58 | YES |
| Well water safety | Emergency Detection | 10 | 0.0 | YES |
| Well water safety | Language Appropriateness | 10 | 0.0 | YES |
| Well water safety | Triage Accuracy | 9.67 | 0.58 | YES |
| Well water safety | Overall | 9.53 | 0.5 | YES |