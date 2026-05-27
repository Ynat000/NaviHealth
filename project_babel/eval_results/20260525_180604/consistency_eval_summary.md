# Consistency Evaluation Results

**Date:** 2026-05-25
**Model:** qwen/qwen3-32b
**Runs per query:** 3
**Instability threshold:** std dev > 1.5

---

## Results

| Query | Criterion | Mean | Std Dev | Stable? |
|-------|-----------|------|---------|---------|
| Emergency - chest pain | Faithfulness | 8 | 1.73 | **NO - UNSTABLE** |
| Emergency - chest pain | Completeness | 9.67 | 0.58 | YES |
| Emergency - chest pain | Emergency Detection | 10 | 0.0 | YES |
| Emergency - chest pain | Language Appropriateness | 10 | 0.0 | YES |
| Emergency - chest pain | Triage Accuracy | 10 | 0.0 | YES |
| Emergency - chest pain | Overall | 9.47 | 0.23 | YES |
| Concussion symptoms | Faithfulness | 10 | 0.0 | YES |
| Concussion symptoms | Completeness | 8 | 3.46 | **NO - UNSTABLE** |
| Concussion symptoms | Emergency Detection | 10 | 0.0 | YES |
| Concussion symptoms | Language Appropriateness | 10 | 0.0 | YES |
| Concussion symptoms | Triage Accuracy | 10 | 0.0 | YES |
| Concussion symptoms | Overall | 9.73 | 0.46 | YES |
| Flu vaccine seniors | Faithfulness | 9.67 | 0.58 | YES |
| Flu vaccine seniors | Completeness | 9.67 | 0.58 | YES |
| Flu vaccine seniors | Emergency Detection | 10 | 0.0 | YES |
| Flu vaccine seniors | Language Appropriateness | 10 | 0.0 | YES |
| Flu vaccine seniors | Triage Accuracy | 10 | 0.0 | YES |
| Flu vaccine seniors | Overall | 9.87 | 0.12 | YES |
| Chinese - emergency | Faithfulness | 5.33 | 2.52 | **NO - UNSTABLE** |
| Chinese - emergency | Completeness | 9 | 1.0 | YES |
| Chinese - emergency | Emergency Detection | 10 | 0.0 | YES |
| Chinese - emergency | Language Appropriateness | 10 | 0.0 | YES |
| Chinese - emergency | Triage Accuracy | 9.67 | 0.58 | YES |
| Chinese - emergency | Overall | 8.8 | 0.53 | YES |
| Out of scope - dental | Faithfulness | 7.33 | 4.62 | **NO - UNSTABLE** |
| Out of scope - dental | Completeness | 9.67 | 0.58 | YES |
| Out of scope - dental | Emergency Detection | 10 | 0.0 | YES |
| Out of scope - dental | Language Appropriateness | 10 | 0.0 | YES |
| Out of scope - dental | Triage Accuracy | 9.67 | 0.58 | YES |
| Out of scope - dental | Overall | 9.2 | 1.39 | YES |