======================================================================
A/B RUBRIC TEST SUMMARY
======================================================================

qwen/qwen3-32b  (evaluated 15/15)
  Faithfulness                 mean=9.8/10  min=9  max=10
  Completeness                 mean=9.2/10  min=7  max=10
  Emergency Detection          mean=10.0/10  min=10  max=10
  Language Appropriateness     mean=10.0/10  min=10  max=10
  Triage Accuracy              mean=9.9333/10  min=9  max=10
  Overall                      mean=9.7733/10

llama-3.3-70b-versatile  (evaluated 15/15)
  Faithfulness                 mean=7.6/10  min=3  max=10
  Completeness                 mean=7.7333/10  min=5  max=10
  Emergency Detection          mean=10.0/10  min=10  max=10
  Language Appropriateness     mean=9.9333/10  min=9  max=10
  Triage Accuracy              mean=9.5333/10  min=7  max=10
  Overall                      mean=8.8933/10

STATISTICAL TEST (paired t-test, p < 0.05 = significant)
Criterion                    Model A mean Model B mean    p-value  Significant Winner
------------------------------------------------------------------------------------------
Faithfulness                       9.8000       7.6000     0.0024          YES qwen3-32b
Completeness                       9.2000       7.7333     0.0129          YES qwen3-32b
Emergency Detection               10.0000      10.0000        nan           no qwen3-32b
Language Appropriateness          10.0000       9.9333     0.3343           no qwen3-32b
Triage Accuracy                    9.9333       9.5333     0.1383           no qwen3-32b
Overall                            9.7733       8.8933     0.0064          YES qwen3-32b