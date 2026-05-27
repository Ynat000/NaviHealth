======================================================================
A/B RUBRIC TEST SUMMARY
======================================================================

qwen/qwen3-32b  (evaluated 15/15)
  Faithfulness                 mean=9.1333/10  min=5  max=10
  Completeness                 mean=8.8667/10  min=5  max=10
  Emergency Detection          mean=9.8667/10  min=8  max=10
  Language Appropriateness     mean=8.8667/10  min=1  max=10
  Triage Accuracy              mean=9.4/10  min=6  max=10
  Overall                      mean=9.0667/10

llama-3.3-70b-versatile  (evaluated 15/15)
  Faithfulness                 mean=7.7333/10  min=2  max=10
  Completeness                 mean=7.0667/10  min=3  max=10
  Emergency Detection          mean=10.0/10  min=10  max=10
  Language Appropriateness     mean=10.0/10  min=10  max=10
  Triage Accuracy              mean=9.4667/10  min=6  max=10
  Overall                      mean=8.7733/10


STATISTICAL TEST (paired t-test, p < 0.05 = significant)
Criterion                    Model A mean Model B mean    p-value  Significant Winner
------------------------------------------------------------------------------------------
Faithfulness                       9.1333       7.7333     0.0592           no qwen3-32b
Completeness                       8.8667       7.0667     0.0007          YES qwen3-32b
Emergency Detection                9.8667      10.0000     0.3343           no llama-3.3-70b-versatile
Language Appropriateness           8.8667      10.0000     0.1651           no llama-3.3-70b-versatile
Triage Accuracy                    9.4000       9.4667     0.7513           no llama-3.3-70b-versatile
Overall                            9.0667       8.7733     0.4141           no qwen3-32b