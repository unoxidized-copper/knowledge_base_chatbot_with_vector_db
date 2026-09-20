# Bonus Retrieval Comparison

The same multilingual embedding model was used for both approaches. Only the chunking settings changed.

| Approach | Chunk Size | Overlap | Hit Rate | Hits |
|---|---:|---:|---:|---:|
| A: compact chunks | 700 | 100 | 78% | 7/9 |
| B: balanced chunks | 900 | 120 | 89% | 8/9 |

Best result: **B: balanced chunks**.

A hit means that the expected chapter appeared in the top-5 retrieved passages, or that one of the expected answer keywords appeared in those passages. The no-answer question is excluded from the hit-rate calculation because it has no correct book passage.