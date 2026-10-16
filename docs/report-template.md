# Research Report Template

Fill per experiment/study; keep reports linked to experiment IDs (result
filenames in experiments/results/). Export Markdown now, PDF later (§12).

```markdown
# <Study title>

## Experiment
- Experiment ID(s): <result filename(s)>
- Dataset: <name@version> (<n> docs, <n> queries)
- Git commit: <hash>

## Research question

## Hypothesis

## Methodology
- Retrieval strategies compared:
- Evaluation metrics:
- Seed / splits:

## Results
| strategy | recall@5 | recall@10 | mrr@10 | ndcg@10 |
|---|---|---|---|---|

## Statistical analysis

## Error analysis
| failure class | count | example query |
|---|---|---|

## Limitations

## Conclusion
```
