# Reviewer-requested statistics

### fiqa
- canonical test queries: 648
- unified best uniform: hybrid @ 0.3711
  (strategy means: bm25 0.2312, dense 0.3687, hybrid 0.3711)
- hard oracle 0.4338 | soft oracle 0.4413 | tuned constant 0.3695
- hedging: soft beats hard on 9.6% of queries;
  tuned constant captures 54.8% of the positive hedge mass
- pick accuracy 44.1% (95% CI [40.3%, 48.0%])
- break-even p* 44.7% (95% CI [40.5%, 48.9%])
- paired bootstrap P(pick accuracy > p*) = 0.413

### nfcorpus
- canonical test queries: 323
- unified best uniform: hybrid @ 0.3368
  (strategy means: bm25 0.3111, dense 0.3186, hybrid 0.3368)
- hard oracle 0.3842 | soft oracle 0.3884 | tuned constant 0.3414
- hedging: soft beats hard on 14.2% of queries;
  tuned constant captures 42.2% of the positive hedge mass
- pick accuracy 57.6% (95% CI [52.3%, 62.9%])
- break-even p* 46.0% (95% CI [40.1%, 52.3%])
- paired bootstrap P(pick accuracy > p*) = 0.999
### pooled (971 queries)
- pick accuracy 48.6% (95% CI [45.5%, 51.8%])
- break-even p* 45.1% (95% CI [41.7%, 48.5%])
- paired bootstrap P(pick accuracy > p*) = 0.950
- soft oracle beats hard on 11.1% of queries (weighted);
  tuned constant captures 50.6% of positive hedge mass

```
Reviewer-stats verdict for the paper:
  pick accuracy 48.6% vs break-even 45.1%; P(acc > p*) = 0.95 over 10,000 paired resamples.
  -> cannot reject break-even operation: the CI overlaps p*.
  The tuned constant weight already captures 51% of the hedging
  mass that makes the soft oracle exceed the hard oracle.
```
