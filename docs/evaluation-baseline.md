# Initial retrieval baseline (2026-09-12)

This records the first successful runs of the 36-query development seed described
in [Retrieval evaluation](evaluation.md). It is a within-suite comparison, not a
held-out evaluation or a measurement of agent task success and token savings.
The suite and sources are preserved at
[5988de0](https://github.com/scarletkc/vexor/tree/5988de0bceb8c903405adefac99be05a9e2fe762).
No ranking or retrieval behavior was changed for these runs.

## Remote embedding model

Configuration: `BAAI/bge-m3` through the configured `custom` provider, top 10,
2,000 characters per result, 8,000 characters total, three measured repetitions
after a warmup for every query/arm. Each arm therefore has 36 queries and 108
observations. Timings are warm Python API calls on Windows and
should not be treated as general hardware-independent performance claims.

```bash
uv run python scripts/eval_retrieval.py --output .cache/evaluation/bge-m3-baseline.json
```

| Ranking | File MRR@10 | Evidence MRR@10 | All evidence hit@10 | Mean returned chars | Source overlap | Warm p50 / p95 (ms) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Dense (`off`) | 0.948 | 0.756 | 86.1% | 5,898 | 9.8% | 19.5 / 24.7 |
| BM25 rerank | 0.909 | 0.716 | 83.3% | 5,885 | 10.3% | 21.5 / 26.8 |
| Hybrid | 0.921 | 0.797 | 88.9% | 5,897 | 10.6% | 30.3 / 37.2 |

Per-corpus evidence MRR@10:

| Corpus | Dense | BM25 rerank | Hybrid |
| --- | ---: | ---: | ---: |
| Code | 0.750 | 0.736 | 0.833 |
| Documentation | 0.558 | 0.454 | 0.600 |
| Authored Chinese records | 0.958 | 0.958 | 0.958 |

The difference between file MRR and evidence MRR is the main reason to retain
both metrics. Dense search frequently returns the expected file without supplying
the annotated answer evidence. Hybrid improves evidence MRR on this seed even
though its file MRR is lower. This does not contradict the old 30-query
`eval_hybrid.py` result: the corpus, query set, and relevance criterion differ.

All returned hits without a body in this run reported `budget_exhausted`: 249,
246, and 258 results for dense, BM25, and hybrid respectively. These are counts
across three repetitions, not distinct failed queries or file-read errors.
In each arm, 23 of 36 queries have at least one result without a body.
This does not prove the content budget caused every evidence miss; ranking and
chunk boundaries also determine which text enters the budget.

The Chinese corpus is short and comparatively easy: all three arms cover all
annotated evidence at top 10. That ceiling is a limitation of the seed, not
evidence that the strategies perform equally well on real Chinese collections.

The [compact result](../benchmarks/retrieval/baselines/bge-m3.json) records exact
metrics, package versions, safe configuration, fingerprints, and the SHA-256 of
the full observation report. The full JSON is generated in the ignored
`.cache/evaluation/` directory; it includes all returned source text for auditing.
Private provider endpoints and keys are absent from both reports.

## Local embedding model

The second run uses `intfloat/multilingual-e5-small` with the local provider and
the same suite, corpus snapshots, ranking arms, content budgets, and repetitions.
FastEmbed 0.8.0 and ONNX Runtime 1.29.0 were installed from the repository lockfile;
the model was already cached locally. Its
[compact result](../benchmarks/retrieval/baselines/e5-small.json) includes the
fingerprints and a digest of the full observation report.

```bash
uv run --extra local python scripts/eval_retrieval.py --provider local \
  --model intfloat/multilingual-e5-small --output .cache/evaluation/e5-small-baseline.json
```

| Ranking | File MRR@10 | Evidence MRR@10 | All evidence hit@10 | Mean returned chars | Source overlap | Warm p50 / p95 (ms) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Dense (`off`) | 0.912 | 0.746 | 91.7% | 5,905 | 10.2% | 975.8 / 1,067.2 |
| BM25 rerank | 0.864 | 0.691 | 80.6% | 5,867 | 9.1% | 969.8 / 1,058.8 |
| Hybrid | 0.940 | 0.834 | 91.7% | 5,896 | 10.5% | 975.2 / 1,072.0 |

Per-corpus evidence MRR@10:

| Corpus | Dense | BM25 rerank | Hybrid |
| --- | ---: | ---: | ---: |
| Code | 0.764 | 0.720 | 0.944 |
| Documentation | 0.558 | 0.438 | 0.642 |
| Authored Chinese records | 0.917 | 0.917 | 0.917 |

Hybrid improves where the evidence appears for the local model, while its
all-evidence coverage remains equal to dense at top 10. BM25 reranking loses on
both models in this seed. Neither observation establishes a general model or
strategy ranking; the short Chinese corpus again has complete top-10 coverage.

Local timings include the current API's per-call searcher/backend construction,
even with query embeddings cached. They are not isolated ONNX inference or
matrix-product measurements. The host was not reserved exclusively for the run;
offline tests also ran during part of it. Treat these latency samples as
diagnostic observations, not controlled cross-model speed comparisons.

## Next experiments

Freeze these labels while comparing a change. Inspect individual misses before
selecting an implementation: test whether a better candidate order, a different
chunk boundary, or a different content budget recovers the missing evidence.
Report the associated increase in returned content and time, not just hit rate.

Add longer Chinese sources and real held-out projects with independent relevance
review before using the benchmark to select a default ranking. Use the
[agent evaluation protocol](agent-evaluation.md) before claiming token savings
or fewer follow-up file reads.
