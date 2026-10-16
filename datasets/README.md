# Datasets

Datasets are declared in `manifest.yaml` and referenced from experiment configs
by `name` + `version`. The runner loads anything in **BEIR format**:

```text
<path>/
├── corpus.jsonl    # {"_id": str, "text": str, "title": str?}
├── queries.jsonl   # {"_id": str, "text": str}
└── qrels.tsv       # query-id <TAB> doc-id <TAB> relevance
```

## Smoke test (no download)

```bash
python scripts/make_synthetic_dataset.py
```

Deterministic (seed 42), generated locally, tiny. Use only to verify the pipeline.

## Real benchmarks (BEIR)

FiQA and NFCorpus are the first research datasets (finance + science domains).
Download options:

**Option A — `ir-datasets` (recommended, one command per dataset):**

```bash
pip install ir-datasets
python - <<'PY'
import ir_datasets, json, pathlib
for name, out in [("beir/fiqa", "fiqa"), ("beir/nfcorpus", "nfcorpus")]:
    dst = pathlib.Path("datasets/downloads") / out
    dst.mkdir(parents=True, exist_ok=True)
    ds = ir_datasets.load(name)
    with open(dst / "corpus.jsonl", "w") as fc, open(dst / "queries.jsonl", "w") as fq:
        for d in ds.docs_iter(): fc.write(json.dumps({"_id": d.doc_id, "text": d.text, "title": getattr(d, "title", "")}) + "\n")
        for q in ds.queries_iter(): fq.write(json.dumps({"_id": q.query_id, "text": q.text}) + "\n")
    with open(dst / "qrels.tsv", "w") as fr:
        fr.write("query-id\tdoc-id\trelevance\n")
        for qid, docid, rel in ds.qrels_iter(): fr.write(f"{qid}\t{docid}\t{rel}\n")
    print("wrote", dst)
PY
```

**Option B — Hugging Face datasets:** the `BeIR/fiqa` and `BeIR/nfcorpus`
repos ship corpus/queries/qrels; convert to the layout above.

Downloads go to `datasets/downloads/` (gitignored — regenerate, don't commit).
For the property-routing study we need a third, tables/figures-heavy source;
candidates: arXiv papers (PDFs, profiled with `core/profiling`) and financial
annual reports. See `papers/property-routing/proposal.md` §Datasets.

## Multimodal documents

For Phase 9+ experiments, raw PDFs live under `datasets/downloads/<name>/pdfs/`
alongside a BEIR-style corpus whose `text` fields come from the document
profiler. Document properties (pages, text density, tables, figures) are stored
per document and become routing features — always tagged with
`profiler_version`.
