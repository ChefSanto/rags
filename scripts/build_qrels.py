import csv, pathlib, sys

src, out = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])
rows = []
for split in ["train", "dev", "test"]:
    p = src / "qrels" / f"{split}.tsv"
    if not p.exists():
        continue
    with open(p, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f, delimiter="\t"):
            rows.append((r["query-id"], r["corpus-id"], r["score"]))
with open(out, "w", encoding="utf-8") as f:
    f.write("query-id\tdoc-id\trelevance\n")
    for q, d, s in rows:
        f.write(f"{q}\t{d}\t{s}\n")
print("wrote", out, "-", len(rows), "judgments")
