"""Merge BEIR qrels splits into one qrels.tsv (all splits) plus per-split
files qrels-{train,dev,test}.tsv for train/test separation in router
experiments. The per-split files use the same schema as qrels.tsv.

Usage:
    python scripts/build_qrels.py datasets/downloads/fiqa datasets/downloads/fiqa/qrels.tsv
"""

import csv
import pathlib
import sys

SPLITS = ["train", "dev", "test"]


def read_split(p: pathlib.Path) -> list[tuple[str, str, str]]:
    with open(p, newline="", encoding="utf-8") as f:
        return [(r["query-id"], r["corpus-id"], r["score"])
                for r in csv.DictReader(f, delimiter="\t")]


def write_tsv(path: pathlib.Path, rows: list[tuple[str, str, str]]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write("query-id\tdoc-id\trelevance\n")
        for q, d, s in rows:
            f.write(f"{q}\t{d}\t{s}\n")


def main() -> None:
    src, out = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])
    all_rows: list[tuple[str, str, str]] = []
    for split in SPLITS:
        p = src / "qrels" / f"{split}.tsv"
        if not p.exists():
            continue
        rows = read_split(p)
        split_out = out.parent / f"qrels-{split}.tsv"
        write_tsv(split_out, rows)
        print("wrote", split_out, "-", len(rows), "judgments")
        all_rows.extend(rows)
    write_tsv(out, all_rows)
    print("wrote", out, "-", len(all_rows), "judgments (all splits)")


if __name__ == "__main__":
    main()
