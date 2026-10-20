#!/usr/bin/env python
"""Ingest local documents (PDF / PPTX) into the RAG pipeline.

Extracts text, chunks it (~1000 chars with overlap), and writes a
corpus-style JSONL that scripts/upload_corpus.mjs can upload into the
personal Vectorize index:

    pip install -e ".[pdf]"                      # pymupdf + python-pptx
    python scripts/ingest_documents.py report.pdf slides.pptx \
        --out artifacts/personal/chunks.jsonl
    VECTORIZE_INDEX=ai-lab-personal node worker/scripts/upload_corpus.mjs \
        artifacts/personal/chunks.jsonl

PDF  -> pymupdf (page text; chunk ids file-p<page>-c<chunk>)
PPTX -> python-pptx (slide text; chunk ids file-s<slide>-c<chunk>)
Legacy .ppt (binary) is NOT supported by python-pptx — convert to .pptx first
(e.g. open in Keynote/PowerPoint and export).

Chunk metadata includes the source filename so answers can cite origins later.
"""

import argparse
import json
import re
import sys
from pathlib import Path

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200


def chunk_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Split text into roughly `size`-char chunks on whitespace boundaries."""
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + size, len(text))
        if end < len(text):
            boundary = text.rfind(" ", start + size // 2, end)
            if boundary > start:
                end = boundary
        chunks.append(text[start:end].strip())
        if end == len(text):
            break
        start = end - overlap
    return [c for c in chunks if len(c) > 40]


def extract_pdf(path: Path) -> list[tuple[str, str]]:
    import pymupdf

    out = []
    doc = pymupdf.open(path)
    for i, page in enumerate(doc):
        out.append((f"p{i}", page.get_text()))
    doc.close()
    return out


def extract_pptx(path: Path) -> list[tuple[str, str]]:
    from pptx import Presentation

    prs = Presentation(path)
    out = []
    for i, slide in enumerate(prs.slides):
        parts = []
        for shape in slide.shapes:
            if shape.has_text_frame:
                parts.append(shape.text_frame.text)
            if shape.has_table:
                for row in shape.table.rows:
                    parts.append(" | ".join(cell.text for cell in row.cells))
        out.append((f"s{i}", "\n".join(parts)))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("files", nargs="+", help="PDF and/or PPTX files")
    ap.add_argument("--out", required=True, help="output JSONL path")
    ap.add_argument("--chunk-size", type=int, default=CHUNK_SIZE)
    ap.add_argument("--overlap", type=int, default=CHUNK_OVERLAP)
    args = ap.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    n_chunks = 0
    with open(out_path, "w", encoding="utf-8") as f:
        for file in args.files:
            p = Path(file)
            suffix = p.suffix.lower()
            if suffix == ".pdf":
                segments = extract_pdf(p)
            elif suffix in (".pptx", ".ppt"):
                if suffix == ".ppt":
                    print(f"skipping {p.name}: legacy .ppt not supported; convert to .pptx",
                          file=sys.stderr)
                    continue
                segments = extract_pptx(p)
            else:
                print(f"skipping {p.name}: unsupported type {suffix}", file=sys.stderr)
                continue

            for seg_id, text in segments:
                for j, chunk in enumerate(chunk_text(text, args.chunk_size, args.overlap)):
                    doc_id = f"{p.stem}-{seg_id}-c{j}"[:200]
                    f.write(json.dumps({"_id": doc_id,
                                        "text": f"[{p.name} {seg_id}] {chunk}"}) + "\n")
                    n_chunks += 1
            print(f"{p.name}: {len(segments)} segments -> chunks so far {n_chunks}")

    print(f"done: {n_chunks} chunks in {out_path}")
    print("upload with: VECTORIZE_INDEX=ai-lab-personal node worker/scripts/upload_corpus.mjs " + str(out_path))


if __name__ == "__main__":
    main()
