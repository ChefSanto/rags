#!/usr/bin/env python
"""Ingest a web page into the personal Vectorize index.

Fetch + article extraction runs locally (residential IP, browser UA), which
passes the bot protection that blocks datacenter fetches. Pages that still
refuse (login walls, heavy JS apps) error clearly -- download the PDF and use
the dashboard's Upload docs instead.

Usage:
    pip install trafilatura
    AILAB_APP_KEY='your-passphrase' python scripts/ingest_url.py URL [--name slug]
"""

import argparse
import json
import os
import re
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from ingest_documents import chunk_text  # reuse the same chunker

WORKER = "https://ai-lab-worker.antwibrand.workers.dev"


def fetch_bytes(url: str) -> bytes:
    import urllib.request

    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0 Safari/537.36"})
    with urllib.request.urlopen(req, timeout=90) as r:
        return r.read()


def fetch_text(url: str) -> str:
    import trafilatura

    try:
        raw = fetch_bytes(url)
    except Exception as e:
        raise SystemExit(f"fetch failed: {e}\nfallback: open the page, save as PDF, use the dashboard Upload docs.")
    if raw[:4] == b"%PDF":
        import fitz

        doc = fitz.open(stream=raw, filetype="pdf")
        text = "\n".join(page.get_text() for page in doc)
        doc.close()
        if len(text.strip()) < 200:
            raise SystemExit("PDF has no text layer (scanned) - OCR needed.")
        return text
    downloaded = raw.decode("utf-8", errors="replace")
    low = downloaded.lstrip().lower()
    if "<html" not in low[:500] and not low.startswith("<"):
        return downloaded.strip()          # plain text / markdown (e.g. raw.githubusercontent)
    text = trafilatura.extract(downloaded, include_comments=False, include_tables=True)
    if not text or len(text.strip()) < 200:
        raise SystemExit("page fetched but no readable article text found (likely a JS-only app).")
    return text


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("url")
    ap.add_argument("--name", default=None)
    args = ap.parse_args()

    key = os.environ.get("AILAB_APP_KEY", "")
    if not key:
        raise SystemExit("set AILAB_APP_KEY to your dashboard passphrase")

    slug = args.name or re.sub(r"[^\w.-]+", "-", args.url.split("//")[-1])[:60].strip("-")
    print(f"fetching {args.url} ...")
    text = fetch_text(args.url)
    print(f"extracted {len(text)} chars")

    chunks = [{"id": f"{slug}-u0-c{j}"[:200], "text": f"[{slug} web] {c}"}
              for j, c in enumerate(chunk_text(text))]
    print(f"uploading {len(chunks)} chunks ...")
    for i in range(0, len(chunks), 500):
        part = chunks[i:i + 500]
        req = urllib.request.Request(
            WORKER + "/ingest-text",
            data=json.dumps({"chunks": part, "name": slug, "total_chunks": len(chunks)}).encode(),
            headers={"Content-Type": "application/json", "X-Ingest-Key": key,
                     "User-Agent": "curl/8.5.0"},  # Cloudflare bot-management rejects urllib UAs
        )
        try:
            with urllib.request.urlopen(req, timeout=120) as r:
                r.read()
        except urllib.error.HTTPError as e:
            raise SystemExit(f"worker error {e.code}: {e.read().decode()[:300]}")
    print(f"done — '{slug}' is in your personal index; ask about it in the dashboard.")


if __name__ == "__main__":
    main()
