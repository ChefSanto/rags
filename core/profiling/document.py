"""Document profiler v0 — measurable PDF properties for routing experiments.

Scope deliberately small (roadmap §15): pages, words, text density, figures,
tables, image density, scanned-page estimate. Quality of extraction is less
important than VERSIONING: every profile carries profiler_version so routing
experiments stay reproducible when extraction improves.

Requires: pip install -e ".[pdf]"  (pymupdf)
"""

from __future__ import annotations

from pathlib import Path

PROFILER_VERSION = "0.1"


def profile_pdf(path: str | Path) -> dict:
    try:
        import pymupdf
    except ImportError as e:
        raise ImportError(
            "Document profiler requires pymupdf: pip install -e '.[pdf]'"
        ) from e

    doc = pymupdf.open(path)
    pages = len(doc)

    words = 0
    text_chars = 0
    page_area = 0.0
    images = 0
    tables = 0
    scanned_pages = 0

    for page in doc:
        page_words = page.get_text("words")
        words += len(page_words)
        text = page.get_text()
        text_chars += len(text.strip())
        page_area += max(page.rect.width * page.rect.height, 1.0)
        images += len(page.get_images())
        if len(page_words) < 5:
            scanned_pages += 1
        try:
            tables += len(page.find_tables().tables)
        except Exception:
            pass  # table detection is best-effort in v0

    doc.close()

    return {
        "profiler_version": PROFILER_VERSION,
        "pages": pages,
        "words": words,
        "text_density": round(text_chars / page_area, 6) if page_area else 0.0,
        "tables": tables,
        "figures": images,  # v0 proxy: embedded images
        "image_density": round(images / pages, 4) if pages else 0.0,
        "scanned_pages": scanned_pages,
        "scanned_ratio": round(scanned_pages / pages, 4) if pages else 0.0,
    }
