"""Reproducibility metadata collected for every experiment run.

Goal (roadmap §9): a previous experiment should be reproducible as closely as
practical. We record what we can observe automatically; anything missing
(e.g. git unavailable) is stored as null, never silently omitted.
"""

from __future__ import annotations

import importlib.metadata as importlib_metadata
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path

TRACKED_PACKAGES = (
    "ai-lab",
    "numpy",
    "pyyaml",
    "scipy",
    "pandas",
    "scikit-learn",
    "torch",
    "sentence-transformers",
    "pymupdf",
)


def git_commit(repo_root: str | Path | None = None) -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            cwd=repo_root or ".",
            timeout=5,
        )
        return out.stdout.strip() if out.returncode == 0 else None
    except Exception:
        return None


def git_dirty(repo_root: str | Path | None = None) -> bool | None:
    """True if the working tree has uncommitted changes (None if git unavailable)."""
    try:
        out = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True,
            text=True,
            cwd=repo_root or ".",
            timeout=5,
        )
        if out.returncode != 0:
            return None
        return bool(out.stdout.strip())
    except Exception:
        return None


def package_versions() -> dict:
    versions = {}
    for pkg in TRACKED_PACKAGES:
        try:
            versions[pkg] = importlib_metadata.version(pkg)
        except importlib_metadata.PackageNotFoundError:
            versions[pkg] = None
    return versions


def collect(seed: int, repo_root: str | Path | None = None) -> dict:
    return {
        "git_commit": git_commit(repo_root),
        "git_dirty": git_dirty(repo_root),
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "packages": package_versions(),
        "seed": seed,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
