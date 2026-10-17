#!/usr/bin/env bash
# Idempotent Kaggle session bootstrap: repo + datasets + baseline artifacts.
# Every step skips itself when already satisfied — safe to run every session.
#
# Cell usage (works on fresh AND persistent sessions):
#   cd /kaggle/working && ([ -d rags ] || git clone https://github.com/ChefSanto/rags.git) && bash rags/scripts/kaggle_bootstrap.sh
set -euo pipefail
cd /kaggle/working

[ -d rags ] || git clone https://github.com/ChefSanto/rags.git
cd rags
git pull
pip install -q -e ".[ml]"

mkdir -p datasets/downloads

fetch_beir() {  # $1 = dataset name (fiqa | nfcorpus)
  if [ ! -f "datasets/downloads/$1/corpus.jsonl" ]; then
    echo ">>> downloading $1"
    curl -sLO "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/$1.zip"
    unzip -o -q "$1.zip" -d datasets/downloads
    rm "$1.zip"
    python scripts/build_qrels.py "datasets/downloads/$1" "datasets/downloads/$1/qrels.tsv"
  fi
}
fetch_beir fiqa
fetch_beir nfcorpus

# baselines: skipped automatically when run-artifacts are present (git or local)
python scripts/ensure_baselines.py

echo ">>> bootstrap complete: repo + datasets + baseline artifacts ready"
