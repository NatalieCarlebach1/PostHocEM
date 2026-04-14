#!/usr/bin/env bash
# Rebuild the self-contained Overleaf upload bundle from neurips/paper/.
#
# Usage:
#   bash neurips/scripts/build_overleaf_zip.sh
#
# Output: neurips/pem-neurips-overleaf.zip (gitignored — regenerate on demand).
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PAPER_DIR="$REPO_ROOT/neurips/paper"
STAGING="$(mktemp -d -t pem-overleaf-XXXXXX)/pem-neurips-overleaf"
OUT="$REPO_ROOT/neurips/pem-neurips-overleaf.zip"

if [[ ! -d "$PAPER_DIR" ]]; then
  echo "error: $PAPER_DIR not found" >&2
  exit 1
fi

mkdir -p "$STAGING"
rsync -a \
  --exclude='build/' \
  --exclude='*.aux' --exclude='*.log' --exclude='*.out' \
  --exclude='*.bbl' --exclude='*.blg' --exclude='*.toc' \
  --exclude='*.synctex.gz' --exclude='*.fls' --exclude='*.fdb_latexmk' \
  "$PAPER_DIR/" "$STAGING/"

cat > "$STAGING/README-OVERLEAF.md" <<'EOF'
# PEM NeurIPS paper — Overleaf import bundle

This bundle was built by `neurips/scripts/build_overleaf_zip.sh`.

## First-time upload
1. https://overleaf.com → New Project → **Upload Project** → drop this zip.
2. Overleaf auto-detects `main.tex`. It compiles in review (anonymized) mode.

## Updating an existing Overleaf project
If you already uploaded once and want to push a new iteration:
  **Option A (manual):** delete the project on Overleaf and re-upload the new zip.
  **Option B (git sync, Overleaf Premium):** use `git push` to the project's
  git remote instead of re-uploading zips. See `neurips/WORKFLOW.md` for
  instructions.

## What's in here
- `main.tex` — NeurIPS 2024 review mode (`\usepackage{neurips_2024}`)
- `neurips_2024.sty` — NeurIPS 2024 style file (bundled)
- `sections/00_abstract.tex .. 07_discussion.tex` — paper body (`\input`'d)
- `references.bib` — bibliography (natbib, loaded by the NeurIPS style)
- `macros.tex` — shared LaTeX macros

## Switching submission mode
Edit `main.tex`:
- Review / anonymized:      `\usepackage{neurips_2024}`           (default)
- Camera-ready / final:     `\usepackage[final]{neurips_2024}`
- Preprint / arXiv:         `\usepackage[preprint]{neurips_2024}`
EOF

(
  cd "$(dirname "$STAGING")"
  zip -qr "$OUT" "$(basename "$STAGING")"
)

rm -rf "$(dirname "$STAGING")"

echo "built $OUT ($(du -h "$OUT" | cut -f1))"
echo "upload to https://overleaf.com → New Project → Upload Project"
