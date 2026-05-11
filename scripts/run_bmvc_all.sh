#!/bin/bash
# BMVC orchestrator — runs every experiment in dependency order with
# graceful skip if outputs already exist. Re-runnable. Honest about
# what it has and hasn't done.
#
# Usage:
#   bash scripts/run_bmvc_all.sh                  # run everything
#   bash scripts/run_bmvc_all.sh --dry-run        # print what would run
#   bash scripts/run_bmvc_all.sh --tier 1         # only Tier 1
#   bash scripts/run_bmvc_all.sh --skip-long      # skip BCP retrain + UA-MT
#
# Designed to be safe to interrupt and restart. Each step checks for
# its own output artifact and skips if present.
set -e
cd /home/tals/Documents/PostHocEM
PY=/home/tals/miniconda3/envs/ns-sam3/bin/python

DRY=0
SKIP_LONG=0
TIER=2  # 1 or 2

while [[ $# -gt 0 ]]; do
  case $1 in
    --dry-run)   DRY=1; shift ;;
    --skip-long) SKIP_LONG=1; shift ;;
    --tier)      TIER=$2; shift 2 ;;
    *)           echo "unknown flag $1" >&2; exit 1 ;;
  esac
done

step() {
  local name="$1"; local artifact="$2"; shift 2
  if [[ -e "$artifact" ]]; then
    echo "[$name] SKIP — artifact exists at $artifact"
    return
  fi
  if [[ $DRY -eq 1 ]]; then
    echo "[$name] WOULD RUN: $*"
    return
  fi
  echo "============================================================"
  echo "[$name] running:"
  echo "  $*"
  echo "============================================================"
  "$@"
}

echo "BMVC orchestrator starting at $(date)"
echo "tier=$TIER  skip_long=$SKIP_LONG  dry_run=$DRY"

# ── TIER 1 ─────────────────────────────────────────────────────────────────

# T1.5 — TENT head-to-head (fast, ~10 min total)
step "T1.5 TENT (Pancreas)"    "result/tent_pancreas20_s2020/best_model.pth"      bash scripts/run_tent_all.sh

# T1.4 — gradient distribution (fast, ~5 min)
step "T1.4 gradient dist"      "result/gradient_dist/summary.json"                $PY scripts/gradient_distribution.py

# T1.6 — 5-seed extension (only new seeds run; old seeds reused)
step "T1.6 5-seed"             "result/pem_seed_pancreas_11/train.log"            bash scripts/run_multiseed.sh

# Aggregate multi-seed
step "T1.6 aggregate"          "result/multiseed_aggregate.csv"                   $PY scripts/aggregate_multiseed.py

# T1.3 — BCP trajectory retrain (LONG, ~8h). Skipped via --skip-long.
if [[ $SKIP_LONG -eq 0 ]]; then
  step "T1.3 BCP trajectory"   "result/bcp_trajectory/trajectory/epoch_260.pth"   bash scripts/run_bcp_trajectory.sh
  step "T1.3 PEM sweep"        "result/pem_trajectory_sweep/trajectory.csv"       $PY scripts/trajectory_pem_sweep.py
fi

# T1.7 — figures (fast, runs even with partial data; each script skips gracefully)
echo "[T1.7 figures]"
if [[ $DRY -eq 0 ]]; then
  $PY neurips/figures/fig_gradient_maps.py    || true
  $PY neurips/figures/fig_pem_learning_curve.py || true
  $PY neurips/figures/fig_trajectory_curve.py || true
fi

# ── TIER 2 ─────────────────────────────────────────────────────────────────

if [[ $TIER -ge 2 && $SKIP_LONG -eq 0 ]]; then
  step "T2.1 UA-MT base"       "result/uamt_pancreas20/best_model.pth"            bash scripts/run_uamt_pancreas.sh
  step "T2.1 PEM on UA-MT"     "result/pem_on_uamt_pancreas_2020/train.log"       bash scripts/run_pem_on_uamt.sh
fi

# ── Significance + entropy diagnostics that already exist ──────────────────
step "paired significance"     "result/paired_significance.csv"                   $PY scripts/paired_significance.py
step "entropy_stats"           "result/entropy_stats.csv"                         $PY scripts/entropy_stats.py

echo "============================================================"
echo "BMVC orchestrator finished at $(date)"
echo "Inspect:"
echo "  result/multiseed_aggregate.csv"
echo "  result/gradient_dist/summary.json"
echo "  result/pem_trajectory_sweep/trajectory.csv (if T1.3 ran)"
echo "  neurips/paper/figures/*.pdf"
