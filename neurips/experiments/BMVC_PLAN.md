# BMVC 2026 experiments — implementation plan

Branch: `feature/bmvc-submission`. Target: full paper submission, 29 May 2026.
Budget: 19 days from 2026-05-10.

## Status of each experiment

| ID | What | Status | Owner script | Output |
|---|---|---|---|---|
| T1.5 | TENT head-to-head | **infra ready** | `train_posthoc_em.py --tent_mode` | `result/tent_<dataset>_<lab>/` |
| T1.4 | $\|\partial H/\partial z\|$ distribution | **infra ready** | `scripts/gradient_distribution.py` | `result/gradient_dist/<ckpt>.json` |
| T1.6 | 5-seed PEM | **infra ready** | `scripts/run_multiseed.sh` (extended) | `result/pem_seed_*/` |
| T1.3 | $\rho_f$-trajectory + PEM gain curve | **infra ready** | `train_bcp_baseline.py --save_every` + `scripts/trajectory_pem_sweep.py` | `result/bcp_trajectory/`, `result/pem_trajectory_sweep/` |
| T1.7 | Figures (gain-vs-$\rho_f$, learning curves, gradient maps) | **infra ready** | `figures/fig_*.py` | `neurips/paper/figures/*.pdf` |
| T2.1 | PEM on UA-MT (one non-BCP method) | **infra scaffolded** | `train_uamt_baseline.py` + `scripts/run_pem_on_uamt.sh` | `result/uamt_*/`, `result/pem_on_uamt_*/` |

## Execution order

The order below minimizes wall-clock by parallelizing where the GPU is idle.

1. **Day 1 (today): kick off long-running training jobs first.**
   - `bash scripts/run_bcp_trajectory.sh` — BCP retrain with checkpoint-every-1000-steps. Long-running (~8 hours), launches in background. Required for T1.3.
   - `bash scripts/run_uamt_pancreas.sh` — UA-MT base training. Long-running (~6 hours), background. Required for T2.1.

2. **Day 1, parallel: cheap measurements that don't need new training.**
   - `python scripts/gradient_distribution.py` (T1.4). ~10 min.
   - `bash scripts/run_tent_all.sh` (T1.5). ~10 min, all three configurations.

3. **Day 2: 5-seed extension and figures from existing data.**
   - `bash scripts/run_multiseed.sh` for new seeds 7, 11 (T1.6).
   - `python figures/fig_pem_learning_curve.py` (from existing ablation data).
   - `python figures/fig_gradient_maps.py` (from T1.4 output).

4. **Day 2–3: BCP trajectory becomes available.**
   - `python scripts/trajectory_pem_sweep.py` — applies PEM to each saved checkpoint. ~3 hours.
   - `python figures/fig_trajectory_curve.py` — the key figure.

5. **Day 4–7: UA-MT base done, apply PEM.**
   - `bash scripts/run_pem_on_uamt.sh` — PEM on the UA-MT checkpoint.
   - Update tables with UA-MT row.

6. **Day 8–18: writing.**
   - Section 05 (Experiments) — write properly with real numbers and figures.
   - Section 06 (Ablations) — port + extend.
   - Section 99 (Appendix) — full ablation tables, hyperparameter sensitivity, additional $\rho_f$ measurements.
   - Anonymization pass.

7. **Day 19: format check, dry-run rebuild of Overleaf zip, submit.**

## Compute estimates (single RTX 4070 Ti SUPER)

| Job | Time |
|---|---|
| BCP retrain Pancreas (260 epochs) | ~8 hours |
| UA-MT Pancreas (200 epochs) | ~5 hours |
| One PEM run (E=2 Pancreas) | ~1 min |
| One PEM run (E=5 LA) | ~3 min |
| Gradient distribution (one checkpoint, one volume) | ~5 sec |
| Trajectory sweep (20 checkpoints × 1 PEM run each) | ~30 min |
| Multi-seed extension (2 new seeds × 3 configs) | ~20 min |
| TENT (3 configs × 1 run) | ~10 min |
| **Total new compute** | **~17 GPU-hours** |

Comfortable for the 19-day budget.

## Risk register

| Risk | Mitigation |
|---|---|
| UA-MT reproduction doesn't match published Dice | Use released checkpoint if available, or report our reproduction with explicit note. Drop T2.1 if reproduction is far off — Tier 1 alone is still a strong paper. |
| Trajectory experiment shows non-monotone $\rho_f$ growth | Interesting result either way. If it's noisy, smooth over multiple seeds. |
| TENT closely matches PEM (the closest competitor) | Frame as "PEM ≈ post-convergence TENT on training data" + emphasize the regime framing and the $\rho_f$ diagnostic that TENT doesn't provide. |
| Gradient distribution doesn't show clean shell concentration | Re-examine the V-Net's logits; check for logit clipping. If the result genuinely contradicts the theory, address it in Section 4. |
