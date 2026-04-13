# NeurIPS experiments

One YAML config + one launcher per run. Code is reused from the repo root
(`networks/`, `utils/`, `dataloaders/`, `train_*.py`) — nothing is duplicated here.

## Adding a new experiment

1. Copy `configs/base_pem.yaml` to `configs/<new_exp_id>.yaml` and edit.
2. Set `exp_id` to match the filename.
3. Run from repo root: `bash neurips/experiments/launch/run_single.sh neurips/experiments/configs/<new_exp_id>.yaml`
4. Results land in `../results/<exp_id>/` (relative to this folder = `neurips/results/<exp_id>/`).

## Naming

```
<dataset>_<labelfrac>_<base>_<method>_<variant>_<seed>
```

Examples: `pancreas_20_bcp_pem_tau099_s2020`, `la_5_bcp_plft_s42`,
`acdc_10_most_pem_full_s2020`, `brats_5_dycon_pem_tau095_s42`.

## Sweeps

Sweep configs go under `configs/sweeps/`. Launched via `launch/run_sweep.sh`,
results land under `../results/sweeps/<sweep_id>/` with a `summary.csv`.
