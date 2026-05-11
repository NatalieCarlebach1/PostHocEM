"""
T1.6 — aggregate multi-seed PEM results into a single CSV for the paper.

Scans result/pem_seed_<config>_<seed>/ for every (config, seed) pair, reads
metrics.json (or the per-epoch log if metrics.json is missing), and writes:

  result/multiseed_summary.csv     — one row per (config, seed)
  result/multiseed_aggregate.csv   — one row per config with mean/std/n

The aggregate file is what Section 5's main results table reads.
"""

import csv
import json
from pathlib import Path
from collections import defaultdict
import statistics
import sys

ROOT = Path(__file__).parent.parent

CONFIGS = {
    'pancreas20': {
        'baseline_dice': 82.89,
        'pattern': 'result/pem_seed_pancreas_{seed}',
    },
    'la5': {
        'baseline_dice': 87.32,
        'pattern': 'result/pem_seed_la5_{seed}',
    },
    'la10': {
        'baseline_dice': 89.40,
        'pattern': 'result/pem_seed_la10_{seed}',
    },
}

SEEDS = [2020, 42, 123, 7, 11]


def read_metrics(d: Path):
    """Read final metrics from a PEM result directory. Tries metrics.json,
    falls back to parsing the last [Eval] line from train.log."""
    j = d / 'metrics.json'
    if j.exists():
        try:
            data = json.loads(j.read_text())
            return {
                'dice':    float(data.get('final_dice',    data.get('dice',    'nan'))),
                'jaccard': float(data.get('final_jaccard', data.get('jaccard', 'nan'))),
                'hd95':    float(data.get('final_hd95',    data.get('hd95',    'nan'))),
                'asd':     float(data.get('final_asd',     data.get('asd',     'nan'))),
            }
        except Exception:
            pass

    log = d / 'train.log'
    if not log.exists():
        return None
    # Parse final [Eval] line, format:
    #   [Eval N] Dice=0.xxxx  Jc=0.xxxx  HD95=xx.xx  ASD=xx.xx
    final = None
    for line in log.read_text().splitlines():
        if 'Dice=' in line and 'HD95=' in line:
            final = line
    if final is None:
        return None
    try:
        parts = {}
        for kv in final.replace(',', ' ').split():
            if '=' in kv:
                k, v = kv.split('=', 1)
                parts[k.strip()] = v.strip()
        return {
            'dice':    float(parts.get('Dice',    'nan')) * 100,
            'jaccard': float(parts.get('Jc',      'nan')) * 100,
            'hd95':    float(parts.get('HD95',    'nan')),
            'asd':     float(parts.get('ASD',     'nan')),
        }
    except Exception:
        return None


def main():
    per_run_rows = []
    per_config = defaultdict(lambda: defaultdict(list))

    for config_name, cfg in CONFIGS.items():
        for seed in SEEDS:
            d = ROOT / cfg['pattern'].format(seed=seed)
            if not d.exists():
                print(f"  miss: {d}")
                continue
            m = read_metrics(d)
            if m is None:
                print(f"  no metrics: {d}")
                continue
            per_run_rows.append({
                'config': config_name,
                'seed':   seed,
                'dice':   m['dice'],
                'jaccard': m['jaccard'],
                'hd95':   m['hd95'],
                'asd':    m['asd'],
                'delta_dice': m['dice'] - cfg['baseline_dice'],
            })
            for k in ('dice', 'jaccard', 'hd95', 'asd'):
                per_config[config_name][k].append(m[k])

    out1 = ROOT / 'result/multiseed_summary.csv'
    out1.parent.mkdir(parents=True, exist_ok=True)
    with open(out1, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['config', 'seed',
                                          'dice', 'jaccard', 'hd95', 'asd',
                                          'delta_dice'])
        w.writeheader()
        for r in per_run_rows:
            w.writerow(r)
    print(f"\nWrote {out1}")

    out2 = ROOT / 'result/multiseed_aggregate.csv'
    with open(out2, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['config', 'n_seeds',
                    'dice_mean', 'dice_std',
                    'jaccard_mean', 'jaccard_std',
                    'hd95_mean', 'hd95_std',
                    'asd_mean', 'asd_std'])
        for config_name, vals in per_config.items():
            n = len(vals['dice'])
            row = [config_name, n]
            for k in ('dice', 'jaccard', 'hd95', 'asd'):
                v = vals[k]
                if len(v) >= 2:
                    row.append(f'{statistics.mean(v):.4f}')
                    row.append(f'{statistics.stdev(v):.4f}')
                elif len(v) == 1:
                    row.append(f'{v[0]:.4f}')
                    row.append('nan')
                else:
                    row.append('nan')
                    row.append('nan')
            w.writerow(row)
    print(f"Wrote {out2}")

    # Pretty-print to stdout
    print(f"\n{'config':<12} {'n':>3} {'dice mean ± std':>18} {'Δ vs base':>10}")
    for config_name, vals in per_config.items():
        n = len(vals['dice'])
        if n < 1:
            continue
        mean = statistics.mean(vals['dice'])
        std = statistics.stdev(vals['dice']) if n >= 2 else float('nan')
        base = CONFIGS[config_name]['baseline_dice']
        print(f"{config_name:<12} {n:>3} {mean:>9.2f} ± {std:>5.2f}  "
              f"{mean - base:>+6.2f}")


if __name__ == '__main__':
    main()
