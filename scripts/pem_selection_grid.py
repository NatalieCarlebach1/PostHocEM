"""
PEM hyperparameter selection grid under a labeled-validation protocol.
======================================================================

Motivation
----------
PEM optimizes an entropy loss on the UNLABELED split only. The labeled
training volumes (4 for LA 5%, 8 for LA 10%, 12 for Pancreas 20%) are
therefore never seen by the objective, and form a legitimate held-out
validation set for choosing (learning rate, confidence threshold tau).
This is not cross-validation: no fold is ever trained on, because PEM
does not train on labels at all.

This grid runs every configuration
  - at a FIXED budget (--fixed_budget: no early stopping, no collapse break),
    so configurations are compared at the same number of epochs rather than at
    a per-configuration stopping point, and
  - logging BOTH the labeled-validation metrics (the selection signal) and the
    test metrics (reporting only),
so that the selection and the reporting can be kept strictly separate.

Usage
-----
    python scripts/pem_selection_grid.py --dataset la        # LA 5% + 10%
    python scripts/pem_selection_grid.py --dataset pancreas  # Pancreas 20%
    python scripts/pem_selection_grid.py --dataset la --dry_run
"""

import argparse
import csv
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path


ROOT = Path('/home/tals/Documents/PostHocEM')

# ─── Per-dataset configuration ───────────────────────────────────────────────
# `budget` is the a priori epoch budget E used in the paper for that dataset.
# `lrs` spans the regime in which any update happens: below 1e-6 the update is
# provably nil (established by the earlier sweep) and the top entry documents
# the relocation ceiling.
DATASETS = {
    'la': dict(
        fractions=[5, 10],
        lrs=[5e-6, 1e-5],
        ceiling_lr=5e-5,
        budget=5,
        patch_size='112,112,80',
        stride_xy=18,
        data_root='data/la_h5',
        la_data_root='data/la_h5/2018LA_Seg_Training Set',
        splits_dir='splits/la',
        ckpt=lambda frac: ROOT / 'result' / 'bcp_pretrained' / f'LA_{frac}.pth',
    ),
    'pancreas': dict(
        fractions=[20],
        lrs=[1e-5, 5e-5],
        ceiling_lr=1e-4,
        budget=2,
        patch_size='96',
        stride_xy=16,
        data_root='data/pancreas_h5',
        la_data_root='data/la_h5',
        splits_dir='splits/pancreas',
        ckpt=lambda frac: ROOT / 'result' / 'bcp_baseline_v2' / 'best_model.pth',
    ),
}

THRESHOLDS = [0.9, 0.95, 0.99, 0.999]
NUM_CLASSES = 2
SEED = 2020


def run_name(dataset, fraction, lr, mode, threshold):
    lr_str = f'{lr:.0e}'.replace('-0', '-')
    stem = f'pem_sel_{dataset}_{fraction}pct'
    if mode == 'confident':
        return f'{stem}_conf_t{threshold}_lr{lr_str}'
    return f'{stem}_full_lr{lr_str}'


def parse_metrics(csv_path, budget):
    """Return the row at the fixed budget plus the baseline row, or None.

    A run counts as complete only if it reached `budget` epochs AND carries the
    labeled-validation columns; anything else is re-run.
    """
    if not csv_path.exists():
        return None
    with open(csv_path) as f:
        rows = list(csv.DictReader(f))
    if not rows or 'val_dice' not in rows[0]:
        return None
    by_epoch = {int(r['epoch']): r for r in rows}
    if budget not in by_epoch:
        return None
    return by_epoch[0], by_epoch[budget]


def run_one(dataset, cfg, fraction, lr, mode, threshold, dry_run=False):
    name = run_name(dataset, fraction, lr, mode, threshold)
    save_dir = ROOT / 'result' / name
    budget = cfg['budget']

    existing = parse_metrics(save_dir / 'metrics.csv', budget)
    if existing is not None:
        return name, existing, 'cached'

    checkpoint = cfg['ckpt'](fraction)
    if not checkpoint.exists():
        return name, None, f'missing checkpoint: {checkpoint}'

    cmd = [
        sys.executable, 'train_posthoc_em.py',
        '--dataset', dataset,
        '--checkpoint', str(checkpoint),
        '--data_root', cfg['data_root'],
        '--la_data_root', cfg['la_data_root'],
        '--splits_dir', cfg['splits_dir'],
        '--label_percent', str(fraction),
        '--patch_size', cfg['patch_size'],
        '--stride_xy', str(cfg['stride_xy']),
        '--num_classes', str(NUM_CLASSES),
        '--mode', mode,
        '--lr', str(lr),
        '--epochs', str(budget),
        '--fixed_budget',
        '--val_split', 'labeled',
        '--seed', str(SEED),
        '--save_dir', str(save_dir),
        '--gpu', '0',
    ]
    if mode == 'confident':
        cmd.extend(['--conf_threshold', str(threshold)])

    if dry_run:
        print(' '.join(cmd))
        return name, None, 'dry_run'

    t0 = time.time()
    res = subprocess.run(cmd, capture_output=True, text=True)
    elapsed = time.time() - t0
    if res.returncode != 0:
        tail = '\n'.join((res.stderr or 'no stderr').splitlines()[-8:])
        print(f'  FAILED:\n{tail}')
        return name, None, f'failed (exit {res.returncode})'

    return name, parse_metrics(save_dir / 'metrics.csv', budget), f'ok ({elapsed:.0f}s)'


FIELDS = [
    'dataset', 'fraction', 'lr', 'mode', 'threshold', 'budget',
    'val_base_dice', 'val_dice', 'val_delta',
    'test_base_dice', 'test_dice', 'test_delta',
    'test_jaccard', 'test_hd95', 'test_asd',
    'name', 'status',
]


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', choices=list(DATASETS), required=True)
    p.add_argument('--dry_run', action='store_true')
    args = p.parse_args()

    cfg = DATASETS[args.dataset]
    out = ROOT / 'result' / f'pem_selection_{args.dataset}.csv'
    log_path = ROOT / 'result' / f'pem_selection_{args.dataset}.log'
    log_fh = open(log_path, 'a', buffering=1)

    def log(msg):
        line = f'{datetime.now():%Y-%m-%d %H:%M:%S}  {msg}'
        print(line)
        log_fh.write(line + '\n')

    # Full (lr x mask) grid, plus one run above the working range per fraction
    # to document the relocation ceiling.
    jobs = []
    for frac in cfg['fractions']:
        for lr in cfg['lrs']:
            jobs.append((frac, lr, 'full', None))
            for t in THRESHOLDS:
                jobs.append((frac, lr, 'confident', t))
        jobs.append((frac, cfg['ceiling_lr'], 'confident', 0.99))

    log('=' * 78)
    log(f'PEM SELECTION GRID — {args.dataset} — {len(jobs)} runs, '
        f'budget E={cfg["budget"]}, selection on labeled volumes only')
    log('=' * 78)

    rows = []
    t_start = time.time()
    for i, (frac, lr, mode, thr) in enumerate(jobs, 1):
        label = f'[{i}/{len(jobs)}] {frac}% lr={lr:.0e} {mode}' + (
            f' t={thr}' if thr is not None else '')
        log(label)
        name, metrics, status = run_one(
            args.dataset, cfg, frac, lr, mode, thr, dry_run=args.dry_run)
        row = dict(dataset=args.dataset, fraction=frac, lr=lr, mode=mode,
                   threshold=thr if thr is not None else '',
                   budget=cfg['budget'], name=name, status=status)
        if metrics is not None:
            base, fin = metrics
            row.update(
                val_base_dice=float(base['val_dice']),
                val_dice=float(fin['val_dice']),
                val_delta=float(fin['val_dice']) - float(base['val_dice']),
                test_base_dice=float(base['dice']),
                test_dice=float(fin['dice']),
                test_delta=float(fin['dice']) - float(base['dice']),
                test_jaccard=float(fin['jaccard']),
                test_hd95=float(fin['hd95']),
                test_asd=float(fin['asd']),
            )
            log(f'    {status}  val {row["val_delta"]*100:+.2f}  '
                f'test {row["test_delta"]*100:+.2f}')
        else:
            log(f'    {status}')
        rows.append(row)

        with open(out, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=FIELDS)
            w.writeheader()
            w.writerows(rows)

    log(f'DONE in {(time.time()-t_start)/60:.1f} min → {out}')

    # ── Report the selection the protocol makes, per fraction ───────────────
    for frac in cfg['fractions']:
        cand = [r for r in rows if r['fraction'] == frac and 'val_delta' in r]
        if not cand:
            continue
        best = max(cand, key=lambda r: r['val_delta'])
        log(f'  {args.dataset} {frac}%  selected by labeled val: '
            f'lr={best["lr"]:.0e} mode={best["mode"]} tau={best["threshold"]} '
            f'| val {best["val_delta"]*100:+.2f} '
            f'| test {best["test_delta"]*100:+.2f} '
            f'(Dice {best["test_dice"]*100:.2f})')


if __name__ == '__main__':
    main()
