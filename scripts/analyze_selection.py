"""
Analysis of the labeled-validation selection grid.
==================================================

Answers three questions the SIPAIM reviewer raised, from the grid produced by
scripts/pem_selection_grid.py:

  1. Under a FIXED budget (not the oracle peak), how do the confidence
     thresholds actually order?  In particular, does tau=0.95 beat tau=0.99?
  2. Which configuration does the labeled-validation protocol select, with no
     access to the test split, and what does it score on test?
  3. Is the labeled-validation signal actually informative?  The labeled
     volumes were used to fit the original BCP checkpoint, so their absolute
     Dice is optimistic; what matters is whether the DELTA they report RANKS
     configurations the same way the test delta does.  Reported as Spearman
     and Kendall tau-b between val delta and test delta across the grid.

Usage:
    python scripts/analyze_selection.py                 # both datasets
    python scripts/analyze_selection.py --latex         # also emit LaTeX
"""

import argparse
import csv
from pathlib import Path

ROOT = Path('/home/tals/Documents/PostHocEM')


# ─── Rank statistics (no scipy dependency) ───────────────────────────────────

def _ranks(xs):
    """Fractional (average) ranks, so ties are handled correctly."""
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    ranks = [0.0] * len(xs)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and xs[order[j + 1]] == xs[order[i]]:
            j += 1
        avg = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    return ranks


def pearson(xs, ys):
    n = len(xs)
    if n < 2:
        return float('nan')
    mx, my = sum(xs) / n, sum(ys) / n
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    dx = sum((x - mx) ** 2 for x in xs) ** 0.5
    dy = sum((y - my) ** 2 for y in ys) ** 0.5
    return num / (dx * dy) if dx and dy else float('nan')


def spearman(xs, ys):
    return pearson(_ranks(xs), _ranks(ys))


def kendall_tau_b(xs, ys):
    n = len(xs)
    conc = disc = tx = ty = 0
    for i in range(n):
        for j in range(i + 1, n):
            a, b = xs[i] - xs[j], ys[i] - ys[j]
            if a == 0 and b == 0:
                tx += 1; ty += 1
            elif a == 0:
                tx += 1
            elif b == 0:
                ty += 1
            elif (a > 0) == (b > 0):
                conc += 1
            else:
                disc += 1
    d = ((conc + disc + tx) * (conc + disc + ty)) ** 0.5
    return (conc - disc) / d if d else float('nan')


# ─── Loading ─────────────────────────────────────────────────────────────────

def load(dataset):
    path = ROOT / 'result' / f'pem_selection_{dataset}.csv'
    if not path.exists():
        return []
    rows = []
    with open(path) as f:
        for r in csv.DictReader(f):
            if not r.get('val_delta'):
                continue                      # run failed or still pending
            r['fraction'] = int(r['fraction'])
            r['lr'] = float(r['lr'])
            r['threshold'] = float(r['threshold']) if r['threshold'] else None
            for k in ('val_delta', 'test_delta', 'val_dice', 'test_dice',
                      'test_base_dice', 'val_base_dice',
                      'test_jaccard', 'test_hd95', 'test_asd'):
                r[k] = float(r[k])
            rows.append(r)
    return rows


def mask_label(r):
    return 'Full (no mask)' if r['mode'] == 'full' else f"tau={r['threshold']}"


# ─── Reports ─────────────────────────────────────────────────────────────────

def report_fixed_budget(rows, dataset):
    """Q1: the ablation grid at the fixed budget, test deltas."""
    print(f'\n{"="*72}\n  FIXED-BUDGET ABLATION (test dDice, no oracle early stop) — {dataset}\n{"="*72}')
    for frac in sorted({r['fraction'] for r in rows}):
        sub = [r for r in rows if r['fraction'] == frac]
        lrs = sorted({r['lr'] for r in sub})
        masks = ['full'] + [t for t in (0.9, 0.95, 0.99, 0.999)]
        print(f'\n  {dataset} {frac}%   (budget E={sub[0]["budget"]}, '
              f'baseline Dice {sub[0]["test_base_dice"]*100:.2f})')
        print('    ' + f'{"mask":<16}' + ''.join(f'{f"lr={lr:.0e}":>14}' for lr in lrs))
        for m in masks:
            cells = []
            for lr in lrs:
                hit = [r for r in sub if r['lr'] == lr and (
                    (m == 'full' and r['mode'] == 'full') or
                    (m != 'full' and r['threshold'] == m))]
                cells.append(f'{hit[0]["test_delta"]*100:+14.2f}' if hit else f'{"—":>14}')
            name = 'Full (no mask)' if m == 'full' else f'tau={m}'
            print(f'    {name:<16}' + ''.join(cells))

        # The reviewer's specific question
        for lr in lrs:
            g = {r['threshold']: r for r in sub
                 if r['lr'] == lr and r['mode'] == 'confident'}
            if 0.95 in g and 0.99 in g:
                d95, d99 = g[0.95]['test_delta']*100, g[0.99]['test_delta']*100
                verdict = ('0.95 > 0.99' if d95 > d99 else
                           '0.99 > 0.95' if d99 > d95 else 'tie')
                print(f'      lr={lr:.0e}:  tau=0.95 {d95:+.2f}  vs  '
                      f'tau=0.99 {d99:+.2f}   -> {verdict}')


def report_selection(rows, dataset):
    """Q2: what the labeled-validation protocol picks, and its test score."""
    print(f'\n{"="*72}\n  LABELED-VALIDATION SELECTION (no test access) — {dataset}\n{"="*72}')
    out = []
    for frac in sorted({r['fraction'] for r in rows}):
        sub = [r for r in rows if r['fraction'] == frac]
        best = max(sub, key=lambda r: r['val_delta'])
        oracle = max(sub, key=lambda r: r['test_delta'])
        print(f'\n  {dataset} {frac}%')
        print(f'    selected      : lr={best["lr"]:.0e}  {mask_label(best)}')
        print(f'      val  dDice  : {best["val_delta"]*100:+.2f}')
        print(f'      TEST Dice   : {best["test_dice"]*100:.2f} '
              f'({best["test_delta"]*100:+.2f})  '
              f'Jac {best["test_jaccard"]*100:.2f}  '
              f'HD95 {best["test_hd95"]:.2f}  ASD {best["test_asd"]:.2f}')
        print(f'    test-oracle   : lr={oracle["lr"]:.0e}  {mask_label(oracle)}  '
              f'-> Dice {oracle["test_dice"]*100:.2f} '
              f'({oracle["test_delta"]*100:+.2f})')
        gap = (oracle['test_delta'] - best['test_delta']) * 100
        print(f'    selection gap : {gap:.2f} Dice points '
              f'({"same config" if best["name"] == oracle["name"] else "different config"})')
        out.append((frac, best, oracle))
    return out


def report_signal(rows, dataset):
    """Q3: does the validation delta rank configurations like the test delta?"""
    print(f'\n{"="*72}\n  IS THE VALIDATION SIGNAL INFORMATIVE? — {dataset}\n{"="*72}')
    for frac in sorted({r['fraction'] for r in rows}):
        sub = [r for r in rows if r['fraction'] == frac]
        v = [r['val_delta'] for r in sub]
        t = [r['test_delta'] for r in sub]
        print(f'  {dataset} {frac}%  (n={len(sub)} configs)  '
              f'Spearman={spearman(v, t):+.3f}  '
              f'Kendall tau-b={kendall_tau_b(v, t):+.3f}  '
              f'Pearson={pearson(v, t):+.3f}')
    if len({r['fraction'] for r in rows}) > 1:
        v = [r['val_delta'] for r in rows]
        t = [r['test_delta'] for r in rows]
        print(f'  pooled        (n={len(rows)} configs)  '
              f'Spearman={spearman(v, t):+.3f}  '
              f'Kendall tau-b={kendall_tau_b(v, t):+.3f}  '
              f'Pearson={pearson(v, t):+.3f}')


def emit_latex(rows_by_ds):
    """New Table III: fixed-budget deltas for LA, both learning rates."""
    la = rows_by_ds.get('la', [])
    if not la:
        return
    print(f'\n{"="*72}\n  LATEX (fixed-budget ablation, LA)\n{"="*72}')
    lrs = sorted({r['lr'] for r in la if r['lr'] < 5e-5})
    masks = [('full', 'Full (no mask)'), (0.9, r'Confident $\tau{=}0.90$'),
             (0.95, r'Confident $\tau{=}0.95$'), (0.99, r'Confident $\tau{=}0.99$'),
             (0.999, r'Confident $\tau{=}0.999$')]
    body = []
    for key, name in masks:
        cells = []
        for frac in (5, 10):
            for lr in lrs:
                hit = [r for r in la if r['fraction'] == frac and r['lr'] == lr and (
                    (key == 'full' and r['mode'] == 'full') or
                    (key != 'full' and r['threshold'] == key))]
                cells.append(f'${hit[0]["test_delta"]*100:+.2f}$' if hit else '--')
        body.append(f'{name:<28}& ' + ' & '.join(cells) + r' \\')
    print('\n'.join(body))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--latex', action='store_true')
    args = p.parse_args()

    rows_by_ds = {}
    for ds in ('la', 'pancreas'):
        rows = load(ds)
        if not rows:
            print(f'[{ds}] no completed runs yet')
            continue
        rows_by_ds[ds] = rows
        print(f'\n[{ds}] {len(rows)} completed configurations')
        report_fixed_budget(rows, ds)
        report_selection(rows, ds)
        report_signal(rows, ds)

    if args.latex:
        emit_latex(rows_by_ds)


if __name__ == '__main__':
    main()
