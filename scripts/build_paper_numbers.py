"""
Rebuild every number the SIPAIM paper reports, under the labeled-validation
protocol, and emit the LaTeX for Tables II and III.

Sources
-------
  result/pem_sel_seed_{setting}_{seed}/   3-seed PEM at the selected config
  result/basesel_{method}_{setting}_lr*/  baseline learning-rate parity grid
  result/pem_selection_{la,pancreas}.csv  the (lr x mask) selection grid

Selection rule, identical for every post-hoc method: maximize dDice on the
labeled training volumes, which no post-hoc method optimizes on. The test
split is read once, for reporting.
"""

import csv
import statistics as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from analyze_selection import spearman, kendall_tau_b, load  # noqa: E402

ROOT = Path('/home/tals/Documents/PostHocEM')

SETTINGS = [
    ('Pancreas-CT 20\\%', 'pancreas20', 'pancreas', 20, 82.89, 71.01, 7.81, 2.62),
    ('LA 5\\%',           'la5',        'la',        5, 87.32, 77.65, 13.91, 3.60),
    ('LA 10\\%',          'la10',       'la',       10, 89.40, 80.92, 9.88,  2.86),
]
SEEDS = (2020, 42, 123)


def last_row(path):
    rows = list(csv.DictReader(open(path)))
    return rows[-1] if len(rows) > 1 else None


def pem_rows():
    """3-seed PEM at the validation-selected configuration."""
    out = {}
    for _, stem, *_ in SETTINGS:
        vals = []
        for s in SEEDS:
            p = ROOT / 'result' / f'pem_sel_seed_{stem}_{s}' / 'metrics.csv'
            if not p.exists():
                break
            vals.append(last_row(p))
        if len(vals) == len(SEEDS):
            out[stem] = dict(
                dice=st.mean(float(r['dice']) * 100 for r in vals),
                dice_sd=st.pstdev([float(r['dice']) * 100 for r in vals]),
                jaccard=st.mean(float(r['jaccard']) * 100 for r in vals),
                hd95=st.mean(float(r['hd95']) for r in vals),
                asd=st.mean(float(r['asd']) for r in vals),
            )
    return out


def baseline_rows():
    """Baselines selected by the same labeled-validation rule."""
    out = {}
    for method in ('pl_ft', 'sc'):
        for _, stem, *_ in SETTINGS:
            cands = []
            for p in sorted((ROOT / 'result').glob(f'basesel_{method}_{stem}_lr*')):
                m = p / 'metrics.csv'
                if not m.exists():
                    continue
                rows = list(csv.DictReader(open(m)))
                if len(rows) < 2 or 'val_dice' not in rows[0]:
                    continue
                fin, base = rows[-1], rows[0]
                cands.append(dict(
                    lr=p.name.split('_lr')[-1],
                    val_delta=float(fin['val_dice']) - float(base['val_dice']),
                    dice=float(fin['dice']) * 100,
                    jaccard=float(fin['jaccard']) * 100,
                    hd95=float(fin['hd95']),
                    asd=float(fin['asd']),
                    test_delta=float(fin['dice']) * 100 - float(base['dice']) * 100,
                ))
            if cands:
                out[(method, stem)] = (max(cands, key=lambda c: c['val_delta']), cands)
    return out


def main():
    pem = pem_rows()
    base = baseline_rows()

    print('=' * 74)
    print('  TABLE II — all post-hoc methods selected on labeled volumes only')
    print('=' * 74)
    for name, stem, _, _, bd, bj, bh, ba in SETTINGS:
        print(f'\n  {name.replace(chr(92), "")}   baseline BCP: Dice {bd:.2f}  '
              f'Jac {bj:.2f}  HD95 {bh:.2f}  ASD {ba:.2f}')
        for method, label in (('pl_ft', 'PL-FT'), ('sc', 'SC')):
            hit = base.get((method, stem))
            if not hit:
                print(f'    {label:6s} — pending')
                continue
            sel, cands = hit
            others = ', '.join(f'lr={c["lr"]}:{c["test_delta"]:+.2f}' for c in cands)
            print(f'    {label:6s} Dice {sel["dice"]:.2f} ({sel["test_delta"]:+.2f})  '
                  f'Jac {sel["jaccard"]:.2f}  HD95 {sel["hd95"]:.2f}  '
                  f'ASD {sel["asd"]:.2f}   [selected lr={sel["lr"]}; grid {others}]')
        if stem in pem:
            p = pem[stem]
            print(f'    {"PEM":6s} Dice {p["dice"]:.2f} ({p["dice"]-bd:+.2f}) '
                  f'+-{p["dice_sd"]:.2f}  Jac {p["jaccard"]:.2f}  '
                  f'HD95 {p["hd95"]:.2f}  ASD {p["asd"]:.2f}')
            print(f'           HD95 reduction: {(bh-p["hd95"])/bh*100:.1f}%')

    # ── Winner per setting ──────────────────────────────────────────────────
    print('\n' + '=' * 74)
    print('  WHO WINS, UNDER THE IDENTICAL PROTOCOL')
    print('=' * 74)
    for name, stem, *_ in SETTINGS:
        row = []
        if stem in pem:
            row.append(('PEM', pem[stem]['dice']))
        for method, label in (('pl_ft', 'PL-FT'), ('sc', 'SC')):
            if (method, stem) in base:
                row.append((label, base[(method, stem)][0]['dice']))
        if row:
            row.sort(key=lambda t: -t[1])
            order = '  >  '.join(f'{n} {d:.2f}' for n, d in row)
            print(f'  {name.replace(chr(92), ""):16s} {order}')

    # ── LaTeX for Table II ──────────────────────────────────────────────────
    print('\n' + '=' * 74)
    print('  LATEX — Table II')
    print('=' * 74)
    for name, stem, _, _, bd, bj, bh, ba in SETTINGS:
        print(f'\\multicolumn{{5}}{{l}}{{\\textit{{{name}}}}} \\\\')
        print(f'BCP (released ckpt)          & {bd:.2f} & {bj:.2f} & {bh:5.2f} & {ba:.2f} \\\\')
        for method, label in (('pl_ft', 'PL-FT'), ('sc', 'SC')):
            if (method, stem) in base:
                s = base[(method, stem)][0]
                print(f'\\quad$+$ {label:6s}              & {s["dice"]:.2f} & '
                      f'{s["jaccard"]:.2f} & {s["hd95"]:5.2f} & {s["asd"]:.2f} \\\\')
        if stem in pem:
            p = pem[stem]
            print(f'\\quad$+$ \\textbf{{PEM}}         & {p["dice"]:.2f} & '
                  f'{p["jaccard"]:.2f} & {p["hd95"]:5.2f} & {p["asd"]:.2f} \\\\')
        print('\\midrule')

    # ── Validation-signal quality ───────────────────────────────────────────
    print('\n' + '=' * 74)
    print('  VALIDATION SIGNAL (rank agreement with test, across the grid)')
    print('=' * 74)
    for ds in ('la', 'pancreas'):
        rows = load(ds)
        for frac in sorted({r['fraction'] for r in rows}):
            sub = [r for r in rows if r['fraction'] == frac]
            keep = [r for r in sub if r['test_delta'] * 100 > -5]
            v, t = [r['val_delta'] for r in sub], [r['test_delta'] for r in sub]
            vk, tk = [r['val_delta'] for r in keep], [r['test_delta'] for r in keep]
            print(f'  {ds} {frac}%  all n={len(sub)}: rho={spearman(v,t):+.3f} '
                  f'tau={kendall_tau_b(v,t):+.3f}   |   '
                  f'non-collapsed n={len(keep)}: rho={spearman(vk,tk):+.3f} '
                  f'tau={kendall_tau_b(vk,tk):+.3f}')


if __name__ == '__main__':
    main()
