"""
Seed-averaged per-case paired tests for the SIPAIM revision.
============================================================

Protocol (fixed before looking at any result of this script):
  * Unit of analysis = test case. For each method, each case's metric is
    first averaged over the three seeds {2020, 42, 123}; seed-case
    observations are never pooled. BCP is a single released checkpoint.
  * Models are the FINAL-epoch weights of the validation-selected
    configuration (last_model.pth), i.e. exactly what Table II reports.
  * Two-sided paired Wilcoxon signed-rank tests.
  * Families: HD95 is primary (the paper's claim is boundary refinement),
    Dice secondary. Within each family and each comparison, Holm correction
    across the three settings. Raw and adjusted p-values both reported.
  * Built-in check: the case mean of the seed-averaged per-case values must
    reproduce Table II; the script refuses to report if it does not.
"""

import json, os, sys
from pathlib import Path
import numpy as np
import h5py
import torch
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / 'scripts'))
from utils.metrics import sliding_window_inference, calculate_metric_percase  # noqa
from paired_significance import load_pancreas, load_la                       # noqa

# Same cuDNN settings as the training-time evaluation, so per-case values
# reproduce the logged ones exactly (HD95 is sensitive to single-voxel flips).
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

SEEDS = (2020, 42, 123)
CACHE = ROOT / 'result' / 'percase_cache'

SETTINGS = {
    'pancreas20': dict(name='Pancreas-CT 20%', kind='pancreas',
                       bcp='result/bcp_baseline_v2/best_model.pth',
                       data_root='data/pancreas_h5', split='splits/pancreas/test.txt',
                       patch=(96, 96, 96), stride_xy=16,
                       table=dict(BCP=(82.89, 7.81), PEM=(83.91, 5.87),
                                  **{'PL-FT': (83.18, 7.68), 'SC': (83.04, 7.62)})),
    'la5':        dict(name='LA 5%', kind='la', bcp='result/bcp_pretrained/LA_5.pth',
                       data_root='data/la_h5/2018LA_Seg_Training Set', split='splits/la/test.txt',
                       patch=(112, 112, 80), stride_xy=18,
                       table=dict(BCP=(87.32, 13.91), PEM=(87.99, 7.79),
                                  **{'PL-FT': (88.24, 10.11), 'SC': (87.75, 11.77)})),
    'la10':       dict(name='LA 10%', kind='la', bcp='result/bcp_pretrained/LA_10.pth',
                       data_root='data/la_h5/2018LA_Seg_Training Set', split='splits/la/test.txt',
                       patch=(112, 112, 80), stride_xy=18,
                       table=dict(BCP=(89.40, 9.88), PEM=(90.02, 6.87),
                                  **{'PL-FT': (90.09, 6.37), 'SC': (89.58, 8.04)})),
}
METHOD_DIR = {'PEM': 'pem', 'PL-FT': 'pl_ft', 'SC': 'sc'}


def ckpt(method, stem, seed):
    m = METHOD_DIR[method]
    if seed == 2020:
        return f'result/paired_{m}_{stem}/last_model.pth'
    return f'result/seedlast_{m}_{stem}_{seed}/last_model.pth'


def per_case(tag, path, cfg):
    """Per-case (dice, hd95) for one checkpoint, cached on disk."""
    CACHE.mkdir(parents=True, exist_ok=True)
    cf = CACHE / f'{tag}.json'
    if cf.exists():
        return json.loads(cf.read_text())
    net = (load_pancreas if cfg['kind'] == 'pancreas' else load_la)(path)
    cases = [l.strip() for l in open(ROOT / cfg['split']) if l.strip()]
    out = {}
    for c in cases:
        p = ROOT / cfg['data_root'] / c
        if cfg['kind'] == 'la':
            p = p / 'mri_norm2.h5'
        with h5py.File(str(p), 'r') as f:
            img = f['image'][:].astype(np.float32); lab = f['label'][:].astype(np.uint8)
        pred, _ = sliding_window_inference(net, img, cfg['patch'], cfg['stride_xy'], 4, 2)
        d, _, hd, _ = calculate_metric_percase(pred.astype(np.uint8), lab)
        out[c] = [float(d) * 100, float(hd)]
    del net; torch.cuda.empty_cache()
    cf.write_text(json.dumps(out))
    return out


def holm(ps):
    order = np.argsort(ps); m = len(ps); adj = np.empty(m); run = 0.0
    for rank, i in enumerate(order):
        run = max(run, min(1.0, (m - rank) * ps[i])); adj[i] = run
    return adj


def main():
    os.environ.setdefault('CUDA_VISIBLE_DEVICES', '0')
    data = {}  # stem -> method -> (cases x 2) seed-averaged array
    for stem, cfg in SETTINGS.items():
        cases = [l.strip() for l in open(ROOT / cfg['split']) if l.strip()]
        bcp = per_case(f'bcp_{stem}', cfg['bcp'], cfg)
        data[stem] = {'BCP': np.array([bcp[c] for c in cases])}
        for method in METHOD_DIR:
            per_seed = [per_case(f'{METHOD_DIR[method]}_{stem}_{s}', ckpt(method, stem, s), cfg)
                        for s in SEEDS]
            data[stem][method] = np.array([[np.mean([ps[c][k] for ps in per_seed]) for k in (0, 1)]
                                           for c in cases])

    # ── Built-in check against Table II ─────────────────────────────────────
    print('CHECK: case-mean of seed-averaged per-case values vs Table II')
    bad = False
    for stem, cfg in SETTINGS.items():
        for method, (td, th) in cfg['table'].items():
            md, mh = data[stem][method].mean(0)
            okd, okh = abs(md - td) < 0.006, abs(mh - th) < 0.006
            bad |= not (okd and okh)
            print(f'  {cfg["name"]:16s} {method:6s} Dice {md:6.2f} vs {td:6.2f} {"ok" if okd else "MISMATCH"}'
                  f'   HD95 {mh:6.2f} vs {th:6.2f} {"ok" if okh else "MISMATCH"}')
    if bad:
        sys.exit('Refusing to report tests: per-case data do not reproduce Table II.')

    # ── Tests ───────────────────────────────────────────────────────────────
    comps = [('PEM', 'BCP'), ('PEM', 'PL-FT'), ('PEM', 'SC'), ('PL-FT', 'SC')]
    results = {}
    for fam, k, label in ((0, 1, 'HD95 (primary)'), (1, 0, 'Dice (secondary)')):
        print(f'\n=== {label}: mean paired difference (A - B), two-sided Wilcoxon, Holm over 3 settings ===')
        for a, b in comps:
            raw, rows = [], []
            for stem, cfg in SETTINGS.items():
                x, y = data[stem][a][:, k], data[stem][b][:, k]
                d = x - y
                res = wilcoxon(x, y, zero_method='wilcox', alternative='two-sided')
                better = (d < -1e-9) if k == 1 else (d > 1e-9)   # lower HD95 / higher Dice
                raw.append(res.pvalue)
                rows.append((cfg['name'], d.mean(), res.pvalue, int(better.sum()), len(d)))
            adj = holm(np.array(raw))
            for (name, md, p, w, n), pa in zip(rows, adj):
                tag = f'{a} - {b}'
                print(f'  {tag:12s} {name:16s} {md:+7.2f}  p_raw={p:.4f}  p_holm={pa:.4f}  '
                      f'{a} better on {w}/{n}')
                results.setdefault(label, []).append(dict(comparison=tag, setting=name,
                    mean_diff=round(float(md), 3), p_raw=float(p), p_holm=float(pa), wins=w, n=n))
    out = ROOT / 'result' / 'seed_averaged_tests.json'
    out.write_text(json.dumps(results, indent=1))
    print(f'\nwritten {out}')


if __name__ == '__main__':
    main()
