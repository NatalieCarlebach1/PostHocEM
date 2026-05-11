"""
T1.7 Figure 3 — gradient-magnitude distribution.

Reads result/gradient_dist/summary.json (produced by
scripts/gradient_distribution.py) and renders a single-panel figure
comparing the fraction of |dH/dz| mass within k voxels of the predicted
boundary, for each checkpoint:

  - BCP Pancreas 20%      (expected: ~90% within 3 voxels)
  - BCP LA 5%             (expected: ~95% within 3 voxels)
  - BCP LA 10%            (expected: ~95% within 3 voxels)
  - Supervised V-Net      (expected: ~40-50% within 3 voxels)

Empirically supports Proposition 1 of Section 4: the p(1-p) factor
gates the entropy gradient to a thin boundary shell on converged SSL
networks but not on the supervised-only baseline.

Output: neurips/paper/figures/fig_gradient_distribution.pdf
"""

import json
from pathlib import Path
import sys

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(ROOT))

import style
style.apply()

import matplotlib.pyplot as plt
import numpy as np


LABEL_MAP = {
    'pancreas_supervised': 'Supervised V-Net',
    'pancreas20_bcp':      'BCP Pancreas 20%',
    'la5_bcp':             'BCP LA 5%',
    'la10_bcp':            'BCP LA 10%',
}
ORDER = ['pancreas_supervised', 'pancreas20_bcp', 'la5_bcp', 'la10_bcp']
COLORS = {
    'pancreas_supervised': style.COLOR_FAIL,
    'pancreas20_bcp':      style.COLOR_PEM,
    'la5_bcp':             '#2ca02c',
    'la10_bcp':            '#9467bd',
}


def main():
    summary_path = ROOT / 'result/gradient_dist/summary.json'
    if not summary_path.exists():
        print(f"Missing {summary_path}; run scripts/gradient_distribution.py first.")
        sys.exit(1)
    summary = json.loads(summary_path.read_text())

    distances = [1, 3, 5, 10]
    keys = [f'fraction_within_{k}_vox' for k in distances]

    fig, ax = plt.subplots(1, 1, figsize=(4.0, 2.7))
    for name in ORDER:
        if name not in summary:
            continue
        vals = [100 * summary[name].get(k, 0.0) for k in keys]
        ax.plot(distances, vals, '-o',
                color=COLORS[name], label=LABEL_MAP[name])
    ax.set_xlabel(r'distance to predicted boundary $k$ (voxels)')
    ax.set_ylabel(r'% of $|\partial H/\partial z|$ mass within $k$')
    ax.set_ylim(0, 100)
    ax.set_title(r'Spatial concentration of the entropy gradient')
    ax.legend(loc='lower right')
    fig.tight_layout()
    out = ROOT / 'neurips/paper/figures/fig_gradient_distribution.pdf'
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    print(f"Wrote {out}")


if __name__ == '__main__':
    main()
