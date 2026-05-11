"""
T1.7 Figure 1 — the headline figure.

Reads result/pem_trajectory_sweep/trajectory.csv (produced by
scripts/trajectory_pem_sweep.py) and renders a two-panel figure:

  (a) rho_f and base Dice vs SSL training epoch — shows that
      rho_f rises monotonically during BCP training and plateaus near
      convergence.
  (b) Delta Dice from PEM vs base rho_f — the headline scatter that
      turns rho_f from descriptive to predictive.

Also overlays the supervised-only baseline point if its rho_f and
PEM Delta exist in result/gradient_dist/summary.json (rho_f) and a
companion file result/pem_supervised_pancreas_baseline.json (gain).

Output: neurips/paper/figures/fig_trajectory.pdf
"""

import csv
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


def load_csv(p: Path):
    with open(p) as f:
        rows = list(csv.DictReader(f))
    rows = [r for r in rows if r['rho_f'] and r['pem_dice']]
    rows.sort(key=lambda r: int(r['epoch']))
    return rows


def main():
    csv_path = ROOT / 'result/pem_trajectory_sweep/trajectory.csv'
    if not csv_path.exists():
        print(f"Missing {csv_path}; run scripts/trajectory_pem_sweep.py first.")
        sys.exit(1)
    rows = load_csv(csv_path)
    if not rows:
        print(f"Empty {csv_path}.")
        sys.exit(1)

    epochs = np.array([int(r['epoch']) for r in rows])
    rho    = np.array([float(r['rho_f']) for r in rows])
    base   = np.array([float(r['base_dice']) for r in rows])
    delta  = np.array([float(r['delta_dice']) for r in rows])

    # Optional supervised-only point
    sup = None
    sup_rho_path = ROOT / 'result/gradient_dist/summary.json'
    sup_pem_path = ROOT / 'result/pem_supervised_pancreas_baseline.json'
    if sup_rho_path.exists():
        s = json.loads(sup_rho_path.read_text())
        sup_rho = s.get('pancreas_supervised', {}).get('fraction_in_shell_099')
        if sup_rho is not None and sup_pem_path.exists():
            sup_pem = json.loads(sup_pem_path.read_text())
            sup = (sup_rho, sup_pem.get('delta_dice', 0.0))

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.8))

    # (a) rho_f + base Dice vs epoch
    ax = axes[0]
    color1, color2 = style.COLOR_PEM, style.COLOR_BCP
    ax.plot(epochs, rho, '-', color=color1, marker='o', label=r'$\rho_f(0.99)$')
    ax.set_xlabel('BCP training epoch')
    ax.set_ylabel(r'$\rho_f(0.99)$', color=color1)
    ax.tick_params(axis='y', labelcolor=color1)
    ax.axhline(0.85, color=color1, linestyle='--', linewidth=0.6, alpha=0.5)
    ax.text(epochs[-1], 0.86, r'$\rho^\star=0.85$',
            color=color1, fontsize=7, ha='right', va='bottom')
    ax2 = ax.twinx()
    ax2.plot(epochs, base, '--', color=color2, marker='s',
             markersize=3, label='base Dice')
    ax2.set_ylabel('base Dice (%)', color=color2)
    ax2.tick_params(axis='y', labelcolor=color2)
    ax.set_title('(a) Trajectory along BCP training')

    # (b) Delta Dice vs rho_f
    ax = axes[1]
    ok = delta > -0.5
    ax.scatter(rho[ok], delta[ok], s=18, c=style.COLOR_GAIN,
               edgecolors='black', linewidths=0.4, zorder=3,
               label='BCP trajectory ckpts')
    ax.scatter(rho[~ok], delta[~ok], s=18, c=style.COLOR_FAIL,
               edgecolors='black', linewidths=0.4, zorder=3,
               marker='X', label='collapse / F1')
    if sup is not None:
        ax.scatter([sup[0]], [sup[1]], s=60, c='black',
                   marker='*', zorder=4, label='supervised-only')
    ax.axhline(0, color='black', linestyle='-', linewidth=0.4, alpha=0.6)
    ax.axvline(0.85, color='gray', linestyle=':', linewidth=0.6, alpha=0.6)
    ax.set_xlabel(r'base checkpoint $\rho_f(0.99)$')
    ax.set_ylabel(r'$\Delta$ Dice from PEM (%)')
    ax.set_title('(b) PEM gain vs shell concentration')
    ax.legend(loc='best')

    fig.tight_layout()
    out = ROOT / 'neurips/paper/figures/fig_trajectory.pdf'
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    print(f"Wrote {out}")


if __name__ == '__main__':
    main()
