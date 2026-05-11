"""
T1.7 Figure 2 — PEM as an interpretable refinement process.

Reads per-epoch Dice and rho_f from PEM result directories and renders
a three-panel learning-curve figure for each of the three main
configurations. Each row shows:
  - test Dice across PEM epochs (0..E)
  - rho_f(0.99) on the unlabeled set across PEM epochs (should decrease)
  - mean unlabeled-set loss across PEM epochs

Output: neurips/paper/figures/fig_pem_curves.pdf

Expects per-epoch logs at result/pem_*_s2020/train.log with lines:
  [Eval N] Dice=0.xxxx Jc=0.xxxx HD95=xx.xx ASD=xx.xx
  [EM eN/E] loss=x.xxxx  rho_f=0.xxxx
"""

import re
from pathlib import Path
import sys

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(ROOT))

import style
style.apply()

import matplotlib.pyplot as plt
import numpy as np

CONFIGS = [
    ('Pancreas-CT 20%', 'result/pem_seed_pancreas_2020/train.log'),
    ('LA 5%',           'result/pem_seed_la5_2020/train.log'),
    ('LA 10%',          'result/pem_seed_la10_2020/train.log'),
]


def parse_log(path: Path):
    if not path.exists():
        return None
    eval_pat = re.compile(r'\[Eval\s+(\d+)\].*?Dice=([0-9.]+).*?HD95=([0-9.]+)',
                          re.IGNORECASE)
    em_pat = re.compile(r'\[EM\s+e?(\d+).*?loss=([\-0-9.eE]+)'
                        r'(?:.*?rho_f=([0-9.]+))?',
                        re.IGNORECASE)
    epochs, dices, hd95s = [], [], []
    em_epochs, em_losses, em_rhos = [], [], []
    for line in path.read_text().splitlines():
        m = eval_pat.search(line)
        if m:
            epochs.append(int(m.group(1)))
            dices.append(float(m.group(2)) * 100)
            hd95s.append(float(m.group(3)))
        m = em_pat.search(line)
        if m:
            em_epochs.append(int(m.group(1)))
            em_losses.append(float(m.group(2)))
            if m.group(3):
                em_rhos.append(float(m.group(3)))
    return {
        'epoch': epochs,
        'dice': dices,
        'hd95': hd95s,
        'em_epoch': em_epochs,
        'em_loss': em_losses,
        'em_rho': em_rhos,
    }


def main():
    fig, axes = plt.subplots(len(CONFIGS), 3, figsize=(8.5, 2.2 * len(CONFIGS)))
    if len(CONFIGS) == 1:
        axes = axes[None, :]
    for i, (name, log_rel) in enumerate(CONFIGS):
        d = parse_log(ROOT / log_rel)
        if d is None or not d['epoch']:
            for j in range(3):
                axes[i, j].text(0.5, 0.5, '(no data)', ha='center',
                                transform=axes[i, j].transAxes)
                axes[i, j].set_xticks([]); axes[i, j].set_yticks([])
            axes[i, 0].set_ylabel(name)
            continue

        axes[i, 0].plot(d['epoch'], d['dice'], '-o', color=style.COLOR_PEM)
        axes[i, 0].set_xlabel('PEM epoch')
        axes[i, 0].set_ylabel('Dice (%)')
        if i == 0:
            axes[i, 0].set_title('Test Dice')

        if d['em_rho']:
            axes[i, 1].plot(d['em_epoch'][:len(d['em_rho'])],
                            d['em_rho'], '-o', color=style.COLOR_GAIN)
            axes[i, 1].set_xlabel('PEM epoch')
            axes[i, 1].set_ylabel(r'$\rho_f(0.99)$')
        else:
            axes[i, 1].text(0.5, 0.5, r'$\rho_f$ not logged',
                            ha='center', transform=axes[i, 1].transAxes)
            axes[i, 1].set_xticks([]); axes[i, 1].set_yticks([])
        if i == 0:
            axes[i, 1].set_title(r'Shell concentration $\rho_f$')

        if d['em_loss']:
            axes[i, 2].plot(d['em_epoch'], d['em_loss'], '-o',
                            color=style.COLOR_PLFT)
            axes[i, 2].set_xlabel('PEM epoch')
            axes[i, 2].set_ylabel('mean H on unlab.')
            axes[i, 2].set_yscale('log')
        if i == 0:
            axes[i, 2].set_title('Unlabeled-set entropy')

        # Row label
        axes[i, 0].text(-0.25, 0.5, name, rotation=90, fontsize=9,
                        ha='center', va='center',
                        transform=axes[i, 0].transAxes)

    fig.tight_layout()
    out = ROOT / 'neurips/paper/figures/fig_pem_curves.pdf'
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    print(f"Wrote {out}")


if __name__ == '__main__':
    main()
