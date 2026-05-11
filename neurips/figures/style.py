"""Shared matplotlib style for BMVC paper figures.

Import this at the top of every figure script. Establishes a consistent
look (serif body matching LaTeX, tight margins, modest font sizes for
single-column captions).
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


def apply():
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif'],
        'font.size': 9,
        'axes.titlesize': 9,
        'axes.labelsize': 9,
        'xtick.labelsize': 8,
        'ytick.labelsize': 8,
        'legend.fontsize': 8,
        'legend.frameon': False,
        'lines.linewidth': 1.2,
        'lines.markersize': 4,
        'axes.linewidth': 0.6,
        'xtick.direction': 'in',
        'ytick.direction': 'in',
        'xtick.major.size': 3,
        'ytick.major.size': 3,
        'pdf.fonttype': 42,  # embed TrueType so the PDF is text-searchable
        'ps.fonttype': 42,
        'figure.dpi': 300,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.02,
    })


COLOR_BCP    = '#888888'
COLOR_PEM    = '#1f77b4'  # blue
COLOR_PLFT   = '#d62728'  # red
COLOR_TENT   = '#2ca02c'  # green
COLOR_SC     = '#ff7f0e'  # orange
COLOR_TS     = '#9467bd'  # purple
COLOR_GAIN   = '#1f77b4'
COLOR_FAIL   = '#d62728'
