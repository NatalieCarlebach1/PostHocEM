"""Assets for the SIPAIM 2026 Paper 71 oral deck: equation images and chart data.
Every number comes from the camera-ready paper or is computed from its formulas."""
import json, math, subprocess, glob, os
from pathlib import Path
from PIL import Image

HERE = Path(__file__).parent
A = HERE / 'assets'; A.mkdir(exist_ok=True)
INK = (0x1E, 0x1B, 0x2E)

EQS = {
    'eq_rho':  r'\rho_f(\tau)=\frac{\sum_i H_i\,\mathbf{1}[c_i\le\tau]}{\sum_i H_i}',
    'eq_loss': r'\mathcal{L}_{\mathrm{PEM}}=\frac{\sum_i m_i H_i}{\sum_i m_i+\epsilon},\qquad m_i=\mathbf{1}[c_i\le\tau_m]',
    'eq_grad': r'\frac{\partial H}{\partial z}=-\,z\,p\,(1-p)',
}
tex = [r'\documentclass[12pt]{article}', r'\usepackage{amsmath,amssymb}',
       r'\pagestyle{empty}', r'\begin{document}', r'\Large']
for k in EQS:
    tex += [r'\[' + EQS[k] + r'\]', r'\newpage']
tex += [r'\end{document}']
build = A / 'eqbuild'; build.mkdir(exist_ok=True)
(build / 'eq.tex').write_text('\n'.join(tex))
subprocess.run(['pdflatex', '-interaction=nonstopmode', 'eq.tex'], cwd=build, check=True,
               stdout=subprocess.DEVNULL)
for f in glob.glob(str(build / 'p-*.png')): os.remove(f)
subprocess.run(['pdftoppm', '-r', '600', '-png', 'eq.pdf', 'p'], cwd=build, check=True)
sizes = {}
for k, f in zip(EQS, sorted(glob.glob(str(build / 'p-*.png')))):
    g = Image.open(f).convert('L')
    bbox = Image.eval(g, lambda v: 255 - v).getbbox()
    g = g.crop((bbox[0] - 20, bbox[1] - 20, bbox[2] + 20, bbox[3] + 20))
    out = Image.new('RGBA', g.size, INK + (0,))
    out.putalpha(Image.eval(g, lambda v: 255 - v))          # ink colour, alpha = darkness
    out.save(A / f'{k}.png'); sizes[k] = out.size
print('equations:', sizes)

# |dH/dz| = |z| p (1-p) for a binary voxel, as a function of confidence c = max(p, 1-p)
g = lambda c: abs(math.log(c / (1 - c))) * c * (1 - c)
cs = [round(0.50 + 0.005 * i, 3) for i in range(0, 100)]          # 0.500 .. 0.995
peak = max(g(0.5 + 1e-6 * i) for i in range(1, 499_999))
grad = {'c': cs, 'pct': [round(100 * g(c) / peak, 2) if c > 0.5 else 0.0 for c in cs],
        'peak_c': max(cs, key=g), 'at_090': round(100 * g(0.90) / peak, 1)}
print('gradient peak at c=%.3f; at c=0.90: %.1f%% of peak' % (grad['peak_c'], grad['at_090']))

panels = {}
for st in ('pancreas_ct_20pct', 'la_5pct', 'la_10pct'):
    for k in ('image', 'gt', 'bcp', 'pem', 'entropy', 'corr'):
        p = HERE.parent / 'figures' / 'panels' / f'{st}_{k}.png'
        panels[f'{st}_{k}'] = {'path': str(p.resolve()), 'size': Image.open(p).size}
json.dump({'eq_sizes': sizes, 'grad': grad, 'panels': panels}, open(A / 'data.json', 'w'), indent=1)
