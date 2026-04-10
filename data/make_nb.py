"""Run this to regenerate DGEM_Colab.ipynb"""
import json, os

os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def code(src):
    return {"cell_type":"code","execution_count":None,"metadata":{},"outputs":[],"source":src}

def md(src):
    return {"cell_type":"markdown","metadata":{},"source":src}

cells = [
md("# DGEM — Disagreement-Guided Entropy Minimization\n**MIDL 2025**\n\nSemi-supervised 3D CT segmentation, Pancreas-CT, 20% labeled.\n\n> **Runtime → Change runtime type → T4 GPU** before running.\n\nSet `MINI=True` (default) for a full pipeline run with synthetic data in ~10 minutes."),

md("## 0. GPU Check"),
code("import torch\nprint('CUDA:', torch.cuda.is_available())\nif torch.cuda.is_available():\n    print('GPU:', torch.cuda.get_device_name(0))\n    print('VRAM:', round(torch.cuda.get_device_properties(0).total_memory/1e9,1), 'GB')\nelse:\n    raise RuntimeError('No GPU — Runtime -> Change runtime type -> T4')"),

md("## 1. Install & Clone"),
code("%%capture\n!pip install tensorboardX nibabel h5py medpy SimpleITK scikit-image matplotlib"),
code("import os, sys\nREPO = '/content/PostHocEM'\nif not os.path.exists(REPO):\n    !git clone https://github.com/NatalieCarlebach1/PostHocEM.git {REPO}\nelse:\n    !git -C {REPO} pull --quiet\nsys.path.insert(0, REPO)\nos.chdir(REPO)\n!ls"),

md("## 2. Configuration\n\n| `MINI` | Data | Drive needed | Time |\n|--------|------|-------------|------|\n| `True` | Synthetic 20-case dataset (auto-generated) | No | ~10 min |\n| `False` | Real Pancreas-CT from TCIA | Yes | ~12 hrs |\n\nAll paths and hyperparameters are set here."),
code("MINI = True  # True=synthetic (no Drive) | False=real Pancreas-CT (requires Drive)\n\nSPLITS_DIR = f'{REPO}/splits/pancreas'\n\nif MINI:\n    # Everything lives in Colab's local /content — no Drive needed\n    DATA_ROOT   = '/content/synthetic_h5'\n    RESULT_ROOT = '/content/results'\n    FIG_DIR     = '/content/figures'\n    PATCH_SIZE  = 64\n    MAX_EPOCHS  = 10\n    EVAL_EVERY  = 2\n    N_TEST      = 4\nelse:\n    # Mount Drive for real data persistence across sessions\n    from google.colab import drive\n    drive.mount('/content/drive')\n    DRIVE_ROOT  = '/content/drive/MyDrive/DGEM'\n    DATA_ROOT   = f'{DRIVE_ROOT}/pancreas_h5'\n    RESULT_ROOT = f'{DRIVE_ROOT}/results'\n    FIG_DIR     = f'{DRIVE_ROOT}/figures'\n    PATCH_SIZE  = 96\n    MAX_EPOCHS  = 300\n    EVAL_EVERY  = 10\n    N_TEST      = 20\n\nBATCH_SIZE = 2\n\nfor d in [DATA_ROOT, RESULT_ROOT, SPLITS_DIR, FIG_DIR]:\n    os.makedirs(d, exist_ok=True)\n\nBCP_SAVE  = f'{RESULT_ROOT}/bcp_baseline'\nDGEM_SAVE = f'{RESULT_ROOT}/dgem_20p'\nSUP_SAVE  = f'{RESULT_ROOT}/supervised_only'\n\nmode = 'MINI — synthetic data, no Drive' if MINI else 'FULL — real Pancreas-CT + Drive'\nprint(f'Mode        : {mode}')\nprint(f'Data root   : {DATA_ROOT}')\nprint(f'Result root : {RESULT_ROOT}')\nprint(f'Patch size  : {PATCH_SIZE}  |  Max epochs: {MAX_EPOCHS}')"),

md("## 3. Data"),
code("if MINI:\n    print('Generating synthetic data (ellipsoid pancreas phantoms)...')\n    !python data/make_synthetic.py \\\n        --output_dir {DATA_ROOT} \\\n        --n_cases    20 \\\n        --vol_size   {PATCH_SIZE}\nelse:\n    # Upload raw TCIA data to Drive first, then set paths:\n    RAW_DATA  = f'{DRIVE_ROOT}/raw/Pancreas-CT/PANCREAS'\n    RAW_LABEL = f'{DRIVE_ROOT}/raw/TCIA_pancreas_labels'\n    existing  = list(__import__('pathlib').Path(DATA_ROOT).glob('*.h5'))\n    if not existing:\n        !python data/preprocess_pancreas.py \\\n            --data_root  {RAW_DATA} \\\n            --label_root {RAW_LABEL} \\\n            --output_dir {DATA_ROOT}\n    else:\n        print(f'Found {len(existing)} H5 files, skipping preprocess.')"),

code("# Generate train/test splits\n!python data/generate_splits.py \\\n    --h5_dir        {DATA_ROOT} \\\n    --splits_dir    {SPLITS_DIR} \\\n    --label_percent 20 \\\n    --n_test        {N_TEST} \\\n    --seed          2020\n\nprint('Split sizes:')\nfor fname in ['train_lab_20.txt', 'train_unlab_20.txt', 'test.txt']:\n    n = len(open(f'{SPLITS_DIR}/{fname}').readlines())\n    print(f'  {fname}: {n} cases')"),

md("### Data Sanity Check"),
code("import h5py, numpy as np, matplotlib.pyplot as plt\nfrom pathlib import Path\n\ncases = sorted(Path(DATA_ROOT).glob('*.h5'))\nassert cases, f'No H5 files in {DATA_ROOT}'\n\nwith h5py.File(str(cases[0]), 'r') as f:\n    image = f['image'][:]\n    label = f['label'][:]\n\nz = int(np.argmax(label.sum(axis=(0,1))))\nslices = [\n    (image[:,:,z],               label[:,:,z],               'Axial'),\n    (image[:,image.shape[1]//2,:], label[:,label.shape[1]//2,:], 'Coronal'),\n    (image[image.shape[0]//2,:,:], label[label.shape[0]//2,:,:], 'Sagittal'),\n]\nfig, axes = plt.subplots(1, 3, figsize=(13, 4))\nfor ax, (img_sl, lbl_sl, title) in zip(axes, slices):\n    ax.imshow(img_sl, cmap='gray', vmin=0, vmax=1)\n    if lbl_sl.sum() > 0:\n        ax.imshow(np.ma.masked_equal(lbl_sl, 0), cmap='Reds', alpha=0.6)\n    ax.set_title(title)\n    ax.axis('off')\nplt.suptitle(f'Case: {cases[0].stem}  |  Shape: {image.shape}  |  Mask voxels: {label.sum()}')\nplt.tight_layout()\nplt.show()\nprint(f'Total cases: {len(cases)}')"),

md("## 4. Train BCP Baseline (SOTA — CVPR 2023)"),
code("!python train_bcp_baseline.py \\\n    --data_root     {DATA_ROOT} \\\n    --splits_dir    {SPLITS_DIR} \\\n    --label_percent 20 \\\n    --max_epochs    {MAX_EPOCHS} \\\n    --patch_size    {PATCH_SIZE} \\\n    --batch_size    {BATCH_SIZE} \\\n    --eval_every    {EVAL_EVERY} \\\n    --save_dir      {BCP_SAVE} \\\n    --gpu           0"),

md("## 5. Train DGEM (Our Method)"),
code("!python train_dgem.py \\\n    --data_root          {DATA_ROOT} \\\n    --splits_dir         {SPLITS_DIR} \\\n    --label_percent      20 \\\n    --max_epochs         {MAX_EPOCHS} \\\n    --patch_size         {PATCH_SIZE} \\\n    --batch_size         {BATCH_SIZE} \\\n    --em_weight          1.0 \\\n    --consistency_rampup 40 \\\n    --ema_decay          0.99 \\\n    --eval_every         {EVAL_EVERY} \\\n    --save_dir           {DGEM_SAVE} \\\n    --gpu                0"),

md("## 6. Ablation — Supervised Only"),
code("!python train_dgem.py \\\n    --data_root     {DATA_ROOT} \\\n    --splits_dir    {SPLITS_DIR} \\\n    --label_percent 20 \\\n    --max_epochs    {MAX_EPOCHS} \\\n    --patch_size    {PATCH_SIZE} \\\n    --batch_size    {BATCH_SIZE} \\\n    --em_weight     0.0 \\\n    --eval_every    {EVAL_EVERY} \\\n    --save_dir      {SUP_SAVE} \\\n    --gpu           0"),

md("## 7. Evaluate — Paper Results Table"),
code("!python evaluate.py \\\n    --data_root   {DATA_ROOT} \\\n    --test_file   {SPLITS_DIR}/test.txt \\\n    --num_classes 2 \\\n    --patch_size  {PATCH_SIZE} \\\n    --compare \\\n        \"Supervised only:{SUP_SAVE}/best_model.pth\" \\\n        \"BCP (CVPR 2023):{BCP_SAVE}/best_model.pth\" \\\n        \"DGEM (ours):{DGEM_SAVE}/best_model.pth\""),

md("## 8. Loss Curves"),
code("""
import re, os
import numpy as np
import matplotlib.pyplot as plt

def parse_log(log_path):
    \"\"\"Parse train.log → (epochs, sup_losses, [(epoch, dice), ...]).\"\"\"
    epochs, sup_losses, dice_pts = [], [], []
    with open(log_path) as f:
        for line in f:
            # DGEM / Supervised: "Epoch [001/010]  sup=0.4521  em=..."
            m = re.search(r'Epoch \\[(\\d+)/\\d+\\].*?sup=([\\.\\d]+)', line)
            if m:
                epochs.append(int(m.group(1)))
                sup_losses.append(float(m.group(2)))
            # BCP: "Epoch [001/010]  loss1=0.3210  loss2=..."
            if not m:
                m2 = re.search(r'Epoch \\[(\\d+)/\\d+\\].*?loss1=([\\.\\d]+)', line)
                if m2:
                    epochs.append(int(m2.group(1)))
                    sup_losses.append(float(m2.group(2)))
            # Eval: "[Eval 002]  Dice=0.1234 ..."
            m3 = re.search(r'\\[Eval\\s+(\\d+)\\].*Dice=([\\.\\d]+)', line)
            if m3:
                dice_pts.append((int(m3.group(1)), float(m3.group(2))))
    return epochs, sup_losses, dice_pts

methods = [
    ('Supervised Only', SUP_SAVE,  '#4CAF50'),
    ('BCP (CVPR 2023)', BCP_SAVE,  '#F44336'),
    ('DGEM (ours)',     DGEM_SAVE, '#2196F3'),
]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle('Training Curves — Semi-supervised 3D Segmentation', fontsize=13)

for label, save_dir, color in methods:
    log_file = f'{save_dir}/train.log'
    if not os.path.exists(log_file):
        print(f'No log yet: {log_file}')
        continue
    epochs, losses, dice_pts = parse_log(log_file)
    if epochs:
        ax1.plot(epochs, losses, color=color, label=label, linewidth=2)
    if dice_pts:
        xs, ys = zip(*dice_pts)
        ax2.plot(xs, ys, color=color, label=label, linewidth=2, marker='o', markersize=5)
        best = max(ys)
        ax2.axhline(best, color=color, linestyle='--', alpha=0.35, linewidth=1)
        ax2.annotate(f'{best:.3f}', xy=(xs[-1], best), color=color, fontsize=8, va='bottom')

ax1.set_title('Supervised Loss per Epoch'); ax1.set_xlabel('Epoch'); ax1.set_ylabel('Loss')
ax2.set_title('Test Dice Score');           ax2.set_xlabel('Epoch'); ax2.set_ylabel('Dice')
ax1.legend(); ax1.grid(alpha=0.3)
ax2.legend(); ax2.grid(alpha=0.3)
ax2.set_ylim(bottom=0)
plt.tight_layout()
out_path = f'{FIG_DIR}/loss_curves.png'
plt.savefig(out_path, dpi=150, bbox_inches='tight')
plt.show()
print(f'Saved: {out_path}')
"""),

md("## 9. Qualitative Results — GT vs BCP vs DGEM\n\nFor each random test case: three views (axial / coronal / sagittal) through the lesion centroid.\nEach row shows the CT image with overlays: **green = GT**, **red = BCP**, **blue = DGEM**.\nDice score is shown in the subtitle of each prediction column."),
code("""
import torch, torch.nn as nn, torch.nn.functional as F
import numpy as np, h5py, random, os
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from networks import VNet
from utils.metrics import sliding_window_inference, calculate_metric_percase

# ── helpers ──────────────────────────────────────────────────────────────────
def load_model(ckpt_path, n_classes=2):
    net = VNet(n_classes=n_classes, normalization='instancenorm', has_dropout=True)
    net = nn.DataParallel(net).cuda()
    state = torch.load(ckpt_path, map_location='cuda')
    if isinstance(state, dict) and 'net' in state:
        state = state['net']
    net.load_state_dict(state)
    net.eval()
    return net

def get_centroid(label):
    coords = np.argwhere(label > 0)
    if len(coords) == 0:
        return tuple(s // 2 for s in label.shape)
    return tuple(coords.mean(0).astype(int))

def overlay(ax, img_sl, gt_sl, pred_sl, pred_color, title, dice=None):
    ax.imshow(img_sl, cmap='gray', vmin=0, vmax=1, interpolation='bilinear')
    # GT: green semi-transparent
    if gt_sl.sum() > 0:
        gt_rgba = np.zeros((*gt_sl.shape, 4))
        gt_rgba[gt_sl > 0] = [0.0, 0.9, 0.2, 0.35]
        ax.imshow(gt_rgba)
    # Prediction: coloured contour fill
    if pred_sl.sum() > 0:
        pred_rgba = np.zeros((*pred_sl.shape, 4))
        pred_rgba[pred_sl > 0] = [*pred_color, 0.55]
        ax.imshow(pred_rgba)
    lbl = title if dice is None else f'{title}\\nDice={dice:.3f}'
    ax.set_title(lbl, fontsize=8, pad=3)
    ax.axis('off')

def ct_only(ax, img_sl, title):
    ax.imshow(img_sl, cmap='gray', vmin=0, vmax=1, interpolation='bilinear')
    ax.set_title(title, fontsize=8, pad=3)
    ax.axis('off')

# ── config ───────────────────────────────────────────────────────────────────
N_CASES  = 4
SEED     = 42
BCP_CLR  = [1.0, 0.2, 0.1]   # red
DGEM_CLR = [0.1, 0.4, 1.0]   # blue

random.seed(SEED)
with open(f'{SPLITS_DIR}/test.txt') as f:
    all_cases = [l.strip() for l in f if l.strip()]
cases = random.sample(all_cases, min(N_CASES, len(all_cases)))

# ── load models ──────────────────────────────────────────────────────────────
bcp_ckpt  = f'{BCP_SAVE}/best_model.pth'
dgem_ckpt = f'{DGEM_SAVE}/best_model.pth'

bcp_net  = load_model(bcp_ckpt)  if os.path.exists(bcp_ckpt)  else None
dgem_net = load_model(dgem_ckpt) if os.path.exists(dgem_ckpt) else None

if bcp_net is None:  print('WARNING: BCP checkpoint not found, skipping.')
if dgem_net is None: print('WARNING: DGEM checkpoint not found, skipping.')

# ── per-case figure ───────────────────────────────────────────────────────────
VIEWS = ['Axial', 'Coronal', 'Sagittal']

for case in cases:
    with h5py.File(str(Path(DATA_ROOT) / case), 'r') as f:
        image = f['image'][:].astype(np.float32)
        label = f['label'][:].astype(np.uint8)

    cx, cy, cz = get_centroid(label)

    bcp_pred  = sliding_window_inference(bcp_net,  image, PATCH_SIZE, 16, 8, 2)[0].astype(np.uint8) if bcp_net  else np.zeros_like(label)
    dgem_pred = sliding_window_inference(dgem_net, image, PATCH_SIZE, 16, 8, 2)[0].astype(np.uint8) if dgem_net else np.zeros_like(label)

    bcp_dice  = calculate_metric_percase(bcp_pred,  label)[0]
    dgem_dice = calculate_metric_percase(dgem_pred, label)[0]

    # 3 views x 4 cols: [CT only | GT overlay | BCP | DGEM]
    fig, axes = plt.subplots(3, 4, figsize=(14, 9))
    fig.patch.set_facecolor('#111111')
    fig.suptitle(f'Case: {Path(case).stem}   |   BCP Dice={bcp_dice:.3f}   DGEM Dice={dgem_dice:.3f}',
                 fontsize=11, color='white', y=0.98)

    for row, view in enumerate(VIEWS):
        if view == 'Axial':
            img_sl = image[:, :, cz];  gt_sl = label[:, :, cz]
            bp_sl  = bcp_pred[:, :, cz]; dp_sl = dgem_pred[:, :, cz]
        elif view == 'Coronal':
            img_sl = image[:, cy, :];  gt_sl = label[:, cy, :]
            bp_sl  = bcp_pred[:, cy, :]; dp_sl = dgem_pred[:, cy, :]
        else:
            img_sl = image[cx, :, :];  gt_sl = label[cx, :, :]
            bp_sl  = bcp_pred[cx, :, :]; dp_sl = dgem_pred[cx, :, :]

        ct_only(axes[row, 0], img_sl, f'{view} — CT')
        overlay(axes[row, 1], img_sl, gt_sl, gt_sl,  [0.0, 0.9, 0.2], f'{view} — Ground Truth')
        overlay(axes[row, 2], img_sl, gt_sl, bp_sl,  BCP_CLR,         f'{view} — BCP (SOTA)',  bcp_dice)
        overlay(axes[row, 3], img_sl, gt_sl, dp_sl,  DGEM_CLR,        f'{view} — DGEM (ours)', dgem_dice)

        for ax in axes[row]:
            ax.set_facecolor('#111111')
            for spine in ax.spines.values():
                spine.set_edgecolor('#333333')

    # Legend
    legend_els = [
        mpatches.Patch(color=[0.0, 0.9, 0.2], alpha=0.5, label='Ground Truth'),
        mpatches.Patch(color=BCP_CLR,          alpha=0.6, label='BCP (SOTA)'),
        mpatches.Patch(color=DGEM_CLR,         alpha=0.6, label='DGEM (ours)'),
    ]
    fig.legend(handles=legend_els, loc='lower center', ncol=3,
               fontsize=9, framealpha=0.2, labelcolor='white',
               facecolor='#222222', bbox_to_anchor=(0.5, 0.01))

    plt.subplots_adjust(wspace=0.04, hspace=0.18, bottom=0.07)
    out_path = f'{FIG_DIR}/qual_{Path(case).stem}.png'
    plt.savefig(out_path, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.show()
    print(f'Saved: {out_path}')
"""),

md("## 10. Smoke Test (No Data, <60s)\n\nFull forward/backward pass with random tensors. Confirms imports and shapes are correct."),
code("import torch\nimport torch.nn.functional as F\nimport numpy as np\nfrom networks import VNet\nfrom utils.losses import SupLoss, entropy_loss_masked\nfrom utils.ramps  import get_current_consistency_weight\nfrom utils.metrics import sliding_window_inference, calculate_metric_percase\n\ntorch.manual_seed(42)\nP, B, dev = 64, 2, 'cuda'\n\n# Build models\nnet     = VNet(n_classes=2, normalization='instancenorm', has_dropout=True).to(dev)\nnet_ema = VNet(n_classes=2, normalization='instancenorm', has_dropout=True).to(dev)\nnet_ema.load_state_dict(net.state_dict())\nfor param in net_ema.parameters(): param.detach_()\n\noptimizer = torch.optim.Adam(net.parameters(), lr=1e-3)\ncriterion = SupLoss(n_classes=2)\n\nlab_img   = torch.randn(B,1,P,P,P).to(dev)\nlab_lbl   = torch.randint(0,2,(B,P,P,P)).to(dev)\nunlab_img = torch.randn(B,1,P,P,P).to(dev)\n\nnet.train()\nnet_ema.eval()\n\n# 1. Supervised loss\nsup_loss = criterion(net(lab_img)[0], lab_lbl)\n\n# 2. DGEM: entropy on disagreement voxels\nstudent_probs = F.softmax(net(unlab_img)[0], dim=1)\nstudent_pred  = student_probs.argmax(dim=1)\nwith torch.no_grad():\n    teacher_pred = F.softmax(net_ema(unlab_img)[0], dim=1).argmax(dim=1)\ndisagree = (student_pred != teacher_pred).float()\nem_loss  = entropy_loss_masked(student_probs, disagree)\nlam      = get_current_consistency_weight(1, 1.0, 40)\ntotal    = sup_loss + lam * em_loss\noptimizer.zero_grad()\ntotal.backward()\noptimizer.step()\n\n# 3. EMA update\nwith torch.no_grad():\n    for sp, tp in zip(net.parameters(), net_ema.parameters()):\n        tp.data = 0.99 * tp.data + 0.01 * sp.data\n\n# 4. Sliding window inference\nnet.eval()\nvol  = np.random.rand(P,P,P).astype(np.float32)\npred, score = sliding_window_inference(net, vol, P, 16, 8, 2)\nassert pred.shape == (P,P,P), 'Pred shape wrong'\nassert score.shape == (2,P,P,P), 'Score shape wrong'\n\n# 5. Metrics\ngt = (np.random.rand(P,P,P) > 0.8).astype(np.uint8)\npr = (np.random.rand(P,P,P) > 0.8).astype(np.uint8)\nd, jc, hd, asd = calculate_metric_percase(pr, gt)\n\nprint('=' * 45)\nprint('ALL CHECKS PASSED')\nprint(f'  sup_loss       : {sup_loss.item():.4f}')\nprint(f'  em_loss        : {em_loss.item():.4f}')\nprint(f'  disagree_ratio : {disagree.mean().item():.3f}')\nprint(f'  lam            : {lam:.4f}')\nprint(f'  total_loss     : {total.item():.4f}')\nprint(f'  pred shape     : {pred.shape}')\nprint(f'  score shape    : {score.shape}')\nprint(f'  Dice (random)  : {d:.4f}')\nprint('=' * 45)"),

md("## 11. Download Results\n\n- **MINI mode**: downloads a zip of all figures directly to your browser.\n- **FULL mode**: figures are already saved on Drive. The zip is also downloaded."),
code("from google.colab import files\nimport zipfile\nzip_path = '/content/figures.zip'\nwith zipfile.ZipFile(zip_path, 'w') as zf:\n    for f in __import__('pathlib').Path(FIG_DIR).glob('*.png'):\n        zf.write(str(f), f.name)\nfiles.download(zip_path)\nprint('Download started.')\n\n# Also download best checkpoints in MINI mode\nif MINI:\n    ckpt_zip = '/content/checkpoints.zip'\n    with zipfile.ZipFile(ckpt_zip, 'w') as zf:\n        for p in __import__('pathlib').Path(RESULT_ROOT).rglob('best_model.pth'):\n            zf.write(str(p), str(p.relative_to(RESULT_ROOT)))\n    files.download(ckpt_zip)\n    print('Checkpoints download started.')"),
]

nb = {
    "nbformat": 4,
    "nbformat_minor": 0,
    "metadata": {
        "colab": {"provenance": [], "gpuType": "T4"},
        "kernelspec": {"display_name": "Python 3", "name": "python3"},
        "language_info": {"name": "python"},
        "accelerator": "GPU"
    },
    "cells": cells
}

with open("DGEM_Colab.ipynb", "w") as f:
    json.dump(nb, f, indent=1)

with open("DGEM_Colab.ipynb") as f:
    json.load(f)

print(f"Notebook written and validated — {len(cells)} cells")
