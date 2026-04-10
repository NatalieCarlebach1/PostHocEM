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
code("%%capture\n!pip install tensorboardX nibabel h5py medpy SimpleITK scikit-image tensorboard matplotlib"),
code("import os, sys\nREPO = '/content/PostHocEM'\nif not os.path.exists(REPO):\n    !git clone https://github.com/NatalieCarlebach1/PostHocEM.git {REPO}\nelse:\n    !git -C {REPO} pull --quiet\nsys.path.insert(0, REPO)\nos.chdir(REPO)\n!ls"),

md("## 2. Configuration\n\n| `MINI` | Data | Time |\n|--------|------|------|\n| `True` | Synthetic 20-case dataset (auto-generated) | ~10 min |\n| `False` | Real Pancreas-CT from TCIA | ~12 hrs |\n\nAll paths and hyperparameters are set here."),
code("MINI = True  # True=synthetic smoke test | False=real Pancreas-CT\n\nfrom google.colab import drive\ndrive.mount('/content/drive')\n\nDRIVE_ROOT  = '/content/drive/MyDrive/DGEM'\nRESULT_ROOT = f'{DRIVE_ROOT}/results'\nSPLITS_DIR  = f'{REPO}/splits/pancreas'\nFIG_DIR     = f'{DRIVE_ROOT}/figures'\n\nif MINI:\n    DATA_ROOT  = f'{DRIVE_ROOT}/synthetic_h5'\n    PATCH_SIZE = 64\n    MAX_EPOCHS = 10\n    EVAL_EVERY = 2\n    N_TEST     = 4\nelse:\n    DATA_ROOT  = f'{DRIVE_ROOT}/pancreas_h5'\n    PATCH_SIZE = 96\n    MAX_EPOCHS = 300\n    EVAL_EVERY = 10\n    N_TEST     = 20\n\nBATCH_SIZE = 2\n\nfor d in [DATA_ROOT, RESULT_ROOT, SPLITS_DIR, FIG_DIR]:\n    os.makedirs(d, exist_ok=True)\n\nmode = 'MINI (synthetic)' if MINI else 'FULL (real data)'\nprint(f'Mode        : {mode}')\nprint(f'Data root   : {DATA_ROOT}')\nprint(f'Patch size  : {PATCH_SIZE}  |  Max epochs: {MAX_EPOCHS}')"),

md("## 3. Data"),
code("if MINI:\n    print('Generating synthetic data (ellipsoid pancreas phantoms)...')\n    !python data/make_synthetic.py \\\n        --output_dir {DATA_ROOT} \\\n        --n_cases    20 \\\n        --vol_size   {PATCH_SIZE}\nelse:\n    # Upload raw TCIA data to Drive first, then set paths:\n    RAW_DATA  = f'{DRIVE_ROOT}/raw/Pancreas-CT/PANCREAS'\n    RAW_LABEL = f'{DRIVE_ROOT}/raw/TCIA_pancreas_labels'\n    existing  = list(__import__('pathlib').Path(DATA_ROOT).glob('*.h5'))\n    if not existing:\n        !python data/preprocess_pancreas.py \\\n            --data_root  {RAW_DATA} \\\n            --label_root {RAW_LABEL} \\\n            --output_dir {DATA_ROOT}\n    else:\n        print(f'Found {len(existing)} H5 files, skipping preprocess.')"),

code("# Generate train/test splits\n!python data/generate_splits.py \\\n    --h5_dir        {DATA_ROOT} \\\n    --splits_dir    {SPLITS_DIR} \\\n    --label_percent 20 \\\n    --n_test        {N_TEST} \\\n    --seed          2020\n\nprint('Split sizes:')\nfor fname in ['train_lab_20.txt', 'train_unlab_20.txt', 'test.txt']:\n    n = len(open(f'{SPLITS_DIR}/{fname}').readlines())\n    print(f'  {fname}: {n} cases')"),

md("### Data Sanity Check"),
code("import h5py, numpy as np, matplotlib.pyplot as plt\nfrom pathlib import Path\n\ncases = sorted(Path(DATA_ROOT).glob('*.h5'))\nassert cases, f'No H5 files in {DATA_ROOT}'\n\nwith h5py.File(str(cases[0]), 'r') as f:\n    image = f['image'][:]\n    label = f['label'][:]\n\nz = int(np.argmax(label.sum(axis=(0,1))))\nslices = [\n    (image[:,:,z],               label[:,:,z],               'Axial'),\n    (image[:,image.shape[1]//2,:], label[:,label.shape[1]//2,:], 'Coronal'),\n    (image[image.shape[0]//2,:,:], label[label.shape[0]//2,:,:], 'Sagittal'),\n]\nfig, axes = plt.subplots(1, 3, figsize=(13, 4))\nfor ax, (img_sl, lbl_sl, title) in zip(axes, slices):\n    ax.imshow(img_sl, cmap='gray', vmin=0, vmax=1)\n    if lbl_sl.sum() > 0:\n        ax.imshow(np.ma.masked_equal(lbl_sl, 0), cmap='Reds', alpha=0.6)\n    ax.set_title(title)\n    ax.axis('off')\nplt.suptitle(f'Case: {cases[0].stem}  |  Shape: {image.shape}  |  Mask voxels: {label.sum()}')\nplt.tight_layout()\nplt.show()\nprint(f'Total cases: {len(cases)}')"),

md("## 4. Train BCP Baseline (SOTA — CVPR 2023)"),
code("BCP_SAVE = f'{RESULT_ROOT}/bcp_baseline'\n\n!python train_bcp_baseline.py \\\n    --data_root     {DATA_ROOT} \\\n    --splits_dir    {SPLITS_DIR} \\\n    --label_percent 20 \\\n    --max_epochs    {MAX_EPOCHS} \\\n    --patch_size    {PATCH_SIZE} \\\n    --batch_size    {BATCH_SIZE} \\\n    --eval_every    {EVAL_EVERY} \\\n    --save_dir      {BCP_SAVE} \\\n    --gpu           0"),

md("## 5. Train DGEM (Our Method)"),
code("DGEM_SAVE = f'{RESULT_ROOT}/dgem_20p'\n\n!python train_dgem.py \\\n    --data_root          {DATA_ROOT} \\\n    --splits_dir         {SPLITS_DIR} \\\n    --label_percent      20 \\\n    --max_epochs         {MAX_EPOCHS} \\\n    --patch_size         {PATCH_SIZE} \\\n    --batch_size         {BATCH_SIZE} \\\n    --em_weight          1.0 \\\n    --consistency_rampup 40 \\\n    --ema_decay          0.99 \\\n    --eval_every         {EVAL_EVERY} \\\n    --save_dir           {DGEM_SAVE} \\\n    --gpu                0"),

md("## 6. Ablation — Supervised Only"),
code("SUP_SAVE = f'{RESULT_ROOT}/supervised_only'\n\n!python train_dgem.py \\\n    --data_root     {DATA_ROOT} \\\n    --splits_dir    {SPLITS_DIR} \\\n    --label_percent 20 \\\n    --max_epochs    {MAX_EPOCHS} \\\n    --patch_size    {PATCH_SIZE} \\\n    --batch_size    {BATCH_SIZE} \\\n    --em_weight     0.0 \\\n    --eval_every    {EVAL_EVERY} \\\n    --save_dir      {SUP_SAVE} \\\n    --gpu           0"),

md("## 7. Evaluate — Paper Results Table"),
code("!python evaluate.py \\\n    --data_root   {DATA_ROOT} \\\n    --test_file   {SPLITS_DIR}/test.txt \\\n    --num_classes 2 \\\n    --patch_size  {PATCH_SIZE} \\\n    --compare \\\n        \"Supervised only:{SUP_SAVE}/best_model.pth\" \\\n        \"BCP (CVPR 2023):{BCP_SAVE}/best_model.pth\" \\\n        \"DGEM (ours):{DGEM_SAVE}/best_model.pth\""),

md("## 8. Loss Curves"),
code("!python visualize.py losses \\\n    --result_dirs \\\n        \"Supervised only:{SUP_SAVE}\" \\\n        \"BCP (CVPR 2023):{BCP_SAVE}\" \\\n        \"DGEM (ours):{DGEM_SAVE}\" \\\n    --out_dir {FIG_DIR}\n\nfrom IPython.display import Image as IPImage, display\ndisplay(IPImage(f'{FIG_DIR}/loss_curves.png'))"),
code("display(IPImage(f'{FIG_DIR}/em_loss.png'))"),

md("## 9. Prediction Grids [CT | GT | BCP | DGEM]"),
code("!python visualize.py predictions \\\n    --data_root  {DATA_ROOT} \\\n    --test_file  {SPLITS_DIR}/test.txt \\\n    --patch_size {PATCH_SIZE} \\\n    --checkpoints \\\n        \"BCP (CVPR 2023):{BCP_SAVE}/best_model.pth\" \\\n        \"DGEM (ours):{DGEM_SAVE}/best_model.pth\" \\\n    --n_cases 4 \\\n    --out_dir {FIG_DIR}\n\nimport glob\nfrom IPython.display import Image as IPImage, display\nfor p in sorted(glob.glob(f'{FIG_DIR}/pred_*.png')):\n    print(p)\n    display(IPImage(p))"),

md("## 10. TensorBoard"),
code("%load_ext tensorboard\n%tensorboard --logdir {RESULT_ROOT}"),

md("## 11. Smoke Test (No Data, <60s)\n\nFull forward/backward pass with random tensors. Run this first to confirm the code is correct."),
code("import torch\nimport torch.nn.functional as F\nimport numpy as np\nfrom networks import VNet\nfrom utils.losses import SupLoss, entropy_loss_masked\nfrom utils.ramps  import get_current_consistency_weight\nfrom utils.metrics import sliding_window_inference, calculate_metric_percase\n\ntorch.manual_seed(42)\nP, B, dev = 64, 2, 'cuda'\n\n# Build models\nnet     = VNet(n_classes=2, normalization='instancenorm', has_dropout=True).to(dev)\nnet_ema = VNet(n_classes=2, normalization='instancenorm', has_dropout=True).to(dev)\nnet_ema.load_state_dict(net.state_dict())\nfor param in net_ema.parameters(): param.detach_()\n\noptimizer = torch.optim.Adam(net.parameters(), lr=1e-3)\ncriterion = SupLoss(n_classes=2)\n\nlab_img   = torch.randn(B,1,P,P,P).to(dev)\nlab_lbl   = torch.randint(0,2,(B,P,P,P)).to(dev)\nunlab_img = torch.randn(B,1,P,P,P).to(dev)\n\nnet.train()\nnet_ema.eval()\n\n# 1. Supervised loss\nsup_loss = criterion(net(lab_img)[0], lab_lbl)\n\n# 2. DGEM: entropy on disagreement voxels\nstudent_probs = F.softmax(net(unlab_img)[0], dim=1)\nstudent_pred  = student_probs.argmax(dim=1)\nwith torch.no_grad():\n    teacher_pred = F.softmax(net_ema(unlab_img)[0], dim=1).argmax(dim=1)\ndisagree = (student_pred != teacher_pred).float()\nem_loss  = entropy_loss_masked(student_probs, disagree)\nlam      = get_current_consistency_weight(1, 1.0, 40)\ntotal    = sup_loss + lam * em_loss\noptimizer.zero_grad()\ntotal.backward()\noptimizer.step()\n\n# 3. EMA update\nwith torch.no_grad():\n    for sp, tp in zip(net.parameters(), net_ema.parameters()):\n        tp.data = 0.99 * tp.data + 0.01 * sp.data\n\n# 4. Sliding window inference\nnet.eval()\nvol  = np.random.rand(P,P,P).astype(np.float32)\npred, score = sliding_window_inference(net, vol, P, 16, 8, 2)\nassert pred.shape == (P,P,P), 'Pred shape wrong'\nassert score.shape == (2,P,P,P), 'Score shape wrong'\n\n# 5. Metrics\ngt = (np.random.rand(P,P,P) > 0.8).astype(np.uint8)\npr = (np.random.rand(P,P,P) > 0.8).astype(np.uint8)\nd, jc, hd, asd = calculate_metric_percase(pr, gt)\n\nprint('=' * 45)\nprint('ALL CHECKS PASSED')\nprint(f'  sup_loss       : {sup_loss.item():.4f}')\nprint(f'  em_loss        : {em_loss.item():.4f}')\nprint(f'  disagree_ratio : {disagree.mean().item():.3f}')\nprint(f'  lam            : {lam:.4f}')\nprint(f'  total_loss     : {total.item():.4f}')\nprint(f'  pred shape     : {pred.shape}')\nprint(f'  score shape    : {score.shape}')\nprint(f'  Dice (random)  : {d:.4f}')\nprint('=' * 45)"),

md("## 12. Download Results"),
code("from google.colab import files\nimport zipfile\nzip_path = '/content/figures.zip'\nwith zipfile.ZipFile(zip_path, 'w') as zf:\n    for f in __import__('pathlib').Path(FIG_DIR).glob('*.png'):\n        zf.write(str(f), f.name)\nfiles.download(zip_path)\nprint('Download started.')"),
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
