# DGEM — Disagreement-Guided Entropy Minimization

Semi-supervised 3D medical image segmentation for MIDL 2025.

**Core idea:** Apply entropy minimization loss *only* on voxels where the
student model and its EMA teacher disagree. This targets uncertainty exactly
where it exists — no pseudo-labels, no thresholds, no copy-paste.

## Method

```
For each unlabeled batch:
    student_pred = argmax(softmax(net(x)))
    teacher_pred = argmax(softmax(net_ema(x)))   # EMA, no gradient

    disagree_mask = (student_pred ≠ teacher_pred)
    L_em = mean_entropy(student_probs, where=disagree_mask)

Total loss = L_sup(labeled) + λ(t) * L_em(unlabeled)
```

`λ(t)` ramps up from 0 → `em_weight` over `consistency_rampup` epochs
(sigmoid schedule, same as Mean Teacher).

## Why it works

| Method | Where is unlabeled supervision applied? |
|--------|-----------------------------------------|
| Mean Teacher | All voxels (MSE between student/teacher) |
| BCP | All voxels (pseudo-label CE + Dice) |
| **DGEM** | **Only disagreement voxels (entropy minimization)** |

Disagreement = uncertainty. Sharpening predictions exactly there avoids
over-regularizing already-confident regions.

## Setup

```bash
conda env create -f environment.yml
conda activate posthoc_em
```

## Data

### 1. Download Pancreas-CT (NIH)

Register and download from TCIA:
```
https://wiki.cancerimagingarchive.net/display/Public/Pancreas-CT
```
- Dataset name: **Pancreas-CT**
- 82 abdominal CT scans with pancreas segmentation labels
- Labels: `https://zenodo.org/record/7860267` (Roth et al. labels)

### 2. Preprocess → H5

```bash
python data/preprocess_pancreas.py \
    --data_root  /path/to/Pancreas-CT/PANCREAS \
    --label_root /path/to/pancreas_labels \
    --output_dir /path/to/pancreas_h5
```

### 3. Generate splits

```bash
python data/generate_splits.py \
    --h5_dir        /path/to/pancreas_h5 \
    --label_percent 20
```

This creates `splits/pancreas/train_lab_20.txt`, `train_unlab_20.txt`, `test.txt`.

## Training

### DGEM (full method)

```bash
python train_dgem.py \
    --data_root    /path/to/pancreas_h5 \
    --splits_dir   splits/pancreas \
    --label_percent 20 \
    --max_epochs   300 \
    --em_weight    1.0 \
    --save_dir     result/dgem_20p
```

### Supervised-only baseline (ablation: em_weight=0)

```bash
python train_dgem.py \
    --data_root    /path/to/pancreas_h5 \
    --splits_dir   splits/pancreas \
    --label_percent 20 \
    --max_epochs   300 \
    --em_weight    0.0 \
    --save_dir     result/supervised_only
```

### Post-hoc EM on BCP checkpoint

```bash
python train_posthoc_em.py \
    --checkpoint  /path/to/bcp_checkpoint.pth \
    --data_root   /path/to/pancreas_h5 \
    --split_file  splits/pancreas/train_unlab_20.txt \
    --test_file   splits/pancreas/test.txt \
    --epochs 5 --lr 1e-4
```

BCP checkpoint (20% Pancreas-CT):
```
https://pan.baidu.com/s/1kGqRsEF4BX_yChKV3kMNVQ?pwd=hsjb
```

## Ablations (paper Table 2)

| Config | em_weight | consistency_rampup | ema_decay |
|--------|-----------|--------------------|-----------|
| Supervised only | 0.0 | — | — |
| Full EM (no mask) | 1.0 | 40 | 0.99 |
| **DGEM (ours)** | **1.0** | **40** | **0.99** |
| DGEM, slow ramp | 1.0 | 80 | 0.99 |
| DGEM, fast decay | 1.0 | 40 | 0.95 |

## Baselines to compare

All from SSL4MIS ([HiLab-git/SSL4MIS](https://github.com/HiLab-git/SSL4MIS)):
- UA-MT (MICCAI 2019)
- Cross-Teaching CNN+Transformer (MIDL 2022)
- BCP (CVPR 2023) — `github.com/DeepMed-Lab-ECNU/BCP`

## Repo structure

```
PostHocEM/
├── train_dgem.py          # Main DGEM training
├── train_posthoc_em.py    # Post-hoc EM on existing checkpoints
├── networks/
│   └── vnet.py            # VNet (Milletari et al. 2016)
├── dataloaders/
│   └── pancreas_loader.py # H5 dataset + augmentation
├── utils/
│   ├── losses.py          # SupLoss, entropy_loss_masked
│   ├── ramps.py           # Sigmoid ramp-up schedule
│   └── metrics.py         # Dice, HD95, sliding window inference
├── data/
│   ├── preprocess_pancreas.py  # NIfTI → H5
│   └── generate_splits.py      # Train/test/labeled splits
├── splits/pancreas/       # Split text files (generated)
├── environment.yml
└── BCP/                   # Cloned for reference (not required)
```
