# DGEM / PEM — Disagreement-Guided Entropy Minimization for SSL Medical Image Segmentation

## Project Goal

Short paper (4 pages) for **MIDL 2026** (deadline: April 15, 2026 AoE).

**Core claim:** SSL methods leave prediction entropy on the table at convergence. A post-hoc entropy minimization step (PEM) — requiring zero pseudo-labels — consistently improves any SSL checkpoint. The disagreement mask (DGEM) targets sharpening exactly where uncertainty exists.

**Target benchmark:** Pancreas-CT NIH, 20% labeled (BCP/CoraNet splits: 12 lab / 50 unlab / 18 test).
**Baselines:** BCP (CVPR 2023), DyCON (CVPR 2025), MOST (MICCAI 2024).

---

## Method

### Post-hoc Entropy Minimization (PEM) — Main contribution

After any SSL method finishes training:
1. Load the converged SSL checkpoint
2. Fine-tune for 2-5 epochs on **unlabeled data only**
3. Loss = entropy minimization on disagreement voxels (student vs EMA teacher)
4. No pseudo-labels. No labels. No thresholds.

### DGEM — From-scratch training variant

Student + EMA teacher training with entropy loss applied only on disagreement voxels. Supports ablation with `--mask_type`: disagreement, full, random, soft.

---

## Architecture & Training Details

### Model
- **VNet** (Milletari et al. 2016), instancenorm, `has_dropout=False`
- Wrapped in `nn.DataParallel`

### BCP Baseline (must match original exactly)
- **Optimizer:** Adam, lr=1e-3, no weight decay, no LR schedule
- **Phases:** 60 pretrain (CutMix labeled) + 200 self-train (full BCP) = 260 epochs
- **CutMix cube:** 64³ within 96³ crop (corner-based positioning)
- **Loss weights:** l_weight=1.0, u_weight=0.5 (original defaults)
- **Dice loss:** Per-sample, masked per region (NOT micro, NOT composite)
- **Pseudo-labels:** argmax + largest connected component (26-connected)
- **Unlabeled data:** CenterCrop, no flip. Labeled: RandomCrop, no flip.
- **EMA:** decay=0.99, ema_net in train() mode
- **Batch size:** 2, num_workers=0

### DGEM
- **Optimizer:** Adam, lr=1e-3
- **Warmup:** 30 epochs supervised only, then EM ramps over 100 epochs
- **EM weight:** 0.3 max

### PEM (post-hoc)
- **Optimizer:** Adam, lr=1e-4
- **Epochs:** 5
- **Loss:** entropy minimization on unlabeled data only

---

## Data

### Pancreas-CT (NIH)
- 80 valid cases (0025 and 0070 excluded as duplicates)
- **Preprocessing (must match CoraNet/BCP):**
  - Isotropic resampling to 1mm x 1mm x 1mm
  - Bbox crop around pancreas + 25-voxel padding, min 96³
  - HU clip [-125, 275], per-volume min-max normalization to [0,1]
- **Splits (BCP/CoraNet canonical, hardcoded):**
  - Test: 18 cases (0064-0082, excluding 0070)
  - Labeled 20%: 12 cases (0001-0012)
  - Unlabeled: 50 cases (0013-0063, excluding 0025)

### Download
1. TCIA images: NBIA Data Retriever CLI with `Pancreas-CT-20200910.tcia`
2. Labels: `TCIA_pancreas_labels-02-05-2017.zip` from TCIA wiki
3. BCP checkpoint: Baidu Pan (browser only, no CLI)

---

## Evaluation
- Sliding window: patch=96³, stride_xy=16, stride_z=4
- Metrics: Dice (%), Jaccard (%), HD95, ASD
- No test-time augmentation (matching BCP/DyCON)

## Target Numbers (Pancreas-CT 20%)

| Method | Dice (%) | Source |
|--------|----------|--------|
| V-Net supervised | 69.96 | BCP paper |
| UA-MT | 77.26 | BCP paper |
| BCP | **82.91** | BCP paper |
| DyCON | **84.81** | DyCON paper |
| BCP + PEM (ours) | **>83.5** | Target |

---

## Key Resources

- BCP repo: https://github.com/DeepMed-Lab-ECNU/BCP
- SSL4MIS: https://github.com/HiLab-git/SSL4MIS
- CoraNet (splits/preprocess): https://github.com/koncle/CoraNet
- MOST: https://github.com/CUHK-AIM-Group/MOST-SSL4MIS
- DyCON: https://github.com/KU-CVML/DyCON

## Stack

- Python 3.12, PyTorch 2.9+, CUDA
- SimpleITK, nibabel, scipy, h5py, medpy, tensorboardX, matplotlib
- Conda env: `ns-sam3`
