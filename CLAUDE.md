# PostHocEM — Post-hoc Entropy Minimization for SSL Medical Image Segmentation

## Project Goal

Short paper (4 pages) for MIDL 2025.

**Core claim:** SSL methods stop exploiting unlabeled data at convergence. A post-hoc entropy minimization fine-tuning step — requiring zero pseudo-labels — consistently improves any SSL checkpoint by sharpening decision boundaries in minutes of training.

**Target benchmark:** SSL4MIS (Pancreas-CT NIH, 20% labeled) and LA dataset.
**Baseline to beat:** BCP (CVPR 2023) — checkpoint available.

---

## Method

### Post-hoc Entropy Minimization (PEM)

After any SSL method finishes training:

1. Load the converged SSL checkpoint (e.g., BCP)
2. Fine-tune for 2-5 epochs on **unlabeled data only**
3. Loss = entropy minimization: `L = -sum(p * log(p + eps))`
4. No pseudo-labels. No labels. No thresholds.

```python
probs = torch.softmax(model(unlabeled_vol), dim=1)
loss = -(probs * torch.log(probs + 1e-8)).sum(dim=1).mean()
```

### Optional Extension: Disagreement-Masked EM

Run BCP + Cross-Teaching checkpoints in parallel.
Only apply entropy loss on voxels where the two models **disagree**:

```python
disagreement_mask = (bcp_pred != ct_pred).float()
loss = (-(probs * torch.log(probs + 1e-8)).sum(dim=1) * disagreement_mask).mean()
```

This targets sharpening at uncertain regions only — avoiding overconfident collapse.

---

## Experiments

### Datasets
- **Pancreas-CT (NIH):** 82 cases, 62 train / 20 test, 20% labeled = ~12 labeled volumes
- **LA (Left Atrium):** 100 cases, 80 train / 20 test, 20% labeled = 16 labeled volumes

### Baselines (all from SSL4MIS + BCP repo)
- UA-MT (MICCAI 2019)
- Cross-Teaching CNN+Transformer (MIDL 2022)
- BCP (CVPR 2023) ← main baseline, checkpoint available

### Ablations
- Epochs: 1, 2, 3, 5, 10
- Learning rate: 1e-4, 5e-5, 1e-5
- With/without disagreement masking
- Applied to different base checkpoints (UA-MT, Cross-Teaching, BCP)

### Metrics
- Dice Score (%), Jaccard Index (%)
- 95% Hausdorff Distance (HD95)

---

## Key Resources

### Checkpoints
- **BCP Pancreas-CT 20%:** `https://pan.baidu.com/s/1kGqRsEF4BX_yChKV3kMNVQ?pwd=hsjb`
- **SSL4MIS checkpoints:** inside `HiLab-git/SSL4MIS` repo

### Base Repos to Build On
- SSL4MIS: `https://github.com/HiLab-git/SSL4MIS`
- BCP: `https://github.com/DeepMed-Lab-ECNU/BCP`

### Data
- Pancreas-CT (NIH): download via SSL4MIS instructions
- LA dataset: download via SSL4MIS instructions
- KiTS19 (optional extension): `/c/Users/Lenovo/kits19`

---

## Paper Structure (4 pages MIDL short)

1. **Introduction** (~0.5 page): SSL methods abandon unlabeled data at convergence. We propose PEM.
2. **Method** (~0.75 page): Entropy minimization loss, disagreement masking extension.
3. **Experiments** (~1.5 pages): Main results table + ablation table + 1 qualitative figure.
4. **Conclusion** (~0.25 page)

---

## Stack

- Python, PyTorch
- nibabel (for NIfTI loading)
- SimpleITK (preprocessing)
- CUDA GPU required
- Build directly on BCP repo structure

## Timeline (5 days)

- Day 1: Setup repo, download checkpoints, run BCP baseline eval to verify numbers match paper
- Day 2: Implement PEM fine-tuning loop, run on Pancreas-CT 20%
- Day 3: Ablations (epochs, LR, disagreement masking), run on LA dataset
- Day 4: Write paper, generate figures
- Day 5: Polish, proofread, submit
