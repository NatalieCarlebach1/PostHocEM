# DGEM / PEM — Post-hoc Entropy Minimization for SSL Medical Image Segmentation

Semi-supervised 3D medical image segmentation for **MIDL 2026**.

**Core idea:** SSL methods leave prediction entropy on the table at convergence.
A post-hoc entropy minimization (PEM) fine-tuning step — requiring zero
pseudo-labels — consistently improves any SSL checkpoint by sharpening decision
boundaries in minutes of training. The optional disagreement mask (DGEM)
targets sharpening exactly where the student and EMA teacher disagree.

## Setup

```bash
conda activate ns-sam3   # or any env with PyTorch 2.0+ and CUDA
pip install -r requirements.txt
```

## Data

### 1. Download Pancreas-CT images (TCIA)

```bash
# Install NBIA Data Retriever (Ubuntu)
wget -P /tmp https://github.com/CBIIT/NBIA-TCIA/releases/download/nbia-data-retriever-4.4/nbia-data-retriever-4.4.2.deb
sudo mkdir -p /usr/share/desktop-directories/
sudo dpkg -i /tmp/nbia-data-retriever-4.4.2.deb

# Download using the manifest file included in this repo
/opt/nbia-data-retriever/nbia-data-retriever --cli \
    Pancreas-CT-20200910.tcia \
    -d data/raw \
    -v -f --agree-to-license
```

### 2. Download segmentation labels

```bash
wget -P data/raw/labels/ \
    "https://wiki.cancerimagingarchive.net/download/attachments/6261388/TCIA_pancreas_labels-02-05-2017.zip"
cd data/raw/labels && unzip TCIA_pancreas_labels-02-05-2017.zip && cd ../../..
```

### 3. Preprocess DICOM → H5

Resamples to isotropic 1mm, crops around pancreas (25-voxel pad), normalizes to [0,1]:

```bash
python data/preprocess_pancreas.py \
    --input_format dicom \
    --data_root data/raw/Pancreas-CT-20200910 \
    --label_root data/raw/labels \
    --output_dir data/pancreas_h5
```

### 4. Generate canonical splits

```bash
python data/generate_splits.py \
    --use_bcp_splits \
    --splits_dir splits/pancreas
```

Creates the exact BCP/CoraNet splits: 18 test, 12 labeled (20%), 50 unlabeled.

### LA dataset (Left Atrium MRI)

LA H5 files are bundled in the UA-MT repo:

```bash
git clone --depth 1 https://github.com/yulequan/UA-MT /tmp/uamt
cp -r /tmp/uamt/data/2018LA_Seg_Training\ Set data/la_h5/
```

Splits are already provided in `splits/la/` (canonical BCP splits: 80 train / 20 test).
PEM uses the publicly released BCP LA checkpoints — see `result/bcp_pretrained/LA_5.pth`
and `result/bcp_pretrained/LA_10.pth` (downloaded directly from the BCP GitHub repo,
no Baidu Pan needed).

## Training

### BCP Baseline (reproducing CVPR 2023)

Two-phase training matching the original repo: 60 epochs CutMix pretrain + 200 epochs BCP self-train.

```bash
python train_bcp_baseline.py \
    --data_root data/pancreas_h5 \
    --splits_dir splits/pancreas \
    --save_dir result/bcp_baseline
```

### DGEM (from scratch)

30-epoch supervised warmup, then disagreement-guided entropy minimization:

```bash
python train_dgem.py \
    --data_root data/pancreas_h5 \
    --splits_dir splits/pancreas \
    --max_epochs 300 \
    --save_dir result/dgem_20p
```

### Post-hoc PEM on any checkpoint

Fine-tune a converged checkpoint with entropy minimization on unlabeled data.

#### Pancreas-CT (20% labels)

```bash
python train_posthoc_em.py \
    --checkpoint result/bcp_baseline_v2/best_model.pth \
    --data_root data/pancreas_h5 \
    --splits_dir splits/pancreas \
    --label_percent 20 \
    --mode full --lr 5e-5 \
    --epochs 10 --patience 5 \
    --save_dir result/pem_pancreas_20pct
```

#### LA (5% or 10% labels)

```bash
python train_posthoc_em.py \
    --dataset la \
    --checkpoint result/bcp_pretrained/LA_5.pth \
    --data_root data/la_h5 \
    --la_data_root "data/la_h5/2018LA_Seg_Training Set" \
    --splits_dir splits/la \
    --label_percent 5 \
    --patch_size 112,112,80 \
    --num_classes 2 \
    --mode confident --conf_threshold 0.95 --lr 1e-5 \
    --epochs 10 --patience 5 \
    --save_dir result/pem_la_5pct
```

For LA 10%, use `--checkpoint result/bcp_pretrained/LA_10.pth --label_percent 10
--mode confident --conf_threshold 0.9 --lr 5e-6`.

### Hyperparameter sweeps

Two sweep scripts are provided to scan LRs × masking modes:

```bash
python scripts/pem_sweep.py     # Pancreas-CT (all label fractions)
python scripts/pem_sweep_la.py  # LA at 5% and 10%
```

Each sweep writes incremental results to `result/pem_*sweep_summary.csv` and a
human-readable log to `result/pem_*sweep.log`.

### Mask ablations (DGEM)

```bash
python train_dgem.py --mask_type full   --save_dir result/dgem_full ...
python train_dgem.py --mask_type random --save_dir result/dgem_random ...
python train_dgem.py --mask_type soft   --save_dir result/dgem_soft ...
```

### Visualization

```bash
python visualize_disagreement.py \
    --checkpoint result/dgem_20p/best_model.pth \
    --data_root data/pancreas_h5 \
    --test_file splits/pancreas/test.txt
```

## Results

PEM is applied to publicly released BCP pretrained checkpoints on three benchmark
configurations across two datasets. Every configuration shows positive Dice and
HD95 improvements.

### Main results

| Dataset / Labels | Method | Dice (%) | Jaccard (%) | HD95 | ASD |
|---|---|---|---|---|---|
| **Pancreas-CT 20%** | BCP (our reprod.) | 82.89 | 71.01 | 7.81 | 2.62 |
| | **BCP + PEM (ours)** | **84.03** | **72.68** | **5.65** | **1.67** |
| | *Δ* | *+1.14* | *+1.67* | *−2.16 (−28%)* | *−0.95* |
| **LA 5%** | BCP (our reprod.) | 87.32 | 77.65 | 13.91 | 3.60 |
| | **BCP + PEM (ours)** | **88.93** | **80.16** | **7.77** | **1.92** |
| | *Δ* | *+1.61* | *+2.51* | *−6.13 (−44%)* | *−1.68* |
| **LA 10%** | BCP (our reprod.) | 89.40 | 80.92 | 9.88 | 2.86 |
| | **BCP + PEM (ours)** | **90.29** | **82.37** | **6.50** | **2.17** |
| | *Δ* | *+0.89* | *+1.45* | *−3.38 (−34%)* | *−0.69* |

### Best PEM configuration per dataset

| Setting | Mode | LR | Mask % | Δ Dice | Δ HD95 |
|---|---|---|---|---|---|
| Pancreas 20% | full | 5e-5 | 100% | +1.14 | −28% |
| LA 5% | confident (τ=0.95) | 1e-5 | ~7.9% | +1.61 | −44% |
| LA 10% | confident (τ=0.9) | 5e-6 | ~5.9% | +0.89 | −34% |

LA, with its higher base confidence, benefits from the *confident* mode which
restricts entropy minimization to the small subset of voxels with `max_prob < τ`.
Pancreas-CT, with a less saturated base, gets the same improvement from full-volume
entropy because the gradient is naturally dominated by uncertain voxels.

### Base model requirement

| Base Model | Base Dice | +PEM | Δ |
|---|---|---|---|
| Supervised VNet (Pancreas 20%) | 75.66% | 75.66% (no improvement) | 0.00 |
| BCP SSL (Pancreas 20%) | 82.89% | **84.03%** | **+1.14** |

PEM requires a well-converged SSL model. It does not improve supervised-only baselines —
the residual entropy in a supervised model is diffuse, not boundary-concentrated.

## Evaluation

```bash
python evaluate.py \
    --checkpoint result/bcp_baseline_v2/best_model.pth \
    --data_root data/pancreas_h5 \
    --test_file splits/pancreas/test.txt
```

## Repo structure

```
PostHocEM/
├── train_bcp_baseline.py      # BCP reproduction (2-phase, matches original)
├── train_dgem.py              # DGEM training (warmup + disagreement EM)
├── train_posthoc_em.py        # Post-hoc PEM on any checkpoint (Pancreas + LA)
├── evaluate.py                # Standalone evaluation
├── ensemble_eval.py           # Multi-checkpoint ensemble evaluation
├── visualize_disagreement.py  # Qualitative figure generation
├── networks/
│   ├── vnet.py                # VNet (Milletari et al. 2016)
│   └── vnet_bcp_la.py         # BCP-LA-compatible VNet (BatchNorm + extra heads)
├── dataloaders/
│   └── pancreas_loader.py     # H5 dataset + RandomCrop3D + CenterCrop3D
├── utils/
│   ├── losses.py              # SupLoss, DiceLoss (per-sample, masked), entropy
│   ├── ramps.py               # Sigmoid ramp-up schedule
│   └── metrics.py             # Dice, HD95, sliding window inference
├── data/
│   ├── download_pancreas.py   # Pancreas data download (TCIA / ssl4mis)
│   ├── preprocess_pancreas.py # DICOM/NIfTI → isotropic H5
│   └── generate_splits.py     # Canonical BCP/CoraNet splits
├── splits/
│   ├── pancreas/              # 5/10/20/40/60/80% label splits
│   └── la/                    # BCP canonical LA splits at 5% / 10%
├── scripts/
│   ├── pem_sweep.py           # Pancreas LR × mode sweep
│   └── pem_sweep_la.py        # LA LR × mode sweep
├── paper/
│   ├── main.tex               # MIDL 2026 short paper
│   └── references.bib
├── requirements.txt
└── Pancreas-CT-20200910.tcia  # TCIA download manifest
```

## References

- BCP: Bai et al., "Bidirectional Copy-Paste for Semi-Supervised Medical Image Segmentation", CVPR 2023
- DyCON: Assefa et al., "Dynamic Uncertainty-aware Consistency and Contrastive Learning", CVPR 2025
- SSL4MIS: https://github.com/HiLab-git/SSL4MIS
