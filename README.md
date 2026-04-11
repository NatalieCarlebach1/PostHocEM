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

Fine-tune a converged checkpoint with entropy minimization on unlabeled data:

```bash
python train_posthoc_em.py \
    --checkpoint result/bcp_baseline/best_model.pth \
    --data_root data/pancreas_h5 \
    --splits_dir splits/pancreas \
    --epochs 5 --lr 1e-4 \
    --save_dir result/pem_on_bcp
```

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

## Evaluation

```bash
python evaluate.py \
    --checkpoint result/bcp_baseline/best_model.pth \
    --data_root data/pancreas_h5 \
    --test_file splits/pancreas/test.txt
```

## Repo structure

```
PostHocEM/
├── train_bcp_baseline.py      # BCP reproduction (2-phase, matches original)
├── train_dgem.py              # DGEM training (warmup + disagreement EM)
├── train_posthoc_em.py        # Post-hoc PEM on any checkpoint
├── evaluate.py                # Standalone evaluation
├── visualize_disagreement.py  # Qualitative figure generation
├── networks/
│   └── vnet.py                # VNet (Milletari et al. 2016)
├── dataloaders/
│   └── pancreas_loader.py     # H5 dataset + augmentation + CenterCrop
├── utils/
│   ├── losses.py              # SupLoss, DiceLoss (per-sample, masked), entropy
│   ├── ramps.py               # Sigmoid ramp-up schedule
│   └── metrics.py             # Dice, HD95, sliding window inference
├── data/
│   ├── download_pancreas.py   # Data download (TCIA/ssl4mis/synthetic)
│   ├── preprocess_pancreas.py # DICOM/NIfTI → isotropic H5
│   └── generate_splits.py     # Canonical BCP/CoraNet splits
├── splits/pancreas/           # Train/test split files
├── requirements.txt
└── Pancreas-CT-20200910.tcia  # TCIA download manifest
```

## References

- BCP: Bai et al., "Bidirectional Copy-Paste for Semi-Supervised Medical Image Segmentation", CVPR 2023
- DyCON: Assefa et al., "Dynamic Uncertainty-aware Consistency and Contrastive Learning", CVPR 2025
- SSL4MIS: https://github.com/HiLab-git/SSL4MIS
