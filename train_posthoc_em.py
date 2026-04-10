"""
Post-hoc Entropy Minimization (PEM) for SSL Medical Image Segmentation
=======================================================================
Plug-in fine-tuning step on top of any converged SSL checkpoint.
No pseudo-labels. No thresholds. Just entropy minimization on unlabeled data.

Usage (basic):
    python train_posthoc_em.py \
        --checkpoint path/to/bcp_checkpoint.pth \
        --data_root  path/to/pancreas_data \
        --split_file path/to/BCP/data_split/pancreas/train_20.txt \
        --test_file  path/to/BCP/data_split/pancreas/test.txt \
        --epochs 5 --lr 1e-4

Usage (disagreement-masked EM, needs a second checkpoint):
    python train_posthoc_em.py \
        --checkpoint  path/to/bcp_checkpoint.pth \
        --checkpoint2 path/to/crossteaching_checkpoint.pth \
        --data_root   path/to/pancreas_data \
        --split_file  path/to/BCP/data_split/pancreas/train_20.txt \
        --test_file   path/to/BCP/data_split/pancreas/test.txt \
        --epochs 5 --lr 1e-4 --disagreement_mask
"""

import os
import sys
import argparse
import logging
import random
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import h5py
from tensorboardX import SummaryWriter
from medpy import metric

# ── Import BCP pancreas utilities ────────────────────────────────────────────
BCP_PANCREAS = Path(__file__).parent / "BCP" / "code" / "pancreas"
sys.path.insert(0, str(BCP_PANCREAS))
from Vnet import VNet
from test_util import test_DTC_single_case, calculate_metric_percase


# ─────────────────────────────────────────────────────────────────────────────
# Args
# ─────────────────────────────────────────────────────────────────────────────

def get_args():
    p = argparse.ArgumentParser(description="Post-hoc Entropy Minimization")
    p.add_argument("--checkpoint",  required=True, help="Path to converged SSL checkpoint (.pth)")
    p.add_argument("--checkpoint2", default=None,  help="Optional second checkpoint for disagreement masking")
    p.add_argument("--data_root",   required=True, help="Root directory of preprocessed h5 volumes")
    p.add_argument("--split_file",  required=True, help="Train split txt (labeled+unlabeled case names)")
    p.add_argument("--test_file",   required=True, help="Test split txt (case names)")
    p.add_argument("--save_dir",    default="result/posthoc_em", help="Output directory")

    p.add_argument("--epochs",      type=int,   default=5)
    p.add_argument("--lr",          type=float, default=1e-4)
    p.add_argument("--batch_size",  type=int,   default=2)
    p.add_argument("--patch_size",  type=int,   default=64)
    p.add_argument("--num_classes", type=int,   default=2)
    p.add_argument("--label_percent", type=int, default=20)

    # EM loss weight relative to supervised loss on labeled vols
    p.add_argument("--em_weight",   type=float, default=1.0,
                   help="Weight on entropy loss (set 0 to run supervised-only fine-tune)")
    p.add_argument("--disagreement_mask", action="store_true",
                   help="Only apply entropy loss where checkpoint1 and checkpoint2 disagree")

    p.add_argument("--seed", type=int, default=2020)
    p.add_argument("--gpu",  type=str, default="0")
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Reproducibility
# ─────────────────────────────────────────────────────────────────────────────

def seed_everything(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# ─────────────────────────────────────────────────────────────────────────────
# Dataset  (compatible with BCP's h5 format)
# ─────────────────────────────────────────────────────────────────────────────

class PancreasH5Dataset(Dataset):
    """Loads random 3-D patches from h5 volumes.
    Each h5 file must contain 'image' and 'label' keys (BCP format).
    """
    def __init__(self, data_root, case_names, patch_size):
        self.data_root = Path(data_root)
        self.cases     = case_names
        self.patch_size = patch_size

    def __len__(self):
        return len(self.cases)

    def __getitem__(self, idx):
        case = self.cases[idx]
        h5_path = self.data_root / f"{case}.h5"
        with h5py.File(str(h5_path), "r") as f:
            image = f["image"][:]   # (W, H, D)  float32
            label = f["label"][:]   # (W, H, D)  uint8

        image, label = self._random_crop(image, label)

        image = torch.from_numpy(image).unsqueeze(0).float()   # (1, p, p, p)
        label = torch.from_numpy(label).long()                  # (p, p, p)
        return image, label

    def _random_crop(self, image, label):
        p = self.patch_size
        w, h, d = image.shape

        # Pad if smaller than patch
        pw = max((p - w) // 2 + 1, 0)
        ph = max((p - h) // 2 + 1, 0)
        pd = max((p - d) // 2 + 1, 0)
        if pw or ph or pd:
            image = np.pad(image, [(pw, pw), (ph, ph), (pd, pd)], mode="constant")
            label = np.pad(label, [(pw, pw), (ph, ph), (pd, pd)], mode="constant")

        w, h, d = image.shape
        x = np.random.randint(0, w - p + 1)
        y = np.random.randint(0, h - p + 1)
        z = np.random.randint(0, d - p + 1)
        return image[x:x+p, y:y+p, z:z+p], label[x:x+p, y:y+p, z:z+p]


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_cases(txt_path):
    with open(txt_path) as f:
        return [l.strip() for l in f if l.strip()]


def load_vnet(checkpoint_path, num_classes):
    net = VNet(n_channels=1, n_classes=num_classes, normalization="batchnorm", has_dropout=True)
    net = nn.DataParallel(net)
    net = net.cuda()
    state = torch.load(checkpoint_path, map_location="cuda")
    # BCP saves either full dict or bare state_dict
    if isinstance(state, dict) and "net" in state:
        net.load_state_dict(state["net"])
    elif isinstance(state, dict) and "model_state_dict" in state:
        net.load_state_dict(state["model_state_dict"])
    else:
        net.load_state_dict(state)
    return net


def entropy_loss(probs):
    """Shannon entropy: -sum(p * log(p)), averaged over all voxels in batch."""
    return -(probs * torch.log(probs + 1e-8)).sum(dim=1).mean()


def entropy_loss_masked(probs, mask):
    """Entropy loss applied only where mask > 0."""
    H = -(probs * torch.log(probs + 1e-8)).sum(dim=1)   # (B, W, H, D)
    if mask.sum() == 0:
        return torch.tensor(0.0, device=probs.device)
    return (H * mask).sum() / mask.sum()


def test_all_cases(net, test_cases, data_root, patch_size, num_classes, stride_xy=16, stride_z=4):
    net.eval()
    dice_list, jc_list, hd_list, asd_list = [], [], [], []
    with torch.no_grad():
        for case in test_cases:
            h5_path = Path(data_root) / f"{case}.h5"
            with h5py.File(str(h5_path), "r") as f:
                image = f["image"][:]
                label = f["label"][:]

            pred_score, _ = test_DTC_single_case(
                net, image,
                stride_xy=stride_xy, stride_z=stride_z,
                patch_size=(patch_size, patch_size, patch_size),
                num_classes=num_classes
            )
            pred = (pred_score[1] > 0.5).astype(np.uint8)   # class-1 prob map
            if pred.sum() == 0:
                dice_list.append(0); jc_list.append(0)
                hd_list.append(200); asd_list.append(200)
                continue
            d, jc, hd, asd = calculate_metric_percase(pred, label)
            dice_list.append(d); jc_list.append(jc)
            hd_list.append(hd); asd_list.append(asd)

    return (np.mean(dice_list), np.mean(jc_list),
            np.mean(hd_list),  np.mean(asd_list))


# ─────────────────────────────────────────────────────────────────────────────
# Main training loop
# ─────────────────────────────────────────────────────────────────────────────

def main():
    args = get_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    seed_everything(args.seed)

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # ── Logging ──────────────────────────────────────────────────────────────
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(message)s",
        handlers=[
            logging.FileHandler(str(save_dir / "train.log")),
            logging.StreamHandler(sys.stdout),
        ]
    )
    log = logging.getLogger()
    writer = SummaryWriter(str(save_dir))

    log.info(f"Args: {vars(args)}")

    # ── Load cases ───────────────────────────────────────────────────────────
    all_train = load_cases(args.split_file)
    test_cases = load_cases(args.test_file)

    # Labeled: first label_percent%, unlabeled: rest  (same convention as BCP)
    n_labeled   = max(1, int(len(all_train) * args.label_percent / 100))
    labeled_cases   = all_train[:n_labeled]
    unlabeled_cases = all_train[n_labeled:]

    log.info(f"Labeled: {len(labeled_cases)}  |  Unlabeled: {len(unlabeled_cases)}  |  Test: {len(test_cases)}")

    # ── Dataloaders ──────────────────────────────────────────────────────────
    lab_ds   = PancreasH5Dataset(args.data_root, labeled_cases,   args.patch_size)
    unlab_ds = PancreasH5Dataset(args.data_root, unlabeled_cases, args.patch_size)

    lab_loader   = DataLoader(lab_ds,   batch_size=args.batch_size, shuffle=True,
                              num_workers=2, pin_memory=True, drop_last=True)
    unlab_loader = DataLoader(unlab_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=2, pin_memory=True, drop_last=True)

    # ── Model ────────────────────────────────────────────────────────────────
    net = load_vnet(args.checkpoint, args.num_classes)
    log.info(f"Loaded checkpoint: {args.checkpoint}")

    # Optional second model for disagreement masking
    net2 = None
    if args.disagreement_mask:
        if args.checkpoint2 is None:
            raise ValueError("--disagreement_mask requires --checkpoint2")
        net2 = load_vnet(args.checkpoint2, args.num_classes)
        net2.eval()
        for p in net2.parameters():
            p.requires_grad_(False)
        log.info(f"Loaded second checkpoint for disagreement masking: {args.checkpoint2}")

    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr, weight_decay=1e-4)
    dice_loss_fn = nn.CrossEntropyLoss()   # used as supervised CE on labeled batches

    # ── Baseline eval before fine-tuning ────────────────────────────────────
    log.info("=== Baseline evaluation (before PEM) ===")
    base_dice, base_jc, base_hd, base_asd = test_all_cases(
        net, test_cases, args.data_root, args.patch_size, args.num_classes
    )
    log.info(f"[Baseline]  Dice={base_dice:.4f}  Jc={base_jc:.4f}  HD95={base_hd:.2f}  ASD={base_asd:.2f}")
    writer.add_scalar("test/dice", base_dice, 0)

    # ── Fine-tuning loop ─────────────────────────────────────────────────────
    best_dice = base_dice
    unlab_iter = iter(unlab_loader)

    global_step = 0
    for epoch in range(1, args.epochs + 1):
        net.train()
        ep_sup_loss = 0.0
        ep_em_loss  = 0.0
        ep_steps    = 0

        for (lab_img, lab_lbl) in lab_loader:
            # ── Supervised loss on labeled batch ─────────────────────────────
            lab_img = lab_img.cuda()
            lab_lbl = lab_lbl.cuda()
            sup_out = net(lab_img)[0]   # VNet returns (output, ...)
            sup_loss = dice_loss_fn(sup_out, lab_lbl)

            # ── Entropy loss on unlabeled batch ──────────────────────────────
            try:
                unlab_img, _ = next(unlab_iter)
            except StopIteration:
                unlab_iter = iter(unlab_loader)
                unlab_img, _ = next(unlab_iter)

            unlab_img = unlab_img.cuda()
            unlab_out = net(unlab_img)[0]
            probs = F.softmax(unlab_out, dim=1)

            if args.disagreement_mask and net2 is not None:
                with torch.no_grad():
                    probs2 = F.softmax(net2(unlab_img)[0], dim=1)
                pred1  = probs.argmax(dim=1)
                pred2  = probs2.argmax(dim=1)
                mask   = (pred1 != pred2).float()   # 1 where models disagree
                em     = entropy_loss_masked(probs, mask)
            else:
                em = entropy_loss(probs)

            loss = sup_loss + args.em_weight * em

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            ep_sup_loss += sup_loss.item()
            ep_em_loss  += em.item()
            ep_steps    += 1
            global_step += 1

            writer.add_scalar("train/sup_loss", sup_loss.item(), global_step)
            writer.add_scalar("train/em_loss",  em.item(),       global_step)

        log.info(
            f"Epoch [{epoch}/{args.epochs}]  "
            f"sup_loss={ep_sup_loss/ep_steps:.4f}  "
            f"em_loss={ep_em_loss/ep_steps:.4f}"
        )

        # ── Eval at end of each epoch ─────────────────────────────────────
        dice, jc, hd, asd = test_all_cases(
            net, test_cases, args.data_root, args.patch_size, args.num_classes
        )
        log.info(f"[Epoch {epoch}]  Dice={dice:.4f}  Jc={jc:.4f}  HD95={hd:.2f}  ASD={asd:.2f}")
        writer.add_scalar("test/dice", dice, epoch)
        writer.add_scalar("test/hd95", hd,   epoch)

        if dice > best_dice:
            best_dice = dice
            torch.save(net.state_dict(), str(save_dir / "best_model.pth"))
            log.info(f"  *** New best Dice={best_dice:.4f} — model saved ***")

    # ── Final summary ────────────────────────────────────────────────────────
    log.info("=" * 60)
    log.info(f"Baseline Dice : {base_dice:.4f}")
    log.info(f"Best PEM Dice : {best_dice:.4f}  (+{best_dice - base_dice:+.4f})")
    log.info(f"Saved to      : {save_dir}")
    writer.close()


if __name__ == "__main__":
    main()
