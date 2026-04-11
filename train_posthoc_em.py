"""
Post-hoc Entropy Minimization (PEM) for SSL Medical Image Segmentation
=======================================================================
Plug-in fine-tuning step on top of any converged SSL checkpoint.
No pseudo-labels. No thresholds. Just entropy minimization on unlabeled data.

Usage (basic):
    python train_posthoc_em.py \
        --checkpoint path/to/bcp_checkpoint.pth \
        --data_root  path/to/pancreas_h5 \
        --splits_dir splits/pancreas \
        --epochs 5 --lr 1e-4

Usage (disagreement-masked EM, needs a second checkpoint):
    python train_posthoc_em.py \
        --checkpoint  path/to/bcp_checkpoint.pth \
        --checkpoint2 path/to/crossteaching_checkpoint.pth \
        --data_root   path/to/pancreas_h5 \
        --splits_dir  splits/pancreas \
        --epochs 5 --lr 1e-4 --disagreement_mask
"""

import os
import sys
import argparse
import logging
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tensorboardX import SummaryWriter

from networks import VNet
from dataloaders import PancreasDataset, FullVolumeDataset
from utils.losses import entropy_loss_full, entropy_loss_masked
from utils.metrics import evaluate


# ─────────────────────────────────────────────────────────────────────────────
# Args
# ─────────────────────────────────────────────────────────────────────────────

def get_args():
    p = argparse.ArgumentParser(description="Post-hoc Entropy Minimization")
    p.add_argument("--checkpoint",  required=True, help="Path to converged SSL checkpoint (.pth)")
    p.add_argument("--checkpoint2", default=None,  help="Optional second checkpoint for disagreement masking")
    p.add_argument("--data_root",   required=True, help="Root directory of preprocessed h5 volumes")
    p.add_argument("--splits_dir",  required=True, help="Directory with split txt files")
    p.add_argument("--save_dir",    default="result/posthoc_em", help="Output directory")

    p.add_argument("--epochs",      type=int,   default=5)
    p.add_argument("--lr",          type=float, default=1e-4)
    p.add_argument("--batch_size",  type=int,   default=2)
    p.add_argument("--patch_size",  type=int,   default=96)
    p.add_argument("--num_classes", type=int,   default=2)
    p.add_argument("--label_percent", type=int, default=20)

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
# Model loading
# ─────────────────────────────────────────────────────────────────────────────

def load_model(checkpoint_path, num_classes):
    """Load a VNet checkpoint (instancenorm, consistent with train_dgem/evaluate)."""
    net = VNet(n_channels=1, n_classes=num_classes,
               normalization='instancenorm', has_dropout=True)
    net = nn.DataParallel(net).cuda()

    state = torch.load(checkpoint_path, map_location='cuda')
    if isinstance(state, dict):
        if 'net' in state:
            state = state['net']
        elif 'model_state_dict' in state:
            state = state['model_state_dict']
    net.load_state_dict(state)
    return net


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

    # ── CSV log ─────────────────────────────────────────────────────────────
    csv_path = Path(args.save_dir) / 'metrics.csv'
    with open(csv_path, 'w') as f:
        f.write('epoch,em_loss,dice,jaccard,hd95,asd,best_dice\n')

    # ── Data ─────────────────────────────────────────────────────────────────
    splits_dir = Path(args.splits_dir)
    unlab_ds = PancreasDataset(
        args.data_root,
        splits_dir / f'train_unlab_{args.label_percent}.txt',
        patch_size=args.patch_size, augment=True)
    unlab_loader = torch.utils.data.DataLoader(
        unlab_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=2, pin_memory=True, drop_last=True)

    test_ds = FullVolumeDataset(
        args.data_root,
        splits_dir / 'test.txt')

    log.info(f"Unlabeled: {len(unlab_ds)} cases  |  Test: {len(test_ds)} cases")

    # ── Model ────────────────────────────────────────────────────────────────
    net = load_model(args.checkpoint, args.num_classes)
    log.info(f"Loaded checkpoint: {args.checkpoint}")

    # Optional second model for disagreement masking
    net2 = None
    if args.disagreement_mask:
        if args.checkpoint2 is None:
            raise ValueError("--disagreement_mask requires --checkpoint2")
        net2 = load_model(args.checkpoint2, args.num_classes)
        net2.eval()
        for p in net2.parameters():
            p.requires_grad_(False)
        log.info(f"Loaded second checkpoint for disagreement masking: {args.checkpoint2}")

    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr, weight_decay=1e-4)

    # ── Baseline eval before fine-tuning ────────────────────────────────────
    log.info("=== Baseline evaluation (before PEM) ===")
    base_dice, base_jc, base_hd, base_asd = evaluate(
        net, test_ds, args.patch_size, n_classes=args.num_classes)
    log.info(f"[Baseline]  Dice={base_dice:.4f}  Jc={base_jc:.4f}  HD95={base_hd:.2f}  ASD={base_asd:.2f}")
    writer.add_scalar("test/dice", base_dice, 0)

    # ── Fine-tuning loop ─────────────────────────────────────────────────────
    best_dice = base_dice

    global_step = 0
    for epoch in range(1, args.epochs + 1):
        net.train()
        ep_em_loss = 0.0
        ep_steps   = 0

        for (unlab_img, _) in unlab_loader:
            unlab_img = unlab_img.cuda()
            unlab_out = net(unlab_img)[0]
            probs = F.softmax(unlab_out, dim=1)

            if args.disagreement_mask and net2 is not None:
                with torch.no_grad():
                    probs2 = F.softmax(net2(unlab_img)[0], dim=1)
                pred1 = probs.argmax(dim=1)
                pred2 = probs2.argmax(dim=1)
                mask  = (pred1 != pred2).float()
                em = entropy_loss_masked(probs, mask)
            else:
                em = entropy_loss_full(probs)

            loss = args.em_weight * em

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            ep_em_loss += em.item()
            ep_steps   += 1
            global_step += 1
            writer.add_scalar("train/em_loss", em.item(), global_step)

        log.info(f"Epoch [{epoch}/{args.epochs}]  em_loss={ep_em_loss/ep_steps:.4f}")

        # ── Eval at end of each epoch ─────────────────────────────────────
        dice, jc, hd, asd = evaluate(
            net, test_ds, args.patch_size, n_classes=args.num_classes)
        log.info(f"[Epoch {epoch}]  Dice={dice:.4f}  Jc={jc:.4f}  HD95={hd:.2f}  ASD={asd:.2f}")
        writer.add_scalar("test/dice", dice, epoch)
        writer.add_scalar("test/hd95", hd,   epoch)

        if dice > best_dice:
            best_dice = dice
            torch.save(net.state_dict(), str(save_dir / "best_model.pth"))
            log.info(f"  *** New best Dice={best_dice:.4f} -- model saved ***")

        # ── CSV log ──────────────────────────────────────────────────────
        with open(csv_path, 'a') as f:
            f.write(f'{epoch},{ep_em_loss/ep_steps:.6f},'
                    f'{dice:.6f},{jc:.6f},{hd:.4f},{asd:.4f},{best_dice:.6f}\n')

    # ── Final summary ────────────────────────────────────────────────────────
    log.info("=" * 60)
    log.info(f"Baseline Dice : {base_dice:.4f}")
    log.info(f"Best PEM Dice : {best_dice:.4f}  ({best_dice - base_dice:+.4f})")
    log.info(f"Saved to      : {save_dir}")
    writer.close()


if __name__ == "__main__":
    main()
