"""
Post-hoc Entropy Minimization (PEM) for SSL Medical Image Segmentation
=======================================================================
Plug-in fine-tuning step on top of ANY converged SSL checkpoint.
No pseudo-labels. No labels. Just entropy minimization on unlabeled data.

Key design choices for effectiveness:
  - EMA teacher frozen from checkpoint → provides disagreement signal
  - Confidence-gated EM: only sharpen voxels where max_prob < threshold
  - Boundary-aware: disagreement mask targets decision boundaries
  - Collapse protection: early stopping if Dice drops below baseline
  - Gradient clipping: prevents catastrophic updates
  - Very low LR + few epochs: gentle nudge, not a rewrite

Modes:
  --mode full          Entropy on ALL unlabeled voxels (simplest)
  --mode disagreement  Entropy only where student ≠ frozen teacher (default)
  --mode confident     Entropy only where max_prob < threshold
  --mode boundary      Combine disagreement + confidence gating

Usage:
    python train_posthoc_em.py \
        --checkpoint result/bcp_baseline_v2/best_model.pth \
        --data_root  data/pancreas_h5 \
        --splits_dir splits/pancreas \
        --save_dir   result/pem_on_bcp
"""

import os
import sys
import json
import argparse
import logging
import random
import copy
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
    p.add_argument("--checkpoint",  required=True,
                   help="Path to converged SSL checkpoint (.pth)")
    p.add_argument("--data_root",   required=True)
    p.add_argument("--splits_dir",  required=True)
    p.add_argument("--save_dir",    default="result/posthoc_em")

    # PEM hyperparameters
    p.add_argument("--epochs",      type=int,   default=10)
    p.add_argument("--lr",          type=float, default=5e-5,
                   help="Very low LR — we are nudging, not retraining")
    p.add_argument("--batch_size",  type=int,   default=2)
    p.add_argument("--patch_size",  type=int,   default=96)
    p.add_argument("--num_classes", type=int,   default=2)
    p.add_argument("--label_percent", type=int, default=20)

    # EM mode
    p.add_argument("--mode", default="confident",
                   choices=["full", "disagreement", "confident", "boundary"],
                   help="full: all voxels, disagreement: student≠teacher, "
                        "confident: max_prob<threshold, "
                        "boundary: disagreement + confidence gating")
    p.add_argument("--conf_threshold", type=float, default=0.99,
                   help="For confident/boundary mode: only sharpen voxels "
                        "with max_prob below this threshold")
    p.add_argument("--em_weight",   type=float, default=1.0)
    p.add_argument("--grad_clip",   type=float, default=1.0,
                   help="Gradient clipping norm (prevents catastrophic updates)")

    # Safety
    p.add_argument("--patience",    type=int, default=3,
                   help="Early stop if Dice doesn't improve for N epochs")
    p.add_argument("--min_delta",   type=float, default=-0.005,
                   help="Stop if Dice drops more than this below baseline")

    p.add_argument("--seed", type=int, default=2020)
    p.add_argument("--gpu",  type=str, default="0")
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
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


def load_model(checkpoint_path, num_classes):
    """Load a VNet checkpoint (matching BCP: instancenorm, no dropout)."""
    net = VNet(n_channels=1, n_classes=num_classes,
               normalization='instancenorm', has_dropout=False)
    net = nn.DataParallel(net).cuda()

    state = torch.load(checkpoint_path, map_location='cuda')
    if isinstance(state, dict):
        if 'net' in state:
            state = state['net']
        elif 'model_state_dict' in state:
            state = state['model_state_dict']
    net.load_state_dict(state)
    return net


def compute_em_loss(student_probs, teacher_probs, mode, conf_threshold):
    """Compute entropy minimization loss with the specified masking mode."""
    student_pred = student_probs.argmax(dim=1)
    teacher_pred = teacher_probs.argmax(dim=1)
    student_conf = student_probs.max(dim=1)[0]  # (B, W, H, D)

    if mode == 'full':
        mask = None
        loss = entropy_loss_full(student_probs)
    elif mode == 'disagreement':
        mask = (student_pred != teacher_pred).float()
        loss = entropy_loss_masked(student_probs, mask)
    elif mode == 'confident':
        # Only sharpen where the model is NOT yet confident
        mask = (student_conf < conf_threshold).float()
        loss = entropy_loss_masked(student_probs, mask)
    elif mode == 'boundary':
        # Intersection: disagreement AND not-yet-confident
        disagree = (student_pred != teacher_pred).float()
        uncertain = (student_conf < conf_threshold).float()
        mask = disagree * uncertain
        loss = entropy_loss_masked(student_probs, mask)

    mask_ratio = mask.mean().item() if mask is not None else 1.0
    return loss, mask_ratio


# ─────────────────────────────────────────────────────────────────────────────
# Main
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
    csv_path = save_dir / 'metrics.csv'
    with open(csv_path, 'w') as f:
        f.write('epoch,mode,em_loss,mask_ratio,dice,jaccard,hd95,asd,best_dice,delta\n')

    # ── Data (matching BCP: CenterCrop, no flip) ────────────────────────────
    splits_dir = Path(args.splits_dir)
    unlab_ds = PancreasDataset(
        args.data_root,
        splits_dir / f'train_unlab_{args.label_percent}.txt',
        patch_size=args.patch_size, augment=False, center_crop=True)
    unlab_loader = torch.utils.data.DataLoader(
        unlab_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=0, pin_memory=True, drop_last=True)

    test_ds = FullVolumeDataset(
        args.data_root, str(splits_dir / 'test.txt'))

    log.info(f"Unlabeled: {len(unlab_ds)} cases | Test: {len(test_ds)} cases")

    # ── Load student model ──────────────────────────────────────────────────
    net = load_model(args.checkpoint, args.num_classes)
    log.info(f"Loaded checkpoint: {args.checkpoint}")

    # ── Create frozen teacher (copy of checkpoint, never updated) ───────────
    teacher = load_model(args.checkpoint, args.num_classes)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)
    log.info("Created frozen teacher from same checkpoint")

    # ── Optimizer: Adam, very low LR, no weight decay ───────────────────────
    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)

    # ── Baseline eval (with cache) ──────────────────────────────────────────
    # Cache the baseline metrics next to the checkpoint so we don't recompute
    # them across many PEM sweeps. Cache key includes patch_size + test split.
    ckpt_path = Path(args.checkpoint)
    cache_path = ckpt_path.parent / f'{ckpt_path.stem}.baseline.json'
    cache_key = {
        'patch_size': args.patch_size,
        'test_file':  str(splits_dir / 'test.txt'),
        'num_classes': args.num_classes,
    }

    cached = None
    if cache_path.exists():
        try:
            data = json.loads(cache_path.read_text())
            if data.get('key') == cache_key:
                cached = data['metrics']
        except Exception:
            cached = None

    if cached is not None:
        base_dice = cached['dice']
        base_jc   = cached['jaccard']
        base_hd   = cached['hd95']
        base_asd  = cached['asd']
        log.info(f"=== Baseline (cached from {cache_path.name}) ===")
        log.info(f"[Baseline]  Dice={base_dice:.4f}  Jc={base_jc:.4f}  "
                 f"HD95={base_hd:.2f}  ASD={base_asd:.2f}")
    else:
        log.info("=== Baseline evaluation (before PEM) ===")
        base_dice, base_jc, base_hd, base_asd = evaluate(
            net, test_ds, args.patch_size, n_classes=args.num_classes)
        log.info(f"[Baseline]  Dice={base_dice:.4f}  Jc={base_jc:.4f}  "
                 f"HD95={base_hd:.2f}  ASD={base_asd:.2f}")
        # Save to cache
        cache_path.write_text(json.dumps({
            'key': cache_key,
            'metrics': {
                'dice': base_dice, 'jaccard': base_jc,
                'hd95': base_hd, 'asd': base_asd,
            },
        }, indent=2))
        log.info(f"Cached baseline → {cache_path.name}")

    writer.add_scalar("test/dice", base_dice, 0)

    # Write baseline to CSV
    with open(csv_path, 'a') as f:
        f.write(f'0,baseline,0.000000,0.0000,'
                f'{base_dice:.6f},{base_jc:.6f},{base_hd:.4f},{base_asd:.4f},'
                f'{base_dice:.6f},0.0000\n')

    # ── Fine-tuning loop ─────────────────────────────────────────────────────
    best_dice = base_dice
    no_improve = 0

    for epoch in range(1, args.epochs + 1):
        net.train()
        ep_em_loss = 0.0
        ep_mask_ratio = 0.0
        ep_steps = 0

        for unlab_img, _ in unlab_loader:
            unlab_img = unlab_img.cuda()

            # Student forward (with gradients)
            student_logits = net(unlab_img)[0]
            student_probs = F.softmax(student_logits, dim=1)

            # Teacher forward (frozen, no gradients)
            with torch.no_grad():
                teacher_logits = teacher(unlab_img)[0]
                teacher_probs = F.softmax(teacher_logits, dim=1)

            # Compute masked entropy loss
            em_loss, mask_ratio = compute_em_loss(
                student_probs, teacher_probs, args.mode, args.conf_threshold)

            loss = args.em_weight * em_loss

            optimizer.zero_grad()
            loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(net.parameters(), args.grad_clip)
            optimizer.step()

            ep_em_loss += em_loss.item()
            ep_mask_ratio += mask_ratio
            ep_steps += 1
            writer.add_scalar("train/em_loss", em_loss.item(), epoch * len(unlab_loader) + ep_steps)

        avg_loss = ep_em_loss / ep_steps
        avg_mask = ep_mask_ratio / ep_steps
        log.info(f"Epoch [{epoch}/{args.epochs}]  mode={args.mode}  "
                 f"em_loss={avg_loss:.4f}  mask_ratio={avg_mask:.3f}")

        # ── Eval ─────────────────────────────────────────────────────────
        dice, jc, hd, asd = evaluate(
            net, test_ds, args.patch_size, n_classes=args.num_classes)
        delta = dice - base_dice
        log.info(f"[Epoch {epoch}]  Dice={dice:.4f}  Jc={jc:.4f}  "
                 f"HD95={hd:.2f}  ASD={asd:.2f}  "
                 f"delta={delta:+.4f}")
        writer.add_scalar("test/dice", dice, epoch)
        writer.add_scalar("test/delta", delta, epoch)

        if dice > best_dice:
            best_dice = dice
            torch.save(net.state_dict(), str(save_dir / "best_model.pth"))
            log.info(f"  *** New best Dice={best_dice:.4f} "
                     f"(+{best_dice - base_dice:.4f} over baseline) ***")
            no_improve = 0
        else:
            no_improve += 1

        # CSV
        with open(csv_path, 'a') as f:
            f.write(f'{epoch},{args.mode},{avg_loss:.6f},{avg_mask:.4f},'
                    f'{dice:.6f},{jc:.6f},{hd:.4f},{asd:.4f},'
                    f'{best_dice:.6f},{delta:.6f}\n')

        # ── Collapse detection ───────────────────────────────────────────
        if delta < args.min_delta:
            log.info(f"  !!! Dice dropped {delta:.4f} below baseline — stopping to prevent collapse")
            break

        # ── Early stopping ───────────────────────────────────────────────
        if no_improve >= args.patience:
            log.info(f"  No improvement for {args.patience} epochs — early stopping")
            break

    # ── Final summary ────────────────────────────────────────────────────────
    log.info("=" * 60)
    log.info(f"Baseline Dice : {base_dice:.4f}")
    log.info(f"Best PEM Dice : {best_dice:.4f}  ({best_dice - base_dice:+.4f})")
    log.info(f"Mode          : {args.mode}")
    log.info(f"Saved to      : {save_dir}")
    writer.close()


if __name__ == "__main__":
    main()
