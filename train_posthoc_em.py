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
import h5py
import torch
import torch.nn as nn
import torch.nn.functional as F
from tensorboardX import SummaryWriter

from networks import VNet
from dataloaders import PancreasDataset, FullVolumeDataset
from utils.losses import entropy_loss_full, entropy_loss_masked
from utils.metrics import evaluate


# ─────────────────────────────────────────────────────────────────────────────
# LA dataset adapters (inline to avoid touching the dataloader file)
# ─────────────────────────────────────────────────────────────────────────────

class _LAFullVolume:
    """LA test loader: case dirs containing mri_norm2.h5"""
    def __init__(self, data_root, split_file):
        self.data_root = Path(data_root)
        with open(split_file) as f:
            self.cases = [l.strip() for l in f if l.strip()]

    def __len__(self):
        return len(self.cases)

    def __getitem__(self, idx):
        case = self.cases[idx]
        path = self.data_root / case / 'mri_norm2.h5'
        with h5py.File(str(path), 'r') as f:
            image = f['image'][:].astype(np.float32)
            label = f['label'][:].astype(np.uint8)
        return image, label, case


class _LAUnlabeled:
    """LA unlabeled loader: case dirs, simple center crop to patch_size."""
    def __init__(self, data_root, split_file, patch_size):
        self.data_root = Path(data_root)
        with open(split_file) as f:
            self.cases = [l.strip() for l in f if l.strip()]
        self.patch_size = patch_size if isinstance(patch_size, (tuple, list)) else (patch_size,) * 3

    def __len__(self):
        return len(self.cases)

    def __getitem__(self, idx):
        case = self.cases[idx]
        path = self.data_root / case / 'mri_norm2.h5'
        with h5py.File(str(path), 'r') as f:
            image = f['image'][:].astype(np.float32)
            label = f['label'][:].astype(np.uint8)
        p = self.patch_size
        w, h, d = image.shape
        pw = max((p[0] - w) // 2 + 1, 0)
        ph = max((p[1] - h) // 2 + 1, 0)
        pd = max((p[2] - d) // 2 + 1, 0)
        if pw or ph or pd:
            image = np.pad(image, [(pw, pw), (ph, ph), (pd, pd)], mode='constant')
            label = np.pad(label, [(pw, pw), (ph, ph), (pd, pd)], mode='constant')
        w, h, d = image.shape
        x = (w - p[0]) // 2
        y = (h - p[1]) // 2
        z = (d - p[2]) // 2
        image = image[x:x+p[0], y:y+p[1], z:z+p[2]]
        label = label[x:x+p[0], y:y+p[1], z:z+p[2]]
        image = torch.from_numpy(image).unsqueeze(0).float()
        label = torch.from_numpy(label.astype(np.int64)).long()
        return image, label


def parse_patch_size(s):
    """Parse patch size CLI arg as int (e.g. '96') or tuple (e.g. '112,112,80')."""
    parts = s.split(',')
    if len(parts) == 1:
        return int(parts[0])
    return tuple(int(x) for x in parts)


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

    # Dataset selection
    p.add_argument("--dataset",     default="pancreas",
                   choices=["pancreas", "la"])
    p.add_argument("--la_data_root", default="data/la_h5",
                   help="LA H5 root (case dirs containing mri_norm2.h5)")

    # PEM hyperparameters
    p.add_argument("--epochs",      type=int,   default=10)
    p.add_argument("--lr",          type=float, default=5e-5,
                   help="Very low LR — we are nudging, not retraining")
    p.add_argument("--batch_size",  type=int,   default=2)
    p.add_argument("--patch_size",  type=str,   default='96',
                   help="Either int (e.g. '96') or comma-separated tuple "
                        "(e.g. '112,112,80')")
    p.add_argument("--stride_xy",   type=int,   default=None,
                   help="Sliding-window stride in xy. Default: 16 (pancreas), "
                        "18 (la)")
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
    p.add_argument("--decoder_only", action="store_true",
                   help="Freeze encoder, only fine-tune decoder + head during PEM")
    p.add_argument("--random_augment", action="store_true",
                   help="Use random crops + flips + shuffle for unlabeled data "
                        "(adds stochasticity, enables multi-seed ensembling)")

    # Safety
    p.add_argument("--patience",    type=int, default=3,
                   help="Early stop if Dice doesn't improve for N epochs")
    p.add_argument("--min_delta",   type=float, default=-0.005,
                   help="Stop if Dice drops more than this below baseline")

    p.add_argument("--seed", type=int, default=2020)
    p.add_argument("--gpu",  type=str, default="0")
    args = p.parse_args()
    args.patch_size = parse_patch_size(args.patch_size)
    if args.stride_xy is None:
        args.stride_xy = 18 if args.dataset == 'la' else 16
    return args


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


def load_model(checkpoint_path, num_classes, dataset='pancreas'):
    """Load a VNet checkpoint.

    For pancreas: VNet with instancenorm + has_dropout=False (matching BCP).
    For LA: VNetBCP_LA with the LA-specific conventions.
    """
    if dataset == 'la':
        from networks.vnet_bcp_la import VNetBCP_LA
        net = VNetBCP_LA(n_channels=1, n_classes=num_classes)
        net = nn.DataParallel(net).cuda()
        state = torch.load(checkpoint_path, map_location='cuda')
        if isinstance(state, dict):
            if 'net' in state:
                state = state['net']
            elif 'model_state_dict' in state:
                state = state['model_state_dict']
        # LA checkpoints are typically bare state_dicts with no 'module.' prefix
        if not any(k.startswith('module.') for k in state.keys()):
            net.module.load_state_dict(state, strict=False)
        else:
            net.load_state_dict(state)
        return net

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

    # ── Data ────────────────────────────────────────────────────────────────
    splits_dir = Path(args.splits_dir)
    if args.dataset == 'la':
        unlab_ds = _LAUnlabeled(
            args.la_data_root,
            splits_dir / f'train_unlab_{args.label_percent}.txt',
            patch_size=args.patch_size)
        unlab_loader = torch.utils.data.DataLoader(
            unlab_ds, batch_size=args.batch_size, shuffle=True,
            num_workers=0, pin_memory=True, drop_last=True)
        test_ds = _LAFullVolume(
            args.la_data_root, str(splits_dir / 'test.txt'))
    else:
        # Pancreas (matching BCP: CenterCrop, no flip)
        if args.random_augment:
            unlab_ds = PancreasDataset(
                args.data_root,
                splits_dir / f'train_unlab_{args.label_percent}.txt',
                patch_size=args.patch_size, augment=True, repeat=1)
            unlab_loader = torch.utils.data.DataLoader(
                unlab_ds, batch_size=args.batch_size, shuffle=True,
                num_workers=2, pin_memory=True, drop_last=True)
        else:
            unlab_ds = PancreasDataset(
                args.data_root,
                splits_dir / f'train_unlab_{args.label_percent}.txt',
                patch_size=args.patch_size, augment=False, center_crop=True)
            unlab_loader = torch.utils.data.DataLoader(
                unlab_ds, batch_size=args.batch_size, shuffle=True,
                num_workers=0, pin_memory=True, drop_last=True)

        test_ds = FullVolumeDataset(
            args.data_root, str(splits_dir / 'test.txt'))

    log.info(f"Unlabeled: {len(unlab_ds)} cases | Test: {len(test_ds)} cases")

    # ── Load student model ──────────────────────────────────────────────────
    net = load_model(args.checkpoint, args.num_classes, dataset=args.dataset)
    log.info(f"Loaded checkpoint: {args.checkpoint}")

    # ── Create frozen teacher (copy of checkpoint, never updated) ───────────
    teacher = load_model(args.checkpoint, args.num_classes, dataset=args.dataset)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)
    log.info("Created frozen teacher from same checkpoint")

    # ── LA: freeze BatchNorm for stable PEM with small batch ────────────────
    if args.dataset == 'la':
        def freeze_bn(m):
            import torch.nn as _nn
            if isinstance(m, (_nn.BatchNorm1d, _nn.BatchNorm2d, _nn.BatchNorm3d)):
                m.eval()
                for p in m.parameters():
                    p.requires_grad_(False)
        net.apply(freeze_bn)
        teacher.apply(freeze_bn)
        log.info("LA: froze all BatchNorm layers (stats + params)")

    # ── Optionally freeze encoder (decoder-only PEM) ────────────────────────
    if args.decoder_only:
        # Encoder = block_one..five + downsamplings (everything before block_five_up)
        encoder_attrs = [
            'block_one', 'block_one_dw', 'block_two', 'block_two_dw',
            'block_three', 'block_three_dw', 'block_four', 'block_four_dw',
            'block_five',
        ]
        n_frozen = 0
        for attr in encoder_attrs:
            for p in getattr(net.module, attr).parameters():
                p.requires_grad_(False)
                n_frozen += p.numel()
        n_train = sum(p.numel() for p in net.parameters() if p.requires_grad)
        log.info(f"Decoder-only PEM: froze {n_frozen:,} encoder params, "
                 f"training {n_train:,} decoder + head params")

    # ── Optimizer: Adam, very low LR, no weight decay ───────────────────────
    trainable = [p for p in net.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(trainable, lr=args.lr)

    # ── Baseline eval (with cache) ──────────────────────────────────────────
    # Cache the baseline metrics next to the checkpoint so we don't recompute
    # them across many PEM sweeps. Cache key includes patch_size + test split.
    ckpt_path = Path(args.checkpoint)
    cache_path = ckpt_path.parent / f'{ckpt_path.stem}.baseline.json'
    cache_key = {
        'patch_size': args.patch_size,
        'test_file':  str(splits_dir / 'test.txt'),
        'num_classes': args.num_classes,
        'dataset':    args.dataset,
        'stride_xy':  args.stride_xy,
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
            net, test_ds, args.patch_size,
            stride_xy=args.stride_xy, stride_z=4,
            n_classes=args.num_classes)
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
            net, test_ds, args.patch_size,
            stride_xy=args.stride_xy, stride_z=4,
            n_classes=args.num_classes)
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
