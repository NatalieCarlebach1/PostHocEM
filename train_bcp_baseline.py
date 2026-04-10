"""
BCP Baseline — Bidirectional Copy-Paste (CVPR 2023)
====================================================
Re-implemented using our infrastructure (same VNet, same dataloaders, same
eval) so results are directly comparable to DGEM.

Reference: Bai et al., "Bidirectional Copy-Paste for Semi-Supervised Medical
Image Segmentation", CVPR 2023.
GitHub: https://github.com/DeepMed-Lab-ECNU/BCP

Algorithm:
    For each step, sample labeled (img_a, lab_a) and unlabeled (unimg_a):
    1. Generate pseudo-label for unlabeled via EMA model
    2. Mix labeled ← unlabeled patch (foreground=labeled, background=unlabeled)
       Mix unlabeled ← labeled patch (foreground=unlabeled, background=labeled)
    3. Supervised loss on both mixed outputs

Usage:
    python train_bcp_baseline.py \
        --data_root  /path/to/pancreas_h5 \
        --splits_dir splits/pancreas \
        --label_percent 20 \
        --max_epochs 300 \
        --save_dir   result/bcp_baseline
"""

import os
import sys
import random
import logging
import argparse
from pathlib import Path
import numpy as np

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tensorboardX import SummaryWriter

from networks import VNet
from dataloaders import get_loaders, FullVolumeDataset
from utils.losses import SupLoss, DiceLoss
from utils.ramps import get_current_consistency_weight
from utils.metrics import evaluate


# ─────────────────────────────────────────────────────────────────────────────
# Args
# ─────────────────────────────────────────────────────────────────────────────

def get_args():
    p = argparse.ArgumentParser()
    p.add_argument('--data_root',       required=True)
    p.add_argument('--splits_dir',      default='splits/pancreas')
    p.add_argument('--label_percent',   type=int,   default=20)
    p.add_argument('--patch_size',      type=int,   default=96)
    p.add_argument('--max_epochs',      type=int,   default=300)
    p.add_argument('--batch_size',      type=int,   default=2)
    p.add_argument('--lr',              type=float, default=1e-3)
    p.add_argument('--num_classes',     type=int,   default=2)
    p.add_argument('--num_workers',     type=int,   default=2)
    p.add_argument('--ema_decay',       type=float, default=0.99)
    p.add_argument('--consistency',     type=float, default=0.1)
    p.add_argument('--consistency_rampup', type=int, default=40)
    p.add_argument('--eval_every',      type=int,   default=10)
    p.add_argument('--save_dir',        default='result/bcp_baseline')
    p.add_argument('--gpu',             default='0')
    p.add_argument('--seed',            type=int,   default=2020)
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def create_model(n_classes, ema=False):
    model = VNet(n_channels=1, n_classes=n_classes,
                 normalization='instancenorm', has_dropout=True)
    model = nn.DataParallel(model).cuda()
    if ema:
        for p in model.parameters():
            p.detach_()
    return model


@torch.no_grad()
def update_ema(net, net_ema, decay):
    for p, p_ema in zip(net.parameters(), net_ema.parameters()):
        p_ema.data = decay * p_ema.data + (1 - decay) * p.data


def generate_mask(img, patch_size):
    """Random cubic mask — 1 inside cube, 0 outside (BCP convention)."""
    B, _, W, H, D = img.shape
    p = patch_size // 2

    cx = random.randint(p, W - p)
    cy = random.randint(p, H - p)
    cz = random.randint(p, D - p)

    mask = torch.zeros(B, 1, W, H, D, device=img.device)
    mask[:, :, cx-p:cx+p, cy-p:cy+p, cz-p:cz+p] = 1.0
    return mask


def mix_loss(output, lab_img_label, unlab_pseudo, mask,
             l_weight=1.0, u_weight=0.5, unlab=False):
    """
    BCP mix loss: supervised region uses real label, unlabeled region uses pseudo.
    mask=1 → labeled region, mask=0 → unlabeled region.
    """
    CE   = nn.CrossEntropyLoss(reduction='none')
    DICE = DiceLoss(n_classes=2)

    img_w, patch_w = (l_weight, u_weight) if not unlab else (u_weight, l_weight)
    patch_mask = 1 - mask.squeeze(1)   # (B, W, H, D)
    mask_sq    = mask.squeeze(1)

    dice  = DICE(output, lab_img_label) * img_w
    dice += DICE(output, unlab_pseudo)  * patch_w

    ce_lab   = (CE(output, lab_img_label)  * mask_sq).sum()    / (mask_sq.sum()    + 1e-8)
    ce_unlab = (CE(output, unlab_pseudo)   * patch_mask).sum() / (patch_mask.sum() + 1e-8)
    ce       = img_w * ce_lab + patch_w * ce_unlab

    return (dice + ce) / 2


def setup_logging(save_dir):
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s  %(message)s',
        handlers=[
            logging.FileHandler(str(save_dir / 'train.log')),
            logging.StreamHandler(sys.stdout),
        ]
    )
    return logging.getLogger(), SummaryWriter(str(save_dir))


# ─────────────────────────────────────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────────────────────────────────────

def train(args):
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    seed_everything(args.seed)
    log, writer = setup_logging(args.save_dir)
    log.info(f'Args: {vars(args)}')

    # ── Models ───────────────────────────────────────────────────────────────
    net     = create_model(args.num_classes, ema=False)
    net_ema = create_model(args.num_classes, ema=True)
    net_ema.load_state_dict(net.state_dict())

    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr, weight_decay=1e-4)

    # ── Data ─────────────────────────────────────────────────────────────────
    lab_loader, unlab_loader, _ = get_loaders(
        data_root=args.data_root,
        splits_dir=args.splits_dir,
        label_percent=args.label_percent,
        patch_size=args.patch_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    test_dataset = FullVolumeDataset(
        args.data_root,
        str(Path(args.splits_dir) / 'test.txt')
    )
    log.info(f'Labeled: {len(lab_loader)} batches | Unlabeled: {len(unlab_loader)} batches')

    # ── Training loop ────────────────────────────────────────────────────────
    best_dice   = 0.0
    global_step = 0
    unlab_iter  = iter(unlab_loader)
    cut_size    = args.patch_size // 2

    for epoch in range(1, args.max_epochs + 1):
        net.train()
        net_ema.eval()   # EMA teacher always in eval mode

        ep_loss1 = ep_loss2 = 0.0

        for lab_img, lab_lbl in lab_loader:
            lab_img = lab_img.cuda()
            lab_lbl = lab_lbl.cuda()

            try:
                unlab_img, _ = next(unlab_iter)
            except StopIteration:
                unlab_iter   = iter(unlab_loader)
                unlab_img, _ = next(unlab_iter)
            unlab_img = unlab_img.cuda()

            # ── Pseudo-label from EMA ─────────────────────────────────────
            with torch.no_grad():
                unlab_probs  = F.softmax(net_ema(unlab_img)[0], dim=1)
                unlab_pseudo = unlab_probs.argmax(dim=1)   # (B, W, H, D)

            # ── Bidirectional copy-paste mask ─────────────────────────────
            mask = generate_mask(lab_img, cut_size)        # (B,1,W,H,D)

            # Mix 1: labeled foreground + unlabeled background
            mixed_l  = lab_img  * mask + unlab_img * (1 - mask)
            # Mix 2: unlabeled foreground + labeled background
            mixed_u  = unlab_img * mask + lab_img  * (1 - mask)

            # ── Forward & loss ────────────────────────────────────────────
            out_l = net(mixed_l)[0]
            loss1 = mix_loss(out_l, lab_lbl, unlab_pseudo, mask, unlab=False)

            out_u = net(mixed_u)[0]
            loss2 = mix_loss(out_u, lab_lbl, unlab_pseudo, mask, unlab=True)

            loss  = loss1 + loss2

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            update_ema(net, net_ema, args.ema_decay)

            ep_loss1    += loss1.item()
            ep_loss2    += loss2.item()
            global_step += 1

            writer.add_scalar('train/loss1', loss1.item(), global_step)
            writer.add_scalar('train/loss2', loss2.item(), global_step)

        n = len(lab_loader)
        log.info(
            f'Epoch [{epoch:03d}/{args.max_epochs}]  '
            f'loss1={ep_loss1/n:.4f}  loss2={ep_loss2/n:.4f}'
        )

        # ── Evaluation ───────────────────────────────────────────────────────
        if epoch % args.eval_every == 0 or epoch == args.max_epochs:
            dice, jc, hd, asd = evaluate(
                net, test_dataset,
                patch_size=args.patch_size,
                n_classes=args.num_classes
            )
            log.info(
                f'[Eval {epoch:03d}]  '
                f'Dice={dice:.4f}  Jc={jc:.4f}  HD95={hd:.2f}  ASD={asd:.2f}'
            )
            writer.add_scalar('test/dice', dice, epoch)
            writer.add_scalar('test/hd95', hd,   epoch)

            if dice > best_dice:
                best_dice = dice
                torch.save(net.state_dict(),
                           str(Path(args.save_dir) / 'best_model.pth'))
                log.info(f'  *** New best Dice={best_dice:.4f} — saved ***')

    log.info('=' * 60)
    log.info(f'BCP Baseline — Best Dice: {best_dice:.4f}')
    writer.close()


if __name__ == '__main__':
    train(get_args())
