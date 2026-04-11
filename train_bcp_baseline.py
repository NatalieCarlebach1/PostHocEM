"""
BCP Baseline — Bidirectional Copy-Paste (CVPR 2023)
====================================================
Re-implemented using our infrastructure (same VNet, same dataloaders, same
eval) so results are directly comparable to DGEM.

Two-phase training (matching the original BCP repo exactly):
  Phase 1 (pretrain):    CutMix between TWO labeled images, supervised loss (60 epochs)
  Phase 2 (self-train):  Full BCP with pseudo-labels + LCC filtering (200 epochs)

Optimizer: Adam, lr=1e-3, no weight decay, no LR schedule.
Loss weights: l_weight=1.0, u_weight=0.5 (original defaults).
CutMix cube: 64³ within 96³ crop.

Reference: Bai et al., "Bidirectional Copy-Paste for Semi-Supervised Medical
Image Segmentation", CVPR 2023.

Usage:
    python train_bcp_baseline.py \
        --data_root  data/pancreas_h5 \
        --splits_dir splits/pancreas \
        --label_percent 20 \
        --save_dir   result/bcp_baseline_v2
"""

import os
import sys
import random
import logging
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.ndimage import label as cc_label
from tensorboardX import SummaryWriter

from networks import VNet
from dataloaders import PancreasDataset, FullVolumeDataset
from utils.losses import SupLoss, DiceLoss
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
    p.add_argument('--cut_size',        type=int,   default=64,
                   help='CutMix cube size (original BCP uses 64)')
    p.add_argument('--batch_size',      type=int,   default=2)
    p.add_argument('--lr',              type=float, default=1e-3)
    p.add_argument('--num_classes',     type=int,   default=2)
    p.add_argument('--num_workers',     type=int,   default=0)
    p.add_argument('--ema_decay',       type=float, default=0.99)

    # Phase durations (matching the original BCP repo)
    p.add_argument('--pretrain_epochs', type=int, default=60,
                   help='Phase 1: CutMix between two labeled images')
    p.add_argument('--selftrain_epochs', type=int, default=200,
                   help='Phase 2: full BCP with pseudo-labels')

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
                 normalization='instancenorm', has_dropout=False)
    model = nn.DataParallel(model).cuda()
    if ema:
        for p in model.parameters():
            p.detach_()
    return model


@torch.no_grad()
def update_ema(net, net_ema, decay):
    for p, p_ema in zip(net.parameters(), net_ema.parameters()):
        p_ema.data.mul_(decay).add_((1 - decay) * p.data)


def generate_mask(patch_size, cut_size):
    """Random cubic mask — corner-based, matching original BCP.
    Returns img_mask (3D) and loss_mask (3D), both long tensors.
    mask=1 outside cube, 0 inside cube."""
    S = patch_size
    w = np.random.randint(0, S - cut_size)
    h = np.random.randint(0, S - cut_size)
    d = np.random.randint(0, S - cut_size)

    mask = torch.ones(S, S, S).long().cuda()
    mask[w:w+cut_size, h:h+cut_size, d:d+cut_size] = 0
    return mask


def largest_connected_component(pred):
    """Keep only the largest connected component per sample.
    Uses connectivity=2 (26-connected in 3D) matching original BCP."""
    result = torch.zeros_like(pred)
    for i in range(pred.shape[0]):
        p = pred[i].cpu().numpy()
        if p.sum() == 0:
            continue
        labeled, num_features = cc_label(p, structure=np.ones((3, 3, 3)))
        if num_features == 0:
            continue
        largest = 0
        largest_size = 0
        for j in range(1, num_features + 1):
            size = (labeled == j).sum()
            if size > largest_size:
                largest_size = size
                largest = j
        result[i] = torch.from_numpy(
            (labeled == largest).astype(np.float32)).to(pred.device)
    return result.long()


def mix_loss(output, img_label, patch_label, mask, l_weight=1.0, u_weight=0.5,
             unlab=False):
    """
    BCP mix loss with masked Dice per region (matching original exactly).
    mask: (B, W, H, D) long — 1 outside cube, 0 inside cube.
    img_label: target for outside-cube region.
    patch_label: target for inside-cube region.
    """
    CE   = nn.CrossEntropyLoss(reduction='none')
    DICE = DiceLoss(n_classes=2)

    if not unlab:
        img_w, patch_w = l_weight, u_weight
    else:
        img_w, patch_w = u_weight, l_weight

    mask_f     = mask.float()              # outside cube
    patch_mask = (1.0 - mask_f)            # inside cube

    # CE per region
    ce_img   = (CE(output, img_label)   * mask_f).sum()      / (mask_f.sum()      + 1e-8)
    ce_patch = (CE(output, patch_label) * patch_mask).sum()  / (patch_mask.sum()  + 1e-8)
    ce = img_w * ce_img + patch_w * ce_patch

    # Masked Dice per region (separate, weighted)
    dice  = DICE(output, img_label,   mask=mask_f)     * img_w
    dice += DICE(output, patch_label, mask=patch_mask)  * patch_w

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

    max_epochs = args.pretrain_epochs + args.selftrain_epochs

    # ── CSV log ─────────────────────────────────────────────────────────────
    csv_path = Path(args.save_dir) / 'metrics.csv'
    with open(csv_path, 'w') as f:
        f.write('epoch,phase,loss1,loss2,dice,jaccard,hd95,asd,best_dice\n')

    # ── Models ───────────────────────────────────────────────────────────────
    net     = create_model(args.num_classes, ema=False)
    net_ema = create_model(args.num_classes, ema=True)
    net_ema.load_state_dict(net.state_dict())

    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)
    sup_loss_fn = SupLoss(args.num_classes)

    # ── Data ─────────────────────────────────────────────────────────────────
    splits_dir = Path(args.splits_dir)

    # Labeled: RandomCrop, no flip (matching original)
    lab_ds = PancreasDataset(
        args.data_root,
        splits_dir / f'train_lab_{args.label_percent}.txt',
        patch_size=args.patch_size, augment=False, repeat=5)
    lab_loader = torch.utils.data.DataLoader(
        lab_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True, drop_last=True)

    # Unlabeled: CenterCrop, no flip (matching original)
    unlab_ds = PancreasDataset(
        args.data_root,
        splits_dir / f'train_unlab_{args.label_percent}.txt',
        patch_size=args.patch_size, augment=False, center_crop=True)
    unlab_loader = torch.utils.data.DataLoader(
        unlab_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True, drop_last=True)

    test_dataset = FullVolumeDataset(
        args.data_root,
        str(splits_dir / 'test.txt')
    )
    log.info(f'Labeled: {len(lab_loader)} batches | Unlabeled: {len(unlab_loader)} batches')
    log.info(f'Phases: pretrain={args.pretrain_epochs}, '
             f'selftrain={args.selftrain_epochs}, total={max_epochs}')

    # ── Training loop ────────────────────────────────────────────────────────
    best_dice   = 0.0
    global_step = 0
    unlab_iter  = iter(unlab_loader)
    lab_iter_b  = iter(lab_loader)

    for epoch in range(1, max_epochs + 1):
        net.train()
        net_ema.train()  # Original BCP keeps EMA in train mode

        phase = 'pretrain' if epoch <= args.pretrain_epochs else 'selftrain'
        ep_loss1 = ep_loss2 = 0.0

        for lab_img, lab_lbl in lab_loader:
            lab_img = lab_img.cuda()
            lab_lbl = lab_lbl.cuda()

            # Generate mask: corner-based, 64³ cube
            img_mask = generate_mask(args.patch_size, args.cut_size)
            # Expand for batch: (B, W, H, D)
            batch_mask = img_mask.unsqueeze(0).expand(lab_img.size(0), -1, -1, -1)

            if phase == 'pretrain':
                # ── Phase 1: CutMix between TWO labeled images ───────────
                try:
                    lab_img_b, lab_lbl_b = next(lab_iter_b)
                except StopIteration:
                    lab_iter_b = iter(lab_loader)
                    lab_img_b, lab_lbl_b = next(lab_iter_b)
                lab_img_b = lab_img_b.cuda()
                lab_lbl_b = lab_lbl_b.cuda()

                # Mix: img_a outside cube + img_b inside cube
                mask_5d = img_mask.float().unsqueeze(0).unsqueeze(0)  # (1,1,W,H,D)
                mixed = lab_img * mask_5d + lab_img_b * (1.0 - mask_5d)
                mixed_label = (lab_lbl * img_mask.unsqueeze(0) +
                               lab_lbl_b * (1 - img_mask.unsqueeze(0)))

                out = net(mixed)[0]
                loss = sup_loss_fn(out, mixed_label)
                loss1_val = loss.item()
                loss2_val = 0.0

            else:
                # ── Phase 2: Full BCP with pseudo-labels ─────────────────
                try:
                    unlab_img, _ = next(unlab_iter)
                except StopIteration:
                    unlab_iter = iter(unlab_loader)
                    unlab_img, _ = next(unlab_iter)
                unlab_img = unlab_img.cuda()

                # Pseudo-label from EMA + largest connected component (26-conn)
                with torch.no_grad():
                    unlab_probs  = F.softmax(net_ema(unlab_img)[0], dim=1)
                    unlab_pseudo = unlab_probs.argmax(dim=1)
                    unlab_pseudo = largest_connected_component(unlab_pseudo)

                mask_5d = img_mask.float().unsqueeze(0).unsqueeze(0)  # (1,1,W,H,D)

                # Mix 1: unlabeled outside cube + labeled inside cube
                # (original: net3_input_l = unimg_a * mask + img_b * (1-mask))
                mixed_1 = unlab_img * mask_5d + lab_img * (1.0 - mask_5d)
                out_1 = net(mixed_1)[0]
                # img_label=pseudo (outside), patch_label=real (inside), unlab=True
                loss1 = mix_loss(out_1, unlab_pseudo, lab_lbl, batch_mask,
                                 unlab=True)

                # Mix 2: labeled outside cube + unlabeled inside cube
                # (original: net3_input_unlab = img_a * mask + unimg_b * (1-mask))
                mixed_2 = lab_img * mask_5d + unlab_img * (1.0 - mask_5d)
                out_2 = net(mixed_2)[0]
                # img_label=real (outside), patch_label=pseudo (inside), unlab=False
                loss2 = mix_loss(out_2, lab_lbl, unlab_pseudo, batch_mask,
                                 unlab=False)

                loss = loss1 + loss2
                loss1_val = loss1.item()
                loss2_val = loss2.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            update_ema(net, net_ema, args.ema_decay)

            ep_loss1    += loss1_val
            ep_loss2    += loss2_val
            global_step += 1

        n = len(lab_loader)
        log.info(
            f'Epoch [{epoch:03d}/{max_epochs}]  '
            f'phase={phase}  '
            f'loss1={ep_loss1/n:.4f}  loss2={ep_loss2/n:.4f}'
        )

        # ── Evaluation ───────────────────────────────────────────────────────
        if epoch % args.eval_every == 0 or epoch == max_epochs:
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
                torch.save(net_ema.state_dict(),
                           str(Path(args.save_dir) / 'best_ema_model.pth'))
                log.info(f'  *** New best Dice={best_dice:.4f} — saved ***')

            # ── CSV log ──────────────────────────────────────────────────
            with open(csv_path, 'a') as f:
                f.write(f'{epoch},{phase},{ep_loss1/n:.6f},{ep_loss2/n:.6f},'
                        f'{dice:.6f},{jc:.6f},{hd:.4f},{asd:.4f},'
                        f'{best_dice:.6f}\n')

    log.info('=' * 60)
    log.info(f'BCP Baseline — Best Dice: {best_dice:.4f}')
    writer.close()


if __name__ == '__main__':
    train(get_args())
