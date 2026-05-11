"""
T2.1 — UA-MT (Uncertainty-Aware Mean Teacher, MICCAI 2019) reproduction on
Pancreas-CT 20%. Used as a second base SSL method to apply PEM on, so that
the BMVC paper can demonstrate generalization beyond BCP.

This is a minimal, faithful reproduction of Yu et al. 2019, following the
SSL4MIS implementation (https://github.com/HiLab-git/SSL4MIS) with:
  - V-Net (instancenorm) student + EMA teacher
  - Supervised CE + Dice on labeled volumes
  - MSE consistency between student and teacher on unlabeled volumes,
    masked by uncertainty (entropy of K MC-dropout teacher forwards
    averaged; voxels above the uncertainty threshold are excluded)
  - Sigmoid consistency-weight ramp-up
  - Same Pancreas-CT preprocessing and split as BCP

The trained checkpoint is then fed into train_posthoc_em.py exactly like
the BCP checkpoint, with no method-specific adapters.

Reference Dice on Pancreas-CT 20% (BCP paper, Table 1): 77.26.
Our reproduction target: within 0.5 Dice of that.

Note: this is base-method reproduction. Whether PEM helps UA-MT is the
T2.1 question. Even if PEM helps by a smaller amount than on BCP
(because UA-MT's rho_f may differ), the directional result strengthens
the generalization claim.
"""

import argparse
import json
import logging
import math
import os
import random
import sys
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from networks import VNet
from dataloaders.pancreas_loader import PancreasLabeled, PancreasUnlabeled
from utils.losses import SupLoss
from utils.metrics import sliding_window_inference
from utils.ramps import sigmoid_rampup


def get_args():
    p = argparse.ArgumentParser(description='UA-MT reproduction (Yu et al. 2019)')
    p.add_argument('--data_root',  default='data/pancreas_h5')
    p.add_argument('--splits_dir', default='splits/pancreas')
    p.add_argument('--label_percent', type=int, default=20)
    p.add_argument('--patch_size',  type=int, default=96)
    p.add_argument('--batch_size',  type=int, default=4)
    p.add_argument('--lr',          type=float, default=1e-2)
    p.add_argument('--max_iter',    type=int, default=6000)
    p.add_argument('--ema_decay',   type=float, default=0.99)
    p.add_argument('--consistency', type=float, default=0.1)
    p.add_argument('--consistency_rampup', type=float, default=40.0)
    p.add_argument('--threshold_T', type=int, default=8,
                   help='Number of stochastic teacher forwards for uncertainty estimate')
    p.add_argument('--U_threshold', type=float, default=0.75,
                   help='Uncertainty cutoff: voxels with H > U_threshold * ln(C) are masked out')
    p.add_argument('--save_dir',    default='result/uamt_pancreas20')
    p.add_argument('--eval_every',  type=int, default=200)
    p.add_argument('--save_every',  type=int, default=0,
                   help='If >0, dump trajectory checkpoint every Nth iteration')
    p.add_argument('--gpu',         default='0')
    p.add_argument('--seed',        type=int, default=2020)
    return p.parse_args()


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def make_net():
    """V-Net with dropout enabled in the teacher for MC-dropout uncertainty."""
    net = VNet(n_channels=1, n_classes=2,
               normalization='instancenorm', has_dropout=True)
    return nn.DataParallel(net).cuda()


def update_ema(student, teacher, decay):
    with torch.no_grad():
        for p_s, p_t in zip(student.parameters(), teacher.parameters()):
            p_t.data.mul_(decay).add_(p_s.data, alpha=1 - decay)


def uncertainty_mask(teacher_probs_T, threshold):
    """teacher_probs_T: (T, B, C, D, H, W) — T stochastic forwards."""
    mean = teacher_probs_T.mean(dim=0)                                # (B, C, D, H, W)
    eps = 1e-6
    H = -(mean * torch.log(mean + eps)).sum(dim=1, keepdim=True)      # (B, 1, D, H, W)
    Hmax = math.log(mean.shape[1])
    keep = (H < threshold * Hmax).float()                             # 1 where confident
    return mean, keep


def setup_logging(save_dir):
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO,
                        format='[%(asctime)s] %(message)s',
                        datefmt='%H:%M:%S',
                        handlers=[
                            logging.FileHandler(str(save_dir / 'train.log')),
                            logging.StreamHandler(),
                        ])
    return logging.getLogger()


def main():
    args = get_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    seed_everything(args.seed)
    log = setup_logging(args.save_dir)

    # ── Data ─────────────────────────────────────────────────────────────────
    splits_dir = ROOT / args.splits_dir
    labeled_set = PancreasLabeled(
        ROOT / args.data_root, splits_dir / f'labeled_{args.label_percent}.txt',
        patch_size=args.patch_size)
    unlabeled_set = PancreasUnlabeled(
        ROOT / args.data_root, splits_dir / 'unlabeled.txt',
        patch_size=args.patch_size)
    test_cases = [l.strip() for l in open(splits_dir / 'test.txt') if l.strip()]

    labeled_loader = DataLoader(labeled_set, batch_size=args.batch_size // 2,
                                shuffle=True, num_workers=0, drop_last=True)
    unlabeled_loader = DataLoader(unlabeled_set,
                                  batch_size=args.batch_size - args.batch_size // 2,
                                  shuffle=True, num_workers=0, drop_last=True)

    log.info(f'labeled={len(labeled_set)} unlabeled={len(unlabeled_set)} '
             f'test={len(test_cases)}')

    # ── Models ───────────────────────────────────────────────────────────────
    student = make_net()
    teacher = make_net()
    for p in teacher.parameters():
        p.requires_grad_(False)
    teacher.train()  # keep dropout active for MC uncertainty

    optimizer = torch.optim.SGD(student.parameters(), lr=args.lr,
                                momentum=0.9, weight_decay=1e-4)
    sup_loss = SupLoss(n_classes=2)
    mse = nn.MSELoss()

    best_dice = 0.0
    iter_num = 0

    # ── Training loop ────────────────────────────────────────────────────────
    while iter_num < args.max_iter:
        for (img_l, lab_l), (img_u,) in zip(labeled_loader, unlabeled_loader):
            iter_num += 1
            img_l, lab_l = img_l.cuda(), lab_l.cuda().long()
            img_u = img_u.cuda()

            # Supervised forward
            logits_l = student(img_l)
            loss_sup = sup_loss(logits_l, lab_l)

            # Consistency on unlabeled
            noise = torch.clamp(torch.randn_like(img_u) * 0.1, -0.2, 0.2)
            img_u_noisy = img_u + noise

            with torch.no_grad():
                teacher_probs_T = torch.stack([
                    F.softmax(teacher(img_u_noisy), dim=1)
                    for _ in range(args.threshold_T)
                ], dim=0)
                teacher_mean, keep_mask = uncertainty_mask(
                    teacher_probs_T, args.U_threshold)

            student_probs = F.softmax(student(img_u_noisy), dim=1)
            loss_con = ((student_probs - teacher_mean) ** 2 * keep_mask).sum() \
                       / (keep_mask.sum() * student_probs.shape[1] + 1e-6)

            cons_w = args.consistency * sigmoid_rampup(
                iter_num / 200.0, args.consistency_rampup)
            loss = loss_sup + cons_w * loss_con
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            update_ema(student, teacher, args.ema_decay)

            # LR poly decay
            lr_now = args.lr * (1.0 - iter_num / args.max_iter) ** 0.9
            for g in optimizer.param_groups:
                g['lr'] = lr_now

            if iter_num % 100 == 0:
                log.info(f'iter {iter_num:5d}  sup={loss_sup.item():.4f}  '
                         f'con={loss_con.item():.4f}  cw={cons_w:.4f}  lr={lr_now:.4e}')

            # ── Trajectory checkpoint ───────────────────────────────────────
            if args.save_every > 0 and iter_num % args.save_every == 0:
                traj_dir = Path(args.save_dir) / 'trajectory'
                traj_dir.mkdir(parents=True, exist_ok=True)
                torch.save(student.state_dict(),
                           str(traj_dir / f'iter_{iter_num:05d}.pth'))

            if iter_num % args.eval_every == 0:
                student.eval()
                dices = []
                for case in test_cases[:6]:  # quick eval subset
                    path = ROOT / args.data_root / case
                    with h5py.File(str(path), 'r') as f:
                        image = f['image'][:].astype(np.float32)
                        label = f['label'][:].astype(np.uint8)
                    pred, _ = sliding_window_inference(
                        student, image, (96, 96, 96), 16, 4, n_classes=2)
                    pred = pred.astype(np.uint8)
                    if pred.sum() and label.sum():
                        from medpy.metric.binary import dc
                        dices.append(dc(pred, label))
                student.train()
                dice = float(np.mean(dices)) if dices else 0.0
                log.info(f'  [Eval {iter_num:5d}] subset-Dice={dice:.4f}')
                if dice > best_dice:
                    best_dice = dice
                    torch.save(student.state_dict(),
                               str(Path(args.save_dir) / 'best_model.pth'))
                    log.info(f'  *** new best {dice:.4f} saved ***')

            if iter_num >= args.max_iter:
                break

    log.info(f'UA-MT done. Best subset-Dice = {best_dice:.4f}')
    torch.save(student.state_dict(),
               str(Path(args.save_dir) / 'last_model.pth'))


if __name__ == '__main__':
    main()
