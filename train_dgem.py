"""
DGEM — Disagreement-Guided Entropy Minimization
================================================
Full SSL training from scratch. Two models:
  - net     : main model, trained by gradient
  - net_ema : EMA copy of net, frozen weights (like Mean Teacher)

For unlabeled data, entropy loss is applied ONLY on voxels where net and
net_ema DISAGREE. This forces confidence exactly where the model is uncertain,
without generating noisy pseudo-labels.

Loss:
    L = L_sup(net, labeled)
      + λ(t) * L_em_disagree(net, unlabeled)

where λ(t) ramps up via sigmoid over the first `consistency_rampup` epochs.

Usage:
    python train_dgem.py \
        --data_root  /path/to/pancreas_h5 \
        --splits_dir splits/pancreas \
        --label_percent 20 \
        --max_epochs 300 \
        --save_dir result/dgem_20p

Compare baseline (no EM loss):
    python train_dgem.py ... --em_weight 0.0  --save_dir result/supervised_only
"""

import os
import sys
import copy
import random
import logging
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tensorboardX import SummaryWriter

from networks import VNet
from dataloaders import PancreasDataset, FullVolumeDataset
from utils.losses import SupLoss, entropy_loss_masked, entropy_loss_full
from utils.ramps  import get_current_consistency_weight
from utils.metrics import evaluate


# ─────────────────────────────────────────────────────────────────────────────
# Args
# ─────────────────────────────────────────────────────────────────────────────

def get_args():
    p = argparse.ArgumentParser()
    # Data
    p.add_argument('--data_root',      required=True)
    p.add_argument('--splits_dir',     default='splits/pancreas')
    p.add_argument('--label_percent',  type=int,   default=20)
    p.add_argument('--patch_size',     type=int,   default=96)

    # Training
    p.add_argument('--max_epochs',     type=int,   default=300)
    p.add_argument('--batch_size',     type=int,   default=2)
    p.add_argument('--lr',             type=float, default=1e-3)
    p.add_argument('--num_classes',    type=int,   default=2)
    p.add_argument('--num_workers',    type=int,   default=0)

    # DGEM-specific
    p.add_argument('--warmup_epochs',       type=int,   default=30,
                   help='Supervised-only warmup before enabling EM loss.')
    p.add_argument('--em_weight',           type=float, default=0.3,
                   help='Max weight of entropy loss. Set 0 for supervised-only baseline.')
    p.add_argument('--consistency_rampup',  type=int,   default=100,
                   help='Epochs to ramp up EM weight AFTER warmup (sigmoid schedule).')
    p.add_argument('--ema_decay',           type=float, default=0.99,
                   help='EMA decay for the teacher model.')
    p.add_argument('--mask_type',           default='disagreement',
                   choices=['disagreement', 'full', 'random', 'soft'],
                   help='Entropy mask strategy: disagreement (default), '
                        'full (all voxels), random (random mask matching '
                        'disagreement ratio), soft (teacher uncertainty).')

    # Optional: warm-start from a pretrained checkpoint (e.g. BCP)
    p.add_argument('--pretrained_checkpoint', default=None,
                   help='Path to a pretrained .pth (e.g. BCP best_model.pth). '
                        'Both net and net_ema are initialised from it.')

    # Eval / save
    p.add_argument('--eval_every',     type=int,   default=10)
    p.add_argument('--save_dir',       default='result/dgem')
    p.add_argument('--gpu',            default='0')
    p.add_argument('--seed',           type=int,   default=2020)
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

    # ── CSV log ─────────────────────────────────────────────────────────────
    csv_path = Path(args.save_dir) / 'metrics.csv'
    with open(csv_path, 'w') as f:
        f.write('epoch,sup_loss,em_loss,disagree_ratio,lam,mask_type,dice,jaccard,hd95,asd,best_dice\n')

    # ── Models ───────────────────────────────────────────────────────────────
    net     = create_model(args.num_classes, ema=False)
    net_ema = create_model(args.num_classes, ema=True)

    if args.pretrained_checkpoint:
        state = torch.load(args.pretrained_checkpoint, map_location='cuda')
        if isinstance(state, dict) and 'net' in state:
            state = state['net']
        net.load_state_dict(state)
        log.info(f'Warm-started from: {args.pretrained_checkpoint}')

    net_ema.load_state_dict(net.state_dict())   # EMA starts identical to net

    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)
    sup_loss_fn = SupLoss(args.num_classes)

    # ── Data (matching BCP: no flip, CenterCrop for unlabeled) ─────────────
    splits_dir = Path(args.splits_dir)

    lab_ds = PancreasDataset(
        args.data_root,
        splits_dir / f'train_lab_{args.label_percent}.txt',
        patch_size=args.patch_size, augment=False, repeat=5)
    lab_loader = torch.utils.data.DataLoader(
        lab_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True, drop_last=True)

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
    log.info(f'Labeled batches/epoch: {len(lab_loader)} | '
             f'Unlabeled batches/epoch: {len(unlab_loader)}')

    # ── Training loop ────────────────────────────────────────────────────────
    best_dice   = 0.0
    global_step = 0
    unlab_iter  = iter(unlab_loader)

    for epoch in range(1, args.max_epochs + 1):
        net.train()
        net_ema.train()  # Matching BCP: EMA in train mode

        ep_sup = ep_em = ep_disagree_ratio = 0.0

        # Ramp EM weight — zero during warmup, then sigmoid ramp
        if epoch <= args.warmup_epochs:
            lam = 0.0
        else:
            lam = get_current_consistency_weight(
                epoch - args.warmup_epochs, args.em_weight, args.consistency_rampup)

        for lab_img, lab_lbl in lab_loader:
            lab_img = lab_img.cuda()
            lab_lbl = lab_lbl.cuda()

            # ── Supervised loss ───────────────────────────────────────────
            sup_out  = net(lab_img)[0]
            sup_loss = sup_loss_fn(sup_out, lab_lbl)

            # ── Disagreement-guided entropy loss ──────────────────────────
            em_loss = torch.tensor(0.0, device=sup_loss.device)
            disagree_ratio = 0.0

            if lam > 0:
                try:
                    unlab_img, _ = next(unlab_iter)
                except StopIteration:
                    unlab_iter = iter(unlab_loader)
                    unlab_img, _ = next(unlab_iter)
                unlab_img = unlab_img.cuda()

                # Student (net) forward — keep gradients
                student_logits = net(unlab_img)[0]
                student_probs  = F.softmax(student_logits, dim=1)
                student_pred   = student_probs.argmax(dim=1)

                # Teacher (net_ema) forward — no gradients
                with torch.no_grad():
                    teacher_probs = F.softmax(net_ema(unlab_img)[0], dim=1)
                    teacher_pred  = teacher_probs.argmax(dim=1)

                # Disagreement mask: 1 where student ≠ teacher
                disagree_mask = (student_pred != teacher_pred).float()
                disagree_ratio = disagree_mask.mean().item()

                if args.mask_type == 'disagreement':
                    em_loss = entropy_loss_masked(student_probs, disagree_mask)
                elif args.mask_type == 'full':
                    em_loss = entropy_loss_full(student_probs)
                elif args.mask_type == 'random':
                    random_mask = (torch.rand_like(disagree_mask) < disagree_ratio).float()
                    em_loss = entropy_loss_masked(student_probs, random_mask)
                elif args.mask_type == 'soft':
                    soft_mask = 1.0 - teacher_probs.max(dim=1)[0]
                    em_loss = entropy_loss_masked(student_probs, soft_mask)

            loss = sup_loss + lam * em_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            update_ema(net, net_ema, args.ema_decay)

            ep_sup            += sup_loss.item()
            ep_em             += em_loss.item()
            ep_disagree_ratio += disagree_ratio
            global_step       += 1

            writer.add_scalar('train/sup_loss',       sup_loss.item(),   global_step)
            writer.add_scalar('train/em_loss',        em_loss.item(),    global_step)
            writer.add_scalar('train/disagree_ratio', disagree_ratio,    global_step)
            writer.add_scalar('train/lam',            lam,               global_step)

        n = len(lab_loader)
        phase = 'warmup' if epoch <= args.warmup_epochs else 'dgem'
        log.info(
            f'Epoch [{epoch:03d}/{args.max_epochs}]  '
            f'phase={phase}  sup={ep_sup/n:.4f}  em={ep_em/n:.4f}  '
            f'disagree={ep_disagree_ratio/n:.3f}  lam={lam:.4f}  '
            f'mask_type={args.mask_type}  lr={args.lr:.6f}'
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
                torch.save(net_ema.state_dict(),
                           str(Path(args.save_dir) / 'best_ema_model.pth'))
                log.info(f'  *** New best Dice={best_dice:.4f} — saved ***')

            # ── CSV log ──────────────────────────────────────────────────
            with open(csv_path, 'a') as f:
                f.write(f'{epoch},{ep_sup/n:.6f},{ep_em/n:.6f},'
                        f'{ep_disagree_ratio/n:.4f},{lam:.6f},{args.mask_type},'
                        f'{dice:.6f},{jc:.6f},{hd:.4f},{asd:.4f},{best_dice:.6f}\n')

    log.info('=' * 60)
    log.info(f'Training complete. Best Dice: {best_dice:.4f}')
    log.info(f'Checkpoints saved to: {args.save_dir}')
    writer.close()


if __name__ == '__main__':
    train(get_args())
