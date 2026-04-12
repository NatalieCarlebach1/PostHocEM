"""
Post-hoc baselines for PEM comparison.

Three baselines that operate on a converged BCP checkpoint with the same
fixed-budget protocol as PEM (no test-set tuning, fixed E, single LR):

  1. Temperature scaling (TS):
     Freeze all weights. Optimize a single scalar T to minimize the mean
     entropy of softmax(logits/T) on the unlabeled training set. At test
     time, scale logits by 1/T before softmax. Cannot change argmax in
     principle (it can only flatten/sharpen confidence), so it serves as
     a "trivial recalibration" control.

  2. Pseudo-label self-distillation (PL-FT):
     Freeze a copy of the checkpoint as a pseudo-label generator (with
     largest-connected-component filtering, matching BCP's pseudo-label
     procedure). Fine-tune a target copy with supervised CE+Dice against
     these hard pseudo-labels. Same epochs/LR as PEM.

  3. Self-consistency (SC):
     Freeze a teacher copy. For each unlabeled batch, draw a stochastic
     perturbation (random flip + Gaussian noise on inputs), forward the
     student on the perturbed input, and minimize MSE between student
     softmax and teacher softmax (on the un-perturbed input). Same
     epochs/LR as PEM.

Usage (one of):

  python train_baselines.py --method ts \
    --dataset pancreas --checkpoint result/bcp_baseline_v2/best_model.pth \
    --data_root data/pancreas_h5 --splits_dir splits/pancreas \
    --label_percent 20 --epochs 2 --lr 5e-5 \
    --save_dir result/baseline_ts_pancreas
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
from scipy.ndimage import label as cc_label

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from networks import VNet
from dataloaders import PancreasDataset, FullVolumeDataset
from utils.metrics import evaluate, sliding_window_inference, calculate_metric_percase


# ─────────────────────────────────────────────────────────────────────────────
# Inline LA loaders (mirroring train_posthoc_em.py to avoid touching dataloader)
# ─────────────────────────────────────────────────────────────────────────────

class _LAFullVolume:
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
    parts = s.split(',')
    if len(parts) == 1:
        return int(parts[0])
    return tuple(int(x) for x in parts)


# ─────────────────────────────────────────────────────────────────────────────
# Args
# ─────────────────────────────────────────────────────────────────────────────

def get_args():
    p = argparse.ArgumentParser(description="Post-hoc baselines for PEM comparison")
    p.add_argument("--method", required=True, choices=["ts", "pl_ft", "sc"])
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--data_root",  required=True)
    p.add_argument("--splits_dir", required=True)
    p.add_argument("--save_dir",   required=True)

    p.add_argument("--dataset", default="pancreas", choices=["pancreas", "la"])
    p.add_argument("--la_data_root", default="data/la_h5/2018LA_Seg_Training Set")

    p.add_argument("--epochs",     type=int,   default=5)
    p.add_argument("--lr",         type=float, default=5e-5)
    p.add_argument("--batch_size", type=int,   default=2)
    p.add_argument("--patch_size", type=str,   default='96')
    p.add_argument("--stride_xy",  type=int,   default=None)
    p.add_argument("--num_classes", type=int,  default=2)
    p.add_argument("--label_percent", type=int, default=20)
    p.add_argument("--grad_clip",  type=float, default=1.0)
    p.add_argument("--noise_std",  type=float, default=0.05,
                   help="Gaussian input noise std for self-consistency baseline")

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


def freeze_bn(net):
    for m in net.modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            m.eval()
            for p in m.parameters():
                p.requires_grad_(False)


def largest_connected_component_3d(mask_np):
    """Keep the largest connected component in a binary 3D mask. 26-connectivity."""
    if mask_np.sum() == 0:
        return mask_np
    labeled, n = cc_label(mask_np, structure=np.ones((3, 3, 3)))
    if n == 0:
        return mask_np
    sizes = np.bincount(labeled.ravel())
    sizes[0] = 0
    largest = sizes.argmax()
    return (labeled == largest).astype(mask_np.dtype)


def soft_pseudo_labels(teacher, x, num_classes):
    """Hard pseudo-labels via argmax + largest connected component."""
    with torch.no_grad():
        logits = teacher(x)[0]
        probs = F.softmax(logits, dim=1)
        argmax = probs.argmax(dim=1)  # (B, W, H, D)
        out = torch.zeros_like(argmax)
        for i in range(argmax.shape[0]):
            arr = argmax[i].detach().cpu().numpy().astype(np.uint8)
            cc = largest_connected_component_3d(arr)
            out[i] = torch.from_numpy(cc.astype(np.int64)).to(argmax.device)
        return out


# ─────────────────────────────────────────────────────────────────────────────
# Loss functions for the three baselines
# ─────────────────────────────────────────────────────────────────────────────

def loss_ts(student, x, T):
    """Temperature scaling: minimize entropy of softmax(logits/T)."""
    logits = student(x)[0]
    probs = F.softmax(logits / T, dim=1)
    H = -(probs * torch.log(probs + 1e-8)).sum(dim=1)
    return H.mean()


def loss_pl_ft(student, x, pseudo_labels, num_classes):
    """Pseudo-label fine-tune: CE + Dice against frozen pseudo-labels (LCC-filtered)."""
    logits = student(x)[0]
    ce = F.cross_entropy(logits, pseudo_labels)
    # Dice
    probs = F.softmax(logits, dim=1)
    onehot = F.one_hot(pseudo_labels, num_classes).permute(0, 4, 1, 2, 3).float()
    inter = (probs * onehot).sum(dim=(2, 3, 4))
    union = (probs + onehot).sum(dim=(2, 3, 4))
    dice = (2 * inter + 1e-5) / (union + 1e-5)
    dice_loss = 1 - dice.mean()
    return (ce + dice_loss) / 2


def loss_sc(student, teacher, x, noise_std):
    """Self-consistency (post-hoc Mean Teacher): MSE between student(noisy) and teacher(clean)."""
    with torch.no_grad():
        teacher_logits = teacher(x)[0]
        teacher_probs  = F.softmax(teacher_logits, dim=1)
    noise = torch.randn_like(x) * noise_std
    student_logits = student(x + noise)[0]
    student_probs  = F.softmax(student_logits, dim=1)
    return F.mse_loss(student_probs, teacher_probs)


# ─────────────────────────────────────────────────────────────────────────────
# Temperature-scaled evaluation (only used by TS)
# ─────────────────────────────────────────────────────────────────────────────

class TemperatureWrapper(nn.Module):
    """Wraps a base network and divides its logits by T before returning."""
    def __init__(self, base, T):
        super().__init__()
        self.base = base
        self.T = T
    def forward(self, x):
        out = self.base(x)
        logits = out[0] if isinstance(out, tuple) else out
        return (logits / self.T,)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    args = get_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    seed_everything(args.seed)

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(message)s",
        handlers=[
            logging.FileHandler(str(save_dir / "train.log")),
            logging.StreamHandler(sys.stdout),
        ]
    )
    log = logging.getLogger()
    log.info(f"Args: {vars(args)}")

    # CSV
    csv_path = save_dir / 'metrics.csv'
    with open(csv_path, 'w') as f:
        f.write('epoch,method,loss,dice,jaccard,hd95,asd,best_dice,delta\n')

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
        test_ds = _LAFullVolume(args.la_data_root, str(splits_dir / 'test.txt'))
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

    # ── Models ──────────────────────────────────────────────────────────────
    student = load_model(args.checkpoint, args.num_classes, dataset=args.dataset)
    log.info(f"Loaded student: {args.checkpoint}")

    if args.method in ('pl_ft', 'sc'):
        teacher = load_model(args.checkpoint, args.num_classes, dataset=args.dataset)
        teacher.eval()
        for p in teacher.parameters():
            p.requires_grad_(False)
        log.info("Loaded frozen teacher (same checkpoint)")
    else:
        teacher = None

    if args.dataset == 'la':
        freeze_bn(student)
        if teacher is not None:
            freeze_bn(teacher)
        log.info("LA: froze BatchNorm running stats")

    # ── Optimizer / parameters ──────────────────────────────────────────────
    if args.method == 'ts':
        # Freeze all weights, optimize only T
        for p in student.parameters():
            p.requires_grad_(False)
        T = nn.Parameter(torch.ones(1, device='cuda'))
        optimizer = torch.optim.Adam([T], lr=args.lr * 100)  # T can move faster
    else:
        T = None
        optimizer = torch.optim.Adam(student.parameters(), lr=args.lr)

    # ── Baseline eval (with cache, mirroring train_posthoc_em.py) ───────────
    ckpt_path = Path(args.checkpoint)
    cache_path = ckpt_path.parent / f'{ckpt_path.stem}.baseline.json'
    cache_key = {
        'patch_size': args.patch_size,
        'test_file': str(splits_dir / 'test.txt'),
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
        base_jc = cached['jaccard']
        base_hd = cached['hd95']
        base_asd = cached['asd']
        log.info(f"=== Baseline (cached) === Dice={base_dice:.4f}")
    else:
        log.info("=== Baseline evaluation ===")
        base_dice, base_jc, base_hd, base_asd = evaluate(
            student, test_ds, args.patch_size,
            stride_xy=args.stride_xy, stride_z=4,
            n_classes=args.num_classes)
        log.info(f"[Baseline] Dice={base_dice:.4f}")

    with open(csv_path, 'a') as f:
        f.write(f'0,baseline,0.000000,'
                f'{base_dice:.6f},{base_jc:.6f},{base_hd:.4f},{base_asd:.4f},'
                f'{base_dice:.6f},0.0000\n')

    # ── Fine-tuning loop ────────────────────────────────────────────────────
    best_dice = base_dice

    for epoch in range(1, args.epochs + 1):
        student.train()
        if args.dataset == 'la':
            freeze_bn(student)  # re-freeze each epoch (train() resets eval mode)
        if teacher is not None:
            teacher.eval()

        ep_loss = 0.0
        ep_steps = 0
        for unlab_img, _ in unlab_loader:
            unlab_img = unlab_img.cuda()

            if args.method == 'ts':
                loss = loss_ts(student, unlab_img, T)
            elif args.method == 'pl_ft':
                pseudo = soft_pseudo_labels(teacher, unlab_img, args.num_classes)
                loss = loss_pl_ft(student, unlab_img, pseudo, args.num_classes)
            elif args.method == 'sc':
                loss = loss_sc(student, teacher, unlab_img, args.noise_std)

            optimizer.zero_grad()
            loss.backward()
            if args.grad_clip > 0 and args.method != 'ts':
                torch.nn.utils.clip_grad_norm_(
                    [p for p in student.parameters() if p.requires_grad], args.grad_clip)
            optimizer.step()

            ep_loss += loss.item()
            ep_steps += 1

        avg_loss = ep_loss / max(ep_steps, 1)
        if args.method == 'ts':
            log.info(f"Epoch [{epoch}/{args.epochs}]  loss={avg_loss:.4f}  T={T.item():.4f}")
        else:
            log.info(f"Epoch [{epoch}/{args.epochs}]  loss={avg_loss:.4f}")

        # ── Eval ─────────────────────────────────────────────────────────
        if args.method == 'ts':
            # Wrap student so logits are scaled by 1/T at inference
            eval_net = TemperatureWrapper(student, T.detach()).cuda()
            eval_net.eval()
            dice, jc, hd, asd = evaluate(
                eval_net, test_ds, args.patch_size,
                stride_xy=args.stride_xy, stride_z=4,
                n_classes=args.num_classes)
        else:
            dice, jc, hd, asd = evaluate(
                student, test_ds, args.patch_size,
                stride_xy=args.stride_xy, stride_z=4,
                n_classes=args.num_classes)
        delta = dice - base_dice
        log.info(f"[Epoch {epoch}] Dice={dice:.4f}  HD95={hd:.2f}  delta={delta:+.4f}")

        if dice > best_dice:
            best_dice = dice
            torch.save(student.state_dict(), str(save_dir / "best_model.pth"))
            if args.method == 'ts':
                torch.save({'T': T.item()}, str(save_dir / "T.pt"))

        with open(csv_path, 'a') as f:
            f.write(f'{epoch},{args.method},{avg_loss:.6f},'
                    f'{dice:.6f},{jc:.6f},{hd:.4f},{asd:.4f},'
                    f'{best_dice:.6f},{delta:.6f}\n')

    log.info("=" * 60)
    log.info(f"Method        : {args.method}")
    log.info(f"Baseline Dice : {base_dice:.4f}")
    log.info(f"Best Dice     : {best_dice:.4f}  ({best_dice - base_dice:+.4f})")
    log.info(f"Saved to      : {save_dir}")


if __name__ == '__main__':
    main()
