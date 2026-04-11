"""
Visualize disagreement mask between student (net) and teacher (net_ema).

Generates a qualitative figure with representative axial slices showing:
  - Row 1: CT image
  - Row 2: Ground truth overlay
  - Row 3: Student prediction overlay
  - Row 4: Disagreement mask (red/yellow highlight)

Usage:
    python visualize_disagreement.py \
        --checkpoint result/dgem_20p/best_model.pth \
        --data_root  /path/to/pancreas_h5 \
        --case_id    0 \
        --output_dir figures/

If a matching *_ema_model.pth exists alongside the checkpoint, it is loaded as
the teacher model and the disagreement mask is computed. Otherwise only the
student predictions are shown (3 rows).
"""

import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from pathlib import Path

from networks import VNet
from dataloaders import FullVolumeDataset
from utils.metrics import sliding_window_inference


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint',  required=True,
                   help='Path to student checkpoint (.pth)')
    p.add_argument('--data_root',   required=True)
    p.add_argument('--test_file',   default='splits/pancreas/test.txt')
    p.add_argument('--case_id',     type=int, default=0,
                   help='Index of the test case to visualize')
    p.add_argument('--output_dir',  default='figures/')
    p.add_argument('--patch_size',  default='96,96,96',
                   help='Patch size as W,H,D (default: 96,96,96)')
    p.add_argument('--num_classes', type=int, default=2)
    p.add_argument('--num_slices',  type=int, default=5,
                   help='Number of representative axial slices')
    p.add_argument('--gpu',         default='0')
    return p.parse_args()


def load_model(checkpoint_path, n_classes):
    """Load a VNet checkpoint (matches evaluate.py pattern)."""
    net = VNet(n_channels=1, n_classes=n_classes,
               normalization='instancenorm', has_dropout=True)
    net = nn.DataParallel(net).cuda()

    state = torch.load(checkpoint_path, map_location='cuda')
    if isinstance(state, dict):
        if 'net' in state:
            state = state['net']
        elif 'model_state_dict' in state:
            state = state['model_state_dict']
    net.load_state_dict(state)
    net.eval()
    return net


def find_ema_checkpoint(checkpoint_path):
    """Try to find a matching EMA checkpoint next to the student checkpoint."""
    ckpt = Path(checkpoint_path)
    # Convention from train_dgem.py: best_model.pth -> best_ema_model.pth
    ema_path = ckpt.parent / ckpt.name.replace('best_model', 'best_ema_model')
    if ema_path.exists() and ema_path != ckpt:
        return str(ema_path)
    # Also try: model.pth -> ema_model.pth
    ema_path = ckpt.parent / ('ema_' + ckpt.name)
    if ema_path.exists():
        return str(ema_path)
    return None


def pick_representative_slices(label_vol, n_slices):
    """Pick axial slices that best show the organ (spread across foreground)."""
    # label_vol shape: (W, H, D) — axial slices along D axis
    fg_per_slice = label_vol.sum(axis=(0, 1))  # sum over W, H for each D
    fg_indices = np.where(fg_per_slice > 0)[0]

    if len(fg_indices) == 0:
        # No foreground — just pick evenly spaced slices
        total = label_vol.shape[2]
        return np.linspace(0, total - 1, n_slices, dtype=int)

    # Spread n_slices evenly across the foreground range
    start, end = fg_indices[0], fg_indices[-1]
    if end - start < n_slices:
        indices = fg_indices[:n_slices]
    else:
        indices = np.linspace(start, end, n_slices, dtype=int)
    return indices


def overlay_mask(ax, image_slice, mask_slice, color, alpha=0.35):
    """Show image with a colored mask overlay."""
    ax.imshow(image_slice.T, cmap='gray', origin='lower')
    if mask_slice.any():
        overlay = np.ma.masked_where(mask_slice == 0, mask_slice)
        cmap = ListedColormap([color])
        ax.imshow(overlay.T, cmap=cmap, alpha=alpha, origin='lower',
                  vmin=0.5, vmax=1.5)


def main():
    args = get_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    patch_size = tuple(int(x) for x in args.patch_size.split(','))

    # ── Load data ────────────────────────────────────────────────────────────
    test_ds = FullVolumeDataset(args.data_root, args.test_file)
    assert args.case_id < len(test_ds), (
        f'case_id={args.case_id} out of range (test set has {len(test_ds)} cases)')

    image_np, label_np, case_name = test_ds[args.case_id]
    print(f'Case: {case_name}  volume shape: {image_np.shape}')

    # ── Load student model and run inference ─────────────────────────────────
    net = load_model(args.checkpoint, args.num_classes)
    student_pred, student_score = sliding_window_inference(
        net, image_np, patch_size, stride_xy=16, stride_z=4,
        n_classes=args.num_classes)
    student_pred = student_pred.astype(np.uint8)
    print(f'Student prediction — foreground voxels: {student_pred.sum()}')

    # ── Try to load teacher (EMA) model ──────────────────────────────────────
    ema_path = find_ema_checkpoint(args.checkpoint)
    has_teacher = False
    if ema_path is not None:
        print(f'Found EMA checkpoint: {ema_path}')
        net_ema = load_model(ema_path, args.num_classes)
        teacher_pred, _ = sliding_window_inference(
            net_ema, image_np, patch_size, stride_xy=16, stride_z=4,
            n_classes=args.num_classes)
        teacher_pred = teacher_pred.astype(np.uint8)
        disagree_mask = (student_pred != teacher_pred).astype(np.uint8)
        has_teacher = True
        print(f'Teacher prediction — foreground voxels: {teacher_pred.sum()}')
        print(f'Disagreement voxels: {disagree_mask.sum()} '
              f'({100 * disagree_mask.mean():.2f}%)')
    else:
        print('No EMA checkpoint found — skipping disagreement mask.')

    # ── Pick representative slices ───────────────────────────────────────────
    slice_indices = pick_representative_slices(label_np, args.num_slices)
    n_slices = len(slice_indices)
    print(f'Axial slices: {slice_indices.tolist()}')

    # ── Build figure ─────────────────────────────────────────────────────────
    n_rows = 4 if has_teacher else 3
    row_labels = ['CT Image', 'Ground Truth', 'Student Pred']
    if has_teacher:
        row_labels.append('Disagreement')

    fig, axes = plt.subplots(n_rows, n_slices, figsize=(3 * n_slices, 3 * n_rows))
    if n_slices == 1:
        axes = axes[:, np.newaxis]

    for col, si in enumerate(slice_indices):
        img_sl = image_np[:, :, si]
        gt_sl  = label_np[:, :, si]
        stu_sl = student_pred[:, :, si]

        # Row 0: CT image
        axes[0, col].imshow(img_sl.T, cmap='gray', origin='lower')
        axes[0, col].set_title(f'Slice {si}', fontsize=9)

        # Row 1: Ground truth overlay (green)
        overlay_mask(axes[1, col], img_sl, gt_sl, color='lime', alpha=0.4)

        # Row 2: Student prediction overlay (cyan)
        overlay_mask(axes[2, col], img_sl, stu_sl, color='cyan', alpha=0.4)

        # Row 3: Disagreement mask (red/yellow)
        if has_teacher:
            dis_sl = disagree_mask[:, :, si]
            axes[3, col].imshow(img_sl.T, cmap='gray', origin='lower')
            if dis_sl.any():
                overlay = np.ma.masked_where(dis_sl == 0, dis_sl)
                cmap_ry = ListedColormap(['#FF4444'])
                axes[3, col].imshow(overlay.T, cmap=cmap_ry, alpha=0.55,
                                    origin='lower', vmin=0.5, vmax=1.5)

    # Row labels
    for r, label in enumerate(row_labels):
        axes[r, 0].set_ylabel(label, fontsize=10, fontweight='bold')

    # Clean up axes
    for ax in axes.flat:
        ax.set_xticks([])
        ax.set_yticks([])

    title = f'Disagreement Visualization — {case_name}'
    if not has_teacher:
        title += '  (single model, no disagreement)'
    fig.suptitle(title, fontsize=12, fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    # ── Save ─────────────────────────────────────────────────────────────────
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f'disagreement_{case_name}.png'
    fig.savefig(str(out_path), dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved: {out_path}')


if __name__ == '__main__':
    main()
