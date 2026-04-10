"""Pancreas-CT dataloader (NIH, SSL4MIS / BCP convention).
H5 files must contain 'image' (float32, W×H×D) and 'label' (uint8, W×H×D).
Split text files contain one relative h5 path per line, e.g.:
    pancreas/pancreas_001.h5
"""

import numpy as np
import h5py
import torch
from pathlib import Path
from torch.utils.data import Dataset, DataLoader


# ── Transforms ────────────────────────────────────────────────────────────────

class RandomCrop3D:
    def __init__(self, size):
        self.size = size if isinstance(size, (tuple, list)) else (size, size, size)

    def __call__(self, image, label):
        p = self.size
        w, h, d = image.shape

        pw = max((p[0] - w) // 2 + 1, 0)
        ph = max((p[1] - h) // 2 + 1, 0)
        pd = max((p[2] - d) // 2 + 1, 0)
        if pw or ph or pd:
            image = np.pad(image, [(pw, pw), (ph, ph), (pd, pd)], mode='constant')
            label = np.pad(label, [(pw, pw), (ph, ph), (pd, pd)], mode='constant')

        w, h, d = image.shape
        x = np.random.randint(0, max(w - p[0], 1))
        y = np.random.randint(0, max(h - p[1], 1))
        z = np.random.randint(0, max(d - p[2], 1))
        return (image[x:x+p[0], y:y+p[1], z:z+p[2]],
                label[x:x+p[0], y:y+p[1], z:z+p[2]])


class RandomFlip3D:
    def __call__(self, image, label):
        axes = [ax for ax in range(3) if np.random.rand() > 0.5]
        if axes:
            image = np.flip(image, axis=axes).copy()
            label = np.flip(label, axis=axes).copy()
        return image, label


# ── Dataset ──────────────────────────────────────────────────────────────────

class PancreasDataset(Dataset):
    """
    Args:
        data_root: root directory containing h5 files
        split_file: text file with one relative h5 path per line
        patch_size: int or (W, H, D) tuple
        augment: apply random crop + flip (True for train, False for test/unlab)
        repeat: artificially repeat labeled set to match unlabeled length
    """
    def __init__(self, data_root, split_file, patch_size=96,
                 augment=True, repeat=1):
        self.data_root  = Path(data_root)
        self.patch_size = patch_size if isinstance(patch_size, tuple) else (patch_size,) * 3
        self.augment    = augment
        self.repeat     = repeat

        with open(split_file) as f:
            self.cases = [l.strip() for l in f if l.strip()]

        self.crop = RandomCrop3D(self.patch_size)
        self.flip = RandomFlip3D()

    def __len__(self):
        return len(self.cases) * self.repeat

    def __getitem__(self, idx):
        case = self.cases[idx % len(self.cases)]
        path = self.data_root / case
        with h5py.File(str(path), 'r') as f:
            image = f['image'][:].astype(np.float32)
            label = f['label'][:].astype(np.uint8)

        if self.augment:
            image, label = self.crop(image, label)
            image, label = self.flip(image, label)
        else:
            image, label = self.crop(image, label)

        image = torch.from_numpy(image).unsqueeze(0).float()   # (1, W, H, D)
        label = torch.from_numpy(label.astype(np.int64)).long() # (W, H, D)
        return image, label


# ── Factory ──────────────────────────────────────────────────────────────────

def get_loaders(data_root, splits_dir, label_percent=20,
                patch_size=96, batch_size=2, num_workers=2):
    """
    Returns (lab_loader, unlab_loader, test_loader).

    Expected split files:
        splits_dir/train_lab_{label_percent}.txt
        splits_dir/train_unlab_{label_percent}.txt
        splits_dir/test.txt
    """
    splits_dir = Path(splits_dir)

    lab_ds   = PancreasDataset(data_root,
                               splits_dir / f'train_lab_{label_percent}.txt',
                               patch_size=patch_size, augment=True, repeat=5)
    unlab_ds = PancreasDataset(data_root,
                               splits_dir / f'train_unlab_{label_percent}.txt',
                               patch_size=patch_size, augment=True)
    test_ds  = PancreasDataset(data_root,
                               splits_dir / 'test.txt',
                               patch_size=patch_size, augment=False)

    lab_loader   = DataLoader(lab_ds,   batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True, drop_last=True)
    unlab_loader = DataLoader(unlab_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True, drop_last=True)
    test_loader  = DataLoader(test_ds,  batch_size=1,          shuffle=False,
                              num_workers=num_workers, pin_memory=True)

    return lab_loader, unlab_loader, test_loader
