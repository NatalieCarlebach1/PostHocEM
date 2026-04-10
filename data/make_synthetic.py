"""
Generate synthetic Pancreas-CT-like H5 data for smoke testing.
Creates small volumes with a random ellipsoid 'pancreas' mask.
No real data needed — use this to verify the full pipeline works.

Usage:
    python data/make_synthetic.py \
        --output_dir /path/to/synthetic_h5 \
        --n_cases    20 \
        --vol_size   96

    Then run generate_splits.py on the output dir.
"""

import argparse
import numpy as np
import h5py
from pathlib import Path
from tqdm import tqdm


def make_ellipsoid_mask(shape, center_frac=(0.5, 0.5, 0.5),
                        radius_frac=(0.12, 0.08, 0.12)):
    """Binary ellipsoid mask at given fractional centre and radius."""
    W, H, D = shape
    cx, cy, cz = int(center_frac[0]*W), int(center_frac[1]*H), int(center_frac[2]*D)
    rx, ry, rz = int(radius_frac[0]*W), int(radius_frac[1]*H), int(radius_frac[2]*D)
    rx, ry, rz = max(rx, 2), max(ry, 2), max(rz, 2)

    zz, yy, xx = np.mgrid[:W, :H, :D]
    mask = ((zz - cx)**2 / rx**2 +
            (yy - cy)**2 / ry**2 +
            (xx - cz)**2 / rz**2) <= 1.0
    return mask.astype(np.uint8)


def make_case(vol_size, rng):
    shape = (vol_size,) * 3
    # Background CT-like noise
    image = rng.normal(0.3, 0.15, shape).astype(np.float32)
    image = np.clip(image, 0, 1)

    # Random ellipsoid pancreas
    cx = rng.uniform(0.35, 0.65)
    cy = rng.uniform(0.35, 0.65)
    cz = rng.uniform(0.35, 0.65)
    rx = rng.uniform(0.08, 0.16)
    ry = rng.uniform(0.06, 0.12)
    rz = rng.uniform(0.08, 0.16)

    label = make_ellipsoid_mask(shape, (cx, cy, cz), (rx, ry, rz))

    # Make pancreas region slightly brighter
    image += label * rng.uniform(0.1, 0.25)
    image = np.clip(image, 0, 1).astype(np.float32)

    return image, label


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output_dir', required=True)
    p.add_argument('--n_cases',    type=int, default=20)
    p.add_argument('--vol_size',   type=int, default=96,
                   help='Cube side length in voxels (96 = real size, 64 = faster)')
    p.add_argument('--seed',       type=int, default=42)
    args = p.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.RandomState(args.seed)

    for i in tqdm(range(1, args.n_cases + 1), desc='Generating'):
        image, label = make_case(args.vol_size, rng)
        path = out / f'pancreas_{i:03d}.h5'
        with h5py.File(str(path), 'w') as f:
            f.create_dataset('image', data=image, compression='gzip')
            f.create_dataset('label', data=label, compression='gzip')

    print(f'\nGenerated {args.n_cases} synthetic cases in {out}')
    print(f'Volume shape: ({args.vol_size}, {args.vol_size}, {args.vol_size})')
    print('\nNext: python data/generate_splits.py --h5_dir', args.output_dir)


if __name__ == '__main__':
    main()
