"""
Preprocess NIH Pancreas-CT NIfTI volumes → h5 files.
Follows the same convention as SSL4MIS / BCP.

Expected input layout:
    data_root/
        PANCREAS_0001/
            PANCREAS_0001.nii.gz   (image)
        ...
    label_root/
        label0001.nii.gz
        ...

Output:
    output_dir/
        pancreas_001.h5      (keys: 'image', 'label')
        pancreas_002.h5
        ...

Usage:
    python data/preprocess_pancreas.py \
        --data_root  /path/to/Pancreas_CT/PANCREAS \
        --label_root /path/to/TCIA_pancreas_labels \
        --output_dir /path/to/pancreas_h5

Then run generate_splits.py to create train/test split files.
"""

import argparse
import os
import numpy as np
import nibabel as nib
import h5py
from pathlib import Path
from tqdm import tqdm


def normalize(image, lower=-125, upper=275):
    """Clip HU values and min-max normalize to [0, 1]."""
    image = np.clip(image, lower, upper)
    image = (image - lower) / (upper - lower)
    return image.astype(np.float32)


def process_case(img_path, lbl_path, out_path):
    img_nib = nib.load(str(img_path))
    lbl_nib = nib.load(str(lbl_path))

    image = img_nib.get_fdata(dtype=np.float32)
    label = lbl_nib.get_fdata().astype(np.uint8)

    image = normalize(image)

    with h5py.File(str(out_path), 'w') as f:
        f.create_dataset('image', data=image, compression='gzip')
        f.create_dataset('label', data=label, compression='gzip')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data_root',  required=True,
                   help='Dir containing PANCREAS_XXXX subdirs')
    p.add_argument('--label_root', required=True,
                   help='Dir containing labelXXXX.nii.gz files')
    p.add_argument('--output_dir', required=True)
    args = p.parse_args()

    data_root  = Path(args.data_root)
    label_root = Path(args.label_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Collect all cases
    cases = sorted(data_root.glob('PANCREAS_*'))
    print(f'Found {len(cases)} cases.')

    for case_dir in tqdm(cases, desc='Preprocessing'):
        case_id = case_dir.name  # e.g. PANCREAS_0001
        num     = case_id.split('_')[1]  # e.g. 0001

        img_files = list(case_dir.glob('*.nii.gz'))
        if not img_files:
            print(f'  WARNING: no .nii.gz in {case_dir}, skipping')
            continue
        img_path = img_files[0]

        lbl_path = label_root / f'label{num}.nii.gz'
        if not lbl_path.exists():
            print(f'  WARNING: label not found for {case_id}, skipping')
            continue

        out_name = f'pancreas_{int(num):03d}.h5'
        out_path = output_dir / out_name
        process_case(img_path, lbl_path, out_path)

    print(f'\nDone. H5 files saved to: {output_dir}')
    print('Next step: run data/generate_splits.py')


if __name__ == '__main__':
    main()
