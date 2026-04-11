"""
Download and prepare Pancreas-CT (NIH CT-82) for DGEM experiments.
===================================================================

The NIH Pancreas-CT dataset is hosted on TCIA (The Cancer Imaging Archive)
and requires a free registration at:
    https://wiki.cancerimagingarchive.net/display/Public/Pancreas-CT

This script offers two paths:

  1. TCIA API download  — uses the public NBIA REST API (no login needed for
     this collection) to fetch the DICOM series, then converts to H5.

  2. SSL4MIS preprocessed H5  — download already-preprocessed H5 files from
     the SSL4MIS Google Drive mirror and skip DICOM conversion entirely.
     (Same format used by BCP / DGEM.)

Usage:
    # Option A: download preprocessed H5 from SSL4MIS (recommended, ~2 GB)
    python data/download_pancreas.py \
        --method  ssl4mis \
        --out_dir /path/to/pancreas_h5

    # Option B: download raw DICOM from TCIA, then preprocess
    python data/download_pancreas.py \
        --method  tcia \
        --out_dir /path/to/pancreas_h5 \
        --dicom_dir /tmp/pancreas_dicom \
        --label_dir /path/to/TCIA_pancreas_labels

    # Generate synthetic data (no download, for pipeline testing only)
    python data/download_pancreas.py \
        --method  synthetic \
        --out_dir /path/to/synthetic_h5 \
        --n_cases 20 \
        --vol_size 64
"""

import argparse
import os
import sys
import subprocess
from pathlib import Path


# ─────────────────────────────────────────────────────────────────────────────
# Args
# ─────────────────────────────────────────────────────────────────────────────

def get_args():
    p = argparse.ArgumentParser()
    p.add_argument('--method',   required=True,
                   choices=['ssl4mis', 'tcia', 'synthetic'],
                   help='Download method (see module docstring)')
    p.add_argument('--out_dir',  required=True,
                   help='Directory to write H5 files')
    # TCIA-specific
    p.add_argument('--dicom_dir', default=None,
                   help='[tcia] Directory to save raw DICOM series')
    p.add_argument('--label_dir', default=None,
                   help='[tcia] Directory containing NIfTI segmentation labels')
    # Synthetic-specific
    p.add_argument('--n_cases',  type=int, default=20,
                   help='[synthetic] Number of phantom volumes to generate')
    p.add_argument('--vol_size', type=int, default=64,
                   help='[synthetic] Cube side length (64=fast, 96=real size)')
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Method A: SSL4MIS preprocessed H5 (recommended)
# ─────────────────────────────────────────────────────────────────────────────

# Google Drive file ID for SSL4MIS preprocessed Pancreas-CT H5 archive.
# Source: https://github.com/HiLab-git/SSL4MIS (Pancreas-CT preprocessed data)
_SSL4MIS_GDRIVE_ID = '1qVrpV7vmhIsUxFiH189LmAn0ALbAPrgS'
_SSL4MIS_FILENAME  = 'pancreas_h5.zip'


def download_ssl4mis(out_dir: Path):
    """Download preprocessed H5 archive from SSL4MIS Google Drive."""
    try:
        import gdown
    except ImportError:
        print('Installing gdown...')
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', 'gdown'])
        import gdown

    out_dir.mkdir(parents=True, exist_ok=True)
    zip_path = out_dir.parent / _SSL4MIS_FILENAME

    if zip_path.exists():
        print(f'Archive already exists: {zip_path}')
    else:
        print(f'Downloading Pancreas-CT H5 archive from SSL4MIS Google Drive...')
        url = f'https://drive.google.com/uc?id={_SSL4MIS_GDRIVE_ID}'
        gdown.download(url, str(zip_path), quiet=False)

    print(f'Extracting to {out_dir} ...')
    import zipfile
    import tarfile
    if tarfile.is_tarfile(str(zip_path)):
        with tarfile.open(str(zip_path), 'r:*') as tf:
            tf.extractall(str(out_dir.parent))
    else:
        with zipfile.ZipFile(str(zip_path), 'r') as zf:
            zf.extractall(str(out_dir.parent))

    h5_files = list(out_dir.glob('*.h5'))
    print(f'Done. Found {len(h5_files)} H5 files in {out_dir}')
    if not h5_files:
        print('WARNING: no .h5 files found. Check that extraction succeeded and '
              'that --out_dir points to the folder containing the H5 files.')
    return h5_files


# ─────────────────────────────────────────────────────────────────────────────
# Method B: Raw TCIA DICOM download
# ─────────────────────────────────────────────────────────────────────────────

# TCIA collection name for Pancreas-CT (NIH CT-82)
_TCIA_COLLECTION = 'Pancreas-CT'
_TCIA_API_BASE   = 'https://services.cancerimagingarchive.net/nbia-api/services/v1'


def download_tcia(out_dir: Path, dicom_dir: Path, label_dir: Path):
    """
    Download Pancreas-CT DICOM series from TCIA using the public REST API,
    then convert to H5 using preprocess_pancreas.py.

    Note: the segmentation labels must be downloaded separately from:
        https://wiki.cancerimagingarchive.net/display/Public/Pancreas-CT
      → "TCIA_pancreas_labels-02-05-2017.zip"
    """
    try:
        import requests
        from tqdm import tqdm
    except ImportError:
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q',
                               'requests', 'tqdm'])
        import requests
        from tqdm import tqdm

    dicom_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Get series UIDs
    print(f'Fetching series list for collection: {_TCIA_COLLECTION}')
    resp = requests.get(
        f'{_TCIA_API_BASE}/getSeries',
        params={'Collection': _TCIA_COLLECTION, 'format': 'json'},
        timeout=30
    )
    resp.raise_for_status()
    series = resp.json()
    print(f'Found {len(series)} series')

    # 2. Download each series as ZIP
    for i, s in enumerate(tqdm(series, desc='Downloading series')):
        uid = s['SeriesInstanceUID']
        zip_path = dicom_dir / f'{uid}.zip'
        if zip_path.exists():
            continue
        dl_resp = requests.get(
            f'{_TCIA_API_BASE}/getImage',
            params={'SeriesInstanceUID': uid},
            stream=True, timeout=120
        )
        dl_resp.raise_for_status()
        with open(str(zip_path), 'wb') as f:
            for chunk in dl_resp.iter_content(chunk_size=8192):
                f.write(chunk)

    print(f'\nDICOM download complete. Saved to: {dicom_dir}')
    print(f'\nNow converting DICOM → H5...')

    if label_dir is None or not Path(label_dir).exists():
        print('ERROR: --label_dir is required for TCIA method.')
        print('Download the label archive from:')
        print('  https://wiki.cancerimagingarchive.net/display/Public/Pancreas-CT')
        print('  → "TCIA_pancreas_labels-02-05-2017.zip"')
        sys.exit(1)

    # Run the existing preprocess script
    script = Path(__file__).parent / 'preprocess_pancreas.py'
    subprocess.check_call([
        sys.executable, str(script),
        '--data_root',  str(dicom_dir),
        '--label_root', str(label_dir),
        '--output_dir', str(out_dir),
    ])
    print(f'H5 files written to: {out_dir}')


# ─────────────────────────────────────────────────────────────────────────────
# Method C: Synthetic phantoms (no download needed)
# ─────────────────────────────────────────────────────────────────────────────

def generate_synthetic(out_dir: Path, n_cases: int, vol_size: int):
    script = Path(__file__).parent / 'make_synthetic.py'
    subprocess.check_call([
        sys.executable, str(script),
        '--output_dir', str(out_dir),
        '--n_cases',    str(n_cases),
        '--vol_size',   str(vol_size),
    ])


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    args = get_args()
    out_dir = Path(args.out_dir)

    if args.method == 'ssl4mis':
        print('=== Method: SSL4MIS preprocessed H5 ===')
        h5_files = download_ssl4mis(out_dir)
        if h5_files:
            print(f'\nNext step:')
            print(f'  python data/generate_splits.py --h5_dir {out_dir} '
                  f'--splits_dir splits/pancreas --label_percent 20')

    elif args.method == 'tcia':
        print('=== Method: TCIA DICOM download + preprocess ===')
        if not args.dicom_dir:
            print('ERROR: --dicom_dir is required for tcia method.')
            sys.exit(1)
        download_tcia(
            out_dir,
            dicom_dir=Path(args.dicom_dir),
            label_dir=Path(args.label_dir) if args.label_dir else None,
        )
        print(f'\nNext step:')
        print(f'  python data/generate_splits.py --h5_dir {out_dir} '
              f'--splits_dir splits/pancreas --label_percent 20')

    elif args.method == 'synthetic':
        print('=== Method: Synthetic phantom data (no download) ===')
        generate_synthetic(out_dir, args.n_cases, args.vol_size)
        print(f'\nNext step:')
        print(f'  python data/generate_splits.py --h5_dir {out_dir} '
              f'--splits_dir splits/pancreas --label_percent 20 '
              f'--n_test 4')


if __name__ == '__main__':
    main()
