"""
Preprocess NIH Pancreas-CT volumes → h5 files.
Follows the same convention as BCP / CoraNet preprocessing pipeline.

Supports two input formats (--input_format):

  nifti (default):
    data_root/
        PANCREAS_0001/
            PANCREAS_0001.nii.gz   (image)
        ...

  dicom (TCIA download):
    data_root/
        "Pancreas-CT"/"PANCREAS_0001"/<date-study>/<series>/1-001.dcm ...
    Directory names may contain literal quote characters.

Label root always expects NIfTI:
    label_root/
        label0001.nii.gz
        ...

Output:
    output_dir/
        pancreas_001.h5      (keys: 'image', 'label')
        pancreas_002.h5
        ...

Pipeline (matching BCP/CoraNet):
  1. Load image + label
  2. Resample to isotropic 1mm x 1mm x 1mm spacing
  3. Crop tightly around the pancreas label (25-voxel pad, min 96^3)
  4. Clip HU to [-125, 275], then per-volume min-max normalize to [0, 1]
  5. Save as gzip-compressed h5

Usage:
    # NIfTI (default):
    python data/preprocess_pancreas.py \\
        --data_root  /path/to/Pancreas_CT/PANCREAS \\
        --label_root /path/to/TCIA_pancreas_labels \\
        --output_dir /path/to/pancreas_h5

    # DICOM:
    python data/preprocess_pancreas.py \\
        --input_format dicom \\
        --data_root  /path/to/raw/Pancreas-CT-20200910 \\
        --label_root /path/to/TCIA_pancreas_labels \\
        --output_dir /path/to/pancreas_h5

Then run generate_splits.py to create train/test split files.
"""

import argparse
import os
import re
import numpy as np
import nibabel as nib
import h5py
from scipy.ndimage import zoom
from pathlib import Path
from tqdm import tqdm

TARGET_SPACING = np.array([1.0, 1.0, 1.0])
HU_LOWER = -125
HU_UPPER = 275
CROP_PAD = 25
MIN_SIZE = 96


def resample_volume(image, original_spacing, target_spacing=TARGET_SPACING,
                    order=3):
    """Resample a volume to target_spacing using scipy.ndimage.zoom.

    Parameters
    ----------
    image : np.ndarray
        Input volume.
    original_spacing : array-like of length 3
        Voxel spacing of the input volume (x, y, z) in mm.
    target_spacing : array-like of length 3
        Desired output spacing in mm.
    order : int
        Interpolation order (3 = cubic for images, 0 = nearest for labels).

    Returns
    -------
    np.ndarray
        Resampled volume.
    """
    original_spacing = np.array(original_spacing, dtype=np.float64)
    target_spacing = np.array(target_spacing, dtype=np.float64)
    zoom_factors = original_spacing / target_spacing
    resampled = zoom(image, zoom_factors, order=order)
    return resampled


def crop_around_label(image, label, pad=CROP_PAD, min_size=MIN_SIZE):
    """Crop image and label tightly around the non-zero region of the label.

    Parameters
    ----------
    image : np.ndarray
    label : np.ndarray
    pad : int
        Padding (voxels) to add on each side of the bounding box.
    min_size : int
        Minimum extent along each axis; padding is expanded if needed.

    Returns
    -------
    (image_crop, label_crop)
    """
    coords = np.argwhere(label > 0)
    if len(coords) == 0:
        # No foreground -- return as-is (should not happen for pancreas)
        return image, label

    bb_min = coords.min(axis=0)
    bb_max = coords.max(axis=0)

    # Add padding and enforce minimum size
    starts = []
    ends = []
    for axis in range(3):
        lo = bb_min[axis] - pad
        hi = bb_max[axis] + pad + 1  # +1 because max is inclusive
        extent = hi - lo
        if extent < min_size:
            deficit = min_size - extent
            lo -= deficit // 2
            hi += deficit - deficit // 2
        # Clip to volume bounds
        lo = max(lo, 0)
        hi = min(hi, image.shape[axis])
        starts.append(lo)
        ends.append(hi)

    slices = tuple(slice(s, e) for s, e in zip(starts, ends))
    return image[slices], label[slices]


def normalize(image, lower=HU_LOWER, upper=HU_UPPER):
    """Clip HU values and per-volume min-max normalize to [0, 1].

    Matches BCP/CoraNet: clip to [-125, 275], then per-volume min-max.
    """
    image = np.clip(image, lower, upper)
    img_min = image.min()
    img_max = image.max()
    image = (image - img_min) / (img_max - img_min + 1e-8)
    return image.astype(np.float32)


def load_dicom_volume(case_dir):
    """Load a DICOM series from *case_dir* using SimpleITK.

    *case_dir* should be the PANCREAS_XXXX directory.  The actual .dcm
    files may be nested several levels deep (TCIA layout:
    ``<study>/<series>/*.dcm``).  We walk downward to find the deepest
    directory that contains .dcm files.

    Returns
    -------
    image : np.ndarray (float32)
        Volume in (W, H, D) layout to match nibabel convention.
    spacing : tuple of float
        Voxel spacing in (W, H, D) order — i.e. (x, y, z) in mm.
    """
    import SimpleITK as sitk

    # Walk to find the deepest directory containing .dcm files
    dicom_dir = None
    for root, _dirs, files in os.walk(str(case_dir)):
        if any(f.endswith('.dcm') for f in files):
            dicom_dir = root  # keep going; last found is deepest

    if dicom_dir is None:
        raise FileNotFoundError(
            f'No .dcm files found under {case_dir}')

    reader = sitk.ImageSeriesReader()
    dicom_names = reader.GetGDCMSeriesFileNames(str(dicom_dir))
    if not dicom_names:
        raise FileNotFoundError(
            f'SimpleITK found no DICOM series in {dicom_dir}')
    reader.SetFileNames(dicom_names)
    sitk_image = reader.Execute()

    # sitk.GetSpacing() returns (x, y, z)
    spacing_xyz = sitk_image.GetSpacing()

    # sitk.GetArrayFromImage returns (D, H, W); transpose to (W, H, D)
    image = sitk.GetArrayFromImage(sitk_image).astype(np.float32)
    image = image.transpose(2, 1, 0)

    # spacing needs to match the (W, H, D) = (x, y, z) layout
    spacing = (spacing_xyz[0], spacing_xyz[1], spacing_xyz[2])
    return image, spacing


def process_case_nifti(img_path, lbl_path, out_path):
    """Process a single case from NIfTI input."""
    img_nib = nib.load(str(img_path))
    lbl_nib = nib.load(str(lbl_path))

    image = img_nib.get_fdata(dtype=np.float32)
    label = lbl_nib.get_fdata().astype(np.uint8)

    # Get spacing from NIfTI header (x, y, z)
    spacing = np.array(img_nib.header.get_zooms()[:3], dtype=np.float64)

    # Step 1: Resample to isotropic 1mm
    image = resample_volume(image, spacing, TARGET_SPACING, order=3)
    label = resample_volume(label.astype(np.float32), spacing, TARGET_SPACING,
                            order=0).astype(np.uint8)

    # Step 2: Crop around pancreas label
    image, label = crop_around_label(image, label)

    # Step 3: Normalize (clip + per-volume min-max)
    image = normalize(image)

    with h5py.File(str(out_path), 'w') as f:
        f.create_dataset('image', data=image, compression='gzip')
        f.create_dataset('label', data=label, compression='gzip')


def process_case_dicom(case_dir, lbl_path, out_path):
    """Process a single case from DICOM input."""
    image, spacing = load_dicom_volume(case_dir)
    lbl_nib = nib.load(str(lbl_path))
    label = lbl_nib.get_fdata().astype(np.uint8)

    # Step 1: Resample to isotropic 1mm
    image = resample_volume(image, spacing, TARGET_SPACING, order=3)
    label = resample_volume(label.astype(np.float32), spacing, TARGET_SPACING,
                            order=0).astype(np.uint8)

    # Step 2: Crop around pancreas label
    image, label = crop_around_label(image, label)

    # Step 3: Normalize (clip + per-volume min-max)
    image = normalize(image)

    with h5py.File(str(out_path), 'w') as f:
        f.create_dataset('image', data=image, compression='gzip')
        f.create_dataset('label', data=label, compression='gzip')


def _extract_case_number(name):
    """Extract the numeric part from a case directory name.

    Handles both clean names (``PANCREAS_0001``) and TCIA names that
    may contain literal quote characters (``"PANCREAS_0001"``).
    """
    # Strip any surrounding quote characters from the directory name
    clean = name.strip('"').strip("'")
    m = re.search(r'PANCREAS_(\d+)', clean)
    if m is None:
        return None
    return m.group(1)


def _collect_dicom_cases(data_root):
    """Find PANCREAS_XXXX directories under a TCIA-style data_root.

    TCIA layout example::

        data_root/"Pancreas-CT"/"PANCREAS_0001"/<study>/<series>/*.dcm

    Directories may have literal quote characters in their names.
    We recursively search for directories whose (cleaned) name matches
    ``PANCREAS_\\d+``.
    """
    cases = []
    for root, dirs, _files in os.walk(str(data_root)):
        for d in dirs:
            clean = d.strip('"').strip("'")
            if re.fullmatch(r'PANCREAS_\d+', clean):
                cases.append(Path(root) / d)
    # De-duplicate (shouldn't happen, but just in case)
    seen = set()
    unique = []
    for c in sorted(cases):
        num = _extract_case_number(c.name)
        if num not in seen:
            seen.add(num)
            unique.append(c)
    return unique


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data_root',  required=True,
                   help='Dir containing PANCREAS_XXXX subdirs')
    p.add_argument('--label_root', required=True,
                   help='Dir containing labelXXXX.nii.gz files')
    p.add_argument('--output_dir', required=True)
    p.add_argument('--input_format', choices=['nifti', 'dicom'],
                   default='nifti',
                   help='Input image format (default: nifti)')
    args = p.parse_args()

    data_root  = Path(args.data_root)
    label_root = Path(args.label_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.input_format == 'dicom':
        cases = _collect_dicom_cases(data_root)
    else:
        cases = sorted(data_root.glob('PANCREAS_*'))

    print(f'Found {len(cases)} cases ({args.input_format} mode).')

    for case_dir in tqdm(cases, desc='Preprocessing'):
        num = _extract_case_number(case_dir.name)
        if num is None:
            print(f'  WARNING: cannot parse case number from {case_dir.name}, skipping')
            continue

        lbl_path = label_root / f'label{num}.nii.gz'
        if not lbl_path.exists():
            print(f'  WARNING: label not found for PANCREAS_{num}, skipping')
            continue

        out_name = f'pancreas_{int(num):03d}.h5'
        out_path = output_dir / out_name

        if args.input_format == 'dicom':
            process_case_dicom(case_dir, lbl_path, out_path)
        else:
            img_files = list(case_dir.glob('*.nii.gz'))
            if not img_files:
                print(f'  WARNING: no .nii.gz in {case_dir}, skipping')
                continue
            process_case_nifti(img_files[0], lbl_path, out_path)

    print(f'\nDone. H5 files saved to: {output_dir}')
    print('Next step: run data/generate_splits.py')


if __name__ == '__main__':
    main()
