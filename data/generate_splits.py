"""
Generate train/test split files for Pancreas-CT (SSL4MIS / BCP convention).
Creates:
    splits/pancreas/train_lab_20.txt    (labeled cases, 20% of train)
    splits/pancreas/train_unlab_20.txt  (unlabeled cases)
    splits/pancreas/test.txt            (test cases)

Standard SSL4MIS split:
    Total 82 cases → 62 train, 20 test.
    20% labeled = 12 cases out of 62.

Usage:
    # Random split:
    python data/generate_splits.py --h5_dir /path/to/pancreas_h5

    # BCP/CoraNet canonical split (deterministic, no h5_dir needed):
    python data/generate_splits.py --use_bcp_splits
"""

import argparse
import random
from pathlib import Path


# ── BCP / CoraNet canonical splits ───────────────────────────────────────────
# Cases 0025 and 0070 are excluded (known duplicates in the NIH dataset).

BCP_TEST = [64, 65, 66, 67, 68, 69, 71, 72, 73, 74,
            75, 76, 77, 78, 79, 80, 81, 82]

BCP_LABELED_20 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]

BCP_UNLABELED_20 = [
    13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24,
    26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39,
    40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53,
    54, 55, 56, 57, 58, 59, 60, 61, 62, 63,
]


def _case_num_to_h5(num):
    """Convert integer case number to h5 filename, e.g. 1 → 'pancreas_001.h5'."""
    return f'pancreas_{num:03d}.h5'


def write_split(path, cases):
    with open(path, 'w') as f:
        f.write('\n'.join(cases) + '\n')


def generate_bcp_splits(splits_dir, label_percent=20):
    """Write deterministic BCP-style split files for any label fraction.
    Test set is always the BCP canonical 18 cases.
    Train cases (62 total) are split by taking the first N as labeled.
    For 20% this matches the BCP/CoraNet canonical split exactly.
    """
    splits_dir = Path(splits_dir)
    splits_dir.mkdir(parents=True, exist_ok=True)

    # All 62 train cases in canonical order (excluding 0025, 0070)
    all_train = sorted(BCP_LABELED_20 + BCP_UNLABELED_20)
    n_labeled = max(1, int(len(all_train) * label_percent / 100))

    # For 100%, all are labeled and unlabeled list is empty
    lab_nums   = all_train[:n_labeled]
    unlab_nums = all_train[n_labeled:]

    test_cases  = [_case_num_to_h5(n) for n in BCP_TEST]
    lab_cases   = [_case_num_to_h5(n) for n in lab_nums]
    unlab_cases = [_case_num_to_h5(n) for n in unlab_nums]

    print(f'BCP-style split (Pancreas-CT {label_percent}% labeled)')
    print(f'  Train labeled   : {len(lab_cases)} cases')
    print(f'  Train unlabeled : {len(unlab_cases)} cases')
    print(f'  Test            : {len(test_cases)} cases')

    write_split(splits_dir / f'train_lab_{label_percent}.txt',   lab_cases)
    write_split(splits_dir / f'train_unlab_{label_percent}.txt', unlab_cases)
    write_split(splits_dir / 'test.txt',                         test_cases)
    print(f'\nSplit files saved to {splits_dir}/')


def generate_random_splits(h5_dir, splits_dir, n_test, label_percent, seed):
    """Write random split files (original behaviour)."""
    random.seed(seed)
    h5_dir = Path(h5_dir)
    splits_dir = Path(splits_dir)
    splits_dir.mkdir(parents=True, exist_ok=True)

    all_cases = sorted([f.name for f in h5_dir.glob('*.h5')])
    print(f'Total cases found: {len(all_cases)}')

    random.shuffle(all_cases)
    test_cases  = all_cases[:n_test]
    train_cases = all_cases[n_test:]

    n_labeled = max(1, int(len(train_cases) * label_percent / 100))
    lab_cases   = train_cases[:n_labeled]
    unlab_cases = train_cases[n_labeled:]

    print(f'Train labeled   : {len(lab_cases)}')
    print(f'Train unlabeled : {len(unlab_cases)}')
    print(f'Test            : {len(test_cases)}')

    write_split(splits_dir / f'train_lab_{label_percent}.txt',   lab_cases)
    write_split(splits_dir / f'train_unlab_{label_percent}.txt', unlab_cases)
    write_split(splits_dir / 'test.txt',                         test_cases)
    print(f'\nSplit files saved to {splits_dir}/')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--h5_dir',       default=None,
                   help='Dir containing pancreas_XXX.h5 files (not needed with --use_bcp_splits)')
    p.add_argument('--splits_dir',   default='splits/pancreas')
    p.add_argument('--n_test',       type=int, default=20)
    p.add_argument('--label_percent',type=int, default=20)
    p.add_argument('--seed',         type=int, default=2020)
    p.add_argument('--use_bcp_splits', action='store_true',
                   help='Use the deterministic BCP/CoraNet canonical splits '
                        'instead of a random split. No --h5_dir needed.')
    args = p.parse_args()

    if args.use_bcp_splits:
        generate_bcp_splits(args.splits_dir, args.label_percent)
    else:
        if args.h5_dir is None:
            p.error('--h5_dir is required unless --use_bcp_splits is set')
        generate_random_splits(args.h5_dir, args.splits_dir,
                               args.n_test, args.label_percent, args.seed)


if __name__ == '__main__':
    main()
