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
    python data/generate_splits.py --h5_dir /path/to/pancreas_h5
"""

import argparse
import random
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--h5_dir',       required=True,
                   help='Dir containing pancreas_XXX.h5 files')
    p.add_argument('--splits_dir',   default='splits/pancreas')
    p.add_argument('--n_test',       type=int, default=20)
    p.add_argument('--label_percent',type=int, default=20)
    p.add_argument('--seed',         type=int, default=2020)
    args = p.parse_args()

    random.seed(args.seed)
    h5_dir = Path(args.h5_dir)
    splits_dir = Path(args.splits_dir)
    splits_dir.mkdir(parents=True, exist_ok=True)

    all_cases = sorted([f.name for f in h5_dir.glob('*.h5')])
    print(f'Total cases found: {len(all_cases)}')

    random.shuffle(all_cases)
    test_cases  = all_cases[:args.n_test]
    train_cases = all_cases[args.n_test:]

    n_labeled = max(1, int(len(train_cases) * args.label_percent / 100))
    lab_cases   = train_cases[:n_labeled]
    unlab_cases = train_cases[n_labeled:]

    print(f'Train labeled   : {len(lab_cases)}')
    print(f'Train unlabeled : {len(unlab_cases)}')
    print(f'Test            : {len(test_cases)}')

    def write(path, cases):
        with open(path, 'w') as f:
            f.write('\n'.join(cases) + '\n')

    write(splits_dir / f'train_lab_{args.label_percent}.txt',   lab_cases)
    write(splits_dir / f'train_unlab_{args.label_percent}.txt', unlab_cases)
    write(splits_dir / 'test.txt',                              test_cases)
    print(f'\nSplit files saved to {splits_dir}/')


if __name__ == '__main__':
    main()
