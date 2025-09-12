# -*- coding: utf-8 -*-
# @Author : liang
# @File : run_dataloader.py


import argparse
import os

from diffclipRL.mattergen.common.data.dataset import CrystalDataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--csv-folder",
        type=str,
        default='/home/zl/DiffclipRL/dataset/alex_mp_20_xrd',
        help="Path to the folder containing the csv files. All csv files in the folder will be processed (e.g., 'train.csv', 'val.csv', 'test.csv') and the resulting datasets will be placed under {cache_path/dataset_name/filename_without_extension}, e.g, /path/to/project/dataset/mp_20/train.",
    )

    parser.add_argument(
        "--cache-folder",
        type=str,
        # required=True,
        default=f"/home/zl/DiffclipRL/dataset/debug_alex_mp_20_cache",
        help="Path to the cache folder. Defaults to datasets folder in the project root.",
    )

    parser.add_argument(
        '--min_2_theta',
        default=0,
        type=float)

    parser.add_argument(
        '--max_2_theta',
        default=180,
        type=float)

    parser.add_argument(
        '--wavesource',
        default='CuKa',
        type=str)

    parser.add_argument(
        '--nanomaterial_size_angstrom',
        default=10,
        type=int)

    parser.add_argument(
        '--n_postsubsample',
        default=512,
        type=int)

    args = parser.parse_args()
    for file in os.listdir(f"{args.csv_folder}"):
        if file.endswith(".csv"):
            print(f"Processing {args.csv_folder}/{file}")
            CrystalDataset.from_csv(
                csv_path=f"{args.csv_folder}/{file}",
                cache_path=f"{args.cache_folder}/{file.split('.')[0]}",
                min_2_theta=args.min_2_theta,
                max_2_theta=args.max_2_theta,
                wavesource=args.wavesource,
                nanomaterial_size_angstrom=args.nanomaterial_size_angstrom,  # 10, 50, 100, 1000
                n_postsubsample=args.n_postsubsample,
                horizontal_noise_range=(1e-2, 1.1e-2),  # (1e-3, 1.1e-3)
                vertical_noise=1e-3,
            )


if __name__ == "__main__":

    main()
