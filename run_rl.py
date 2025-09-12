# -*- coding: utf-8 -*-
# @Author : liang
# @File : run_rl.py


import argparse
import os

from torch_geometric.data import DataLoader
from tqdm import tqdm
import numpy as np
import random

import pandas as pd
import torch
from pymatgen.io.cif import CifParser
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from diffclipRL.mattergen.common.data.dataset import CrystalDataset
from diffclipRL.mattergen.common.data.transform import symmetrize_lattice, set_chemical_system_string
from diffclipRL.mattergen.common.data.dataset_transform import filter_sparse_properties
from diffclipRL.mattergen.common.data.collate import collate
from diffclipRL.reinforce.agent import Agent


import warnings
warnings.filterwarnings('ignore')

def worker_init_fn(id: int):
    """
    DataLoaders workers init function.

    Initialize the numpy.random seed correctly for each worker, so that
    random augmentations between workers and/or epochs are not identical.

    If a global seed is set, the augmentations are deterministic.

    https://pytorch.org/docs/stable/notes/randomness.html#dataloader
    """
    uint64_seed = torch.initial_seed()
    ss = np.random.SeedSequence([uint64_seed])
    # More than 128 bits (4 32-bit words) would be overkill.
    np.random.seed(ss.generate_state(4))
    random.seed(uint64_seed)


def main(args):

    Crystaldataset = CrystalDataset.from_cache_path(
        cache_path=args.test_path,
        transforms=[symmetrize_lattice, set_chemical_system_string],
        properties=[],
        dataset_transforms=[filter_sparse_properties],
    )

    Crystal_dataloader = DataLoader(
        Crystaldataset,
        shuffle=args.shuffle,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        worker_init_fn=worker_init_fn,
        collate_fn=collate
    )

    agent = Agent(
        args=args,
        batch_size=args.batch_size,
        num_batches=args.num_batches,
        score_model_path=args.score_model_path,
        latent_dim=args.latent_dim,
        xrd_dim=args.xrd_dim,
        max_steps=args.max_steps,
        lr=args.lr,
        weight_decay=args.weight_decay,
        device=args.device,
        checkpoint_epoch=args.checkpoint_epoch,
        record_trajectories=args.record_trajectories,
        dataloader=Crystal_dataloader
    )

    agent.load_agent(args.agent_model_path)

    agent.train()




if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--device', default='cuda:3', type=str)

    parser.add_argument('--score_model_path',
                        default='/home/zl/DiffclipRL/ckpt/clip/alex_mp_20/clip_checkpoint_best.pt',
                        type=str)
    parser.add_argument('--agent_model_path',
                        default='/home/zl/DiffclipRL/ckpt/diffusion/alex_mp_20',
                        type=str)
    parser.add_argument('--test_path',
                        default='/home/zl/DiffclipRL/dataset/debug_alex_mp_20_cache/test',
                        type=str)

    parser.add_argument('--xrd_filter', default='both', type=str)
    parser.add_argument('--n_postsubsample', default=512, type=int)
    parser.add_argument('--nanomaterial_size', default=10, type=int)
    parser.add_argument('--checkpoint_epoch', default='last', type=str)
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--latent_dim', default=512, type=int)
    parser.add_argument('--num_batches', default=1, type=int)
    parser.add_argument('--xrd_dim', default=512, type=int)
    parser.add_argument('--lr', default=1.0e-5, type=float)
    parser.add_argument('--weight_decay', default=0.05, type=float)
    parser.add_argument('--max_steps', default=1000, type=int)
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--shuffle', default=True, type=bool)
    parser.add_argument('--num_workers', default=0, type=int)
    parser.add_argument('--record_trajectories', default=True, type=bool)

    parser.add_argument('--clip_ratio', default=1e-4, type=float)
    parser.add_argument('--clip_adv', default=10.0, type=float)
    parser.add_argument('--inner_eps', default=1, type=int)
    parser.add_argument('--buffer', default=32, type=int)
    parser.add_argument('--min_count', default=16, type=int)
    parser.add_argument('--use_amp', default=True, type=bool)

    args = parser.parse_args()

    main(args)