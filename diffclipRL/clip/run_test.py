# -*- coding: utf-8 -*-
# @Author : liang
# @File : run_test.py


# -*- coding: utf-8 -*-
# @Author : liang
# @File : run.py



import argparse
import os, yaml, datetime
import random
from pathlib import Path

import pandas as pd
import torch
import numpy as np
import torch.nn as nn
from sklearn.metrics import mean_absolute_error
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from diffclipRL.clip.module import CLIP
from diffclipRL.clip.trainer import CLIPTrainer
from diffclipRL.mattergen.common.data.dataset import CrystalDataset
from diffclipRL.mattergen.common.data.transform import symmetrize_lattice, set_chemical_system_string
from diffclipRL.mattergen.common.data.dataset_transform import filter_sparse_properties
from diffclipRL.mattergen.common.data.collate import collate

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

    seed = args.seed
    if seed is not None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)


    Testdataset = CrystalDataset.from_cache_path(
        cache_path=args.test_path,
        transforms=[symmetrize_lattice, set_chemical_system_string],
        properties=[],
        dataset_transforms=[filter_sparse_properties],
    )

    # ------------------------------------------------------------------
    # 2. Build DataLoaders
    # ------------------------------------------------------------------
    test_loader = DataLoader(
        Testdataset,
        shuffle=args.shuffle,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        worker_init_fn=worker_init_fn,
        collate_fn=collate
    )

    # ------------------------------------------------------------------
    # 3. Instantiate model
    # ------------------------------------------------------------------
    model = CLIP(
        latent_dim=args.latent_dim,
        xrd_dim=args.xrd_dim,
        expand_dim=args.expand_dim,
        num_blocks=args.num_blocks,
        device=args.device,
        batch_size=args.batch_size,
        temperature=args.temperature,
        use_cosine_similarity=args.use_cosine_similarity,
        use_contrastive_loss=args.use_contrastive_loss,
        task=args.task,
    )

    # Load checkpoint
    ckpt_path = Path(args.model_path)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Score-model weights not found at {ckpt_path}")

    state_dict = torch.load(ckpt_path, map_location=args.device, weights_only=False)
    model.load_state_dict(state_dict["model_state_dict"], strict=False)
    model.eval().to(args.device)

    print("model loaded successfully.")

    # ------------------------------------------------------------------
    # 4. Build trainer and configure optimizer / scheduler
    # ------------------------------------------------------------------
    trainer = CLIPTrainer(
        model=model,
        device=args.device,
        patience=args.patience,
        T_0=args.T_0,
        T_mult=args.T_mult,
        eta_min=args.eta_min,
        lr=args.lr,
        weight_decay=args.weight_decay
    )

    trainer.set_optimizer(optim=args.optimizer)
    trainer.set_scheduler(sched=args.scheduler)

    # Initialize best metrics
    best_epoch, best_loss = args.best_epoch, args.best_loss
    train_losses, val_losses = [], []

    # ------------------------------------------------------------------
    # 5. Training loop
    # ------------------------------------------------------------------
    for epoch_counter in range(1, args.epoch_counter + 1):
        print(f'Epoch: {epoch_counter}')

        # --- Training step ---
        loss, graph_acc, xrd_acc = trainer.test(test_loader)

        print(loss, graph_acc, xrd_acc)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()


    parser.add_argument('--name', default='alex_mp_20_cache', type=str)
    parser.add_argument('--task', default='clip', type=str)
    parser.add_argument('--prop', default="xrd", type=str)

    parser.add_argument('--model_path',
                        default='/home/zl/DiffclipRL/ckpt/clip/alex_mp_20'
                                '/clip_checkpoint_38_2025-07-15_18-53-00_loss_0.0023748775345970104.pt',
                        type=str)
    parser.add_argument('--dataset', default='/home/zl/DiffclipRL/dataset', type=str)
    parser.add_argument('--preprocess_workers', default=30, type=int)
    parser.add_argument('--xrd_filter', default='both', type=str)
    parser.add_argument('--device', default='cuda:0', type=str)

    parser.add_argument('--batch_size', default=48, type=int)
    parser.add_argument('--latent_dim', default=512, type=int)
    parser.add_argument('--xrd_dim', default=512, type=int)
    parser.add_argument('--expand_dim', default=1024, type=int)
    parser.add_argument('--epoch_counter', default=500, type=int)
    parser.add_argument('--lr', default=1.0e-5, type=float)
    parser.add_argument('--weight_decay', default=0.05, type=float)
    parser.add_argument('--num_workers', default=0, type=int)
    parser.add_argument('--seed', default=42, type=int)

    parser.add_argument('--drop', default=True, type=bool)
    parser.add_argument('--shuffle', default=True, type=bool)
    parser.add_argument('--num_blocks', default=4, type=int)
    parser.add_argument('--temperature', default=1.0, type=float)
    parser.add_argument('--use_cosine_similarity', default=True, type=bool)
    parser.add_argument('--use_contrastive_loss', default=False, type=bool)

    parser.add_argument('--optimizer', default="adamw", type=str)
    parser.add_argument('--scheduler', default="cosineWR", type=str)
    parser.add_argument('--best_epoch', default=0, type=float)
    parser.add_argument('--best_loss', default=100, type=float)
    parser.add_argument('--T_0', default=10, type=int)
    parser.add_argument('--T_mult', default=2, type=int)
    parser.add_argument('--eta_min', default=1e-6, type=float)
    parser.add_argument('--boundary', default=10, type=int)
    parser.add_argument('--patience', default=200, type=int)

    args = parser.parse_args()
    args.test_path = os.path.join(args.dataset, args.name, 'test')

    main(args)