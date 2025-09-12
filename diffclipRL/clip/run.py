# -*- coding: utf-8 -*-
# @Author : liang
# @File : run.py



import argparse
import os, yaml, datetime
import random

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
    """
    1. Loads training and validation datasets from cached LMDB files.
    2. Builds PyTorch DataLoaders for both splits.
    3. Instantiates the CLIP model and the training facilitator (CLIPTrainer).
    4. Runs the requested number of epochs:
         - Performs one training pass and one validation pass per epoch.
         - Saves the best checkpoint (lowest validation loss) to disk.
    5. Persists the training and validation loss history to a pickle file.

    Parameters
    ----------
    args : argparse.Namespace
        Command-line / configuration arguments containing hyper-param,
        file paths, and training options.
    """

    seed = args.seed
    if seed is not None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)

    # ------------------------------------------------------------------
    # 1. Load datasets
    # ------------------------------------------------------------------
    Traindataset = CrystalDataset.from_cache_path(
        cache_path=args.train_path,
        transforms=[symmetrize_lattice, set_chemical_system_string],
        properties=[],
        dataset_transforms=[filter_sparse_properties],
    )

    Valdataset = CrystalDataset.from_cache_path(
        cache_path=args.val_path,
        transforms=[symmetrize_lattice, set_chemical_system_string],
        properties=[],
        dataset_transforms=[filter_sparse_properties],
    )

    # ------------------------------------------------------------------
    # 2. Build DataLoaders
    # ------------------------------------------------------------------
    train_loader = DataLoader(
        Traindataset,
        shuffle=args.shuffle,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        worker_init_fn=worker_init_fn,
        collate_fn=collate
    )

    valid_loader = DataLoader(
        Valdataset,
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
        train_loss, lr, train_graph_acc, train_xrd_acc = trainer.train(train_loader)

        # --- Validation step ---
        val_loss, val_graph_acc, val_xrd_acc = trainer.validate(valid_loader)

        # --- Scheduler step (after a warm-up period) ---
        if epoch_counter >= args.boundary:
            trainer.scheduler.step()

        # --- Checkpointing ---
        if val_loss < best_loss:
            best_loss = val_loss
            best_epoch = epoch_counter

            current_time = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
            checkpoint_filename = f'clip_checkpoint_{best_epoch}_{current_time}_loss_{best_loss}.pt'

            torch.save(
                {
                    'epoch': best_epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': trainer.optimizer.state_dict(),
                    'scheduler_state_dict': trainer.scheduler.state_dict(),
                    'learning_rate': lr,
                },
                os.path.join(str(args.ckpt_save_path), checkpoint_filename)
            )

            print(
                f"Epoch: {epoch_counter}, Best Val Loss = {round(best_loss, 5)}, checkpoint saved.")
            print()

            # Accumulate losses for logging
            train_losses.append(train_loss)
            val_losses.append(val_loss)

    # ------------------------------------------------------------------
    # 6. Save loss history to disk
    # ------------------------------------------------------------------
    df = pd.DataFrame(columns=['train_losses', 'val_losses'])
    df['train_losses'] = train_losses
    df['val_losses'] = val_losses
    df.to_pickle(os.path.join(args.results_save_path, str(args.name) + '_clip_loss.pkl'))

    print('Best Epoch is:', best_epoch, 'Best Loss is:', best_loss)