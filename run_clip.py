# -*- coding: utf-8 -*-
# @Author : liang
# @File : run_clip.py

import os
import argparse
from diffclipRL.clip.run import main

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    # parser.add_argument('--train_path', default="/home/zl/DiffclipRL_old/dataset/alex_mp_20_xrd/train.csv", type=str)
    # parser.add_argument('--val_path', default="/home/zl/DiffclipRL_old/dataset/alex_mp_20_xrd/val.csv", type=str)
    # parser.add_argument('--dataset_save_path', default='/home/zl/DiffclipRL_old/dataset/alex_mp_20_pt', type=str)
    # os.environ["CUDA_VISIBLE_DEVICES"] = '1'
    # os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

    parser.add_argument('--name', default='alex_mp_20', type=str)
    parser.add_argument('--task', default='clip', type=str)
    parser.add_argument('--prop', default="xrd", type=str)

    parser.add_argument('--dataset', default='/home/zl/DiffclipRL/dataset', type=str)
    parser.add_argument('--ckpt_save_path', default='/home/zl/DiffclipRL/ckpt/clip', type=str)
    parser.add_argument('--results_save_path', default='/home/zl/DiffclipRL/results', type=str)
    parser.add_argument('--preprocess_workers', default=30, type=int)
    parser.add_argument('--xrd_filter', default='both', type=str)
    parser.add_argument('--device', default='cuda:2', type=str)

    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--latent_dim', default=512, type=int)
    parser.add_argument('--xrd_dim', default=512, type=int)
    parser.add_argument('--expand_dim', default=1024, type=int)
    parser.add_argument('--epoch_counter', default=100, type=int)
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
    args.train_path = os.path.join(args.dataset, str(args.name) + '_cache', 'train')
    args.val_path = os.path.join(args.dataset, str(args.name) + '_cache', 'val')

    # # dataset save path
    # dataset_save_path = os.path.join(args.dataset, str(args.name) + '_pt')
    # if not os.path.exists(dataset_save_path):
    #     os.makedirs(dataset_save_path, exist_ok=True)
    #
    # args.dataset_save_path = dataset_save_path

    # ckpt save path
    ckpt_save_path = os.path.join(args.ckpt_save_path, args.name)
    if not os.path.exists(ckpt_save_path):
        os.makedirs(ckpt_save_path, exist_ok=True)
    args.ckpt_save_path = ckpt_save_path

    main(args)