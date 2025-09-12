# -*- coding: utf-8 -*-
# @Author : liang
# @File : agent.py


import os
from collections import deque
from pathlib import Path
from typing import Literal
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


from diffclipRL.mattergen.common.utils.data_classes import PRETRAINED_MODEL_NAME, MatterGenCheckpointInfo
from diffclipRL.mattergen.generator import CrystalGenerator
from diffclipRL.mattergen.common.data.dataset import structures_to_numpy, CrystalDataset
from diffclipRL.mattergen.common.data.transform import symmetrize_lattice, set_chemical_system_string
from diffclipRL.mattergen.common.data.collate import collate
from diffclipRL.reinforce.scores import ScoreFuncWrapper
from diffclipRL.reinforce.utils import Variable

from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm



class PerCondStatTracker:
    def __init__(self, buffer_size=32, min_count=16):
        self.buffer_size = buffer_size
        self.min_count = min_count
        self.stats = {}  # key -> deque

    def update(self, keys: np.ndarray, rewards: np.ndarray) -> np.ndarray:
        adv = np.empty_like(rewards, dtype=np.float32)
        uniq = np.unique(keys)
        for k in uniq:
            m = (keys == k)
            vals = rewards[m]
            dq = self.stats.get(k)
            if dq is None:
                dq = deque(maxlen=self.buffer_size)
                self.stats[k] = dq
            dq.extend(vals.tolist())
            pool = np.array(dq) if len(dq) >= self.min_count else rewards
            mean, std = pool.mean(), pool.std() + 1e-6
            adv[m] = (vals - mean) / std
        return adv


class Agent:
    def __init__(self,
                 args=None,
                 batch_size=None,
                 num_batches=None,
                 sampling_config_overrides=None,
                 properties_to_condition_on=None,
                 target_compositions=None,
                 checkpoint_epoch=None,
                 strict_checkpoint_loading=None,
                 sampling_config_path=None,
                 record_trajectories=None,
                 output_path=None,
                 lr=None,
                 device=None,
                 weight_decay=None,
                 score_model_path=None,
                 latent_dim=512,
                 xrd_dim=512,
                 max_steps=1000,
                 sampling_config_name='default',
                 diffusion_guidance_factor=2.0,
                 dataloader=None,
                 ):

        super(Agent, self).__init__()

        self.batch_size = batch_size
        self.num_batches = num_batches
        self.checkpoint_epoch = checkpoint_epoch
        self.strict_checkpoint_loading = strict_checkpoint_loading

        self.lr = lr
        self.weight_decay = weight_decay
        # Disable generating element types which are not supported or not in the desired chemical
        # system (if provided).

        self.sampling_config_overrides = sampling_config_overrides  or []
        self.config_overrides = ["++lightning_module.diffusion_module.model.element_mask_func={_target_:"
                             "'diffclipRL.mattergen.denoiser.mask_disallowed_elements',_partial_:True}"]

        self.properties_to_condition_on = properties_to_condition_on or {}
        self.target_compositions = target_compositions or {}

        self._sampling_config_path = Path(sampling_config_path) if sampling_config_path is not None else None

        self.sampling_config_name = sampling_config_name
        self.record_trajectories = record_trajectories
        self.diffusion_guidance_factor = diffusion_guidance_factor

        self.device = device
        self.latent_dim = latent_dim
        self.xrd_dim = xrd_dim

        self.ScoreModel = ScoreFuncWrapper(
                score_model_path=score_model_path,
                latent_dim=self.latent_dim,
                xrd_dim=self.xrd_dim,
                device=self.device,
                batch_size=self.batch_size,
        )

        self.dataloader = dataloader
        self.max_steps = max_steps

        self.clip_ratio = args.clip_ratio
        self.clip_adv = args.clip_adv
        self.inner_eps = args.inner_eps
        self.buffer = args.buffer
        self.min_count = args.min_count
        self.use_amp = args.use_amp


    def load_sample_model(self, PriorModelPath):

        prior_checkpoint_info = MatterGenCheckpointInfo(
            model_path=Path(PriorModelPath).resolve(),
            load_epoch=self.checkpoint_epoch,
            config_overrides=self.config_overrides,
            strict_checkpoint_loading=self.strict_checkpoint_loading,
        )

        self.prior_generator = CrystalGenerator(
            device=self.device,
            checkpoint_info=prior_checkpoint_info,
            properties_to_condition_on=self.properties_to_condition_on,
            batch_size=self.batch_size,
            num_batches=self.num_batches,
            sampling_config_name=self.sampling_config_name,
            sampling_config_path=self._sampling_config_path,
            sampling_config_overrides=self.sampling_config_overrides,
            record_trajectories=self.record_trajectories,
            diffusion_guidance_factor=(self.diffusion_guidance_factor if self.diffusion_guidance_factor is not None else 0.0),
            target_compositions_dict=self.target_compositions,
        )

        for param in self.prior_generator.model.parameters():
            param.requires_grad = False

    def load_agent(self, AgentModelPath):

        agent_checkpoint_info = MatterGenCheckpointInfo(
            model_path=Path(AgentModelPath).resolve(),
            load_epoch=self.checkpoint_epoch,
            config_overrides=self.config_overrides,
            strict_checkpoint_loading=self.strict_checkpoint_loading,
            device=self.device,
        )

        self.agent_generator = CrystalGenerator(
            device=self.device,
            checkpoint_info=agent_checkpoint_info,
            properties_to_condition_on=self.properties_to_condition_on,
            batch_size=self.batch_size,
            num_batches=self.num_batches,
            sampling_config_name=self.sampling_config_name,
            sampling_config_path=self._sampling_config_path,
            sampling_config_overrides=self.sampling_config_overrides,
            record_trajectories=self.record_trajectories,
            diffusion_guidance_factor=(self.diffusion_guidance_factor if self.diffusion_guidance_factor is not None else 0.0),
            target_compositions_dict=self.target_compositions,
        )

        for param in self.agent_generator.model.parameters():
            param.requires_grad = True

        self.tracker = PerCondStatTracker(buffer_size=self.buffer, min_count=self.min_count)

    def train(self, ):
        # self.agent_generator.model.eval()

        optimizer = torch.optim.AdamW(
            self.agent_generator.model.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay
        )

        # ===== RL options =====
        rl_config = {
            "clip_ratio": self.clip_ratio,  # PPO ratio clipping parameter
            "clip_adv": self.clip_adv,  # Advantage clipping range [-10, 10]
            "inner_eps": self.inner_eps,  # Number of PPO inner-update epochs per rollout
            "buffer": self.buffer,  # Buffer size for advantage normalization
            "min_count": self.min_count,  # Minimum samples per condition before using per-condition stats
            "use_amp": self.use_amp,
        }


        for step in range(self.max_steps):
            print('the step is {}'.format(step + 1))
            samples = self.agent_generator.generate_batch(batch_size=self.batch_size,
                                                          num_batches=self.num_batches,
                                                          optimizer=optimizer,
                                                          score_model=self.ScoreModel,
                                                          rl_config=rl_config,
                                                          dataloader=self.dataloader,
                                                          tracker=self.tracker
                                                          )

            # xrd_list = np.tile(xrd, (len(samples), 1))
            # structure_infos, _ = structures_to_numpy(samples)
            #
            # dataset = CrystalDataset(
            #     pos=structure_infos["pos"],
            #     cell=structure_infos["cell"],
            #     atomic_numbers=structure_infos["atomic_numbers"],
            #     num_atoms=structure_infos["num_atoms"],
            #     structure_id=structure_infos["structure_id"],
            #     xrd=xrd_list,
            #     transforms=[symmetrize_lattice, set_chemical_system_string],
            #     properties=[],
            # )
            #
            # batch = next(iter(DataLoader(
            #     dataset,
            #     shuffle=False,
            #     batch_size=self.batch_size,
            #     collate_fn=collate))
            # )
            #
            # loss = self.ScoreModel(batch)  # shape: [batch_size] or scalar
            #
            # optimizer.zero_grad()
            # loss.backward()
            # optimizer.step()
            #
            # print('loss:', loss.item())

        new_samples = self.agent_generator.generate(batch_size=self.batch_size,
                                                    num_batches=self.num_batches)




