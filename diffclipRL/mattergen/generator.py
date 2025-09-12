# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import io
import os
from dataclasses import dataclass
from pathlib import Path
from zipfile import ZipFile

import ase.io
import hydra
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from pymatgen.core.structure import Structure
from pymatgen.io.ase import AseAtomsAdaptor
from tqdm import tqdm

from diffclipRL.mattergen.common.data.chemgraph import ChemGraph
from diffclipRL.mattergen.common.data.collate import collate
from diffclipRL.mattergen.common.data.condition_factory import ConditionLoader
from diffclipRL.mattergen.common.data.num_atoms_distribution import NUM_ATOMS_DISTRIBUTIONS
from diffclipRL.mattergen.common.data.custom_types import TargetProperty
from diffclipRL.mattergen.common.utils.data_utils import lattice_matrix_to_params_torch
from diffclipRL.mattergen.common.utils.eval_utils import (
    MatterGenCheckpointInfo,
    get_crystals_list,
    load_model_diffusion,
    make_structure,
    save_structures,
)
from diffclipRL.mattergen.common.utils.globals import DEFAULT_SAMPLING_CONFIG_PATH
from diffclipRL.mattergen.diffusion.lightning_module import DiffusionLightningModule
from diffclipRL.mattergen.diffusion.sampling.pc_sampler import PredictorCorrector
from diffclipRL.mattergen.common.data.dataset import structures_to_numpy, CrystalDataset
from diffclipRL.mattergen.common.data.transform import symmetrize_lattice, set_chemical_system_string
from torch.utils.data import Dataset, DataLoader

import math
import numpy as np
import torch
from collections import deque


def gaussian_logprob(x, mean, std, reduce_non_batch=True):
    std = torch.clamp(std, min=1e-6)
    lp = -((x - mean) ** 2) / (2 * std ** 2) - torch.log(std) - math.log(math.sqrt(2 * math.pi))
    if reduce_non_batch:
        dims = tuple(range(1, lp.ndim))
        lp = lp.mean(dim=dims)
    return lp  # 形状：(B,) 或与 x 对齐

def ppo_surrogate(new_logps, old_logps, advantages, clip_ratio):
    """
    new_logps/old_logps: (T, B)
    advantages:          (B,)
    """
    # 按时间步平均 PPO 损失（也可加权）
    T = new_logps.shape[0]
    adv = advantages.unsqueeze(0).expand_as(new_logps)  # (T,B)
    ratio = torch.exp(new_logps - old_logps.detach())
    unclipped = -adv * ratio
    clipped = -adv * torch.clamp(ratio, 1.0 - clip_ratio, 1.0 + clip_ratio)
    loss = torch.max(unclipped, clipped).mean()  # 先 (T,B) 平均
    return loss

def _to_device(batch, device):
    # 递归把 batch 搬到 device；支持 Tensor / dict / list / 自定义含 .to()
    if torch.is_tensor(batch):
        return batch.to(device, non_blocking=True)
    if isinstance(batch, dict):
        return {k: _to_device(v, device) for k, v in batch.items()}
    if isinstance(batch, (list, tuple)):
        t = [_to_device(v, device) for v in batch]
        return type(batch)(t) if not isinstance(batch, tuple) else tuple(t)
    if hasattr(batch, "to"):
        try:
            return batch.to(device)
        except Exception:
            return batch
    return batch

def draw_samples_from_sampler(
    sampler: PredictorCorrector,
    condition_loader = None,
    properties_to_condition_on = None,
    output_path: Path | None = None,
    cfg: DictConfig | None = None,
    record_trajectories: bool = True,
    rl_config: dict | None = None,
    optimizer: torch.optim.Optimizer | None = None,
    cond_key_fn=None,
    score_model=None,
    tracker=None

):

    # Dict
    properties_to_condition_on = properties_to_condition_on or {}

    # we cannot conditional sample on something on which the model was not trained to condition on
    assert all([key in sampler.diffusion_module.model.cond_fields_model_was_trained_on for key in properties_to_condition_on.keys()])  # type: ignore

    all_samples_list, sample, x_traj, logprobs_pos, logprobs_cell, rewards, adv_np = [], None, None, None, None, None, None

    for conditioning_data in tqdm(condition_loader):
        # Generate rollout; record trajectory and old log-probs if needed by RL
        if record_trajectories or rl_config:
            sample, mean, x_traj, logprobs_pos, logprobs_cell = sampler.sample_with_record(conditioning_data, mask=None)
            # all_trajs_list.extend(list_of_time_steps_to_list_of_trajectories(intermediate_samples))
        else:
            sample, mean = sampler.sample(conditioning_data, mask=None)

        # ===== RL: compute rewards & advantages =====
        if rl_config:
            # 1) Build structures from mean (rollout result)
            tmp_collated = collate(sample.to_data_list())
            assert isinstance(tmp_collated, ChemGraph)
            # lengths, angles = lattice_matrix_to_params_torch(tmp_collated.cell)
            # tmp_collated = tmp_collated.replace(lengths=lengths, angles=angles)

            # structs = structure_from_model_output(
            #     tmp_collated["pos"].reshape(-1, 3),
            #     tmp_collated["atomic_numbers"].reshape(-1),
            #     tmp_collated["lengths"].reshape(-1, 3),
            #     tmp_collated["angles"].reshape(-1, 3),
            #     tmp_collated["num_atoms"].reshape(-1),
            # )

            dataset = CrystalDataset(
                pos=tmp_collated["pos"].reshape(-1, 3).detach().cpu().numpy(),
                cell=tmp_collated["cell"].reshape(-1, 3, 3).detach().cpu().numpy(),
                atomic_numbers=tmp_collated["atomic_numbers"].reshape(-1).detach().cpu().numpy(),
                num_atoms=tmp_collated["num_atoms"].reshape(-1).detach().cpu().numpy(),
                structure_id=np.array([None for _ in range(len(tmp_collated["num_atoms"]))]),
                xrd=conditioning_data['xrd'].detach().cpu().numpy(),
                transforms=[symmetrize_lattice, set_chemical_system_string],
                properties=None,
            )

            dataloader = DataLoader(
                dataset,
                shuffle=False,
                batch_size=len(tmp_collated["num_atoms"]),
                collate_fn=collate,
                drop_last=False,
                # pin_memory=True if device.type == "cuda" else False,
            )

            # 2) Rewards = -loss (numerically sanitized)
            for batch in tqdm(dataloader, total=len(dataloader), desc="Computing rewards", position=0):
                # batch = _to_device(batch, device=device)

                losses, graph_acc, xrd_acc = score_model(batch)  # shape: [batch_size] or scalar
                rewards = torch.nan_to_num((losses).detach(), nan=0.0, posinf=0.0, neginf=0.0)
                rewards = torch.clip(rewards, min=-1e6, max=1e6)
                # all_rewards.append(rewards.cpu().numpy())

                # 3) Keys for per-condition normalization
                # rewards_re = np.concatenate([r.reshape(-1) for r in rewards.detach().cpu().numpy()],
                #                             axis=0).astype(np.float32)  # (B,)
                #
                # if cond_key_fn is not None:
                #     keys = np.asarray(cond_key_fn(conditioning_data))
                #     if keys.shape[0] != rewards_re.shape[0]:
                #         raise ValueError(
                #             f"cond_key_fn length {keys.shape[0]} != rewards length {rewards_re.shape[0]}"
                #         )
                # else:
                #     keys = np.zeros_like(rewards_re, dtype=np.int64)
                #
                # # 4) Advantages (group-wise normalization) + clip
                # adv_np = tracker.update(keys, rewards_re)  # (B,)
                # if rl_config['clip_adv'] is not None:
                #     adv_np = np.clip(adv_np, -rl_config['clip_adv'], rl_config['clip_adv'])
                # adv_th = torch.from_numpy(adv_np).to(device=logprobs_pos.device, dtype=torch.float32)

                try:
                    tqdm.write(f"loss = {float(rewards.detach().cpu().numpy().mean()):.4f}, graph acc = {graph_acc:.4f}, xrd acc = {xrd_acc:.4f}")
                except Exception:
                    print(f"loss = {float(rewards.detach().cpu().numpy().mean()):.4f}, graph acc = {graph_acc:.4f}, xrd acc = {xrd_acc:.4f}")

            # 5) PPO updates (streaming/chunked to control memory); print loss each inner epoch
            for e in range(rl_config['inner_eps']):
                optimizer.zero_grad(set_to_none=True)

                avg_reward = sampler.recompute_logps(
                    x_traj=x_traj,
                    logprobs_pos=logprobs_pos,
                    logprobs_cell=logprobs_cell,
                    advantages=rewards,
                    clip_ratio=rl_config['clip_ratio'],
                    use_amp=rl_config['use_amp'],                       # AMP if you use GradScaler outside
                    mask=None
                )
                optimizer.step()

                # Print reward without breaking tqdm bars
                try:
                    tqdm.write(f"[PPO] inner_epoch {e+1}/{rl_config['inner_eps']} | reward = {float(avg_reward):.6f}")
                except Exception:
                    print(f"[PPO] inner_epoch {e+1}/{rl_config['inner_eps']} | reward = {float(avg_reward):.6f}")

            # ===== Collect samples normally =====
            all_samples_list.extend(mean.to_data_list())

        return all_samples_list


def list_of_time_steps_to_list_of_trajectories(
    list_of_time_steps: list[ChemGraph],
) -> list[list[ChemGraph]]:
    # Rearrange the shapes of the recorded intermediate samples and predictions
    # We get a list of <num_timesteps> many ChemGraphBatches, each containing <batch_size>
    # many ChemGraphs. Instead, we group all the ChemGraphs of the same trajectory together,
    # i.e., we construct lists of <batch_size> many lists of
    # <num_timesteps * (1 + num_corrector_steps)> many ChemGraphs.

    # <num_timesteps * (1 + num_corrector_steps)> many lists of <batch_size> many ChemGraphs
    data_lists_per_timesteps = [x.to_data_list() for x in list_of_time_steps]

    # <batch_size> many lists of <num_timesteps * (1 + num_corrector_steps)> many ChemGraphs.
    data_lists_per_sample = [
        [data_lists_per_timesteps[ix_t][ix_traj] for ix_t in range(len(data_lists_per_timesteps))]
        for ix_traj in range(len(data_lists_per_timesteps[0]))
    ]
    return data_lists_per_sample


def dump_trajectories(
    output_path: Path,
    all_trajs_list: list[list[ChemGraph]],
) -> None:
    try:
        # We gather all trajectories in a single zip file as .extxyz files.
        # This way we can view them easily after downloading.
        with ZipFile(output_path / "generated_trajectories.zip", "w") as zip_obj:
            for ix, traj in enumerate(all_trajs_list):
                strucs = structures_from_trajectory(traj)
                ase_atoms = [AseAtomsAdaptor.get_atoms(crystal) for crystal in strucs]
                str_io = io.StringIO()
                ase.io.write(str_io, ase_atoms, format="extxyz")
                str_io.flush()
                zip_obj.writestr(f"gen_{ix}.extxyz", str_io.getvalue())
    except IOError as e:
        print(f"Got error {e} writing the trajectory to disk.")
    except ValueError as e:
        print(f"Got error ValueError '{e}' writing the trajectory to disk.")


def structure_from_model_output(
    frac_coords, atom_types, lengths, angles, num_atoms
) -> list[Structure]:
    structures = [
        make_structure(
            lengths=d["lengths"],
            angles=d["angles"],
            atom_types=d["atom_types"],
            frac_coords=d["frac_coords"],
        )
        for d in get_crystals_list(
            frac_coords.cpu(),
            atom_types.cpu(),
            lengths.cpu(),
            angles.cpu(),
            num_atoms.cpu(),
        )
    ]
    return structures


def structures_from_trajectory(traj: list[ChemGraph]) -> list[Structure]:
    all_strucs = []
    for batch in traj:
        cell = batch.cell
        lengths, angles = lattice_matrix_to_params_torch(cell)
        all_strucs.extend(
            structure_from_model_output(
                frac_coords=batch.pos,
                atom_types=batch.atomic_numbers,
                lengths=lengths,
                angles=angles,
                num_atoms=batch.num_atoms,
            )
        )

    return all_strucs


@dataclass
class CrystalGenerator:
    checkpoint_info: MatterGenCheckpointInfo

    # These may be set at runtime
    batch_size: int | None = None
    num_batches: int | None = None
    target_compositions_dict: list[dict[str, float]] | None = None
    num_atoms_distribution: str = "ALEX_MP_20"

    # Conditional generation
    diffusion_guidance_factor: float = 0.0
    properties_to_condition_on: TargetProperty | None = None

    # Additional overrides, only has an effect when using a diffusion-codebase model
    sampling_config_overrides: list[str] | None = None

    # These only have an effect when using a legacy model
    num_samples_per_batch: int = 1
    niggli_reduction: bool = False

    # Config path, if None will default to DEFAULT_SAMPLING_CONFIG_PATH
    sampling_config_path: Path | None = None
    sampling_config_name: str = "default"

    record_trajectories: bool = True  # store all intermediate samples by default

    # These attributes are set when prepare() method is called.
    _model: DiffusionLightningModule | None = None
    _cfg: DictConfig | None = None

    device: torch.device | None = None

    def __post_init__(self) -> None:
        assert self.num_atoms_distribution in NUM_ATOMS_DISTRIBUTIONS, (
            f"num_atoms_distribution must be one of {list(NUM_ATOMS_DISTRIBUTIONS.keys())}, "
            f"but got {self.num_atoms_distribution}. To add your own distribution, "
            "please add it to mattergen.common.data.num_atoms_distribution.NUM_ATOMS_DISTRIBUTIONS."
        )
        if self.target_compositions_dict:
            assert self.cfg.lightning_module.diffusion_module.loss_fn.weights.get(
                "atomic_numbers", 0.0
            ) == 0.0 and "atomic_numbers" not in self.cfg.lightning_module.diffusion_module.corruption.get(
                "discrete_corruptions", {}
            ), "Input model appears to have been trained for crystal generation (i.e., with atom type denoising), not crystal structure prediction. Please use a model trained for crystal structure prediction instead."
            sampling_cfg = self._load_sampling_config(
                sampling_config_name=self.sampling_config_name,
                sampling_config_overrides=self.sampling_config_overrides,
                sampling_config_path=self.sampling_config_path,
            )
            if (
                "atomic_numbers" in sampling_cfg.sampler_partial.predictor_partials
                or "atomic_numbers" in sampling_cfg.sampler_partial.corrector_partials
            ):
                raise ValueError(
                    "Incompatible sampling config for crystal structure prediction: found atomic_numbers in predictor_partials or corrector_partials. Use the 'csp' sampling config instead, e.g., via --sampling-config-name=csp."
                )

    @property
    def model(self) -> DiffusionLightningModule:
        self.prepare()
        assert self._model is not None
        return self._model

    @property
    def cfg(self) -> DictConfig:
        self._cfg = self.checkpoint_info.config
        assert self._cfg is not None
        return self._cfg

    @property
    def num_structures_to_generate(self) -> int:
        """Returns the total number of structures to generate if `batch_size` and `num_batches` are specified at construction time;
        otherwise, raises an AssertionError.
        """
        assert self.batch_size is not None
        assert self.num_batches is not None
        return self.batch_size * self.num_batches

    @property
    def sampling_config(self) -> DictConfig:
        """Returns the sampling config if `batch_size` and `num_batches` are specified at construction time;
        otherwise, raises an AssertionError.
        """
        assert self.batch_size is not None
        assert self.num_batches is not None
        return self.load_sampling_config(
            batch_size=self.batch_size,
            num_batches=self.num_batches,
            target_compositions_dict=self.target_compositions_dict,
        )

    def get_condition_loader(
        self,
        sampling_config: DictConfig,
        target_compositions_dict: list[dict[str, float]] | None = None,
    ) -> ConditionLoader:
        condition_loader_partial = instantiate(sampling_config.condition_loader_partial)
        if not target_compositions_dict:
            return condition_loader_partial(properties=self.properties_to_condition_on)

        return condition_loader_partial(target_compositions_dict=target_compositions_dict)

    def load_sampling_config(
        self,
        batch_size: int,
        num_batches: int,
        num_atoms: int = int,
        target_compositions_dict: list[dict[str, float]] | None = None,
        atom_types: list[dict[str, float]] | None = None,
        xrd = None,
    ) -> DictConfig:
        """
        Create a sampling config from the given param.
        We specify certain sampling hyperparameters via the sampling config that is loaded via hydra.
        """
        if self.sampling_config_overrides is None:
            sampling_config_overrides = []
        else:
            # avoid modifying the original list
            sampling_config_overrides = self.sampling_config_overrides.copy()
        if not target_compositions_dict:
            # Default `condition_loader_partial` is
            # mattergen.common.data.condition_factory.get_number_of_atoms_condition_loader
            sampling_config_overrides += [
                f"+condition_loader_partial.num_atoms_distribution={self.num_atoms_distribution}",
                f"+condition_loader_partial.batch_size={batch_size}",
                f"+condition_loader_partial.num_samples={num_batches * batch_size}",
                f"sampler_partial.guidance_scale={self.diffusion_guidance_factor}",
            ]
        else:
            # `condition_loader_partial` for fixed atom type (crystal structure prediction)
            num_structures_to_generate_per_composition = (
                num_batches * batch_size // len(target_compositions_dict)
            )
            sampling_config_overrides += [
                "condition_loader_partial._target_=mattergen.common.data.condition_factory.get_composition_data_loader",
                f"+condition_loader_partial.num_structures_to_generate_per_composition={num_structures_to_generate_per_composition}",
                f"+condition_loader_partial.batch_size={batch_size}",
            ]
        return self._load_sampling_config(
            sampling_config_overrides=sampling_config_overrides,
            sampling_config_path=self.sampling_config_path,
            sampling_config_name=self.sampling_config_name,
        )

    def _load_sampling_config(
        self,
        sampling_config_path: Path | None = None,
        sampling_config_name: str = "default",
        sampling_config_overrides: list[str] | None = None,
    ) -> DictConfig:
        if sampling_config_path is None:
            sampling_config_path = DEFAULT_SAMPLING_CONFIG_PATH

        if sampling_config_overrides is None:
            sampling_config_overrides = []

        with hydra.initialize_config_dir(os.path.abspath(str(sampling_config_path))):
            sampling_config = hydra.compose(
                config_name=sampling_config_name, overrides=sampling_config_overrides
            )
        return sampling_config

    def prepare(self) -> None:
        """Loads the model from checkpoint and prepares for generation."""
        if self._model is not None:
            return
        model = load_model_diffusion(self.checkpoint_info)
        model = model.to(self.device)
        self._model = model
        self._cfg = self.checkpoint_info.config

    # def generate(
    #     self,
    #     batch_size: int | None = None,
    #     num_batches: int = 1,
    #     target_compositions_dict: list[dict[str, float]] | None = None,
    #     output_dir: str = "results",
    #     num_atoms: int | None = None,
    #     atom_types: list[dict[str, float]] | None = None,
    #     xrd = None,
    #     score_model=None,
    # ) -> list[Structure]:
    #     # Prioritize the runtime provided batch_size, num_batches and target_compositions_dict
    #     batch_size = batch_size or self.batch_size
    #     num_batches = num_batches or self.num_batches
    #     target_compositions_dict = target_compositions_dict or self.target_compositions_dict
    #
    #     assert batch_size is not None
    #     assert num_batches is not None
    #     assert num_atoms is not None
    #
    #     # print config for debugging and reproducibility
    #     # print("\nModel config:")
    #     # print(OmegaConf.to_yaml(self.cfg, resolve=True))
    #
    #     sampling_config = self.load_sampling_config(
    #         batch_size=batch_size,
    #         num_batches=num_batches,
    #         target_compositions_dict=target_compositions_dict,
    #         num_atoms=num_atoms,
    #         atom_types=atom_types,
    #     )
    #
    #     # print("\nSampling config:")
    #     # print(OmegaConf.to_yaml(sampling_config, resolve=True))
    #     condition_loader = self.get_condition_loader(sampling_config, target_compositions_dict)
    #
    #     sampler_partial = instantiate(sampling_config.sampler_partial)
    #     sampler = sampler_partial(pl_module=self.model)
    #
    #     generated_structures = draw_samples_from_sampler(
    #         sampler=sampler,
    #         condition_loader=condition_loader,
    #         cfg=self.cfg,
    #         output_path=Path(output_dir),
    #         properties_to_condition_on=self.properties_to_condition_on,
    #         record_trajectories=self.record_trajectories,
    #         xrd=xrd,
    #         score_model=score_model,
    #         rl_config=rl_config,
    #         optimizer=optimizer,
    #     )
    #
    #     return generated_structures

    def generate_batch(
            self,
            batch_size: int | None = None,
            num_batches: int | None = None,
            output_dir: str = "results",
            target_compositions_dict: list[dict[str, float]] | None = None,
            score_model=None,
            optimizer=None,
            rl_config=None,
            dataloader=None,
            tracker=None
    ) -> list[Structure]:
        """
        Generate a batch of structures using the sampler and optional reinforcement learning setup.

        Args:
            batch_size (int, optional): Number of samples per batch. Defaults to self.batch_size if not provided.
            num_batches (int, optional): Number of batches to generate.
            output_dir (str, optional): Directory to save generated results. Default: "results".
            target_compositions_dict (list[dict[str, float]], optional): Target compositions to condition on.
            score_model: Model used for scoring (e.g., in RL training). Defaults to self._model if not provided.
            optimizer: Optimizer used for RL fine-tuning (must be provided if RL is enabled).
            rl_config: Reinforcement learning configuration dictionary.
            dataloader: DataLoader providing conditional data (if conditioning is used).

        Returns:
            list[Structure]: A list of generated crystal/molecular structures.
        """

        # Prioritize runtime arguments; fallback to defaults if not provided
        batch_size = batch_size or self.batch_size
        score_model = score_model or self._model
        target_compositions_dict = target_compositions_dict or self.target_compositions_dict

        # Ensure essential parameters are available
        assert batch_size is not None
        assert score_model is not None
        assert optimizer is not None

        # (Optional) Print model configuration for debugging and reproducibility
        # print("\nModel config:")
        # print(OmegaConf.to_yaml(self.cfg, resolve=True))

        # Load the sampling configuration (sampler, parameters, etc.)
        sampling_config = self.load_sampling_config(
            batch_size=batch_size,
            num_batches=num_batches,
            target_compositions_dict=target_compositions_dict,
        )

        # Instantiate the sampler with the model
        sampler_partial = instantiate(sampling_config.sampler_partial)
        sampler = sampler_partial(pl_module=self.model)

        # Generate structures using the sampler
        generated_structures = draw_samples_from_sampler(
            sampler=sampler,
            condition_loader=dataloader,
            cfg=self.cfg,
            output_path=Path(output_dir),
            properties_to_condition_on=self.properties_to_condition_on,
            record_trajectories=self.record_trajectories,
            score_model=score_model,
            rl_config=rl_config,
            optimizer=optimizer,
            tracker=tracker
        )

        return generated_structures



