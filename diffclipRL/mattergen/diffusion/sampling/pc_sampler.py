# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from __future__ import annotations

import contextlib
from typing import Generic, Mapping, Tuple, TypeVar

import math
import torch
from tqdm import tqdm
from typing import Any, Dict, Optional, Tuple, List

from diffclipRL.mattergen.diffusion.corruption.multi_corruption import MultiCorruption, apply
from diffclipRL.mattergen.diffusion.data.batched_data import BatchedData
from diffclipRL.mattergen.diffusion.diffusion_module import DiffusionModule
from diffclipRL.mattergen.diffusion.lightning_module import DiffusionLightningModule
from diffclipRL.mattergen.diffusion.sampling.pc_partials import CorrectorPartial, PredictorPartial

Diffusable = TypeVar(
    "Diffusable", bound=BatchedData
)  # Don't use 'T' because it clashes with the 'T' for time

SampleAndMean = Tuple[Diffusable, Diffusable]
SampleAndMeanAndStd = Tuple[Diffusable, Diffusable, Diffusable]
SampleAndMeanAndMaybeRecords = Tuple[Diffusable, Diffusable, list[Diffusable] | None]
SampleAndMeanAndRecords = Tuple[Diffusable, Diffusable, list[Diffusable]]


import math
import numpy as np
import torch
from collections import deque


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


def gaussian_logprob(x, mean, std, reduce_non_batch=True):
    std = torch.clip(std, min=1e-6)
    lp = -((x.detach() - mean) ** 2) / (2 * std ** 2) - torch.log(std) - math.log(math.sqrt(2 * math.pi))
    if reduce_non_batch:
        dims = tuple(range(1, lp.ndim))
        lp = lp.mean(dim=dims)
    return lp


def ppo_surrogate(new_logps, old_logps, advantages, clip_ratio):
    """
    new_logps/old_logps: (T, B)
    advantages:          (B,)
    """
    T = new_logps.shape[0]
    adv = advantages.unsqueeze(0).expand_as(new_logps)  # (T,B)
    ratio = torch.exp(new_logps - old_logps.detach())
    unclipped = -adv * ratio
    clipped = -adv * torch.clamp(ratio, 1.0 - clip_ratio, 1.0 + clip_ratio)
    loss = torch.max(unclipped, clipped).mean()  # 先 (T,B) 平均
    return loss


def _ppo_step_loss(
    new_lp_pos: torch.Tensor,   # node-wise (N,) OR graph-wise (B,)
    new_lp_cell: torch.Tensor,  # graph-wise (B,)
    old_lp_pos: torch.Tensor,   # graph-wise (B,)
    old_lp_cell: torch.Tensor,  # graph-wise (B,)
    advantages: torch.Tensor,   # (B,)
    clip_ratio: float,
    node_batch: torch.Tensor,   # (N,)
    LOGP_CLIP = 20.0
) -> torch.Tensor:
    """
    Aggregate pos (node-wise) to graph-wise using `node_batch`, then compute clipped PPO loss.
    Cell is assumed to already be graph-wise.
    """
    # infer B
    B = int(node_batch.max().item()) + 1

    # combine in log-space for stability
    delta_pos  = (new_lp_pos  - old_lp_pos.detach()).clamp(-LOGP_CLIP, LOGP_CLIP)
    delta_cell = (new_lp_cell - old_lp_cell.detach()).clamp(-LOGP_CLIP, LOGP_CLIP)

    new_delta_pos = _aggregate_nodes_to_graph(delta_pos, node_batch, B, how="mean")  # (B,)

    delta = new_delta_pos + delta_cell
    ratio = delta.exp()

    # adv = torch.nan_to_num(advantages, nan=0.0, posinf=0.0, neginf=0.0)
    unclipped = -advantages * ratio
    clipped   = -advantages * torch.clip(ratio, 1.0 - clip_ratio, 1.0 + clip_ratio)
    return torch.max(unclipped, clipped).mean()


class PredictorCorrector(Generic[Diffusable]):
    """Generates samples using predictor-corrector sampling."""

    def __init__(
        self,
        *,
        diffusion_module: DiffusionModule,
        predictor_partials: dict[str, PredictorPartial] | None = None,
        corrector_partials: dict[str, CorrectorPartial] | None = None,
        device: torch.device,
        n_steps_corrector: int,
        N: int,
        eps_t: float = 1e-3,
        max_t: float | None = None,
    ):
        """
        Args:
            diffusion_module: diffusion module
            predictor_partials: partials for constructing predictors. Keys are the names of the corruptions.
            corrector_partials: partials for constructing correctors. Keys are the names of the corruptions.
            device: device to run on
            n_steps_corrector: number of corrector steps
            N: number of noise levels
            eps_t: diffusion time to stop denoising at
            max_t: diffusion time to start denoising at. If None, defaults to the maximum diffusion time. You may want to start at T-0.01, say, for numerical stability.
        """
        self._diffusion_module = diffusion_module
        self.N = N

        if max_t is None:
            max_t = self._multi_corruption.T
        assert max_t <= self._multi_corruption.T, "Denoising cannot start from beyond T"

        self._max_t = max_t
        assert (
            corrector_partials or predictor_partials
        ), "Must specify at least one predictor or corrector"
        corrector_partials = corrector_partials or {}
        predictor_partials = predictor_partials or {}
        if self._multi_corruption.discrete_corruptions:
            # These all have property 'N' because they are D3PM type
            assert set(c.N for c in self._multi_corruption.discrete_corruptions.values()) == {N}  # type: ignore

        self._predictors = {
            k: v(corruption=self._multi_corruption.corruptions[k], score_fn=None)
            for k, v in predictor_partials.items()
        }

        self._correctors = {
            k: v(
                corruption=self._multi_corruption.corruptions[k],
                n_steps=n_steps_corrector,
                score_fn=None,
            )
            for k, v in corrector_partials.items()
        }
        self._eps_t = eps_t
        self._n_steps_corrector = n_steps_corrector
        self._device = device

    @property
    def diffusion_module(self) -> DiffusionModule:
        return self._diffusion_module

    @property
    def _multi_corruption(self) -> MultiCorruption:
        return self._diffusion_module.corruption

    def _score_fn(self, x: Diffusable, t: torch.Tensor) -> Diffusable:
        return self._diffusion_module.score_fn(x, t)

    @classmethod
    def from_pl_module(cls, pl_module: DiffusionLightningModule, **kwargs) -> PredictorCorrector:
        return cls(diffusion_module=pl_module.diffusion_module, device=pl_module.device, **kwargs)

    @torch.no_grad()
    def sample(
        self, conditioning_data: BatchedData, mask: Mapping[str, torch.Tensor] | None = None
    ):
        """Create one sample for each of a batch of conditions.
        Args:
            conditioning_data: batched conditioning data. Even if you think you don't want conditioning, you still need to pass a batch of conditions
               because the sampler uses these to determine the shapes of things to generate.
            mask: for inpainting. Keys should be a subset of the keys in `data`. 1 indicates data that should be fixed, 0 indicates data that should be replaced with sampled values.
                Shapes of values in `mask` must match the shapes of values in `conditioning_data`.
        Returns:
           (batch, mean_batch). The difference between these is that `mean_batch` has no noise added at the final denoising step.

        """
        return self._sample_maybe_record(conditioning_data, mask=mask, record=False)[:2]

    @torch.no_grad()
    def sample_with_record(
        self, conditioning_data: BatchedData, mask: Mapping[str, torch.Tensor] | None = None
    ):
        """Create one sample for each of a batch of conditions.
        Args:
            conditioning_data: batched conditioning data. Even if you think you don't want conditioning, you still need to pass a batch of conditions
               because the sampler uses these to determine the shapes of things to generate.
            mask: for inpainting. Keys should be a subset of the keys in `data`. 1 indicates data that should be fixed, 0 indicates data that should be replaced with sampled values.
                Shapes of values in `mask` must match the shapes of values in `conditioning_data`.
        Returns:
           (batch, mean_batch). The difference between these is that `mean_batch` has no noise added at the final denoising step.

        """
        return self._sample_maybe_record(conditioning_data, mask=mask, record=True)

    @torch.no_grad()
    def _sample_maybe_record(
        self,
        conditioning_data: BatchedData,
        mask: Mapping[str, torch.Tensor] | None = None,
        record: bool = False,
    ):
        """Create one sample for each of a batch of conditions.
        Args:
            conditioning_data: batched conditioning data. Even if you think you don't want conditioning, you still need to pass a batch of conditions
               because the sampler uses these to determine the shapes of things to generate.
            mask: for inpainting. Keys should be a subset of the keys in `data`. 1 indicates data that should be fixed, 0 indicates data that should be replaced with sampled values.
                Shapes of values in `mask` must match the shapes of values in `conditioning_data`.
        Returns:
           (batch, mean_batch, recorded_samples, recorded_predictions).
           The difference between the former two is that `mean_batch` has no noise added at the final denoising step.
           The latter two are only returned if `record` is True, and contain the samples and predictions from each step of the diffusion process.

        """
        if isinstance(self._diffusion_module, torch.nn.Module):
            self._diffusion_module.eval()
        mask = mask or {}
        conditioning_data = conditioning_data.to(self._device)
        mask = {k: v.to(self._device) for k, v in mask.items()}
        batch = _sample_prior(self._multi_corruption, conditioning_data, mask=mask)
        return self._denoise(batch=batch, mask=mask, record=record)

    @torch.no_grad()
    def _denoise(
        self,
        batch: Diffusable,
        mask: dict[str, torch.Tensor],
        record: bool = True,
        record_traj: bool = True,        # x
        record_logprob: bool = True,     # log-prob（ predictor）

    ):

        """Denoise from a prior sample to a t=eps_t sample."""
        # recorded_samples = [] if record else None

        for k in self._predictors:
            mask.setdefault(k, None)
        for k in self._correctors:
            mask.setdefault(k, None)
        mean_batch = batch.clone()

        # Decreasing timesteps from T to eps_t
        timesteps = torch.linspace(self._max_t, self._eps_t, self.N, device=self._device)
        dt = -torch.tensor((self._max_t - self._eps_t) / (self.N - 1)).to(self._device)

        # [DDPO]
        x_traj = [] if record_traj else None
        logprobs_pos = [] if record_logprob else None
        logprobs_cell = [] if record_logprob else None

        if record_traj:
            x_traj.append(batch.clone())

        for i in tqdm(range(self.N), total=self.N, position=0, desc="Getting trajectories"):
            # Set the timestep
            t = torch.full((batch.get_batch_size(),), timesteps[i], device=self._device)

            # # Corrector updates.
            # if self._correctors:
            #     for _ in range(self._n_steps_corrector):
            #         score = self._score_fn(batch, t)
            #         fns = {k: corrector.step_given_score for k, corrector in self._correctors.items()}
            #
            #         samples_means = apply(
            #             fns=fns,
            #             broadcast={"t": t, "dt": dt},
            #             x=batch,
            #             score=score,
            #             batch_idx=self._multi_corruption._get_batch_indices(batch),
            #         )
            #         if record:
            #             recorded_samples.append(batch.clone().to("cpu"))
            #
            #         batch, mean_batch, _ = _mask_replace(
            #             samples_means=samples_means, batch=batch, mean_batch=mean_batch, mask=mask
            #         )

            # Predictor updates
            score = self._score_fn(batch, t)
            predictor_fns = {k: predictor.update_given_score for k, predictor in self._predictors.items()}

            samples_means = apply(
                fns=predictor_fns,
                x=batch,
                score=score,
                broadcast=dict(t=t, batch=batch, dt=dt),
                batch_idx=self._multi_corruption._get_batch_indices(batch),
            )
            # if record:
            #     recorded_samples.append(batch.clone())

            # sample/mean/std
            batch, mean_batch, std_batch = _mask_replace(
                samples_means=samples_means, batch=batch, mean_batch=mean_batch, mask=mask
            )

            if record_traj:
                x_traj.append(batch.clone())

            if record_logprob:
                if std_batch is None:
                    raise RuntimeError(
                        "Predictor did not return `std`. Cannot compute log-prob. "
                        "Please make sure that `predictor.update_given_score` returns (sample, mean, std)."
                    )

                # pos logprobs
                x_pos, m_pos, std_pos, node_batch, B = flatten_pos_atomwise(batch, mean_batch, std_batch)
                pos_lp = gaussian_logprob(x_pos, m_pos, std_pos, reduce_non_batch=True)
                logprobs_pos.append(pos_lp.detach())

                # cell logprobs
                x_cell, m_cell, std_cell = flatten_cell_atomwise(batch, mean_batch, std_batch)
                cell_lp = gaussian_logprob(x_cell, m_cell, std_cell, reduce_non_batch=True)
                logprobs_cell.append(cell_lp.detach())

        if record_logprob:
            logprobs_pos = torch.stack(logprobs_pos, dim=0)
            logprobs_cell = torch.stack(logprobs_cell, dim=0)

        return batch, mean_batch, x_traj, logprobs_pos, logprobs_cell
        #return batch, mean_batch, recorded_samples

    def recompute_logps(
            self,
            x_traj: list,  # expected [x0, x1, ..., xN]
            mask: dict[str, torch.Tensor] | None,
            logprobs_pos: torch.Tensor,  # (T_pred, B)
            logprobs_cell: torch.Tensor,  # (T_pred, B)
            advantages: torch.Tensor,  # (B,)
            clip_ratio: float,
            use_amp: bool = False,  # set True if you use GradScaler outside
    ) -> torch.Tensor:
        device = self._device

        # ---- mask fallback / normalization ----
        mask = mask or {}
        mask = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in mask.items()}
        for k in self._predictors:
            mask.setdefault(k, None)
        for k in self._correctors:
            mask.setdefault(k, None)

        # ---- time grid ----
        timesteps = torch.linspace(self._max_t, self._eps_t, self.N, device=device)
        dt = -torch.tensor((self._max_t - self._eps_t) / (self.N - 1), device=device)

        if len(x_traj) < 2:
            raise ValueError("x_traj must contain at least two states (ideally includes x0).")

        T_pred = min(self.N, len(x_traj) - 1)
        if logprobs_pos.shape[0] != T_pred:
            raise ValueError(f"logprobs_pos time dim {logprobs_pos.shape[0]} != available steps {T_pred}.")
        if logprobs_cell.shape[0] != T_pred:
            raise ValueError(f"logprobs_cell time dim {logprobs_cell.shape[0]} != available steps {T_pred}.")

        total_loss_val = 0.0  # logging only; no gradients

        # ---- optional AMP context ----
        amp_ctx = torch.cuda.amp.autocast if use_amp else contextlib.nullcontext
        with amp_ctx():
            for i in tqdm(range(T_pred), total=T_pred, desc="PPO recompute", position=0):
                x_t = x_traj[i].to(device)  # x_i (from no_grad sampling; no grads)
                x_next = x_traj[i + 1].to(device)  # x_{i+1} (observation)
                t = torch.full((x_t.get_batch_size(),), timesteps[i], device=device)

                # one predictor forward (requires gradients)
                score = self._score_fn(x_t, t)
                predictor_fns = {k: p.update_given_score for k, p in self._predictors.items()}
                samples_means = apply(
                    fns=predictor_fns,
                    x=x_t,
                    score=score,
                    broadcast=dict(t=t, batch=x_t, dt=dt),
                    batch_idx=self._multi_corruption._get_batch_indices(x_t),
                )

                # merge using current x_t as base to break cross-step graph deps
                base = x_t.clone().detach()
                _, mean_i, std_i = _mask_replace(
                    samples_means=samples_means,
                    batch=x_t.clone(),
                    mean_batch=base,
                    mask=mask,
                )

                # pos logprobs
                x_pos, m_pos, s_pos, node_batch, B = flatten_pos_atomwise(x_next, mean_i, std_i)
                pos_lp = gaussian_logprob(x_pos, m_pos, s_pos, reduce_non_batch=True)  # (N,)

                # cell logprobs
                x_cell, m_cell, std_cell = flatten_cell_atomwise(x_next, mean_i, std_i)
                cell_lp = gaussian_logprob(x_cell, m_cell, std_cell, reduce_non_batch=True)  # (N,)

                # PPO clipped surrogate for this step (returns scalar)
                loss_i = _ppo_step_loss(pos_lp, cell_lp, logprobs_pos[i], logprobs_cell[i], advantages, clip_ratio, node_batch)
                loss_i.backward()

                # accumulate (graph only covers this step)
                total_loss_val += float(loss_i.detach().item())

                # explicit deref
                del x_t, x_next, t, score, samples_means, mean_i, std_i, base
                del x_pos, m_pos, s_pos, node_batch, B, pos_lp
                del x_cell, m_cell, std_cell, cell_lp
                del loss_i

        # return an average loss value (tensor, no grad) for logging
        return torch.tensor(total_loss_val / T_pred, device=device)


def _mask_replace(
    samples_means,
    batch: BatchedData,
    mean_batch: BatchedData,
    mask: dict[str, torch.Tensor | None],
) -> SampleAndMeanAndStd:

    # Apply masks
    samples_means = apply(
        fns={k: _mask_both for k in samples_means},
        broadcast={},
        sample_and_mean=samples_means,
        mask=mask,
        old_x=batch,
    )

    # Put the updated values in `batch` and `mean_batch`
    batch = batch.replace(**{k: v[0] for k, v in samples_means.items()})
    mean_batch = mean_batch.replace(**{k: v[1] for k, v in samples_means.items()})

    std_batch_dict: Dict[str, torch.Tensor] = {
        k: v[2] for k, v in samples_means.items() if isinstance(v, tuple) and len(v) == 3
    }
    std_batch = std_batch_dict if len(std_batch_dict) > 0 else None

    return batch, mean_batch, std_batch


def _mask_both(
    *, sample_and_mean: Tuple[torch.Tensor, torch.Tensor], old_x: torch.Tensor, mask: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    return tuple(_mask(old_x=old_x, new_x=x, mask=mask) for x in sample_and_mean)  # type: ignore


def _mask(*, old_x: torch.Tensor, new_x: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
    """Replace new_x with old_x where mask is 1."""
    if mask is None:
        return new_x
    else:
        return new_x.lerp(old_x, mask)


def _sample_prior(
    multi_corruption: MultiCorruption,
    conditioning_data: BatchedData,
    mask: Mapping[str, torch.Tensor] | None,
) -> BatchedData:

    # for k in multi_corruption.corruptions:
    #     sample = multi_corruption.corruptions[k].prior_sampling(shape=conditioning_data[k].shape,
    #                                                             conditioning_data=conditioning_data,
    #                                                             batch_idx=conditioning_data.get_batch_idx(field_name=k))

    samples = {
        k: multi_corruption.corruptions[k]
        .prior_sampling(
            shape=conditioning_data[k].shape,
            conditioning_data=conditioning_data,
            batch_idx=conditioning_data.get_batch_idx(field_name=k),
        )
        .to(conditioning_data[k].device)
        for k in multi_corruption.corruptions
    }
    mask = mask or {}
    for k, msk in mask.items():
        if k in multi_corruption.corrupted_fields:
            samples[k].lerp_(conditioning_data[k], msk)
    return conditioning_data.replace(**samples)


def flatten_pos_atomwise(
    x: Any, mean: Any, std: Dict[str, torch.Tensor]
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int]:
    """
    Atom-wise flattening for 'pos'.
    Returns:
        x_pos_flat, mean_pos_flat, std_pos_flat: (N, D_pos)
        batch_index: (N,) mapping node -> graph id in [0, B-1]
        B: number of graphs in the batch
    """
    def _get(obj, key):  return obj[key] if isinstance(obj, dict) else getattr(obj, key)
    def _has(obj, key):  return (key in obj) if isinstance(obj, dict) else hasattr(obj, key)

    # node -> graph mapping is required
    batch_index = None
    for name in ("batch", "node_batch", "batch_idx"):
        if _has(x, name):
            t = _get(x, name)
            if torch.is_tensor(t):
                batch_index = t
                break
    if batch_index is None:
        raise ValueError("flatten_pos_atomwise requires x.batch / node_batch / batch_idx.")

    if "pos" not in std or not (_has(x, "pos") and _has(mean, "pos")):
        raise ValueError("flatten_pos_atomwise expects 'pos' in std and in x/mean.")

    xt, mt, st = _get(x, "pos"), _get(mean, "pos"), std["pos"]
    dev, dt = xt.device, xt.dtype
    mt = mt.to(dev, dt); st = st.to(dev, dt)

    x_pos_flat = xt.reshape(xt.shape[0], -1)   # (N, D_pos)
    m_pos_flat = mt.reshape(mt.shape[0], -1)
    s_pos_flat = st.reshape(st.shape[0], -1)

    B = int(batch_index.max().item()) + 1
    return x_pos_flat, m_pos_flat, s_pos_flat, batch_index.to(dev), B



def flatten_cell_atomwise(
    x: Any, mean: Any, std: Dict[str, torch.Tensor]
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Broadcast graph-level 'cell' to nodes and flatten.
    Returns:
        x_cell_flat, mean_cell_flat, std_cell_flat: (N, D_cell)
        batch_index: (N,)
        B: number of graphs
    """
    def _get(obj, key):  return obj[key] if isinstance(obj, dict) else getattr(obj, key)
    def _has(obj, key):  return (key in obj) if isinstance(obj, dict) else hasattr(obj, key)

    # need node->graph mapping
    batch_index = None
    for name in ("batch", "node_batch", "batch_idx"):
        if _has(x, name):
            t = _get(x, name)
            if torch.is_tensor(t):
                batch_index = t
                break
    if batch_index is None:
        raise ValueError("flatten_cell_atomwise requires x.batch / node_batch / batch_idx.")

    if "cell" not in std or not (_has(x, "cell") and _has(mean, "cell")):
        raise ValueError("flatten_cell_atomwise expects 'cell' in std and in x/mean.")

    xt = _get(x, "cell")          # (B, ...)
    mt = _get(mean, "cell")
    st = std["cell"]

    dev, dt = xt.device, xt.dtype
    mt = mt.to(dev, dt)
    st = st.to(dev, dt)

    B = xt.shape[0]
    D = int(torch.tensor(xt.shape[1:]).prod().item())

    # (B, D)
    x_cell_B = xt.reshape(B, D)
    m_cell_B = mt.reshape(B, D)
    s_cell_B = st.reshape(B, D)

    return x_cell_B, m_cell_B, s_cell_B


def _aggregate_nodes_to_graph(
    values: torch.Tensor,          # (N,) or (N, D)
    node_batch: torch.Tensor,      # (N,) graph id in [0, B-1]
    B: int,
    how: str = "mean",             # "mean" | "sum"
) -> torch.Tensor:
    """Aggregate node-wise values to graph-wise using node_batch indices."""
    if values.ndim == 1:
        values2 = values.unsqueeze(1)   # (N,1)
        squeeze = True
    else:
        values2 = values.reshape(values.shape[0], -1)  # (N,D)
        squeeze = False

    device, dtype = values2.device, values2.dtype
    node_batch = node_batch.to(device)

    out = torch.zeros((B, values2.shape[1]), device=device, dtype=dtype)
    out.index_add_(0, node_batch, values2)  # sum per graph

    if how == "mean":
        counts = torch.bincount(node_batch, minlength=B).to(device=device, dtype=dtype).clamp_min(1)
        out = out / counts.unsqueeze(1)

    return out.squeeze(1) if squeeze else out