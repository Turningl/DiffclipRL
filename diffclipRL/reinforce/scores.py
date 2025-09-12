# -*- coding: utf-8 -*-
# @Author : liang
# @File : scores.py


import torch
from pathlib import Path

from diffclipRL.clip.module import CLIP as ScoreModel


class ScoreFuncWrapper:
    """
    Lightweight callable wrapper for a pre-trained ScoreModel.

    1. Loads the model weights once during initialization.
    2. Provides a simple __call__(batch) interface to compute
       the average of graph and XRD accuracies.
    """

    def __init__(
        self,
        score_model_path: str,
        latent_dim: int,
        xrd_dim: int,
        device: str,
        batch_size: int,
    ):
        self.device = device

        # Build the model graph
        self.model = ScoreModel(
            latent_dim=latent_dim,
            xrd_dim=xrd_dim,
            device=device,
            batch_size=batch_size,
        )

        # Load checkpoint
        ckpt_path = Path(score_model_path)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Score model weights not found at: {ckpt_path}")

        state_dict = torch.load(ckpt_path, map_location=device, weights_only=False)
        self.model.load_state_dict(state_dict["model_state_dict"], strict=False)
        self.model.eval()

        print(f"INFO: Loading Score model from checkpoint: {ckpt_path}")

    @torch.no_grad()
    def __call__(self, batch):
        """
        Forward pass on a batch and return the average accuracy.

        Args:
            batch: Input batch compatible with ScoreModel.forward.

        Returns:
            float: (graph_acc + xrd_acc) / 2
        """
        # with torch.no_grad():
        loss, graph_acc, xrd_acc = self.model(batch)
        # reward = 1.0 - torch.clamp(loss / 100.0, 0, 1)  # 0 - 1

        return loss, graph_acc, xrd_acc