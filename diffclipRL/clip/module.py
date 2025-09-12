# -*- coding: utf-8 -*-
# @Author : liang
# @File : module.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init

from diffclipRL.mattergen.common.gemnet.gemnet import GemNetT
from diffclipRL.clip.xrd_model import XRDDenseRegressor
from diffclipRL.clip.nnutils import NTXentLoss, cross_entropy, metrics
from diffclipRL.clip.projection import Graph_ProjectionHead, XRD_ProjectionHead
from diffclipRL.mattergen.common.gemnet.layers.embedding_block import AtomEmbedding

class CLIP(nn.Module):
    def __init__(self,
                 latent_dim: int,
                 num_targets=0,
                 xrd_dim=512,
                 expand_dim=1024,
                 num_blocks=4,
                 device='cuda',
                 batch_size=128,
                 temperature=1.0,
                 use_cosine_similarity=True,
                 use_contrastive_loss=False,
                 otf_graph=True,
                 task='clip'):
        """
        Args:
            # num_targets (int): Number of target properties for the graph model.
            latent_dim (int): Dimension of the latent embeddings from graph and XRD models.
            xrd_dim (int): Input dimension for the XRD model.
            expand_dim (int): Dimension of the projected embeddings after the projection heads.
            num_blocks (int): Number of blocks in the XRD dense regressor model.
            device (str): Device to run the model on ('cuda' or 'cpu').
            batch_size (int): Batch size used for training.
            temperature (float): Temperature parameter for contrastive loss scaling.
            use_cosine_similarity (bool): Whether to use cosine similarity in the contrastive loss.
            use_contrastive_loss (bool): Whether to use the NTXent contrastive loss.
        """
        super(CLIP, self).__init__()

        self.num_targets = num_targets
        self.latent_dim = latent_dim
        self.xrd_dim = xrd_dim
        self.expand_dim = expand_dim

        self.num_blocks = num_blocks
        self.temperature = temperature
        self.task = task
        self.device = device
        self.otf_graph = otf_graph

        # Graph encoder (e.g., GemNet-T)
        self.graph_model = GemNetT(num_targets=self.num_targets,
                                   latent_dim=self.latent_dim,
                                   task=self.task,
                                   atom_embedding=AtomEmbedding(emb_size=latent_dim).to(self.device),
                                   otf_graph=self.otf_graph
                                   ).to(self.device)

        # XRD encoder (dense regressor)
        self.xrd_model = (XRDDenseRegressor(
            xrd_dim=self.xrd_dim, expand_dim=self.latent_dim, num_blocks=self.num_blocks)
                          .to(self.device))

        # Projection heads for graph and XRD embeddings
        self.graph_proj_head = (Graph_ProjectionHead(
            embedding_dim=self.latent_dim,
            projection_dim=self.expand_dim).to(self.device))

        self.xrd_proj_head = (XRD_ProjectionHead(
            embedding_dim=self.latent_dim,
            projection_dim=self.expand_dim).to(self.device))

        # Contrastive loss function (NT-Xent)
        self.use_contrastive_loss = use_contrastive_loss
        self.ntxentloss = NTXentLoss(device=device,
                                     batch_size=batch_size,
                                     temperature=temperature,
                                     use_cosine_similarity=use_cosine_similarity)

    def forward(self, batch):
        """
        Args:
            batch: Input batch for the graph model (typically a PyTorch Geometric DataBatch).
            xrd: Input tensor for the XRD model of shape (batch_size, xrd_dim).

        Returns:
            loss (torch.Tensor): The computed loss.
            graph_acc (float): Accuracy metric for graph embeddings.
            xrd_acc (float): Accuracy metric for XRD embeddings.
        """

        (frac_coords, lattice, atom_types, num_atoms, xrd, batch_index) = (
            batch["pos"].to(self.device),
            batch["cell"].to(self.device),
            batch["atomic_numbers"].to(self.device),
            batch["num_atoms"].to(self.device),
            batch['xrd'].to(self.device),
            batch.get_batch_idx("pos").to(self.device),
        )

        # Encode graph and XRD inputs into latent embeddings
        graph_embeddings = self.graph_model(
            z=None,
            frac_coords=frac_coords,
            atom_types=atom_types,
            num_atoms=num_atoms,
            batch=batch_index,
            lengths=None,
            angles=None,
            lattice=lattice,
            # we construct the graph on the fly, hence pass None for these:
            edge_index=None,
            to_jimages=None,
            num_bonds=None,
        )

        xrd_embeddings = self.xrd_model(xrd)

        # Project embeddings into a shared space
        graph_proj = self.graph_proj_head(graph_embeddings)
        xrd_proj = self.xrd_proj_head(xrd_embeddings)

        if self.use_contrastive_loss:
            # Use NT-Xent contrastive loss
            loss, graph_acc, xrd_acc = self.ntxentloss(graph_proj, xrd_proj)

        else:
            # Compute similarity logits
            logits = graph_proj @ xrd_proj.T / self.temperature  # (graph_batch, xrd_batch)

            # Compute self-similarity for graph and XRD
            graph_sim = graph_proj @ graph_proj.T  # (graph_batch, graph_batch)
            xrd_sim = xrd_proj @ xrd_proj.T  # (xrd_batch, xrd_batch)

            # import numpy as np
            # from matplotlib.colors import LinearSegmentedColormap
            # import matplotlib.pyplot as plt
            # from matplotlib.ticker import FormatStrFormatter
            # colors = ["#FFFF00", "#00FF00"]  # 黄色到绿色
            # cmap = LinearSegmentedColormap.from_list("custom", colors)
            #
            # log = logits.detach().cpu().numpy()
            # plt.figure(figsize=(100, 80))
            # plt.imshow(log[:400, :400])
            # # sns.heatmap(similarity[:500, :500])
            # plt.title("CLIP model", fontsize=150)
            #
            # cbar = plt.colorbar(format=FormatStrFormatter("%d"))  # 设置小数点后两位
            # cbar.set_label("Similarity", fontsize=150)  # 设置 colorbar 标签
            # cbar.ax.tick_params(labelsize=150)  # 设置 colorbar 刻度字体大小
            #
            # plt.xlabel("Graph Embeddings", fontsize=150)
            # plt.ylabel("XRD Embeddings", fontsize=150)
            #
            # plt.savefig("similarity_matrix.svg", format="svg", bbox_inches="tight")
            #
            # plt.show()

            # Compute soft targets from self-similarities
            targets = F.softmax(
                ((graph_sim + xrd_sim) / 2 / self.temperature).detach(),
                dim=-1)

            # Compute cross-entropy loss in both directions
            loss_graph2xrd = cross_entropy(logits, targets, reduction='none')  # graph→xrd
            loss_xrd2graph = cross_entropy(logits.T, targets.T, reduction='none')  # xrd→graph

            # Average the losses
            loss = (loss_graph2xrd + loss_xrd2graph) / 2.0
            # loss = loss.mean()


            # Compute accuracy metrics
            graph_acc, xrd_acc = metrics(logits)

        return loss, graph_acc, xrd_acc
