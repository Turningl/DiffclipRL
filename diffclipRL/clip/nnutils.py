# -*- coding: utf-8 -*-
# @Author : liang
# @File : nnutils.py


import numpy as np
import torch
import torch.nn.functional as F
from torch import nn, optim


def cross_entropy(preds, targets, reduction='none'):
    log_softmax = nn.LogSoftmax(dim=-1)
    loss = (-targets * log_softmax(preds)).sum(1)
    if reduction == "none":
        return loss
    elif reduction == "mean":
        return loss.mean()


def clip_loss(logits: torch.Tensor) -> torch.Tensor:
    """
    Calculate a custom cross-entropy loss.

    Args:
    - logits (torch.Tensor): The input tensor containing unnormalized logits.

    Returns:
    - torch.Tensor: The computed custom cross-entropy loss.

    Example:
    >>> logits = torch.rand((batch_size, num_classes))
    >>> loss = CLIP_loss(logits)
    """

    n = logits.shape[1]

    # Create labels tensor
    labels = torch.arange(n, device=logits.device)

    # Calculate cross entropy losses along axis 0 and 1
    loss_i = F.cross_entropy(logits.transpose(0, 1), labels, reduction="mean")
    loss_t = F.cross_entropy(logits, labels, reduction="mean")

    # Calculate the final loss
    loss = (loss_i + loss_t) / 2

    return loss


def metrics(similarity: torch.Tensor):
    y = torch.arange(len(similarity)).to(similarity.device)

    g2x_graph_match_idx = similarity.argmax(dim=1)
    x2g_graph_match_idx = similarity.argmax(dim=0)

    graph_acc = (g2x_graph_match_idx == y).float().mean()
    xrd_acc = (x2g_graph_match_idx == y).float().mean()

    return graph_acc, xrd_acc


class NTXentLoss(torch.nn.Module):

    def __init__(self, device, batch_size, temperature=1.0, use_cosine_similarity=True):
        super(NTXentLoss, self).__init__()
        self.batch_size = batch_size
        self.temperature = temperature
        self.device = device
        self.softmax = torch.nn.Softmax(dim=-1)
        self.mask_samples_from_same_repr = self._get_correlated_mask().type(torch.bool)
        self.similarity_function = self._get_similarity_function(use_cosine_similarity)
        self.criterion = torch.nn.CrossEntropyLoss(reduction="sum")

    def _get_similarity_function(self, use_cosine_similarity):
        if use_cosine_similarity:
            self._cosine_similarity = torch.nn.CosineSimilarity(dim=-1)
            return self._cosine_simililarity
        else:
            return self._dot_simililarity

    def _get_correlated_mask(self):
        diag = np.eye(2 * self.batch_size)
        l1 = np.eye((2 * self.batch_size), 2 * self.batch_size, k=-self.batch_size)
        l2 = np.eye((2 * self.batch_size), 2 * self.batch_size, k=self.batch_size)
        mask = torch.from_numpy((diag + l1 + l2))
        mask = (1 - mask).type(torch.bool)
        return mask.to(self.device)

    @staticmethod
    def _dot_simililarity(x, y):
        v = torch.tensordot(x.unsqueeze(1), y.T.unsqueeze(0), dims=2)
        # x shape: (N, 1, C)
        # y shape: (1, C, 2N)
        # v shape: (N, 2N)
        return v

    def _cosine_simililarity(self, x, y):
        # x shape: (N, 1, C)
        # y shape: (1, 2N, C)
        # v shape: (N, 2N)
        v = self._cosine_similarity(x.unsqueeze(1), y.unsqueeze(0))
        return v

    def forward(self, zis, zjs):

        zis = F.normalize(zis, dim=1)
        zjs = F.normalize(zjs, dim=1)

        representations = torch.cat([zjs, zis], dim=0)

        similarity_matrix = self.similarity_function(representations, representations)

        # filter out the scores from the positive samples
        l_pos = torch.diag(similarity_matrix, self.batch_size)
        r_pos = torch.diag(similarity_matrix, -self.batch_size)
        positives = torch.cat([l_pos, r_pos]).view(2 * self.batch_size, 1)

        negatives = similarity_matrix[self.mask_samples_from_same_repr].view(2 * self.batch_size, -1)

        logits = torch.cat((positives, negatives), dim=1)
        logits /= self.temperature

        labels = torch.zeros(2 * self.batch_size).to(self.device).long()
        loss = self.criterion(logits, labels)

        graph_acc, xrd_acc = metrics(similarity_matrix)

        return loss / (2 * self.batch_size), graph_acc, xrd_acc


def pos_loss(x, y, alpha=1):
    x = F.normalize(x, p=2, dim=-1)
    y = F.normalize(y, p=2, dim=-1)

    loss = (1 - (x * y).sum(dim=-1)).pow_(alpha)

    loss = loss.mean()
    return loss