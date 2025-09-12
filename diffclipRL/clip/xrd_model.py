# -*- coding: utf-8 -*-
# @Author : liang
# @File : xrd_model.py


import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from diffclipRL.clip.seblock import SEBlock

# class XRDDenseRegressor(nn.Module):
#     def __init__(self, xrd_dim=512, expand_dim=1024, num_blocks=4):
#         super(XRDDenseRegressor, self).__init__()
#
#         self.xrd_dim = xrd_dim
#         self.num_blocks = num_blocks
#         self.expand_dim = expand_dim
#
#         self.high_dim_proj = nn.Sequential(
#             nn.Linear(xrd_dim, expand_dim),
#             nn.BatchNorm1d(num_features=expand_dim),
#             nn.ReLU(),
#             nn.Linear(expand_dim, expand_dim),
#             nn.BatchNorm1d(num_features=expand_dim),
#             nn.ReLU()
#         )
#
#         self.first_conv = nn.Sequential(
#             nn.Conv1d(in_channels=1, out_channels=8, kernel_size=3, padding=1, bias=False),
#             nn.BatchNorm1d(num_features=8),
#             nn.ReLU()
#         )
#
#         self.dense_blocks = nn.ModuleList(
#             [nn.Sequential(
#                 nn.Conv1d(in_channels=8 * i, out_channels=8, kernel_size=3, padding=1, bias=False),
#                 nn.BatchNorm1d(num_features=8),
#                 nn.ReLU()
#             )
#                 for i in range(1, self.num_blocks + 1)]
#         )
#
#         self.final_conv = (nn.Conv1d(in_channels=8, out_channels=1, kernel_size=3, padding=1, bias=False))
#
#     def forward(self, x):
#         batch_size = x.shape[0]
#
#         # project to XRD dimensionality
#         x = self.high_dim_proj(x)
#         x = x.unsqueeze(1)
#         assert x.shape == (batch_size, 1, self.expand_dim)
#
#         # do first convolution
#         x = self.first_conv(x)
#         assert x.shape == (batch_size, 8, self.expand_dim)
#
#         # densely connected conv blocks
#         x_history = [x]
#         for i, the_block in enumerate(self.dense_blocks):
#             assert len(x_history) == i + 1  # make sure we are updating the history list
#             curr_input = torch.cat(x_history, dim=1)
#             assert curr_input.shape == (batch_size, 8 * (i + 1), self.expand_dim)
#             x = the_block(curr_input)
#             x_history.append(x)  # add new result to running list
#             assert x.shape == (batch_size, 8, self.expand_dim)
#         assert len(x_history) == len(self.dense_blocks) + 1  # make sure we hit all the blocks
#
#         # final conv to get one channel
#         x = self.final_conv(x)
#         # squeeze to get rid of dummy dim
#         x = x.squeeze(1)
#         assert x.shape == (batch_size, self.expand_dim)
#
#         # normalize
#         max_by_xrd = torch.max(x, 1)[0].reshape(batch_size, 1).expand(-1, self.expand_dim)
#         min_by_xrd = torch.min(x, 1)[0].reshape(batch_size, 1).expand(-1, self.expand_dim)
#         assert max_by_xrd.shape == x.shape
#         assert min_by_xrd.shape == x.shape
#         x = (x - min_by_xrd) / (max_by_xrd - min_by_xrd)
#         assert torch.isclose(torch.min(x), torch.tensor(0.0))
#         assert torch.isclose(torch.max(x), torch.tensor(1.0))
#
#         return x


class XRDDenseRegressor(nn.Module):
    def __init__(self, xrd_dim=512, expand_dim=1024, num_blocks=4, dropout=0.2):
        super().__init__()
        self.expand_dim = expand_dim

        self.high_dim_proj = nn.Sequential(
            nn.Linear(xrd_dim, expand_dim),
            nn.GELU(),
            nn.Linear(expand_dim, expand_dim),
            nn.GELU()
        )

        self.first_conv = nn.Conv1d(1, 16, kernel_size=3, padding=1, bias=False)

        self.res_blocks = nn.ModuleList()
        for _ in range(num_blocks):
            block = nn.Sequential(
                nn.Conv1d(16, 16, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm1d(16),
                nn.GELU(),
                nn.Dropout1d(dropout),
                SEBlock(16)
            )
            self.res_blocks.append(block)

        self.final_conv = nn.Conv1d(16, 1, kernel_size=3, padding=1, bias=False)

    def forward(self, x):
        x = self.high_dim_proj(x).unsqueeze(1)  # (B, 1, L)
        x = self.first_conv(x)                  # (B, 16, L)

        for block in self.res_blocks:
            residual = x
            x = block(x)
            x = x + residual  # Residual connection

        x = self.final_conv(x).squeeze(1)  # (B, L)
        x = F.normalize(x, p=2, dim=1)
        return x


class XRDConvRegressor(nn.Module):
    def __init__(self, latent_dim=256, xrd_dim=512):
        # raise ValueError('conv regressor is deprecated')
        super(XRDConvRegressor, self).__init__()

        self.high_dim_proj = nn.Linear(latent_dim, xrd_dim)

        layer_list = [
            nn.Sequential(
                nn.Conv1d(in_channels=1, out_channels=64, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm1d(num_features=64),
                nn.ReLU()
            )
        ]

        curr_num_channels = 64
        for _ in range(3):
            for _ in range(2):
                layer_list.append(nn.Sequential(
                    nn.Conv1d(in_channels=curr_num_channels, out_channels=curr_num_channels // 2, kernel_size=3,
                              padding=1, bias=False),
                    nn.BatchNorm1d(num_features=curr_num_channels // 2),
                    nn.ReLU()
                ))
                curr_num_channels = curr_num_channels // 2
        # dont end on ReLU
        layer_list.append(nn.Conv1d(in_channels=1, out_channels=1, kernel_size=3, padding=1, bias=False))
        self.layers = nn.Sequential(*layer_list)

    def forward(self, x):
        result = self.high_dim_proj(x)
        result = result.unsqueeze(1)
        result = self.layers(result)
        result = result.squeeze(1)
        return result