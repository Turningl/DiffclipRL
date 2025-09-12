# -*- coding: utf-8 -*-
# @Author : liang
# @File : seblock.py

import torch
import torch.nn as nn
import torch.functional as F


class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation Block (SEBlock) - a channel attention mechanism.

    This module adaptively recalibrates channel-wise feature responses by explicitly
    modeling interdependencies between channels. It consists of:
    1. Squeeze: Global average pooling to aggregate spatial information
    2. Excitation: Fully connected layers to learn channel-wise attention weights
    3. Re-weight: Apply the learned weights to the original features
    """

    def __init__(self, channels, reduction=16):
        """
        Initialize the SEBlock.

        Args:
            channels (int): Number of input/output channels (C from the paper)
            reduction (int): Reduction ratio (r) for the bottleneck layer.
                           Default is 16 as suggested in the original paper.
        """
        super(SEBlock, self).__init__()

        # Sequential block containing all operations for the SE mechanism
        self.fc = nn.Sequential(
            # Squeeze operation: Global average pooling across spatial dimensions
            # Input: (B, C, H, W) -> Output: (B, C, 1, 1)
            nn.AdaptiveAvgPool1d(1),

            # Flatten the spatial dimensions (already reduced to 1x1 by pooling)
            # Input: (B, C, 1, 1) -> Output: (B, C)
            nn.Flatten(),

            # First fully connected layer (bottleneck)
            # Reduces dimensionality by reduction factor
            # Input: (B, C) -> Output: (B, C/r)
            nn.Linear(channels, channels // reduction, bias=False),

            # ReLU activation for non-linearity
            nn.ReLU(inplace=True),

            # Second fully connected layer (expansion)
            # Restores original dimensionality
            # Input: (B, C/r) -> Output: (B, C)
            nn.Linear(channels // reduction, channels, bias=False),

            # Sigmoid activation to produce weights between 0 and 1
            # These represent the importance of each channel
            nn.Sigmoid()
        )

    def forward(self, x):
        """
        Forward pass of the SEBlock.

        Args:
            x (torch.Tensor): Input feature map of shape (B, C, H, W)

        Returns:
            torch.Tensor: Output feature map with channel-wise attention applied,
                         same shape as input (B, C, H, W)
        """
        # Get batch size and number of channels
        b, c, _ = x.size()

        # Apply the SE operations to get channel attention weights
        # y shape: (B, C) after passing through fc layers
        y = self.fc(x)

        # Reshape the attention weights to match input dimensions
        # Add spatial dimensions back for broadcasting
        # y shape: (B, C, 1, 1)
        y = y.view(b, c, 1)

        # Apply the learned attention weights to the original features
        # Element-wise multiplication with broadcasting
        # Each channel's features are scaled by its corresponding weight
        return x * y