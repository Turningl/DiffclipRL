# -*- coding: utf-8 -*-
# @Author : liang
# @File : projection.py


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init


class Graph_ProjectionHead(nn.Module):
    def __init__(self, embedding_dim: int, projection_dim: int, dropout: float = 0.0):
        """
        Args:
            embedding_dim (int): The dimension of the input embedding.
            projection_dim (int): The desired dimension of the projected output.
            dropout (float): Dropout probability for regularization.
        """

        super().__init__()
        self.projection = nn.Linear(embedding_dim, projection_dim)  # Initial linear projection layer
        self.act = nn.GELU()  # Non-linear activation function (SiLU)
        self.fc = nn.Linear(projection_dim, projection_dim)  # Second linear layer for further processing
        self.dropout = nn.Dropout(dropout)  # Dropout layer for regularization
        self.layer_norm = nn.LayerNorm(projection_dim)  # Layer normalization

        self._init_weights()  # Initialize weights using Xavier uniform and set biases to zero

    def _init_weights(self):
        """
        Initialize the weights of the linear layers using Xavier uniform initialization and set biases to zero.
        """
        init.xavier_uniform_(self.projection.weight)  # Xavier uniform initialization for projection layer weights
        init.zeros_(self.projection.bias)  # Set projection layer bias to zero
        init.xavier_uniform_(self.fc.weight)  # Xavier uniform initialization for fc layer weights
        init.zeros_(self.fc.bias)  # Set fc layer bias to zero

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, embedding_dim).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, projection_dim).
        """

        projected = self.projection(x)  # Apply initial linear projection

        x = self.dropout(self.fc(self.act(projected)))  # Apply non-linear activation, second linear layer, and dropout
        x = self.layer_norm( x + projected)  # Add residual connection after layer normalization

        return x

class XRD_ProjectionHead(nn.Module):
    def __init__(self, embedding_dim: int, projection_dim: int, dropout: float = 0.1):
        """
        Args:
            embedding_dim (int): The dimension of the input embedding.
            projection_dim (int): The desired dimension of the projected output.
            dropout (float): Dropout probability for regularization.
        """
        super().__init__()
        self.projection_layer = nn.Linear(embedding_dim, projection_dim)  # Initial linear projection layer
        self.mlp = nn.Sequential(
            nn.GELU(),  # First non-linear activation function (GELU)
            nn.Linear(projection_dim, projection_dim),  # Second linear layer
            nn.GELU(),  # Second non-linear activation function (GELU)
            nn.Dropout(dropout),  # Dropout layer for regularization
            nn.LayerNorm(projection_dim)  # Layer normalization
        )
        self._init_weights()  # Initialize weights using Xavier uniform and set biases to zero

    def _init_weights(self):
        """
        Initialize the weights of the linear layers using Xavier uniform initialization and set biases to zero.
        """
        init.xavier_uniform_(self.projection_layer.weight)  # Xavier uniform initialization for projection layer weights
        init.zeros_(self.projection_layer.bias)  # Set projection layer bias to zero
        for layer in self.mlp:
            if isinstance(layer, nn.Linear):
                init.xavier_uniform_(layer.weight)  # Xavier uniform initialization for MLP linear layers
                init.zeros_(layer.bias)  # Set MLP linear layer biases to zero

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, embedding_dim).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, projection_dim).
        """
        x = self.projection_layer(x)  # Apply initial linear projection
        return self.mlp(x)  # Pass through the MLP block and return the result