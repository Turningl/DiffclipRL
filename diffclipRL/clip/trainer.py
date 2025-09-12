# -*- coding: utf-8 -*-
# @Author : liang
# @File : trainer.py


import gc

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm


class CLIPTrainer:
    def __init__(self, model, device, patience, T_0, T_mult, eta_min, lr, weight_decay):
        """
        Initialize the trainer.

        Args:
            model: The model to train.
            optimizer: Optimizer for updating model param.
            device: Device to run training on (e.g., 'cuda' or 'cpu').
        """

        self.model = model.to(device)
        self.device = device
        self.patience = patience
        self.T_0 = T_0
        self.T_mult = T_mult
        self.eta_min = eta_min

        self.optimizer = None
        self.scheduler = None
        self.lr = lr
        self.weight_decay = weight_decay

    def set_optimizer(self, optim):

        # set optimizer
        if optim == "adam":
            self.optimizer = torch.optim.Adam(
                self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay
            )
        elif optim == "sgd":
            self.optimizer = torch.optim.SGD(
                self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay
            )
        elif optim == "adamw":
            self.optimizer = torch.optim.AdamW(
                self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay
            )
        else:
            raise ValueError(f"Invalid optimizer: {self.optimizer}")

        # self.optimizer = optimizer

    def set_scheduler(self, sched):
        """
        Set a learning rate scheduler.

        Args:
            scheduler: Learning rate scheduler (e.g., torch.optim.lr_scheduler).
        """
        # set scheduler

        if sched == "constant":
            self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lambda _: 1.0)

        elif sched == "cosine":
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=self.T_0)

        elif sched == "reduce_on_plateau":
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode="min",
                min_lr=self.lr,
                factor=0.8,
                patience=self.patience
            )

        elif sched == 'cosineWR':
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                self.optimizer,
                T_0=self.T_0,
                T_mult=self.T_mult,
                eta_min=self.eta_min,
            )

        else:
            raise ValueError(f"Invalid scheduler: {self.scheduler}")

        # self.scheduler = scheduler

    def train(self, data_loader):
        """
        Train the model for one epoch.

        Args:
            data_loader: DataLoader providing (batch, xrd) pairs.

        Returns:
            tuple: (average loss, average learning rate)
        """
        self.model.train()
        lr_list, train_losses, graph_acces, xrd_acces = [], [], [], []

        print("Starting training...")
        for step, batch in tqdm(enumerate(data_loader), total=len(data_loader), desc="Training"):

            self.optimizer.zero_grad()
            loss, graph_acc, xrd_acc = self.model(batch)
            loss = loss.mean()
            loss.backward()
            self.optimizer.step()

            train_losses.append(loss.item())
            graph_acces.append(graph_acc.item())
            xrd_acces.append(xrd_acc.item())

            lr_list.append(self.optimizer.param_groups[0]["lr"])

            # if step % 500 == 0 or step == len(data_loader) - 1:
            #     print(f"Train Step {step}, Loss: {np.mean(train_losses):.4f}, GraphAcc: {np.mean(graph_acces):.4f}, XRDAcc: {np.mean(xrd_acces):.4f}")
            #     print()

        # del loss, graph_acc, xrd_acc, batch, xrd
        torch.cuda.empty_cache()
        gc.collect()

        print(
            f"Train Loss: {np.mean(train_losses):.4f}, "
            f"GraphAcc: {np.mean(graph_acces):.4f}, "
            f"XRDAcc: {np.mean(xrd_acces):.4f}")

        # print()

        return np.mean(train_losses), np.mean(lr_list), np.mean(graph_acces), np.mean(xrd_acces)

    def validate(self, data_loader):
        """
        Validate the model.

        Args:
            data_loader: DataLoader for validation data.

        Returns:
            float: Average validation loss.
        """
        self.model.eval()
        val_losses, graph_acces, xrd_acces = [], [], []

        print("Validating...")
        for batch in tqdm(data_loader, total=len(data_loader), desc="Validation"):
            with torch.no_grad():

                loss, graph_acc, xrd_acc = self.model(batch)
                loss = loss.mean()
                val_losses.append(loss.item())
                graph_acces.append(graph_acc.item())
                xrd_acces.append(xrd_acc.item())

        # del loss, graph_acc, xrd_acc, batch, xrd
        torch.cuda.empty_cache()
        gc.collect()

        print(
            f"Val Loss: {np.mean(val_losses):.4f}, "
            f"GraphAcc: {np.mean(graph_acces):.4f}, "
            f"XRDAcc: {np.mean(xrd_acces):.4f}"
        )

        print()

        return np.mean(val_losses), np.mean(graph_acces), np.mean(xrd_acces)

    def test(self, data_loader):
        """
        Test the model.

        Args:
            data_loader: DataLoader for test data.

        Returns:
            float: Average test loss.
        """
        self.model.eval()
        test_losses, graph_acces, xrd_acces = [], [], []

        print("Evaluating...")
        for batch in tqdm(data_loader, total=len(data_loader), desc="Testing"):
            with torch.no_grad():
                loss, graph_acc, xrd_acc = self.model(batch)
                loss = loss.mean()
                test_losses.append(loss.item())
                graph_acces.append(graph_acc.item())
                xrd_acces.append(xrd_acc.item())

        # del loss, graph_acc, xrd_acc, batch, xrd
        torch.cuda.empty_cache()
        gc.collect()

        print(
            f"Test Loss: {np.mean(test_losses):.4f}, "
            f"GraphAcc: {np.mean(graph_acces):.4f}, "
            f"XRDAcc: {np.mean(xrd_acces):.4f}"
        )

        print()

        return np.mean(test_losses), np.mean(graph_acces), np.mean(xrd_acces)

