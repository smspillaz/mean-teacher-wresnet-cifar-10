#!/usr/bin/env python
"""/train.py

Train a standard WResNet-28-2 on CIFAR-10 using Adam,
LeakyReLU with standard procedures for
data augmentation etc.
"""

import argparse
import os
import sys


import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.utils.data import DataLoader

from sklearn.metrics import accuracy_score

from torchvision import datasets
from torchvision.transforms import (
    Compose,
    Normalize,
    RandomAffine,
    RandomHorizontalFlip,
    ToTensor
)

import tqdm

from submodules.wresnet.networks.wide_resnet import Wide_ResNet


def create_dataloaders(Dataset, base_size=32, crop_size=28, batch_size=4):
    train_tfms = Compose([
        RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.7, 1.3)),
        ToTensor(),
        Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.247, 0.243, 0.261))
    ])
    val_tfms = Compose([
        ToTensor(),
        Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.247, 0.243, 0.261))
    ])

    train_dataset = Dataset('data', train=True, transform=train_tfms, download=True)
    val_dataset = Dataset('data', train=False, transform=val_tfms, download=True)

    return (
        # Can't shuffle the training data, since we always want to mask out
        # a certain percentage of it
        DataLoader(train_dataset,
                   batch_size=batch_size,
                   shuffle=False,
                   pin_memory=True),
        DataLoader(val_dataset,
                   batch_size=batch_size,
                   shuffle=False,
                   pin_memory=True)
    )


def compute_accuracy(outputs, labels):
    """Compute accuracy from outputs and labels."""
    outputs_array = outputs.cpu().detach().argmax(dim=1).numpy()
    labels_array = labels.cpu().numpy()

    return accuracy_score(labels_array, outputs_array)



def training_loop(model,
                  train_loader,
                  val_loader,
                  criterion,
                  optimizer,
                  device,
                  epochs):
    """Main training loop."""

    for epoch in tqdm.tqdm(range(0, epochs), desc="Epoch"):
        model.train()

        progress = tqdm.tqdm(train_loader, desc="Train Batch")
        for batch_index, (batch, targets) in enumerate(progress):
            batch = batch.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()

            outputs = model(batch)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            progress.set_postfix({
                'loss': loss.item(),
                'acc': compute_accuracy(outputs, targets)
            })

        progress = tqdm.tqdm(val_loader, desc="Validation Batch")
        for batch_index, (batch, targets) in enumerate(progress):
            batch = batch.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()

            outputs = model(batch)
            loss = criterion(outputs, targets)

            progress.set_postfix({
                'loss': loss.item(),
                'acc': compute_accuracy(outputs, targets)
            })


def main():
    """Entry point."""
    parser = argparse.ArgumentParser("MeanTeacher with CIFAR10.")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--save-to", type=str, default="model.pt")
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--cuda", action='store_true', default=False)
    args = parser.parse_args()

    model = Wide_ResNet(28, 10, args.dropout, 10)
    print(model)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), 0.2)
    device = 'cuda' if args.cuda else 'cpu'

    train_loader, val_loader = create_dataloaders(getattr(datasets, args.dataset), batch_size=args.batch_size)

    train_loader, val_loader = create_dataloaders(CIFAR10, batch_size=args.batch_size)

    training_loop(model,
                  train_loader,
                  val_loader,
                  criterion,
                  optimizer,
                  device,
                  args.epochs)

    if args.save_to:
        torch.save(model.state_dict(), args.save_to)

if __name__ == "__main__":
    main()

