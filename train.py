#!/usr/bin/env python
"""/train.py

Train a standard WResNet-28-2 on CIFAR-10 using Adam,
LeakyReLU with standard procedures for
data augmentation etc.
"""

import abc
import argparse
import copy
import csv
import os
import sys

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from collections import defaultdict
from copy import deepcopy

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
from model import ResNet32x32, ShakeShakeBlock


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

    # Don't compute over unlabelled data
    return accuracy_score([
        l for l in labels_array if l != -1
    ], [
        o for l, o in zip(labels_array, outputs_array) if l != -1
    ])


def explore_module_children(module):
    for child in module.children():
        if not list(child.children()):
            yield child

        if isinstance(child, nn.Module):
            yield from explore_module_children(child)


def iterate_all_parameters(model):
    for module in explore_module_children(model):
        for p in module.parameters():
            if p.requires_grad:
                yield p



class ConsistencyCostRegularizer(metaclass=abc.ABCMeta):
    """An interface for consistency cost regularization."""

    @abc.abstractmethod
    def compute_loss(self, inputs, output):
        """Compute the consistency loss."""
        raise NotImplementedError

    @abc.abstractmethod
    def update(self, student, outputs):
        """Update using the student and the outputs of the student."""
        raise NotImplementedError


def softmax_mse_loss(input_logits, target_logits, criterion):
    """Takes softmax on both sides and returns MSE loss
    Note:
    - Returns the mean over all examples.
    - Sends gradients to inputs but not the targets.
    """
    assert input_logits.size() == target_logits.size()
    input_softmax = F.softmax(input_logits, dim=1)
    target_softmax = F.softmax(target_logits, dim=1)
    num_classes = input_logits.size()[1]
    return criterion(input_softmax, target_softmax) / num_classes


class MeanTeacherConsistencyCostRegularizer(ConsistencyCostRegularizer):
    """A class that takes the Mean Teacher approach to consistency cost."""

    def __init__(self, teacher, beta):
        """Initialize with teacher model."""
        super().__init__()
        self.teacher = teacher
        self.mse = nn.MSELoss(reduction='sum')
        self.beta = beta

    def compute_loss(self, inputs, outputs):
        """Compute the loss by comparing the outputs with outputs from the teacher."""
        teacher_outputs = self.teacher(inputs)
        return softmax_mse_loss(outputs, teacher_outputs.detach(), self.mse)

    def update(self, student, outputs, step):
        """Update the teacher using weight averaging.

        This will apply weight averaging to all parameters, including Batch
        Normalization layers.
        """
        beta = min(1 - (1 / (step + 1)), self.beta)
        for param, new_param in zip(self.teacher.parameters(),
                                    student.parameters()):
            param.data.mul_(self.beta).add_(1 - beta, new_param.data)


class NullRegularizer(ConsistencyCostRegularizer):
    """A class that doesn't do any regularization."""

    def __init__(self):
        super().__init__()

    def compute_loss(self, inputs, outputs):
        return torch.tensor([0.0])


    def update(self, student, outputs, step):
        pass


def training_loop(model,
                  train_loader,
                  val_loader,
                  criterion,
                  optimizer,
                  scheduler,
                  regularizer,
                  device,
                  epochs,
                  consistency_cost_curve,
                  labelling_func,
                  noise,
                  test_only,
                  write_func):
    """Main training loop."""
    step = 0

    for epoch in tqdm.tqdm(range(0, 1 if test_only else epochs), desc="Epoch"):
        model.train()

        if not test_only:
            losses = []
            consistency_losses = []
            accuracies = []
            progress = tqdm.tqdm(train_loader, desc="Train Batch")
            for batch_index, (batch, targets) in enumerate(progress):
                batch = batch.to(device)
                targets = labelling_func(targets.to(device))

                optimizer.zero_grad()

                outputs = model(batch + torch.rand(batch.size()).to(device) * noise)
                classification_loss = criterion(outputs, targets)
                consistency_loss = regularizer.compute_loss(batch, outputs).to(device)
                consistency_discount = consistency_cost_curve(epoch)
                loss = classification_loss + consistency_loss * consistency_discount
                loss.backward()

                optimizer.step()
                scheduler.step()
                regularizer.update(model, outputs, step)
                step += 1

                accuracy = compute_accuracy(outputs, targets)
                progress.set_postfix({
                    'loss': loss.item(),
                    'const': consistency_loss.item() * consistency_discount,
                    'acc': accuracy
                })
                losses.append(loss.item())
                consistency_losses.append(consistency_loss.item() * consistency_discount)
                accuracies.append(accuracy)
                write_func('train', epoch, loss.item(), consistency_loss.item() * consistency_discount, accuracy)

            tqdm.tqdm.write("Training (Epoch {}): Loss: {}, Consistency: {}, Acc: {}".format(epoch,
                                                                                             np.mean(losses),
                                                                                             np.mean(consistency_losses),
                                                                                             np.mean(accuracies)))

        progress = tqdm.tqdm(val_loader, desc="Validation Batch")

        val_model = getattr(regularizer, 'teacher', model)
        val_model.eval()

        losses = []
        accuracies = []
        for batch_index, (batch, targets) in enumerate(progress):
            batch = batch.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()

            outputs = val_model(batch)
            loss = criterion(outputs, targets)

            accuracy = compute_accuracy(outputs, targets)
            progress.set_postfix({
                'loss': loss.item(),
                'acc': accuracy
            })
            losses.append(loss.item())
            accuracies.append(accuracy.item())
            write_func('valid', epoch, loss.item(), 0.0, accuracy)

        tqdm.tqdm.write("Validation (Epoch {}): Loss: {}, Acc: {}".format(epoch, np.mean(losses), np.mean(accuracies)))

        # Done evaluating, set it back to train
        val_model.train()


IN_CHANNELS = defaultdict(lambda: 3, **{
    "MNIST": 1,
    "CIFAR10": 3
})


def result_writer(base_model_name):
    """Write result log to something can be visualized later."""
    log_filename = "{}.log".format(base_model_name)

    csv_file = open(log_filename, "w")
    writer = csv.writer(csv_file)

    def write(*args):
        writer.writerow(args)
        csv_file.flush()

    return write


def main():
    """Entry point."""
    parser = argparse.ArgumentParser("MeanTeacher with CIFAR10.")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--learning-rate", type=float, default=0.02)
    parser.add_argument("--save-to", type=str, default="model.pt")
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--cuda", action='store_true', default=False)
    parser.add_argument("--supervised-ratio", type=float, default=1.0)
    parser.add_argument("--regularizer", type=str, default="mt")
    parser.add_argument("--dataset", type=str, default="CIFAR10")
    parser.add_argument("--ema-decay", type=float, default=0.999)
    parser.add_argument("--load", type=str, default=None)
    parser.add_argument("--test-only", action='store_true')
    parser.add_argument("--consistency-weight", type=float, default=100)
    args = parser.parse_args()

    device = 'cuda' if args.cuda else 'cpu'
    # model = Wide_ResNet(28, 10, args.dropout, IN_CHANNELS[args.dataset], 10).to(device)
    model = ResNet32x32(ShakeShakeBlock, layers=[4, 4, 4], channels=96, in_channels=3, downsample='shift_conv', num_classes=10).to(device)
    print(model)

    if args.load:
        model.load_state_dict(torch.load(args.load))

    train_loader, val_loader = create_dataloaders(getattr(datasets, args.dataset), batch_size=args.batch_size)

    criterion = nn.CrossEntropyLoss(ignore_index=-1)
    optimizer = optim.SGD(model.parameters(),
                          args.learning_rate,
                          weight_decay=2e-4,
                          nesterov=True,
                          momentum=0.9)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                     len(train_loader) * (args.epochs + 50),
                                                     eta_min=0,
                                                     last_epoch=-1)
    regularizer = MeanTeacherConsistencyCostRegularizer(model, args.ema_decay) if args.regularizer == "mt" else NullRegularizer()

    training_loop(model,
                  train_loader,
                  val_loader,
                  criterion,
                  optimizer,
                  scheduler,
                  regularizer,
                  device,
                  args.epochs,
                  lambda epoch: (1.0 - np.exp(-25.0 * np.square((epoch + 1) / args.epochs))) * args.consistency_weight,
                  lambda labels: torch.tensor([
                      l if i < args.batch_size * args.supervised_ratio else -1
                      for i, l in enumerate(labels)
                  ]).to(device),
                  0.1,
                  args.test_only,
                  result_writer(args.save_to))

    if args.save_to:
        torch.save(getattr(regularizer, "teacher", model).state_dict(), args.save_to)

if __name__ == "__main__":
    main()

