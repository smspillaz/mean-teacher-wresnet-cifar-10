import numpy as np
import itertools
from torch.utils.data.sampler import Sampler
import torch
from torch.utils.data import TensorDataset

from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split


class TwoStreamBatchSampler(Sampler):
    """Iterate two sets of indices
    An 'epoch' is one iteration through the primary indices.
    During the epoch, the secondary indices are iterated through
    as many times as needed.
    """
    def __init__(self, primary_indices, secondary_indices, batch_size, secondary_batch_size):
        self.primary_indices = primary_indices
        self.secondary_indices = secondary_indices
        self.secondary_batch_size = secondary_batch_size
        self.primary_batch_size = batch_size - secondary_batch_size

        assert len(self.primary_indices) >= self.primary_batch_size > 0
        assert len(self.secondary_indices) >= self.secondary_batch_size > 0

    def __iter__(self):
        primary_iter = iterate_once(self.primary_indices)
        secondary_iter = iterate_eternally(self.secondary_indices)
        return (
            primary_batch + secondary_batch
            for (primary_batch, secondary_batch)
            in  zip(grouper(primary_iter, self.primary_batch_size),
                    grouper(secondary_iter, self.secondary_batch_size))
        )

    def __len__(self):
        return len(self.primary_indices) // self.primary_batch_size


def iterate_once(iterable):
    return np.random.permutation(iterable)


def iterate_eternally(indices):
    def infinite_shuffles():
        while True:
            yield np.random.permutation(indices)
    return itertools.chain.from_iterable(infinite_shuffles())


def grouper(iterable, n):
    "Collect data into fixed-length chunks or blocks"
    # grouper('ABCDEFG', 3) --> ABC DEF"
    args = [iter(iterable)] * n
    return zip(*args)


def get_moon_dataset(n_samples, n_labeled_per_class=None, train=True):
    random_state = 1 if train else 2
    x, y = make_moons(n_samples=n_samples, noise=.1, random_state=random_state)
    print(x.shape, y.shape)

    if train:
        # Hide labels
        np.random.seed(random_state)
        labeled_idxs = [
            np.random.permutation(np.where(y == iclass)[0])[:n_labeled_per_class]
            for iclass in range(2)
        ]

        labeled_idxs = np.hstack(labeled_idxs)
        unlabeled_idxs = np.setdiff1d(np.arange(len(y)), labeled_idxs)
        y[unlabeled_idxs] = -1  # no label

    trainset = TensorDataset(torch.tensor(x, dtype=torch.float32),
                             torch.tensor(y, dtype=torch.int64))

    return trainset


def get_moons_loader(n_samples, n_labeled_per_class=None, batch_size=32):
    trainset = get_moon_dataset(n_samples, n_labeled_per_class=n_labeled_per_class, train=True)
    valset = get_moon_dataset(n_samples // 2, n_labeled_per_class=n_labeled_per_class, train=False)
    batch_size = 32
    labeled_batch_size = 6
    labels = trainset.tensors[1]
    train_labeled_idxs = (labels != -1).nonzero()[:, 0]
    train_unlabeled_idxs = (labels == -1).nonzero()[:, 0]
    batch_sampler = TwoStreamBatchSampler(
        primary_indices=train_unlabeled_idxs, secondary_indices=train_labeled_idxs,
        batch_size=batch_size, secondary_batch_size=labeled_batch_size
    )
    trainloader = torch.utils.data.DataLoader(trainset,
                                              batch_sampler=batch_sampler,
                                              num_workers=0,
                                              pin_memory=True)
    valloader = torch.utils.data.DataLoader(valset, num_workers=0, pin_memory=True)
    return trainloader, valloader

