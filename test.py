import torch
from torch.utils.data import Sampler
import numpy as np


class DoubleSubsetRandomSampler(Sampler):
    def __init__(self, indices_source, indices_target, num_source, num_target):
        """
        Sampler for creating domain-homogeneous batches with balanced sampling.

        Args:
            indices_source (list): Indices of source domain samples.
            indices_target (list): Indices of target domain samples.
            num_source (int): Number of source samples per batch.
            num_target (int): Number of target samples per batch.
        """
        self.indices_source = indices_source
        self.indices_target = indices_target
        self.num_source = num_source
        self.num_target = num_target

    def __iter__(self):
        # Shuffle indices for both domains
        perm_source = torch.randperm(len(self.indices_source))
        perm_target = torch.randperm(len(self.indices_target))

        # Oversample smaller dataset if needed
        max_len = max(len(perm_source), len(perm_target))
        if len(perm_source) < max_len:
            repeats = (max_len + len(perm_source) - 1) // len(perm_source)  # Ceil division
            perm_source = torch.cat([perm_source] * repeats)[:max_len]
        if len(perm_target) < max_len:
            repeats = (max_len + len(perm_target) - 1) // len(perm_target)
            perm_target = torch.cat([perm_target] * repeats)[:max_len]

        # Create batches
        for i in range(0, len(perm_source), self.num_source):
            batch_source = perm_source[i : i + self.num_source]
            yield [self.indices_source[idx] for idx in batch_source]

        for i in range(0, len(perm_target), self.num_target):
            batch_target = perm_target[i : i + self.num_target]
            yield [self.indices_target[idx] for idx in batch_target]

    def __len__(self):
        # Calculate the number of batches
        num_batches_source = (len(self.indices_source) + self.num_source - 1) // self.num_source  # Ceil division
        num_batches_target = (len(self.indices_target) + self.num_target - 1) // self.num_target
        return num_batches_source + num_batches_target

# Example indices for source and target datasets
indices_source = list(range(100))  # 100 samples in source dataset
indices_target = list(range(60))   # 60 samples in target dataset

num_source = 10  # Batch size for source
num_target = 15  # Batch size for target

sampler = DoubleSubsetRandomSampler(indices_source, indices_target, num_source, num_target)

# Iterate over batches
for batch in sampler:
    print(batch)