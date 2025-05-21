import numpy as np
import torch
from torch.utils.data.sampler import Sampler
from torch.utils.data.distributed import DistributedSampler


class DoubleSubsetRandomSampler(Sampler):
    def __init__(
        self, indices_source, indices_target, s_dataset_size, num_source, num_target
    ):
        self.indices_source = indices_source
        self.indices_target = indices_target
        self.s_dataset_size = s_dataset_size
        self.num_source = num_source
        self.num_target = num_target

    def __iter__(self):
        perm = torch.randperm(len(self.indices_source))
        tarperm = torch.randperm(len(self.indices_target))
        while len(tarperm) < (self.num_target // self.num_source) * self.s_dataset_size:
            tp1 = torch.randperm(len(self.indices_target))
            tarperm = torch.cat([tarperm, tp1], dim=0)
        T = 0
        t = 0
        for i, s in enumerate(perm, 1):
            yield self.indices_source[s]
            if i % self.num_source == 0:
                for j in range(self.num_target):
                    t = T + j
                    tidx = t
                    # tidx = t % len(tarperm)
                    yield self.s_dataset_size + self.indices_target[tarperm[tidx]]
                T = t + 1

    def __len__(self):
        full = int(
            np.floor(
                (len(self.indices_source) + len(self.indices_target)) / self.num_source
            )
        )
        last = len(self.indices_source) % self.num_source
        return int(full * self.num_source + last)

class DoubleSubsetRandomSampler_Modified(Sampler):
    def __init__(self, indices_source, indices_target, num_source, num_target):
        """
        Sampler for creating domain-alternating batches.
        
        Args:
            indices_source (list): Indices of source domain samples.
            indices_target (list): Indices of target domain samples.
            num_source (int): Number of source samples per alternating batch.
            num_target (int): Number of target samples per alternating batch.
        """
        self.indices_source = indices_source
        self.indices_target = indices_target
        self.num_source = num_source
        self.num_target = num_target

    def __iter__(self):
        # Shuffle source and target indices
        perm_source = torch.randperm(len(self.indices_source))
        perm_target = torch.randperm(len(self.indices_target))

        # Oversample smaller dataset to match larger one
        max_batches = max(
            (len(perm_source) + self.num_source - 1) // self.num_source,
            (len(perm_target) + self.num_target - 1) // self.num_target,
        )
        perm_source = perm_source.repeat((max_batches * self.num_source + len(perm_source) - 1) // len(perm_source))[:max_batches * self.num_source]
        perm_target = perm_target.repeat((max_batches * self.num_target + len(perm_target) - 1) // len(perm_target))[:max_batches * self.num_target]

        # Produce alternating batches
        for batch_idx in range(max_batches):
            # First yield `num_source` samples from the source
            start_s = batch_idx * self.num_source
            end_s = start_s + self.num_source
            for i in range(start_s, end_s):
                yield self.indices_source[perm_source[i]]

            # Then yield `num_target` samples from the target
            start_t = batch_idx * self.num_target
            end_t = start_t + self.num_target
            for j in range(start_t, end_t):
                yield self.indices_target[perm_target[j]]

    def __len__(self):
        # Calculate the number of batches
        max_batches = max(
            (len(self.indices_source) + self.num_source - 1) // self.num_source,
            (len(self.indices_target) + self.num_target - 1) // self.num_target,
        )
        return max_batches * (self.num_source + self.num_target)
    
    
class DistributedDoubleSubsetRandomSampler(DistributedSampler):
    def __init__(
        self, indices_source, indices_target, s_dataset_size, num_source, num_target,
        num_replicas, rank
    ):
        self.indices_source = indices_source
        self.indices_target = indices_target
        self.s_dataset_size = s_dataset_size
        self.num_source = num_source
        self.num_target = num_target

    def __iter__(self):
        perm = torch.randperm(len(self.indices_source))
        tarperm = torch.randperm(len(self.indices_target))
        while len(tarperm) < (self.num_target // self.num_source) * self.s_dataset_size:
            tp1 = torch.randperm(len(self.indices_target))
            tarperm = torch.cat([tarperm, tp1], dim=0)
        T = 0
        t = 0
        for i, s in enumerate(perm, 1):
            yield self.indices_source[s]
            if i % self.num_source == 0:
                for j in range(self.num_target):
                    t = T + j
                    tidx = t
                    # tidx = t % len(tarperm)
                    yield self.s_dataset_size + self.indices_target[tarperm[tidx]]
                T = t + 1

    def __len__(self):
        full = int(
            np.floor(
                (len(self.indices_source) + len(self.indices_target)) / self.num_source
            )
        )
        last = len(self.indices_source) % self.num_source
        return int(full * self.num_source + last)
