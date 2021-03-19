
import numpy as np
import torch
import random
import os
import torchvision.transforms.functional as F


class ClusteredSampler(torch.utils.data.Sampler):
    def __init__(self, data_source,index_to_cluster,n_cluster,losses,decrease_center = 1):
        self.ds = data_source
        self.index_to_cluster =index_to_cluster
        self.hierarchy = []
        self.decrease_center = decrease_center
        self.n_cluster = n_cluster
        self.center = self.n_cluster + 1
        self.create_distribiouns(losses)

    def create_distribiouns(self, losses):
        losses = np.zeros(self.n_cluster)

        assert self.center > 0

        losses_mean = np.mean(losses)
        new_losses = np.zeros(self.n_cluster)
        for cluster_index, cluster_loss in enumerate(losses):
            if cluster_loss:
                new_losses[cluster_index] = np.mean(cluster_loss)
            else:
                new_losses[cluster_index] = losses_mean

        while len(self.hierarchy) < self.n_cluster:
            max_idx = np.argmax(new_losses)
            self.hierarchy.append(max_idx)
            new_losses[max_idx] = -1
        self.hierarchy = [-1] + self.hierarchy
        assert len(self.hierarchy) == self.n_cluster + 1


    def __iter__(self):
        indexes = list(range(len(self.ds)))
        random.shuffle(indexes)
        print(f"self.center is {self.center}")
        curr_hierarchy = {}
        self.center -= self.decrease_center
        for i in range(len(self.hierarchy)):
            curr_hierarchy[self.hierarchy[i]] = np.exp(-0.2 * abs(self.center - i)) if i < self.center else 1
        diffs = {}
        for i in range(len(self.hierarchy)):
            diffs[i] = []
        for idx in indexes:
            if self.center >= 0:
                cluster = self.index_to_cluster[idx]
                assert cluster in curr_hierarchy
                randi = random.random()
                if randi < curr_hierarchy[cluster]:
                    yield idx
            else:
                yield idx
    def __len__(self):
        return len(self.ds)