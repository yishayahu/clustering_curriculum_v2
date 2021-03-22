import os
import random

import torch
from sklearn.cluster import MiniBatchKMeans
from clustered_sampler import ClusteredSampler
from regular_sampler import RegularSampler
import numpy as np

class DsWrapper(torch.utils.data.Dataset):
    def __init__(self,model,dataset_creator,n_clusters,start_transform,feature_layer_name,warmups,**kwargs):

        #####
        self.future_transform = kwargs.pop("transform")

        kwargs["transform"] = start_transform
        self.ds = dataset_creator(**kwargs)
        self.dataset_creator = dataset_creator
        self.future_kwargs = kwargs
        #####


        #######
        all_indexes = list(range(len(self.ds)))
        random.shuffle(all_indexes)
        train_indexes = all_indexes[:int(len(self.ds)*0.8)]
        clustering_indexes = all_indexes[int(len(self.ds)*0.8):]

        regular_sampler = RegularSampler(data_source=self, train_indexes=train_indexes,
                                         clustering_indexes=clustering_indexes,warmup_epochs=warmups)
        self.current_sampler = regular_sampler



        ######
        self.new_indexes = []
        self.index_to_cluster = {}
        self.n_clusters = n_clusters
        self.losses = [[] for _ in range(n_clusters)]
        self.clustering_algorithm = MiniBatchKMeans(n_clusters=n_clusters)
        def feature_layer_hook(model, input, output):
            if model.training and type(self.current_sampler) == RegularSampler:
                assert output.shape[0] == len(self.new_indexes)
                arrays = []
                for i in range(output.shape[0]):
                    arrays.append(output[i].cpu().detach().flatten().numpy())
                self.clustering_algorithm.partial_fit(arrays)
                for (index,label) in zip(self.new_indexes,self.clustering_algorithm.predict(arrays)):
                    self.index_to_cluster[index] = label
            else:
                if not model.training:
                    assert len(self.new_indexes) == 0
        for module_name, module1 in model.named_modules():
            if module_name == feature_layer_name:
                module1.register_forward_hook(feature_layer_hook)


        #######

    def send_loss(self,loss,train_loader):
        assert loss.shape[0] == len(self.new_indexes)
        if self.current_sampler.get_clustering_flag() == "clustering":
            for i, index in enumerate(self.new_indexes):
                self.losses[self.index_to_cluster[index]].append(loss[i].item())
            self.new_indexes = []
            loss[loss!=0] = 0

        elif self.current_sampler.get_clustering_flag() == "done":
            for i, index in enumerate(self.new_indexes):
                self.losses[self.index_to_cluster[index]].append(loss[i].item())
            self.new_indexes = []
            loss[loss!=0] = 0
            self.future_kwargs["transform"] = self.future_transform
            self.ds = self.dataset_creator(**self.future_kwargs)
            assert len(self.index_to_cluster) == len(self.ds)
            self.current_sampler = ClusteredSampler(data_source=self.ds, index_to_cluster=self.index_to_cluster,
                                                    n_cluster=self.n_clusters, losses=self.losses)
            train_loader.recreate(dataset=self.ds,sampler=self.current_sampler)
            self.new_indexes = []

        else:
            self.new_indexes = []
        return torch.mean(loss)


    def __getitem__(self, item):
        self.new_indexes.append(item)
        return self.ds[item]
    def __len__(self):
        return len(self.ds)



