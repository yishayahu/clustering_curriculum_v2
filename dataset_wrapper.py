import os
import random

import torch
from sklearn.cluster import MiniBatchKMeans
from clustered_sampler import ClusteredSampler
from regular_sampler import RegularSampler
import numpy as np

class DsWrapper(torch.utils.data.Dataset):
    def __init__(self,model,dataset_creator,batch_size,n_clusters,start_transforms,**kwargs):

        #######
        train_dir = kwargs["train_dir"]
        curr_index = 0
        train_indexes = []
        clustering_indexes = []
        for class_name in os.scandir(train_dir):
            for img in os.listdir(class_name.path):
                curr_index += 1
                if random.random() < 0.8:
                    train_indexes.append(curr_index)
                else:
                    clustering_indexes.append(curr_index)
        regular_sampler = RegularSampler(data_source=self, train_indexes=train_indexes,
                                         clustering_indexes=clustering_indexes)
        self.current_sampler = regular_sampler
        #####
        self.future_transforms = kwargs.pop("transforms")

        kwargs["transforms"] = start_transforms
        self.ds = dataset_creator(**kwargs)
        self.dataset_creator = dataset_creator
        self.future_kwargs = kwargs
        #####


        ######
        self.new_indexes = []
        self.index_to_cluster = {}
        self.n_clusters = n_clusters
        self.losses = [[] for _ in range(n_clusters)]
        self.clustering_algorithm = MiniBatchKMeans(n_clusters=n_clusters)
        def feature_layer_hook(model, input, output):
            assert len(self.new_indexes) == batch_size
            assert output.shape[0] == batch_size
            arrays = []
            for i in range(output.shape[0]):
                arrays.append(output[i].cpu().detach().numpy().astype(np.int16))
            self.clustering_algorithm.partial_fit(arrays)
            for (index,label) in zip(self.new_indexes,self.clustering_algorithm.predict(arrays)):
                self.index_to_cluster[index] = label
        features_layer =  model.named_children()[-2][1]
        features_layer.register_forward_hook(feature_layer_hook)

        #######

    def send_loss(self,loss,train_loader):
        assert loss.shape[0] == len(self.new_indexes)
        if self.current_sampler.get_clustering_flag() == "clustering":
            for i, index in enumerate(self.new_indexes):
                self.losses[self.index_to_cluster[index]].append(loss[i].item())
            self.new_indexes = []
            return 0
        elif self.current_sampler.get_clustering_flag() == "done":
            self.future_kwargs["transforms"] = self.future_transforms
            self.ds = self.dataset_creator(self.future_kwargs)
            self.current_sampler = ClusteredSampler(data_source=self.ds, index_to_cluster=self.index_to_cluster,
                                                    n_cluster=self.n_clusters, losses=self.losses)
            train_loader.dataset = self.ds
            train_loader.sampler = self.current_sampler
        else:
            self.new_indexes = []
            return torch.mean(loss)


    def __getitem__(self, item):
        assert type(self.current_sampler) == RegularSampler
        self.new_indexes.append(item)
        return self.ds[item]




