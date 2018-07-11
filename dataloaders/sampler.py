import torch
import random

class RandSubClassSampler(torch.utils.data.sampler.Sampler):
    r"""Samples a subset of classes for each mini-batch, without replacement.

    Arguments:
        inds (list): a list of indices
        labels (list): a list of class labels
        cls_per_batch (int): number of class in each mini-batch
        batch_size (int): mini-batch size
        num_batch (int): number of mini-batch
    """

    def __init__(self, inds, labels, cls_per_batch, batch_size, num_batch):
        assert len(inds)==len(labels),"Mismatched inputs inds:%d,labels:%d"%(len(inds),len(labels))
        self.batch_size = batch_size
        self.cls_per_batch = cls_per_batch
        self.num_batch = num_batch
        self.sample_per_cls = batch_size//cls_per_batch
        self.inds = inds
        self.labels = labels
        self.cls2ind = {}
        self.label_set = set(labels)
        for l in self.label_set:
            self.cls2ind[l] = []
        for i in range(len(inds)):
            self.cls2ind[labels[i]].append(inds[i])

    def __iter__(self):
        for b in range(self.num_batch):
            rand_cls_set = random.sample(self.label_set,self.cls_per_batch)
            for c in rand_cls_set:
                ind_list = random.sample(self.cls2ind[c],self.sample_per_cls)
                for i in ind_list:
                    yield i

    def __len__(self):
        return self.batch_size * self.num_batch


if __name__ == '__main__':
    # For sanity test
    inds = range(100)
    labels = [random.randint(0,7) for i in range(100)]
    s = RandSubClassSampler(inds,labels,5,20,10)