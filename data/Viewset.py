import os
import random
import numpy as np
import torch
from torch.utils.data import Dataset
from .utils import histEq, process

class ViewsetSingle(Dataset):
    def __init__(self, dir, length, batchsize=24, viewsize=(144,5120)):
        self.dir = dir
        self.len = length
        self.l = [[] for _ in range(batchsize)]
        for item in os.listdir(dir):
            sourceNum = int(item.split('_')[4]) - 1
            self.l[sourceNum].append(item)
        self.size = viewsize
        self.batchsize = batchsize

    def __getitem__(self, item):
        image = torch.zeros(self.batchsize, *self.size)
        for idx, sub in enumerate(self.l):
            catch = random.randint(0, len(sub)-1)
            subImage = np.fromfile(os.path.join(self.dir, sub[catch]), dtype="float32")
            subImage = np.reshape(subImage, self.size)
            subImage = process(subImage)
            image[idx,:,:] = torch.from_numpy(subImage)
        # images = histEq(images)
        # for idx, sub in enumerate(images):
        #     imageEq[idx, :, :] = torch.from_numpy(sub)
        return image

    def __len__(self):
        return self.len

class ViewsetAll(Dataset):
    def __init__(self, dir, batchsize=24, viewsize=(144,5120)):
        self.dir = dir
        self.l = [[] for _ in range(batchsize)]
        for item in os.listdir(dir):
            sourceNum = int(item.split('_')[4]) - 1
            self.l[sourceNum].append(item)
        self.len = len(self.l[0])
        for k in range(1, len(self.l)):
            assert self.len == len(self.l[k])
        self.size = viewsize
        self.batchsize = batchsize

    def __getitem__(self, item):
        image = torch.zeros(self.batchsize, *self.size)
        for idx, sub in enumerate(self.l):
            subImage = np.fromfile(os.path.join(self.dir, sub[item]), dtype="float32")
            subImage = np.reshape(subImage, self.size)
            subImage = process(subImage)
            image[idx,:,:] = torch.from_numpy(subImage)
        return image

    def __len__(self):
        return self.len