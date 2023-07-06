import os
import random
import numpy as np
import torch
from torch.utils.data import Dataset
from .utils import histEq, process

class ViewsetSingle(Dataset):
    def __init__(self, dir, batchsize=24, viewsize=(144,5120)):
        self.dir = dir
        self.l = [[] for _ in range(batchsize)]
        for item in os.listdir(dir):
            sourceNum = int(item.split('_')[4]) - 1
            self.l[sourceNum].append(item)
        self.size = viewsize
        self.batchsize = batchsize

    def __getitem__(self, item):
        image = torch.zeros(self.batchsize, *self.size).cuda()
        imageEq = torch.zeros(self.batchsize, *self.size).cuda()
        images = []
        for idx, sub in enumerate(self.l):
            catch = random.randint(0, len(sub)-1)
            subImage = np.fromfile(os.path.join(self.dir, sub[catch]), dtype="float32")
            subImage = np.reshape(subImage, self.size)
            subImage = process(subImage)
            images.append(subImage)
            image[idx,:,:] = torch.from_numpy(subImage).cuda()
        images = histEq(images)
        for idx, sub in enumerate(images):
            imageEq[idx, :, :] = torch.from_numpy(sub).cuda()
        return image, imageEq

    def __len__(self):
        return 500

