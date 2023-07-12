import os
import setproctitle
import numpy as np
from tqdm import tqdm
import loss
import torch
from datetime import datetime
from torch.utils.data import DataLoader
from models.LUM import DecomNet
from data.Viewset import ViewsetSingle, ViewsetAll
setproctitle.setproctitle("(wyk)HEP light")
lossFunction = torch.nn.L1Loss(reduction="mean")
device = 0

trainSet = ViewsetSingle("/home/nv/wyk/Data/light/proj_1_1", 100)
validSet = ViewsetAll("/home/nv/wyk/Data/light/proj_1_1")
trainLoader = DataLoader(trainSet, batch_size=1, shuffle=False)
validLoader = DataLoader(validSet, batch_size=1, shuffle=False)

net = DecomNet({'norm':"none", 'activ':'relu', 'pad_type':'zero'}).to(device)
perceptual = loss.Perceptual_loss().cuda()
tv = loss.TV_loss().cuda()
mse = torch.nn.MSELoss()
optimizer = torch.optim.Adam(net.parameters(), lr=10e-5)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
# dictionary = torch.load(pretrain3)
# net.load_state_dict(dictionary["model"])
# optimizer.load_state_dict(dictionary["optimizer"])
# scheduler.load_state_dict(dictionary["scheduler"])

checkpointPath = "/home/nv/wyk/Data/light/checkpoint/"
def saveTensor(data, target, i):
    for k in range(data.size(1)):
        data[...,k,:,:].detach().cpu().numpy().tofile(os.path.join(target, "output_{:02d}_{}.raw".format(k+1 ,i)))

trainLoss = []
validLoss = []
net.train()
with tqdm(trainLoader) as iterator:
    for idx,data in enumerate(iterator):
        iterator.set_description("Epoch {}".format(idx))
        input = data
        input = input.to(device)
        l = net(input)
        optimizer.zero_grad()
        perceptual_loss = mse(torch.mean(input.view(-1)), torch.mean((input*l).view(-1)))
        remake_loss = mse(input*l, input)
        tv_loss = tv(l)
        loss = perceptual_loss + remake_loss + tv_loss
        loss.backward()
        optimizer.step()
        trainLoss.append(loss.item())
        iterator.set_postfix_str("loss:{:.3f}(p:{:.3f},r:{:.3f},t:{:.3f}),epoch mean:{:.2f}".format(loss.item(), perceptual_loss.item(), remake_loss.item(), tv_loss.item(), np.mean(np.array(trainLoss))))
net.eval()
with torch.no_grad():
    with tqdm(validLoader) as iterator:
        iterator.set_description("validating...")
        for idx, input in enumerate(iterator):
            input= input.to(device)
            l = net(input)
            output = input/l
            saveTensor(input, "/home/nv/wyk/Data/light/input/", idx)
            saveTensor(output, "/home/nv/wyk/Data/light/output/", idx)
torch.save({
    "epoch": len(trainLoader), "model": net.state_dict(), "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict()
}, "{}/hep_v1.0_{}.dict".format(checkpointPath, datetime.now()))