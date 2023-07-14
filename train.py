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

trainSet = ViewsetSingle("/home/nv/wyk/Data/light/proj_2_1", 1000)
validSet = ViewsetAll("/home/nv/wyk/Data/light/proj_2_1")
trainLoader = DataLoader(trainSet, batch_size=10, shuffle=False)
validLoader = DataLoader(validSet, batch_size=1, shuffle=False)

net = DecomNet({'norm':"none", 'activ':'relu', 'pad_type':'zero'}).to(device)
mse = torch.nn.MSELoss()
optimizer = torch.optim.Adam(net.parameters(), lr=10e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
# dictionary = torch.load(pretrain3)
# net.load_state_dict(dictionary["model"])
# optimizer.load_state_dict(dictionary["optimizer"])
# scheduler.load_state_dict(dictionary["scheduler"])

checkpointPath = "/home/nv/wyk/Data/light/checkpoint/"
def saveTensor(data, target, i, nm="x.raw"):
    # data.detach().cpu().numpy().tofile(os.path.join(target, nm))
    for k in range(data.size(1)):
        data[0,k,:,:].detach().cpu().numpy().tofile(os.path.join(target, "output_{:02d}_{}.raw".format(k+1 ,i)))

net.train()
epoch = 1
finish = False
while not finish:
    with tqdm(trainLoader) as iterator:
        iterator.set_description("Epoch {}".format(epoch))
        epoch += 1
        for idx,data in enumerate(iterator):
            input = data
            input = input.to(device)
            l = net(input)
            output = input * l.view(-1,24,1,1)
            optimizer.zero_grad()
            perceptual_loss = torch.var(torch.mean(output, dim=(0,2,3)))
            remake_loss = mse(output, input)
            totalLoss = perceptual_loss + remake_loss
            totalLoss.backward()
            optimizer.step()
            iterator.set_postfix_str("loss:{:.6f}(p:{:.6f},r:{:.6f})".format(totalLoss.item(), perceptual_loss.item(), remake_loss.item()))
            if totalLoss.item() < 1e-3:
                finish = True
net.eval()
with torch.no_grad():
    with tqdm(validLoader) as iterator:
        iterator.set_description("validating...")
        for idx, input in enumerate(iterator):
            input= input.to(device)
            l = net(input)
            print(l)
            output = input * l.view(-1,24,1,1)
            saveTensor(input, "/home/nv/wyk/Data/light/input", idx, "input.raw")
            saveTensor(output, "/home/nv/wyk/Data/light/output", idx, "output.raw")
# torch.save({
#     "epoch": len(trainLoader), "model": net.state_dict(), "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict()
# }, "{}/hep_v1.0_{}.dict".format(checkpointPath, datetime.now()))