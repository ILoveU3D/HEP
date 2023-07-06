import loss
import torch
from torch.utils.data import DataLoader
from models.LUM import DecomNet
from data.Viewset import ViewsetSingle

model = DecomNet({'norm':"none", 'activ':'relu', 'pad_type':'zero'}).cuda()
set = ViewsetSingle("/home/nv/wyk/Data/proj_1_1", 20)
perceptual = loss.Perceptual_loss().cuda()
tv = loss.TV_loss().cuda()
mse = torch.nn.MSELoss()
loader = DataLoader(set, 1, None)
for i in loader:
    input, inputEq = i
    r, l = model(input)
    loss1 = perceptual(r, inputEq)
    loss2 = mse(r*l, input)
    loss3 = tv(l)
    print("{} {} {}".format(loss1.item(), loss2.item(), loss3.item()))
    del loss1, loss2, loss3
pass