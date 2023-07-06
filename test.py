from torch.utils.data import DataLoader
from models.LUM import DecomNet
from data.Viewset import ViewsetSingle

def saveTensor(data, target):
    data.detach().cpu().numpy().tofile(target)

model = DecomNet({'norm':"wn", 'activ':'prelu', 'pad_type':'zero'}).cuda()
set = ViewsetSingle("/media/wyk/wyk/Data/light/proj_1_1")
loader = DataLoader(set, 1, None)
for i in loader:
    input, inputEq = i
    output = model(input)
pass