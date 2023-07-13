from abc import ABC

from torch import nn
import torch
from models.NDM import Conv2dBlock
try:
    from itertools import izip as zip
except ImportError:
    pass


class DecomNet(nn.Module, ABC):
    def __init__(self, params):
        super(DecomNet, self).__init__()
        self.norm = params['norm']
        self.activ = params['activ']
        self.pad_type = params['pad_type']
        #
        self.conv0 = Conv2dBlock(25, 32, 3, 1, 1, norm=self.norm, activation=self.activ, pad_type=self.pad_type)
        self.conv1 = Conv2dBlock(25, 64, 9, 1, 4, norm=self.norm, activation='none', pad_type=self.pad_type)
        self.conv2 = Conv2dBlock(64, 64, 3, 1, 1, norm=self.norm, activation=self.activ, pad_type=self.pad_type)
        self.conv3 = Conv2dBlock(64, 128, 3, 2, 1, norm=self.norm, activation=self.activ, pad_type=self.pad_type)
        self.conv4 = Conv2dBlock(128, 128, 3, 1, 1, norm=self.norm, activation=self.activ, pad_type=self.pad_type)
        self.conv5 = nn.ConvTranspose2d(128, 64, 3, 2, 1, 1)
        self.activation = nn.ReLU(inplace=True)
        self.xpool = nn.AvgPool2d(3, (1,2), 1)
        self.apool = nn.AvgPool2d(3, 2, 1)
        self.conv6 = Conv2dBlock(128, 64, 3, 1, 1, norm=self.norm, activation=self.activ, pad_type=self.pad_type)
        self.conv7 = Conv2dBlock(96, 64, 3, 1, 1, norm=self.norm, activation='none', pad_type=self.pad_type)
        self.conv8 = Conv2dBlock(64, 1, 3, 1, 1, norm=self.norm, activation='none', pad_type=self.pad_type)
        self.fc = nn.Linear(18*160, 18*160)
        self.oc = nn.Linear(18*160, 24)

    def forward(self, input_im):
        input_max = torch.max(input_im, dim=1, keepdim=True)[0]
        image = torch.cat((input_max, input_im), dim=1)
        x0 = self.conv0(image)
        x1 = self.conv1(image)
        x1 = self.apool(x1)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        x3 = self.apool(x3)
        x4 = self.conv4(x3)
        x5 = self.conv5(x4)
        x5 = self.xpool(x5)
        x5 = self.activation(x5)
        cat5 = torch.cat((x5, self.crop(x2, x5.shape)), dim=1)
        cat5 = self.xpool(cat5)
        x6 = self.conv6(cat5)
        cat6 = torch.cat((x6, self.crop(x0, x6.shape)), dim=1)
        x7 = self.conv7(cat6)
        x8 = self.conv8(x7)
        x8 = self.apool(x8)
        x9 = self.activation(x8)
        x9 = x9.view(-1, 18*160)
        x10 = self.fc(x9)
        x11 = self.oc(x10)
        return x11

    def crop(self, image, s):
        h, w = image.shape[2:]
        h_, w_ = s[2:]
        y, x = (h - h_) // 2, (w - w_) // 2
        image = image[...,y:y+h_,x:x+w_]
        return image