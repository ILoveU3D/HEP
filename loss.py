import torch
from torch import nn as nn
from torch.nn import functional as F
from pylab import *
from torch.autograd import Variable
import torchvision

class vgg_19(nn.Module):
    def __init__(self, index):
        super(vgg_19, self).__init__()
        vgg_model = torchvision.models.vgg19(weights=torchvision.models.VGG19_Weights.IMAGENET1K_V1)
        self.feature_ext = nn.Sequential(*list(vgg_model.features.children())[:index])
        for param in self.feature_ext.parameters():
            param.requires_grad = False

    def forward(self, x):
        if x.size(1) == 1:
            x = torch.cat((x, x, x), 1)
        out = self.feature_ext(x)
        return out


def gradient(input_tensor, direction):
    weights = torch.tensor([[0., 0.],
                            [-1., 1.]]
                           ).cuda()
    weights_x = weights.view(1, 1, 2, 2).repeat(1, 1, 1, 1)
    weights_y = torch.transpose(weights_x, 2, 3)

    if direction == "x":
        weights = weights_x
    elif direction == "y":
        weights = weights_y
    grad_out = torch.abs(F.conv2d(input_tensor, weights, stride=1, padding=1))
    return grad_out


def avg(R, direction):
    return nn.AvgPool2d(kernel_size=3, stride=1, padding=1)(gradient(R, direction))


class R_Smooth_Loss(nn.Module):
    def __init__(self):
        super(R_Smooth_Loss, self).__init__()

    def forward(self, input_R):
        rgb_weights = torch.Tensor([0.2990, 0.5870, 0.1140]).cuda()
        input_R = torch.tensordot(input_R, rgb_weights, dims=([1], [-1]))
        input_R = torch.unsqueeze(input_R, 1)
        loss_R_smooth = torch.mean(torch.abs(gradient(input_R, "x")) + torch.abs(gradient(input_R, "y")))
        return loss_R_smooth


class Smooth_loss(nn.Module):
    def __init__(self):
        super(Smooth_loss, self).__init__()

    def forward(self, input_I, input_R):
        rgb_weights = torch.Tensor([0.2990, 0.5870, 0.1140]).cuda()
        input_gray = torch.tensordot(input_R, rgb_weights, dims=([1], [-1]))
        input_gray = torch.unsqueeze(input_gray, 1)
        return torch.mean(gradient(input_I, "x") * torch.exp(-10 * gradient(input_gray, "x")) +
                          gradient(input_I, "y") * torch.exp(-10 * gradient(input_gray, "y")))


class IS_loss(nn.Module):
    def __init__(self):
        super(IS_loss, self).__init__()

    def forward(self, input_I, input_im):
        rgb_weights = torch.Tensor([0.2990, 0.5870, 0.1140]).cuda()
        input_gray = torch.tensordot(input_im, rgb_weights, dims=([1], [-1]))
        input_gray = torch.unsqueeze(input_gray, 1)
        low_gradient_x = gradient(input_I, "x")
        input_gradient_x = gradient(input_gray, "x")
        k = torch.full(input_gradient_x.shape, 0.01).cuda()
        x_loss = torch.abs(torch.div(low_gradient_x, torch.max(input_gradient_x, k)))
        low_gradient_y = gradient(input_I, "y")
        input_gradient_y = gradient(input_gray, "y")
        y_loss = torch.abs(torch.div(low_gradient_y, torch.max(input_gradient_y, k)))
        mut_loss = torch.mean(x_loss + y_loss)
        return mut_loss


class Exposure_control_loss(nn.Module):
    def __init__(self, patch_size, mean_val):
        super(Exposure_control_loss, self).__init__()
        # print(1)
        self.pool = nn.AvgPool2d(patch_size)
        self.mean_val = mean_val

    def forward(self, x):
        b, c, h, w = x.shape
        x = torch.mean(x, 1, keepdim=True)
        mean = self.pool(x)

        d = torch.mean(torch.pow(mean - torch.FloatTensor([self.mean_val]).cuda(), 2))
        return d


class Color_constancy_loss(nn.Module):
    def __init__(self):
        super(Color_constancy_loss, self).__init__()

    def forward(self, x):
        b, c, h, w = x.shape
        mean_rgb = torch.mean(x, [2, 3], keepdim=True)
        mr, mg, mb = torch.split(mean_rgb, 1, dim=1)
        Drg = torch.pow(mr - mg, 2)
        Drb = torch.pow(mr - mb, 2)
        Dgb = torch.pow(mb - mg, 2)
        k = torch.mean(torch.pow(torch.pow(Drg, 2) + torch.pow(Drb, 2) + torch.pow(Dgb, 2), 0.5))
        return k


class TV_loss(nn.Module):
    def __init__(self, TVLoss_weight=1):
        super(TV_loss, self).__init__()
        self.TVLoss_weight = TVLoss_weight

    def forward(self, x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = (x.size()[2] - 1) * x.size()[3]
        count_w = x.size()[2] * (x.size()[3] - 1)
        h_tv = torch.pow((x[:, :, 1:, :] - x[:, :, :h_x - 1, :]), 2).sum()
        w_tv = torch.pow((x[:, :, :, 1:] - x[:, :, :, :w_x - 1]), 2).sum()
        return self.TVLoss_weight * 2 * (h_tv / count_h + w_tv / count_w) / batch_size

def vgg_preprocess(batch):
    tensor_type = type(batch.data)
    (r, g, b) = torch.chunk(batch, 3, dim=1)
    batch = torch.cat((b, g, r), dim=1)  # convert RGB to BGR
    mean = tensor_type(batch.data.size()).cuda()
    mean[:, 0, :, :] = 103.939
    mean[:, 1, :, :] = 116.779
    mean[:, 2, :, :] = 123.680
    batch = batch.sub(Variable(mean))  # subtract mean
    return batch

class Perceptual_loss(nn.Module):
    def __init__(self):
        super(Perceptual_loss, self).__init__()
        self.map = nn.Conv2d(24, 3, 3, padding=1)
        self.instancenorm = nn.InstanceNorm2d(512, affine=False)
        self.vgg = vgg_19(20)

    def forward(self, img, target):
        img_vgg = self.map(img)
        target_vgg = self.map(target)
        img_vgg = vgg_preprocess(img_vgg)
        target_vgg = vgg_preprocess(target_vgg)
        img_fea = self.vgg(img_vgg)
        target_fea = self.vgg(target_vgg)
        return torch.mean((self.instancenorm(img_fea) - self.instancenorm(target_fea)) ** 2)

