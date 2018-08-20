import torch
import torch.nn as nn
import torch.nn.functional as F

from fastai.conv_learner import *
from fastai.learner import *
from fastai.model import *
from fastai.dataset import *

from torchvision.models import resnet18, resnet34, resnet50, resnet101, resnet152

def conv(ni, nf, ks=3, stride=1, bias=False):
    return nn.Conv2d(ni, nf, kernel_size=ks, stride=stride, padding=ks//2, bias=bias)

data = ImageClassifierData.from_names_and_array()
data.resize()
ConvLearner.pretrained()

def bn(planes, init_zero=False):
    m = nn.BatchNorm2d(planes)
    m.weight.data.fill_(0 if init_zero else 1)
    m.bias.data.zero_()
    return m


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super().__init__()
        self.conv1 = conv(inplanes, planes, stride=stride)
        self.bn1 = bn(planes)
        self.relu = nn.ReLU()
        self.conv2 = conv(planes, planes)
        self.bn2 = bn(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        out = self.conv1(x)
        out = self.relu(out)
        out = self.bn1(out)

        out = self.conv2(out)
        out = self.relu(out)
        out = self.bn2(out)
        return out



class UBlock(nn.Module):
    def __init__(self, inplanes, depth):
        super().__init__()

        loss_fn = torch.nn.BCELoss(reduce=False, size_average=False)

        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv_block1 = BasicBlock(inplanes, inplanes * 2)
        if depth == 1:
            self.mid_block = lambda x: x
        else:
            self.mid_block = UBlock(2 * inplanes, depth - 1)

        # self.upsample = nn.UpsamplingBilinear2d(scale_factor=2)
        self.conv_after_up = conv(inplanes * 2, inplanes)

        self.conv_block2 = BasicBlock(inplanes * 2, inplanes)

    def forward(self, x):
        tmp = x
        x = self.max_pool(x)
        x = self.conv_block1(x)
        x = self.mid_block(x)
        # x = self.upsample(x)
        x = F.interpolate(x, scale_factor=2)
        x = F.relu(self.conv_after_up(x))
        x = torch.cat([tmp, x], dim=1)
        x = self.conv_block2(x)
        return x


class UNetModel(nn.Module):
    def __init__(self, depth=5, start_channels=16):
        super().__init__()
        self.depth = depth
        self.start_channels = start_channels

        self.first_block = BasicBlock(1, start_channels)
        self.u_block = UBlock(start_channels, depth)
        self.out_layer = nn.Sequential(conv(start_channels, 1, ks=1), nn.Sigmoid())

    def forward(self, x):
        # x = x[..., None]
        x = self.first_block(x)
        x = self.u_block(x)
        x = self.out_layer(x)
        return x

def children(m): return m if isinstance(m, (list, tuple)) else list(m.children())


if __name__ == '__main__':
    unet = UNetModel(depth=5, start_channels=16)
    print(unet)

    inputs = torch.rand(2, 1, 128, 128)
    outputs = unet(inputs)

    print(outputs.size())
    # model = resnet34(pretrained=False)
    layers = children(unet)
    print(layers)
    unet.children()
    print(len(layers))


