import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import cv2

import torch
import torch.nn as nn
import torch.nn.functional as F

from fastai.imports import *
from fastai.dataset import *
from fastai.transforms import *
from fastai.conv_learner import *

import imgaug.augmenters as iaa



@contextmanager
def timer(title):
    start =time.time()
    yield
    print("{} - Done in {:.3f} secs.".format(title, time.time() - start))



from torchvision.models import resnet34, resnet101, resnet152


def conv(ni, nf, ks=3, stride=1, bias=True):
    return nn.Conv2d(ni, nf, kernel_size=ks, stride=stride, padding=ks//2, bias=bias)


def bn(planes, init_zero=False):
    m = nn.BatchNorm2d(planes)
    m.weight.data.fill_(0 if init_zero else 1)
    m.bias.data.zero_()
    return m


class ConvRelu(nn.Module):
    def __init__(self, in_, out):
        super().__init__()
        self.conv = conv(in_, out)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.activation(x)
        return x


class NoOperation(nn.Module):
    def forward(self, x):
        return x


class DecoderBlockV2(nn.Module):
    def __init__(self, in_channels, middel_channels, out_channels, is_deconv=True):
        super().__init__()

        if is_deconv:
            self.block = nn.Sequential(
                ConvRelu(in_channels, middel_channels),
                nn.ConvTranspose2d(middel_channels, out_channels, kernel_size=4, stride=2, padding=1)
            )
        else:
            self.block = nn.Sequential(
                nn.Upsample(scale_factor=2, mode="bilinear"),
                ConvRelu(in_channels, middel_channels),
                ConvRelu(middel_channels, out_channels)
            )

    def forward(self, x):
        return self.block(x)


class UNetResNet(nn.Module):
    def __init__(self, encoder_depth, num_classes, num_filters=32, dropout_2d=0.2,
                 pretrained=False, is_deconv=False):
        super().__init__()
        self.num_classes = num_classes
        self.droput_2d = dropout_2d

        if encoder_depth == 34:
            self.encoder = resnet34(pretrained=pretrained)
            bottom_channel_nr = 512
        elif encoder_depth == 101:
            self.encoder = resnet101(pretrained=pretrained)
            bottom_channel_nr = 2048
        elif encoder_depth == 152:
            self.encoder = resnet152(pretrained=pretrained)
            bottom_channel_nr = 2048
        else:
            raise NotImplementedError("only 34, 101, 152 version of Resnet are implemented.")

        self.pool = nn.MaxPool2d(2, 2)
        self.relu = nn.ReLU(inplace=True)
        # pool different from resnet(ks=3, s=2, p=1)
        self.conv1 = nn.Sequential(self.encoder.conv1,
                                   self.encoder.bn1,
                                   self.encoder.relu,
                                   self.pool)

        self.conv2 = self.encoder.layer1
        self.conv3 = self.encoder.layer2
        self.conv4 = self.encoder.layer3
        self.conv5 = self.encoder.layer4

        self.center = DecoderBlockV2(bottom_channel_nr, num_filters * 8 * 2,
                                     num_filters * 8, is_deconv)

        self.dec5 = DecoderBlockV2(bottom_channel_nr + num_filters * 8, num_filters * 8 * 2,
                                   num_filters * 8, is_deconv)

        self.dec4 = DecoderBlockV2(bottom_channel_nr // 2 + num_filters * 8, num_filters * 8 * 2,
                                   num_filters * 8, is_deconv)

        self.dec3 = DecoderBlockV2(bottom_channel_nr // 4 + num_filters * 8, num_filters * 4 * 2,
                                   num_filters * 2, is_deconv)

        self.dec2 = DecoderBlockV2(bottom_channel_nr // 8 + num_filters * 2, num_filters * 2 * 2,
                                   num_filters * 2 * 2, is_deconv)

        self.dec1 = DecoderBlockV2(num_filters * 2 * 2, num_filters * 2 * 2, num_filters, is_deconv)
        self.dec0 = ConvRelu(num_filters, num_filters)
        self.final = nn.Conv2d(num_filters, num_classes, kernel_size=1)

    def forward(self, x):
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)
        print('conv2:', conv2.size())
        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)
        conv5 = self.conv5(conv4)
        print('conv5:', conv5.size())
        pool = self.pool(conv5)
        print('pool:', pool.size())
        center = self.center(pool)
        print('center:', center.size())
        dec5 = self.dec5(torch.cat([center, conv5], dim=1))
        print('dec5:', dec5.size())
        dec4 = self.dec4(torch.cat([dec5, conv4], dim=1))
        dec3 = self.dec3(torch.cat([dec4, conv3], dim=1))
        dec2 = self.dec2(torch.cat([dec3, conv2], dim=1))
        dec1 = self.dec1(dec2)
        dec0 = self.dec0(dec1)

        out = self.final(F.dropout2d(dec0, p=self.droput_2d))

        return out


class SalDataset(BaseDataset):
    def __init__(self, images, masks, ids, transform):
        self.images = images
        self.masks = masks
        self.ids = ids
        self.transform = transform
        super().__init__(transform)

    def get_sz(self):
        return self.transform.sz

    def get_n(self):
        return len(self.ids)

    def get_x(self, i):
        x = self.images[self.ids[i]]
        return x

    def get_y(self, i):
        y = self.masks[self.ids[id]]
        return y

    def get_c(self):
        return 0


class MyTransform:
    def __init__(self, sz, aug_seed, aug=None):
        self.sz = sz
        self.aug_seed = aug_seed
        self.aug = aug
        if aug is not None:
            self.x_aug = aug
            self.x_aug.reseed(aug_seed)
            self.y_aug = aug.deepcopy()
            self.y_aug.reseed(aug_seed)

    def call(self, x, y):
        if self.aug is None:
            return x, y

        x = self.x_aug.augment_image(x)
        y = self.y_aug.augment_image(y)
        return x, y


def load_image(fname):
    return cv2.imread(str(fname))


def load_images(fnames):
    return np.stack([load_image(fname) for fname in fnames], axis=0)


class SalData(ImageData):
    def __init__(self, data_path: Path):
        self.train_imgs_dir = data_path / "train/images"
        self.train_masks_dir = data_path / "train/masks"
        self.ids = [f.stem for f in self.train_imgs_dir.iterdir()]
        self.n_train = len(self.ids)


        with timer("Load train images and masks"):
            self.train_images = load_images(self.train_imgs_dir / idx for idx in self.ids)
            self.train_masks = load_images(self.train_masks_dir / idx for idx in self.ids)







if __name__ == '__main__':
    net = UNetResNet(encoder_depth=34, num_classes=1, num_filters=32)
    # print(net)
    inputs = torch.randn((1, 3, 128, 128))
    out = net(inputs)

    print(out.size())
    print(net.conv2)








