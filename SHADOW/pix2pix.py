import pandas as pd
from seaborn import histplot

from ML_SERVER.birefnet import biref_process
from utils import pic2float, pic2int, pic2pil, sigmoid, swimg, mask_crop, memo, center, memo

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import transforms as tt
from torchvision.transforms import v2
from torchvision.transforms import functional as tf

# dataset
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os

import cv2

import numpy as np
from IPython.display import clear_output

import plotly.express as px

import kornia
from constant import device, ROOT

DTYPE = torch.float16

MODEL = 'SHADOW/models/Generator.pth'
AV_MODEL = 'SHADOW/models/AveragedModel.pth'

USE_AVERAGE = True

from ML_SERVER.sam import sam_process

from constant import device, ROOT


def add_mask(x, mask):
    m = mask[:, 0:1, :, :]
    return torch.cat((x, m), dim=1)


def add_rot(x, rot):
    device = x.device
    r = torch.ones(x.size(0), 2, x.size(2), x.size(3), device=device)
    r[:, 0, :, :] *= rot[:, 0, None, None]
    r[:, 1, :, :] *= rot[:, 1, None, None]
    return torch.cat((x, r), dim=1)


def add_rot_mask(x, rot, mask):
    x = add_mask(x, mask)
    x = add_rot(x, rot)
    return x


def vis(pics):
    # pics = [pic2pil(p) for p in pics]
    # p = torch.concat(pics, dim=2)
    pics = pic2pil(pics)
    swimg([pics])

    # for p in pics:
    # display(p)


# Определяем блоки генератора
class Block(nn.Module):
    def __init__(self, in_channels, out_channels, dropout=False, norm=True, relu=True):
        super(Block, self).__init__()

        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
        ]
        if norm:
            layers.append(nn.BatchNorm2d(out_channels))

        if relu:
            layers.append(nn.ReLU(inplace=True))

        if dropout:
            layers.append(nn.Dropout(0.5))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


# Генератор на основе кодировщика-декодировщика
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        # Кодировщик
        self.enc1 = Block(6, 64, norm=False)  # Входное изображение
        self.enc2 = Block(64, 128)
        self.enc3 = Block(128, 256)
        self.enc4 = Block(256, 256)
        self.enc5 = Block(256, 256)
        self.enc6 = Block(256, 256, dropout=True)
        self.downsample = nn.MaxPool2d(2)

        # Декодировщик
        self.dec6 = Block(256, 256)
        self.dec5 = Block(256 * 2, 256)
        self.dec4 = Block(256 * 2, 256)
        self.dec3 = Block(256 * 2, 128)
        self.dec2 = Block(128 * 2, 64)
        self.dec1 = Block(64 * 2, 3, norm=False, relu=False)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

    def forward(self, x):
        # Кодировка
        enc1 = self.enc1(x)  # 3-64

        enc2 = self.downsample(enc1)
        enc2 = self.enc2(enc2)  # 64-128

        enc3 = self.downsample(enc2)
        enc3 = self.enc3(enc3)  # 128-256

        enc4 = self.downsample(enc3)
        enc4 = self.enc4(enc4)  # 256-512

        enc5 = self.downsample(enc4)
        enc5 = self.enc5(enc5)  # 512-512

        enc6 = self.downsample(enc5)
        enc6 = self.enc6(enc6)  # 512-512

        # Декодировка
        dec6 = self.dec6(enc6)  # 512-512

        dec5 = self.upsample(dec6)
        dec5 = torch.cat([enc5, dec5], dim=1)  # 512+512
        dec5 = self.dec5(dec5)  # 1024-512

        dec4 = self.upsample(dec5)
        dec4 = torch.cat([enc4, dec4], dim=1)  # 512+512
        dec4 = self.dec4(dec4)  # 1024-256

        dec3 = self.upsample(dec4)
        dec3 = torch.cat([enc3, dec3], dim=1)  # 256+256
        dec3 = self.dec3(dec3)  # 512-128

        dec2 = self.upsample(dec3)
        dec2 = torch.cat([enc2, dec2], dim=1)  # 128+128
        dec2 = self.dec2(dec2)  # 256-64

        dec1 = self.upsample(dec2)
        dec1 = torch.cat([enc1, dec1], dim=1)  # 64+64
        dec1 = self.dec1(dec1)  # 128-3

        dec1 = torch.tanh(dec1)

        return dec1


# Дискриминатор
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(9, 64, kernel_size=4, stride=2, padding=1),  # 512 -> 256
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),  # 256 -> 128
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),  # 128 -> 64
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),  # 64 -> 32
            # nn.BatchNorm2d(1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(512, 1, kernel_size=4, stride=2, padding=1),  # 32 -> 16
            # nn.Sigmoid(),
        )
        self.global_pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        x = self.model(x)
        # x = self.global_pool(x).flatten(1)
        return x


class ShadowGenerator:
    def __init__(self, generator=None, device=device, avarage=USE_AVERAGE):
        if generator is None:
            generator = Generator()
            if avarage:
                generator = torch.optim.swa_utils.AveragedModel(generator, multi_avg_fn=torch.optim.swa_utils.get_ema_multi_avg_fn(0.999))
                model_path = AV_MODEL
            else:
                model_path = MODEL
            generator.load_state_dict(torch.load(os.path.join(ROOT, model_path), weights_only=True))
            generator.eval()
            generator.to(DTYPE).to(device)

        self.generator = generator
        self.device = device

    def generate(self, colors, masks, rots=None,):

        if not isinstance(colors, list):
            colors = [colors]
            masks = [masks]

        colors = torch.stack([torch.tensor(i).permute(2, 0, 1) for i in colors])
        masks = 1 - torch.stack([torch.tensor(i).permute(2, 0, 1) for i in masks])
        white = torch.ones_like(colors)
        colors_on_white = white * masks + colors * (1-masks)

        if rots is None:
            rotate_angle = torch.randint(0, 359, (1,)).item()
        else:
            rotate_angle = rots

        rotate_angle = torch.tensor([rotate_angle]) / 180 * 3.1415
        rots = torch.cat([torch.sin(rotate_angle), torch.cos(rotate_angle)], dim=0)
        rots = torch.cat([rots[None, :]] * colors.size(0), dim=0)

        input = add_rot_mask(colors_on_white, rots, masks)

        with torch.inference_mode():
            shadow = self.generator(input.half().to(device))

        shadow_comp = []
        for i, s in enumerate(shadow.cpu()):
            s = s * (masks[i]) + colors[i] * (1 - masks[i])
            shadow_comp.append(s)

        return shadow_comp

print('Loading Shadow model...')
sg = ShadowGenerator()
print('Shadow model loaded')

@memo
def generate_shadow(images, masks, rots=None):
    shadow_comp = sg.generate(images, masks, rots)
    return shadow_comp


def test():
    # sg = ShadowGenerator()
    image_path = os.path.join(ROOT, "image.jpg")
    image = Image.open(image_path)
    image = pic2float(image)
    image, mask = biref_process(image)
    # mask = mask[:,:,None]
    print(image.shape, mask.shape)
    image, mask = center(image, mask)
    print(image.shape, mask.shape)
    # images, masks, _ = sam_process(image)

    shadow_comp = generate_shadow(image, mask, rots=10)
    print(shadow_comp)

    # swimg([*images, *masks, *shadow_comp])

if __name__ == '__main__':
    test()

