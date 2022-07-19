from torchvision import datasets, transforms
from PIL import Image
import numpy as np
import cv2
import os.path as osp
import os
import sys
import torch
import random

def solve_mask(img, img_target):
    img1 = np.asarray(img.permute(1, 2, 0).cpu())
    # print(img1)
    img2 = np.asarray(img_target.permute(1, 2, 0).cpu())
    # print(img2)
    img3 = abs(img1 - img2)
    # print(img3)
    mask = img3.sum(2) > (15.0 / 255.0)
    mask = mask.astype(int)
    # print('oooooooooooooooooooooo')
    # print(mask)
    return mask


def solve_balance(mask):
    height, width = mask.shape
    k = mask.sum()
    # print(k)
    k = (int)(k)

    mask2 = (1.0 - mask) * np.random.rand(height, width)
    mask2 = mask2.flatten()
    pos = np.argsort(mask2)
    balance = np.zeros(height * width)
    balance[pos[:min(250 * 250, 4 * k)]] = 1
    balance = balance.reshape(height, width)
    return balance


def generate_watermark(img,root_logo):
    random.seed(1)
    img = img[0].cuda()
    logo = Image.open(root_logo)
    logo = logo.convert('RGBA')

    rotate_angle = random.randint(0, 360)
    logo_rotate = logo.rotate(rotate_angle, expand=True)
    logo_height, logo_width = logo_rotate.size
    logo_height = random.randint(10, 256)
    logo_width = random.randint(10, 256)
    logo_resize = logo_rotate.resize((logo_height, logo_width))

    transform_totensor = transforms.Compose([transforms.ToTensor()])
    logo = transform_totensor(logo_resize).cuda()

    alpha = random.random() * 0.3 + 0.1
    start_height = random.randint(0, 256 - logo_height)
    start_width = random.randint(0, 256 - logo_width)


    img[:, start_width:start_width + logo_width, start_height:start_height + logo_height] = \
        img[:,start_width:start_width + logo_width, start_height:start_height + logo_height] * (1.0 - alpha * logo[3:4,:,:]) + logo[:3,:,:] * alpha * logo[3:4,:,:]

    return img

def generate_watermark_ori(img,root_logo,seeds):

    random.seed(seeds)
    img = img[0].detach().cpu().numpy()
    img = torch.tensor(img).cuda()
    logo = Image.open(root_logo)
    logo = logo.convert('RGBA')

    rotate_angle = random.randint(0, 360)
    logo_rotate = logo.rotate(rotate_angle, expand=True)
    logo_height, logo_width = logo_rotate.size
    logo_height = random.randint(50, 70)
    logo_width = random.randint(50, 70)
    logo_resize = logo_rotate.resize((logo_height, logo_width))

    transform_totensor = transforms.Compose([transforms.ToTensor()])
    logo = transform_totensor(logo_resize).cuda()

    alpha = random.random() * 0.3 + 0.4
    start_height = random.randint(0, 256 - logo_height)
    start_width = random.randint(0, 256 - logo_width)

    img_target = img.clone()
    img[:, start_width:start_width + logo_width, start_height:start_height + logo_height] = \
        img[:,start_width:start_width + logo_width, start_height:start_height + logo_height] * (1.0 - alpha * logo[3:4,:,:]) + logo[:3,:,:] * alpha * logo[3:4,:,:]

    mask = solve_mask(img, img_target)
    mask = np.concatenate((mask[:,:,np.newaxis],mask[:,:,np.newaxis],mask[:,:,np.newaxis]),2)*256.0

    return img,mask

