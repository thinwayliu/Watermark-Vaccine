import torch
import numpy
import numpy as np
import math
from torchvision import datasets, transforms
import os
from PIL import Image

def clamp(X, lower_limit, upper_limit):
    return torch.max(torch.min(X, upper_limit), lower_limit)


def psnr(img1, img2):
  mse = np.mean((img1 - img2) ** 2)
  if mse == 0:
    return 100
  PIXEL_MAX = 255.0
  return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))

from numpy.lib.stride_tricks import as_strided as ast


def mse(img1, img2):
  mse = np.mean((img1 - img2) ** 2)
  return mse

def block_view(A, block=(3, 3)):
    """Provide a 2D block view to 2D array. No error checking made.
    Therefore meaningful (as implemented) only for blocks strictly
    compatible with the shape of A."""
    # simple shape and strides computations may seem at first strange
    # unless one is able to recognize the 'tuple additions' involved ;-)
    shape = (A.shape[0] / block[0], A.shape[1] / block[1]) + block
    strides = (block[0] * A.strides[0], block[1] * A.strides[1]) + A.strides
    return ast(A, shape=shape, strides=strides)

def image_concat(images):
    width, height = images[0].size
    target_shape = (3 * width, height)
    background = Image.new(mode="RGB", size=target_shape, color='black')
    for i,img in enumerate(images):
        location = ((i) * width,0)
        background.paste(img, location)
    return background

def all_image_concat(images):
    width, height = images[0].size
    target_shape = (width, height*4)
    background = Image.new(mode="RGB", size=target_shape, color='black')
    for i,img in enumerate(images):
        location = (0,(i) * height)
        background.paste(img, location)
    return background




def img_show(inputs,outputs,mask,config):
    inputs = torch.squeeze(inputs)
    outputs = torch.squeeze(outputs)
    mask = torch.squeeze(mask)

    if config == 'Clean':
        img_p = transforms.ToPILImage()(inputs.detach().cpu()).convert('RGB')
        clean_p = transforms.ToPILImage()(outputs.detach().cpu()).convert('RGB')
        clean_mask_p = transforms.ToPILImage()(mask.detach().cpu()).convert('L')
        images = [img_p,clean_p,clean_mask_p]

    elif config == 'DWV':
        adv1 = transforms.ToPILImage()(inputs.detach().cpu()).convert('RGB')
        adv1_p = transforms.ToPILImage()(outputs.detach().cpu()).convert('RGB')
        adv1_mask_p = transforms.ToPILImage()(mask.detach().cpu()).convert('L')
        images = [adv1,adv1_p,adv1_mask_p]

    elif config == 'IWV':
        adv2 = transforms.ToPILImage()(inputs.detach().cpu()).convert('RGB')
        adv2_p = transforms.ToPILImage()(outputs.detach().cpu()).convert('RGB')
        adv2_mask_p = transforms.ToPILImage()(mask.detach().cpu()).convert('L')
        images = [adv2,adv2_p,adv2_mask_p]

    elif config == 'RN':
        random = transforms.ToPILImage()(inputs.detach().cpu()).convert('RGB')
        random_pred = transforms.ToPILImage()(outputs.detach().cpu()).convert('RGB')
        random_mask_p = transforms.ToPILImage()(mask.detach().cpu()).convert('L')
        images = [random,random_pred,random_mask_p]

    return image_concat(images)