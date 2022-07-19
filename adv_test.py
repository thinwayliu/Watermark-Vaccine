from WDNet import generator
from PIL import Image
import numpy as np
import torch
import torch.nn as nn
import cv2
import os.path as osp
import os
import time
from torchvision import datasets, transforms
import torch.nn.functional as F
import pytorch_ssim
import numpy
import math


def psnr(img1, img2):
  mse = numpy.mean((img1 - img2) ** 2)
  if mse == 0:
    return 100
  PIXEL_MAX = 255.0
  return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))

from numpy.lib.stride_tricks import as_strided as ast
def mse(img1, img2):
  mse = numpy.mean((img1 - img2) ** 2)
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


os.environ["CUDA_VISIBLE_DEVICES"] = "0"

G = generator(3, 3)
G.eval()
G.load_state_dict(torch.load(os.path.join('WDNet_G.pkl')))
G.cuda()

root = './universal_data/test/'
train_root = './universal_data/train/'

trainimage_path = osp.join(train_root, 'Watermarked_image', '%s.jpg')
imageJ_path = osp.join(root, 'Watermarked_image', '%s.jpg')
# imageJ_path = osp.join(train_root, 'Watermarked_image', '%s.jpg')
target_path = osp.join(root, 'Watermark_free_image', '%s.jpg')
mask_path= osp.join(root, 'Mask', '%s.png')
watermark_path = './universal_data/test/Watermark/1.png'
img_save_path = osp.join('./universal_demo/adv_result', 'result_img', '%s.jpg')
img_vision_path = osp.join('./universal_demo/adv_result', 'result_vision', '%s.jpg')

train_ids = list()
ids = list()

for file in os.listdir(train_root + '/Watermarked_image'):
    train_ids.append(file.strip('.jpg'))

for file in os.listdir(root + '/Watermarked_image'):
    ids.append(file.strip('.jpg'))



i = 0
j = 0
ans_ssim=0.0
ans_psnr=0.0
rmse_all=0.0
rmse_in=0.0
d_sum = 0.0

cifar10_mean = (0.0, 0.0, 0.0)
cifar10_std = (1.0, 1.0, 1.0)
mu = torch.tensor(cifar10_mean).view(3,1,1).cuda()
std = torch.tensor(cifar10_std).view(3,1,1).cuda()

upper_limit = ((1 - mu)/ std)
lower_limit = ((0 - mu)/ std)
epsilon = (50/ 255.) / std
start_epsilon = (0 / 255.) / std
step_alpha = ( 2/ 255.) / std

def clamp(X, lower_limit, upper_limit):
    return torch.max(torch.min(X, upper_limit), lower_limit)

for img_id in train_ids:
    i += 1
    print('train:',i)
    transform_norm = transforms.Compose([transforms.ToTensor()])
    img_J = Image.open(trainimage_path % img_id)
    img_source = transform_norm(img_J)
    img_source = torch.unsqueeze(img_source.cuda(), 0)
    mask =  Image.open(mask_path % img_id)
    mask = transform_norm(mask)
    mask = torch.unsqueeze(mask.cuda(), 0)
    logo = Image.open(watermark_path)
    logo = transform_norm(logo)
    logo = torch.unsqueeze(logo.cuda(), 0)
    if i == 1:
        delta = torch.zeros_like(img_source).cuda()
        # for j in range(len(epsilon)):
        #     delta[:, j, :, :].uniform_(-start_epsilon[j][0][0].item(), start_epsilon[j][0][0].item())
    delta.requires_grad = True
    for k in range(20):
        clean_image = clamp(img_source + delta*mask, lower_limit, upper_limit)
        start_pred_target, start_mask, start_alpha, start_w, start_I_watermark = G(clean_image)
        loss = F.mse_loss(start_pred_target*mask,clean_image*mask)
        loss.backward()
        print(loss)
        grad = -delta.grad.detach()
        d = delta
        d = clamp(d + step_alpha * torch.sign(grad), -epsilon, epsilon)
        delta.data = d
        delta.grad.zero_()
    d = delta.data
    d_sum += d
    if i>20:
        break
d_ave = d_sum/i
d_record = d_ave.detach().cpu().numpy()
np.save('./delta.npy',d_record)
# print(d_ave.max())

#     adv_img = clamp(img_source + d, lower_limit, upper_limit)


for img_id in ids:
    j += 1
    print('test:',j)
    transform_norm = transforms.Compose([transforms.ToTensor()])
    img_J = Image.open(imageJ_path % img_id)
    img_source = transform_norm(img_J)
    img_source = torch.unsqueeze(img_source.cuda(), 0)
    clean_image = clamp(img_source + d_ave, lower_limit, upper_limit)
    pred_target, mask, alpha, w, I_watermark = G(clean_image)
    p0 = torch.squeeze(clean_image)
    p1 = torch.squeeze(pred_target)
    p2 = mask
    p3 = torch.squeeze(w * mask)
    p2 = torch.squeeze(torch.cat([p2, p2, p2], 1))

    p0 = torch.cat([p0, p1], 1)
    p2 = torch.cat([p2, p3], 1)
    p0 = torch.cat([p0, p2], 2)
    p0 = transforms.ToPILImage()(p0.detach().cpu()).convert('RGB')
    pred_target = transforms.ToPILImage()(p1.detach().cpu()).convert('RGB')
    pred_target.save(img_save_path % img_id)

    '计算指标'
    g_img = np.array(pred_target)
    real_img = cv2.imread(target_path % img_id)
    mask = Image.open(mask_path % img_id)
    mask = numpy.asarray(mask) / 255.0
    ans_psnr += psnr(g_img, real_img)
    mse_all = mse(g_img, real_img)
    mse_in = mse(g_img * mask, real_img * mask) * mask.shape[0] * mask.shape[1] * mask.shape[2] / (
            numpy.sum(mask) + 1e-6)
    rmse_all += numpy.sqrt(mse_all)
    rmse_in += numpy.sqrt(mse_in)
    real_img_tensor = torch.from_numpy(real_img).float().unsqueeze(0) / 255.0
    g_img_tensor = torch.from_numpy(g_img).float().unsqueeze(0) / 255.0
    real_img_tensor = real_img_tensor.cuda()
    g_img_tensor = g_img_tensor.cuda()
    ans_ssim += pytorch_ssim.ssim(g_img_tensor, real_img_tensor)
    if j <= 20:
        p0.save(img_vision_path % img_id)

print('psnr: %.4f, ssim: %.4f, rmse_in: %.4f, rmse_all: %.4f' % (ans_psnr / j, ans_ssim / j, rmse_in / j, rmse_all / j))


#
#
#
