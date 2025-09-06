# -*- coding: utf-8 -*-
import collections
import logging
import math
import os
from datetime import datetime

import dateutil.tz
import torch
import numpy as np
from PIL import Image
from scipy.signal import convolve2d
from torch.utils.data import Dataset
import imageio
import cv2
import glob
import random
from functools import lru_cache
import torch.nn.functional as F

#########################################################################
class SlowDataset(Dataset):
    def __init__(self, args, train=True):  # __init__是初始化该类的一些基础参数
        super(SlowDataset, self).__init__()
        self.args = args
        self.train = train
        self.image_folder = os.path.join('.', args.data_dir)
        self.bin_image_folder = os.path.join('.', args.data_dir+'bin')
        if not os.path.exists(self.bin_image_folder): os.makedirs(self.bin_image_folder, exist_ok=True)
        self.ext = '/*%s' % args.ext
        self.file_names = glob.glob(self.image_folder + self.ext)
        self.bin_file_names = list()
        self.prepare_cache()

    def prepare_cache(self):
        for fname in self.file_names:
            bin_fname = fname.replace(self.image_folder, self.bin_image_folder).replace(self.args.ext, '.npy')
            self.bin_file_names.append(bin_fname)
            if not os.path.exists(bin_fname):
                img = imageio.imread(fname)
                np.save(bin_fname, img)
                print(f'{bin_fname} prepared!')

    def __len__(self):  # 返回整个数据集的大小
        if self.args.data_dir=="cs_train400_png":
            # return len(self.file_names) * 200
            return len(self.file_names) * 20  # 20
        else:
            return len(self.file_names) * 900

    @lru_cache(maxsize=400)
    def get_ndarray(self, fname):
        return np.load(fname)

    def __getitem__(self, index):
        rgb_range = self.args.rgb_range
        n_channels = self.args.n_channels

        img = torch.Tensor(self.get_ndarray(self.bin_file_names[index % len(self.file_names)]))

        if img.numpy().ndim == 2:
            img = img.unsqueeze(2)

        c = img.shape[2]
        # input rgb image output y chanel
        if n_channels == 1 and c == 3:
            img = rgb2ycbcr(img)[:, :, 0].unsqueeze(2)
        elif n_channels == 3 and c == 1:
            img = img.repeat(1, 1, 3)

        w, h, _ = img.shape
        th = tw = self.args.patch_size
        i = random.randint(0, w - tw)
        j = random.randint(0, h - th)
        img = img[i:i + tw, j:j + th, :]
        img_tensor = img.permute(2, 0, 1)
        img_tensor = img_tensor * rgb_range / 255.0
        img_tensor = self.augment(img_tensor)
        return img_tensor

    def augment(self, img, hflip=True, rot=True):

        hflip = hflip and random.random() < 0.5
        vflip = rot and random.random() < 0.5
        rot90 = rot and random.random() < 0.5

        if hflip:
            img = img.flip([1])
        if vflip:
            img = img.flip([0])
        if rot90:
            img = img.permute(0, 2, 1)

        return img

#########################################################################
def imread_CS_py_paddu(imgName, device, block_size):
    Img = cv2.imread(imgName, 1)
    Img_rec_yuv = cv2.cvtColor(Img, cv2.COLOR_BGR2YCrCb)
    Iorg = np.array(Image.open(imgName), dtype='float32')  # 读图
    if len(Iorg.shape) == 3: #rgb转y
        Iorg = test_rgb2ycbcr(Iorg)

    [row, col] = Iorg.shape  # 图像的 形状
    row_pad = block_size-np.mod(row,block_size)  # 求余数操作
    col_pad = block_size-np.mod(col,block_size)  # 求余数操作，用于判断需要补零的数量

    # 计算上下、左右分别需要补的0数量（对称补零）
    pad_top = row_pad // 2
    pad_left = col_pad // 2

    # # 单边小于10个pixel则不采用两边平摊padding
    if  pad_top<10:
        pad_top=0
    if pad_left<10:
        pad_left =0

    # 进行上下补零
    Ipad = F.pad(torch.from_numpy(Iorg).float().to(device), (pad_left, col_pad-pad_left, pad_top, row_pad-pad_top), mode='constant', value=0)
    return [Iorg, row, col, Ipad, pad_top, pad_left,Img_rec_yuv]


def test_rgb2ycbcr(img, only_y=True):
    in_img_type = img.dtype
    img.astype(np.float32)
    # convert
    if only_y:
        rlt = np.dot(img, [65.481, 128.553, 24.966]) / 255.0 + 16.0
    else:
        rlt = np.matmul(img, [[65.481, -37.797, 112.0], [128.553, -74.203, -93.786],
                              [24.966, 112.0, -18.214]]) / 255.0 + [16, 128, 128]
    rlt = rlt.round()
    return rlt.astype(in_img_type)


def img2patches_torch(imgs: torch.Tensor, patch_size: tuple, stride_size: tuple):
    """
    PyTorch GPU加速版：将图像切分为patch。

    Args:
        imgs: Tensor, shape (H,W), (C,H,W), or (B,C,H,W)，值应在GPU上
        patch_size: (patch_h, patch_w)
        stride_size: (stride_h, stride_w)

    Returns:
        patches: Tensor, shape (N_patches_total, C, patch_h, patch_w)
    """
    p_h, p_w = patch_size
    s_h, s_w = stride_size

    # 标准化输入形状到 (B, C, H, W)
    if imgs.ndim == 2:
        imgs = imgs.unsqueeze(0).unsqueeze(0)        # -> (1,1,H,W)
    elif imgs.ndim == 3:
        imgs = imgs.unsqueeze(0)                     # -> (1,C,H,W)
    elif imgs.ndim != 4:
        raise ValueError(f"Unsupported input shape: {imgs.shape}")

    # unfold 提取patch：(B, C * p_h * p_w, L)
    patches = imgs.unfold(2, p_h, s_h).unfold(3, p_w, s_w)  # (B, C, n_h, n_w, p_h, p_w)

    # reshape 为 (N_patches_total, C, p_h, p_w)
    B, C, n_h, n_w, p_h, p_w = patches.shape
    patches = patches.permute(0, 2, 3, 1, 4, 5).contiguous()  # (B, n_h, n_w, C, p_h, p_w)
    patches = patches.view(-1, C, p_h, p_w)  # flatten 所有patch

    return patches


def unpatch2d_torch(patches, imsize: tuple, stride_size: tuple):
    """
    PyTorch GPU版本：从patches重构图像，带重叠加权平均。

    Args:
        patches: (N_patches, C, p_h, p_w) torch.Tensor on GPU
        imsize: (H, W)
        stride_size: (s_h, s_w)

    Returns:
        Reconstructed image tensor of shape (B, C, H, W)
    """
    assert patches.ndim == 4, "Expected shape (N, C, p_h, p_w)"
    N, C, p_h, p_w = patches.shape
    H, W = imsize
    s_h, s_w = stride_size

    # 计算每张图需要的 patch 数量
    n_patches_y = (H - p_h) // s_h + 1
    n_patches_x = (W - p_w) // s_w + 1
    n_patches_per_img = n_patches_y * n_patches_x
    B = N // n_patches_per_img

    # 把 patch reshape 成 (B, C * p_h * p_w, n_patches_y * n_patches_x)
    patches = patches.view(B, n_patches_per_img, C, p_h, p_w)
    patches = patches.permute(0, 2, 3, 4, 1)  # (B, C, p_h, p_w, n_patches)
    patches = patches.reshape(B, C * p_h * p_w, n_patches_y * n_patches_x)

    # 使用 fold 重建图像
    output = torch.nn.functional.fold(
        patches,
        output_size=(H, W),
        kernel_size=(p_h, p_w),
        stride=(s_h, s_w)
    )

    # 构建一个全1的 patch 用于计算加权平均
    ones = torch.ones((B, C * p_h * p_w, n_patches_y * n_patches_x),
                      dtype=patches.dtype, device=patches.device)

    weight = torch.nn.functional.fold(
        ones,
        output_size=(H, W),
        kernel_size=(p_h, p_w),
        stride=(s_h, s_w)
    )

    output = output / weight.clamp(min=1e-8)  # 避免除0

    return output  # shape: (B, C, H, W)

#####################################################################################
def psnr(img1, img2):
    img1.astype(np.float32)
    img2.astype(np.float32)
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return 100
    PIXEL_MAX = 255.0
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))
