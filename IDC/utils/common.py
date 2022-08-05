import random
import numpy as np
import torch
import torch.nn.functional as F
from torchvision.utils import save_image

IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp')
MEANS = {'cifar': [0.4914, 0.4822, 0.4465], 'imagenet': [0.485, 0.456, 0.406]}
STDS = {'cifar': [0.2023, 0.1994, 0.2010], 'imagenet': [0.229, 0.224, 0.225]}
MEANS['cifar10'] = MEANS['cifar']
STDS['cifar10'] = STDS['cifar']
MEANS['cifar100'] = MEANS['cifar']
STDS['cifar100'] = STDS['cifar']
MEANS['svhn'] = [0.4377, 0.4438, 0.4728]
STDS['svhn'] = [0.1980, 0.2010, 0.1970]
MEANS['mnist'] = [0.1307]
STDS['mnist'] = [0.3081]
MEANS['fashion'] = [0.2861]
STDS['fashion'] = [0.3530]


def set_seed(seed=0):
    """set random seed if seed<0 close the function"""
    if seed >= 0:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.enabled = True


def dataset_mean_var_init():
    mean_torch = {}
    std_torch = {}
    for key, val in MEANS.items():
        mean_torch[key] = torch.tensor(val).reshape(1, len(val), 1, 1)
    for key, val in STDS.items():
        std_torch[key] = torch.tensor(val).reshape(1, len(val), 1, 1)
    return mean_torch, std_torch


def img_denormlaize(img, dataname='imagenet'):
    """Scaling and shift a batch of images (NCHW)
    """
    mean = MEANS[dataname]
    std = STDS[dataname]
    nch = img.shape[1]

    mean = torch.tensor(mean, device=img.device).reshape(1, nch, 1, 1)
    std = torch.tensor(std, device=img.device).reshape(1, nch, 1, 1)

    return img * std + mean


def save_img(save_dir, img, unnormalize=True, max_num=200, size=64, nrow=10, dataname='imagenet'):
    img = img[:max_num].detach()
    if unnormalize:
        img = img_denormlaize(img, dataname=dataname)
    img = torch.clamp(img, min=0., max=1.)

    if img.shape[-1] > size:
        img = F.interpolate(img, size)
    save_image(img.cpu(), save_dir, nrow=nrow)
