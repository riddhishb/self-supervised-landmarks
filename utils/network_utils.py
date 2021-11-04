import torch.nn.functional as F
import torch
import os.path as osp
import numpy as np
from torch.autograd import Variable
import math
import torch.nn as nn

def distance(input_dt, input_pts, imgL, imgW, imgH, dim='3d'):
    # remove the corner positions
    input_pts[:, :, 0] = (input_pts[:, :, 0]+1)*0.5*(imgL - 1)
    input_pts[:, :, 1] = (input_pts[:, :, 1]+1)*0.5*(imgW - 1)
    input_pts[:, :, 2] = (input_pts[:, :, 2]+1)*0.5*(imgH - 1)
    inpts = torch.round(input_pts).long()
    btsz = inpts.shape[0]
    # concat the torch arange so it fits into a 4D array
    if dim == '3d':
        import pdb; pdb.set_trace()
        all_dist = input_dt[torch.arange(btsz), 0, inpts[torch.arange(btsz), :, 0], inpts[torch.arange(btsz), :, 1], inpts[torch.arange(btsz), :, 2]]
    else:
        all_dist = input_dt[torch.arange(btsz), 0, inpts[torch.arange(btsz), :, 0], inpts[torch.arange(btsz), :, 1]]
    return all_dist

def grid_sample(input, grid, canvas = None):
    '''
    Sampling the grid from the 2D data
    '''
    output = F.grid_sample(input, grid)
    if canvas is None:
        return output
    else:
        input_mask = Variable(input.data.new(input.size()).fill_(1))
        output_mask = F.grid_sample(input_mask, grid)
        padded_output = output * output_mask + canvas * (1 - output_mask)
        return padded_output

def dice_loss(x,reg_x):
    
    numerator = 2 * torch.sum(F.sigmoid(reg_x) * x)
    denominator = torch.sum(F.sigmoid(reg_x) + x)
    return 1 - (numerator + 1) / (denominator + 1)



def atanh(x):
    return 0.5*torch.log((1+x)/(1-x))

def save_checkpoint(model, optim, saveDir, fnm, step, silent=True):
    model_states = {'net': model.state_dict(),}
    optim_states = {'optim': optim.state_dict(),}
    states = {'iter':step,
                  'model_states':model_states,
                  'optim_states':optim_states}

    file_path = osp.join(saveDir, fnm)
    with open(file_path, mode='wb+') as f:
        torch.save(states, f)
    if not silent:
        print("=> saved checkpoint '{}' (iter {})".format(file_path, step))

def load_checkpoint(model, optim, saveDir, fnm, step, silent=True):
    try:
        file_path = osp.join(saveDir, fnm)
        if osp.isfile(file_path):
            checkpoint = torch.load(file_path)
            model.load_state_dict(checkpoint['model_states']['net'])
            optim.load_state_dict(checkpoint['optim_states']['optim'])
            print("=> loaded checkpoint '{}'".format(file_path))
            return model, optim
        else:
            print("=> no checkpoint found at '{}'".format(file_path))
    except Exception as e:
        print('model couldnt load')

def online_mean_and_sd(loader):
    """Compute the mean and sd in an online fashion

        Var[x] = E[X^2] - E^2[X]
    """
    cnt = 0
    fst_moment = torch.empty(1)
    snd_moment = torch.empty(1)

    for [images, x, xx] in loader:
        images = images.float()
        # [imgS, imgT] = images.split([1, 1], 1)
        # print(images.shape)
        b, c, h, w = images.shape
        nb_pixels = b * h * w
        sum_ = torch.sum(images, dim=[0, 2, 3])
        sum_of_square = torch.sum(images ** 2, dim=[0, 2, 3])
        fst_moment = (cnt * fst_moment + sum_) / (cnt + nb_pixels)
        
        snd_moment = (cnt * snd_moment + sum_of_square) / (cnt + nb_pixels)

        cnt += nb_pixels

    return fst_moment, torch.sqrt(snd_moment - fst_moment ** 2)

def online_mean_and_sd_3d(loader):
    """Compute the mean and sd in an online fashion

        Var[x] = E[X^2] - E^2[X]
    """
    cnt = 0
    fst_moment = torch.empty(1)
    snd_moment = torch.empty(1)

    for [images,sx] in loader:
        images = images.float()
        b, c, l, h, w = images.shape
        nb_pixels = b * h * w * l
        sum_ = torch.sum(images, dim=[0, 2, 3, 4])
        sum_of_square = torch.sum(images ** 2, dim=[0, 2, 3, 4])
        fst_moment = (cnt * fst_moment + sum_) / (cnt + nb_pixels)
        snd_moment = (cnt * snd_moment + sum_of_square) / (cnt + nb_pixels)

        cnt += nb_pixels

    return fst_moment, torch.sqrt(snd_moment - fst_moment ** 2)