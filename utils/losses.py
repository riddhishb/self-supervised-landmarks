import torch.nn.functional as F
import torch
import os.path as osp
import numpy as np
from torch.autograd import Variable
import math
import torch.nn as nn

def pdist_squared(x):
    xx = (x**2).sum(dim=1).unsqueeze(2)
    yy = xx.permute(0, 2, 1)
    dist = xx + yy - 2.0 * torch.bmm(x.permute(0, 2, 1), x)
    dist[dist != dist] = 0
    dist = torch.clamp(dist, 0.0, 10000)
    return dist

def MINDSSC(img, radius=2, dilation=2):
    # kernel size
    kernel_size = radius * 2 + 1

    # define start and end locations for self-similarity pattern
    six_neighbourhood = torch.Tensor([[0,1,1],
                                      [1,1,0],
                                      [1,0,1],
                                      [1,1,2],
                                      [2,1,1],
                                      [1,2,1]]).long()

    # squared distances
    dist = pdist_squared(six_neighbourhood.t().unsqueeze(0)).squeeze(0)
    
    # define comparison mask
    x, y = torch.meshgrid(torch.arange(6), torch.arange(6))
    mask = ((x > y).view(-1) & (dist == 2).view(-1))
    
    # build kernel
    idx_shift1 = six_neighbourhood.unsqueeze(1).repeat(1,6,1).view(-1,3)[mask,:]
    idx_shift2 = six_neighbourhood.unsqueeze(0).repeat(6,1,1).view(-1,3)[mask,:]
    
    mshift1 = torch.zeros(12, 1, 3, 3, 3).cuda()
    mshift1.view(-1)[torch.arange(12) * 27 + idx_shift1[:,0] * 9 + idx_shift1[:, 1] * 3 + idx_shift1[:, 2]] = 1
    mshift2 = torch.zeros(12, 1, 3, 3, 3).cuda()
    mshift2.view(-1)[torch.arange(12) * 27 + idx_shift2[:,0] * 9 + idx_shift2[:, 1] * 3 + idx_shift2[:, 2]] = 1
    rpad1 = nn.ReplicationPad3d(dilation)
    rpad2 = nn.ReplicationPad3d(radius)

    # compute patch-ssd
    ssd = F.avg_pool3d(rpad2((F.conv3d(rpad1(img), mshift1, dilation=dilation) - F.conv3d(rpad1(img), mshift2, dilation=dilation)) ** 2), kernel_size, stride=1)

    # MIND equation
    mind = ssd - torch.min(ssd, 1, keepdim=True)[0]
    mind_var = torch.mean(mind, 1, keepdim=True)
    # import pdb; pdb.set_trace()
    mind_var = torch.clamp(mind_var, mind_var.mean().item()*0.001, mind_var.mean().item()*1000)
    mind /= mind_var
    mind = torch.exp(-mind)

    #permute to have same ordering as C++ code
    mind = mind[:, torch.Tensor([6, 8, 1, 11, 2, 10, 0, 7, 9, 4, 5, 3]).long(), :, :, :]

    return mind

def MINDSSC_2d(img, radius=2, dilation=2):
#     # see http://mpheinrich.de/pub/miccai2013_943_mheinrich.pdf for details on the MIND-SSC descriptor

    # kernel size
    kernel_size = radius * 2 + 1

    # define start and end locations for self-similarity pattern
    six_neighbourhood = torch.Tensor([[0,1],
                                      [1,0],
                                      [1,2],
                                      [2,1]]).long()

    # squared distances
    dist = pdist_squared(six_neighbourhood.t().unsqueeze(0)).squeeze(0)
    
    # define comparison mask
    x, y = torch.meshgrid(torch.arange(4), torch.arange(4))
    mask = ((x > y).view(-1) & (dist == 2).view(-1))

    # build kernel
    idx_shift1 = six_neighbourhood.unsqueeze(1).repeat(1,4,1).view(-1,2)[mask,:]
    idx_shift2 = six_neighbourhood.unsqueeze(0).repeat(4,1,1).view(-1,2)[mask,:]
    # import pdb; pdb.set_trace()
    mshift1 = torch.zeros(8, 1, 2, 2).cuda()
    mshift1.view(-1)[torch.arange(4) * 4 + idx_shift1[:,0] * 2 + idx_shift1[:, 1]] = 1
    mshift2 = torch.zeros(8, 1, 2, 2).cuda()
    mshift2.view(-1)[torch.arange(4) * 4 + idx_shift2[:,0] * 2 + idx_shift2[:, 1]] = 1
    rpad1 = nn.ReplicationPad2d(dilation)
    rpad2 = nn.ReplicationPad2d(radius)
    # import pdb; pdb.set_trace()
    # compute patch-ssd
    ssd = F.avg_pool2d(rpad2((F.conv2d(rpad1(img), mshift1, dilation=dilation) - F.conv2d(rpad1(img), mshift2, dilation=dilation)) ** 2), kernel_size, stride=1)

    # MIND equation
    mind = ssd - torch.min(ssd, 1, keepdim=True)[0]
    mind_var = torch.mean(mind, 1, keepdim=True)
    mind_var = torch.clamp(mind_var, mind_var.mean().item()*0.001, mind_var.mean().item()*1000)
    # mind = nn.functional.relu(mind - 1e-20)
    mind /= mind_var
    mind = torch.exp(-mind)
    # import pdb; pdb.set_trace()
    #permute to have same ordering as C++ code
    mind = mind[:, torch.Tensor([6, 1, 2, 0, 7, 4, 5, 3]).long(), :, :]

    return mind

def mind_loss_3d(x, y, msk=None):
    if msk is not None:
        # import pdb; pdb.set_trace()
        msk = msk.unsqueeze(1).unsqueeze(1).unsqueeze(1).unsqueeze(1).float()
        msk = msk.repeat(1, 12, x.shape[2], x.shape[3], x.shape[4])
        # import pdb; pdb.set_trace()
        return torch.mean( (MINDSSC(x)*msk - MINDSSC(y)*msk) ** 2 )
    else:
        return torch.mean( (MINDSSC(x) - MINDSSC(y)) ** 2 )

def mind_loss_2d(x, y, msk=None):
    # import pdb; pdb.set_trace()
    if msk is not None:
        # import pdb; pdb.set_trace()
        msk = msk.unsqueeze(1).unsqueeze(1).unsqueeze(1).float()
        msk = msk.repeat(1, 8, 177, 210)
        return torch.mean( (MINDSSC_2d(x, radius=2, dilation=1)*msk - MINDSSC_2d(y, radius=2, dilation=1)*msk) ** 2 )
    else:
        return torch.mean( (MINDSSC_2d(x, radius=2, dilation=1) - MINDSSC_2d(y, radius=2, dilation=1)) ** 2 )

def ncc_loss(I, J, device, win=None):
    """
    calculate the normalize cross correlation between I and J
    assumes I, J are sized [batch_size, *vol_shape, nb_feats]
    """

    ndims = len(list(I.size())) - 2
    assert ndims in [1, 2, 3], "volumes should be 1 to 3 dimensions. found: %d" % ndims

    if win is None:
        win = [3] * ndims

    conv_fn = getattr(F, 'conv%dd' % ndims)
    sum_filt = torch.ones([1, 1, *win]).to(device)

    pad_no = math.floor(win[0]/2)

    if ndims == 1:
        stride = (1)
        padding = (pad_no)
    elif ndims == 2:
        stride = (1,1)
        padding = (pad_no, pad_no)
    else:
        stride = (1,1,1)
        padding = (pad_no, pad_no, pad_no)
    
    # I_var, J_var, cross = compute_local_sums(I, J, sum_filt, stride, padding, win)
    I2 = I * I
    J2 = J * J
    IJ = I * J

    I_sum = conv_fn(I, sum_filt, stride=stride, padding=padding)
    J_sum = conv_fn(J, sum_filt, stride=stride, padding=padding)
    I2_sum = conv_fn(I2, sum_filt, stride=stride, padding=padding)
    J2_sum = conv_fn(J2, sum_filt, stride=stride, padding=padding)
    IJ_sum = conv_fn(IJ, sum_filt, stride=stride, padding=padding)

    win_size = np.prod(win) * 1
    u_I = I_sum / win_size
    u_J = J_sum / win_size

    cross = IJ_sum - u_J * I_sum - u_I * J_sum + u_I * u_J * win_size
    I_var = I2_sum - 2 * u_I * I_sum + u_I * u_I * win_size
    J_var = J2_sum - 2 * u_J * J_sum + u_J * u_J * win_size
    # msk = I_var*J_var != 0 
    cc = cross*cross / (I_var*J_var + 0.00000001)
    # print('This')
    # print(cross, I_var, J_var)
    MSE = F.mse_loss(I, J, reduction="mean")
    denon = F.mse_loss(0*J, I, reduction="mean")
    relMSE = MSE/F.mse_loss(0*J, I, reduction="mean")
    loss = 1 - torch.mean(cc) + 0.00001*MSE
    return loss, relMSE, denon


def seg_loss(x, reg_x, device):
    BCE = -1*(x.to(device)*torch.log(reg_x + 0.000000001) + (1-x.to(device))*torch.log(1 - reg_x + 0.000000001))
    BCE = BCE.mean()
    MSE = F.mse_loss(reg_x, x.to(device), reduction="mean")
    denon = F.mse_loss(0*reg_x, x, reduction="mean")
    relMSE = MSE/F.mse_loss(0*reg_x, x, reduction="mean")
    return BCE, relMSE, denon

def l2_loss(x, reg_x, device):
    MSE = F.mse_loss(reg_x.to(device), x.to(device), reduction="mean")
    denon = F.mse_loss(0*reg_x, x, reduction="mean")
    relMSE = MSE/F.mse_loss(0*reg_x, x, reduction="mean")
    return MSE, relMSE, denon

def cond_num_loss(A, device):
    # print(x.shape, reg_x.shape)
    B = torch.transpose(A, 1, 2)
    orthoA = torch.bmm(A, B).to(device)
    x = torch.eye(A.size(1)).to(device)
    x = x.reshape((1, x.size(0), x.size(1)))
    y = x.repeat((A.size(0), 1, 1))
    out_loss = F.mse_loss(orthoA, y, reduction='mean')
    return out_loss

def cond_num_loss_v2(A, device):
    # print(x.shape, reg_x.shape)
    A =A.to(device)
    B = torch.inverse(A).to(device)
    nA = torch.norm(A, p='fro', dim=[1,2])
    nB = torch.norm(B, p='fro', dim=[1,2])
    # print(nA.shape, nB.shape)
    kb = (nA*nB).mean()
    return kb

def l1_loss(x, reg_x, device):
    # print(x.shape, reg_x.shape)
    loss = F.l1_loss(reg_x.to(device), x.to(device), reduction='mean')
    MSE = F.mse_loss(reg_x.to(device), x.to(device), reduction="mean")
    denon = F.mse_loss(0*reg_x.to(device), x.to(device), reduction="mean")
    relMSE = MSE/F.mse_loss(0*reg_x.to(device), x.to(device), reduction="mean")
    return loss, relMSE, denon

