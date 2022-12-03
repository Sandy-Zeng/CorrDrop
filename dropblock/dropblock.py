import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.distributions import Bernoulli
import sys


class DropBlock2D(nn.Module):
    r"""Randomly zeroes 2D spatial blocks of the input tensor.
    As described in the paper
    `DropBlock: A regularization method for convolutional networks`_ ,
    dropping whole blocks of feature map allows to remove semantic
    information as compared to regular dropout.
    Args:
        drop_prob (float): probability of an element to be dropped.
        block_size (int): size of the block to drop
    Shape:
        - Input: `(N, C, H, W)`
        - Output: `(N, C, H, W)`
    .. _DropBlock: A regularization method for convolutional networks:
       https://arxiv.org/abs/1810.12890
    """

    def __init__(self, drop_prob, block_size, att=False):
        super(DropBlock2D, self).__init__()

        self.drop_prob = drop_prob
        self.block_size = block_size
        self.att = att
        self.i = 0
        self.drop_values = np.linspace(start=0, stop=drop_prob, num=200)

    def step(self):
        if self.i < len(self.drop_values):
            self.drop_prob = self.drop_values[self.i]
        self.i += 1

    def normalize(self, input):
        min_c, max_c = input.min(1, keepdim=True)[0], input.max(1, keepdim=True)[0]
        input_norm = (input - min_c) / (max_c - min_c + 1e-8)
        return input_norm

    def forward(self, x):
        # shape: (bsize, channels, height, width)
        # print (x.shape)

        assert x.dim() == 4, \
            "Expected input with 4 dimensions (bsize, channels, height, width)"

        if self.training:
            self.step()
        if not self.training or self.drop_prob == 0.:
            mask = torch.ones_like(x)
            return x
        else:
            # sample from a mask
            # print (x.shape)
            # count_block,sim = self.selective_block(x)
            # if self.block_size > x.shape[-1]:
            #     self.block_size = x.shape[-1] - 1
            N, C, H, W = x.shape
            if self.block_size > H:
                self.block_size = H - 1

            mask_reduction = self.block_size // 2
            mask_height = x.shape[-2] - mask_reduction
            mask_width = x.shape[-1] - mask_reduction
            mask_sizes = [mask_height, mask_width]

            if any([x <= 0 for x in mask_sizes]):
                raise ValueError('Input of shape {} is too small for block_size {}'
                                 .format(tuple(x.shape), self.block_size))

            # get gamma value
            gamma = self._compute_gamma(x, mask_sizes)
            if self.att:
                mask = Bernoulli(gamma).sample((x.shape[0], *mask_sizes)).float()
                # print (mask.shape)

                # place mask on input device
                mask = mask.to(x.device)

                # compute block mask
                block_mask = self._compute_block_mask(mask)

                # apply block mask
                out = x * block_mask[:, None, :, :]
                # print (block_mask.shape)
                block_vis = block_mask.view(x.shape[0],1,x.shape[2],x.shape[3])

            else:
                # sample mask
                # print (mask_sizes)
                x_norm = self.normalize(x)
                mask = Bernoulli(gamma).sample((x.shape[0], *mask_sizes)).float()

                # place mask on input device
                mask = mask.to(x.device)

                # compute block mask
                block_mask = self._compute_block_mask(mask)
                block_mask = 1 - ((1 - block_mask[:,None,:,:])*x_norm)

                # apply block mask
                out = x * block_mask
                # out = x * block_mask[:, None, :, :]
                block_vis = torch.mean(block_mask,dim=1,keepdim=True)

            # scale output
            out = out * block_mask.numel() / block_mask.sum()

            return out

    def selective_block(self, x):
        N, C, H, W = x.shape
        block_sizes = [5, 7, 9]
        pool1 = nn.AvgPool2d(kernel_size=5, stride=1, padding=2)
        pool2 = nn.AvgPool2d(kernel_size=7, stride=1, padding=3)
        pool3 = nn.AvgPool2d(kernel_size=9, stride=1, padding=4)
        x_vec = self.unit_normal(x.view(N, C, H * W))
        x1 = self.unit_normal(pool1(x).view(N, C, H * W))
        x2 = self.unit_normal(pool2(x).view(N, C, H * W))
        x3 = self.unit_normal(pool3(x).view(N, C, H * W))
        dev_x = torch.var(x_vec, dim=-1)
        # print (dev_x)
        dev_x = torch.clamp(dev_x, 0, 500)
        # print ('Block 5')
        sim_x_x1 = self.variance(x_vec, x1, dev_x)
        # print ('Block 7')
        sim_x_x2 = self.variance(x_vec, x2, dev_x)
        # print ('Block 9')
        sim_x_x3 = self.variance(x_vec, x3, dev_x)
        simList = [sim_x_x1, sim_x_x2, sim_x_x3]
        sim = torch.stack(simList, dim=0)
        # print (sim)
        max_sim, max_index = torch.min(sim, dim=0)  # larger and much similar
        block = block_sizes[max_index]
        simList = sim.detach().cpu().numpy()
        # if H==8:
        #     print (simList)
        #     print (block)
        return block, simList

    def unit_normal(self, x):
        mean = torch.mean(x, dim=1, keepdim=True)
        var = torch.var(x, dim=1, keepdim=True)
        x = (x - mean) / var
        return x

    def L2_norm(self, x):
        x_norm = x / torch.sqrt(torch.sum(torch.pow(x, 2), dim=1, keepdim=True))
        return x_norm

    def variance(self, x, y, x_driv):
        y_driv = torch.var(y, dim=-1)
        y_driv = torch.clamp(y_driv, 0, 500)
        x_norm = self.L2_norm(x)
        y_norm = self.L2_norm(y)
        L2_sim = torch.sqrt(torch.sum(torch.pow((x_norm - y_norm), 2), dim=1))
        L2_sim = torch.clamp(L2_sim, 0, 500)
        L2_sim = torch.mean(L2_sim)
        # print ('L2:',L2_sim)
        # print (L2_sim)
        dev_x_y = torch.mean(torch.abs(torch.matmul(x, torch.transpose(y, 1, 2))) / x.shape[-1], dim=-1)
        # dev_x_y = torch.var(x,y)
        # print (dev_x_y)
        dev_x_y = torch.clamp(dev_x_y, 0, 500)
        # assert False
        structural_sim = torch.mean(dev_x_y / (x_driv * y_driv))  # N,C
        # print (structural_sim)
        # print ('Struct Sim:',structural_sim)
        sim = 0.2 * L2_sim + 0.8 * structural_sim
        # print ('Sim',sim)
        return sim

    def save_mask(self, x):
        # shape: (bsize, channels, height, width)
        # print (x.shape)

        assert x.dim() == 4, \
            "Expected input with 4 dimensions (bsize, channels, height, width)"

        # sample from a mask
        # print (x.shape)
        mask_reduction = self.block_size // 2
        mask_height = x.shape[-2] - mask_reduction
        mask_width = x.shape[-1] - mask_reduction
        mask_sizes = [mask_height, mask_width]

        if any([x <= 0 for x in mask_sizes]):
            raise ValueError('Input of shape {} is too small for block_size {}'
                                 .format(tuple(x.shape), self.block_size))

        # get gamma value
        gamma = self._compute_gamma(x, mask_sizes)

        mask = Bernoulli(gamma).sample((x.shape[0], *mask_sizes)).float()

        # place mask on input device
        mask = mask.to(x.device)

        # compute block mask
        block_mask = self._compute_block_mask(mask)

        # apply block mask
        out = x * block_mask[:, None, :, :]
        block_vis = block_mask.view(x.shape[0],1,x.shape[2],x.shape[3])

        # scale output
        out = out * block_mask.numel() / block_mask.sum()
        return out,block_vis

    def _compute_block_mask(self, mask):
        block_mask = F.conv2d(mask[:, None, :, :],
                              torch.ones((1, 1, self.block_size, self.block_size)).to(
                                  mask.device),
                              padding=int(np.ceil(self.block_size / 2) + 1))

        delta = self.block_size // 2
        input_height = mask.shape[-2] + delta
        input_width = mask.shape[-1] + delta

        height_to_crop = block_mask.shape[-2] - input_height
        width_to_crop = block_mask.shape[-1] - input_width

        if height_to_crop > 0:
            block_mask = block_mask[:, :, :-height_to_crop, :]

        if width_to_crop > 0:
            block_mask = block_mask[:, :, :, :-width_to_crop]

        if height_to_crop < 0:
            height_pad = torch.zeros((block_mask.shape[0],1,-height_to_crop,block_mask.shape[-1])).cuda()
            block_mask = torch.cat([block_mask,height_pad],dim=-2)
        if width_to_crop < 0:
            width_pad = torch.zeros((block_mask.shape[0],1, block_mask.shape[-2], -width_to_crop)).cuda()
            block_mask = torch.cat([block_mask, width_pad], dim=-1)

        block_mask = (block_mask >= 1).to(device=block_mask.device, dtype=block_mask.dtype)
        block_mask = 1 - block_mask.squeeze(1)

        return block_mask

    def _compute_gamma(self, x, mask_sizes):
        feat_area = x.shape[-2] * x.shape[-1]
        mask_area = mask_sizes[-2] * mask_sizes[-1]
        return (self.drop_prob / (self.block_size ** 2)) * (feat_area / mask_area)


class DropBlock3D(DropBlock2D):
    r"""Randomly zeroes 3D spatial blocks of the input tensor.
    An extension to the concept described in the paper
    `DropBlock: A regularization method for convolutional networks`_ ,
    dropping whole blocks of feature map allows to remove semantic
    information as compared to regular dropout.
    Args:
        drop_prob (float): probability of an element to be dropped.
        block_size (int): size of the block to drop
    Shape:
        - Input: `(N, C, D, H, W)`
        - Output: `(N, C, D, H, W)`
    .. _DropBlock: A regularization method for convolutional networks:
       https://arxiv.org/abs/1810.12890
    """

    def __init__(self, drop_prob, block_size):
        super(DropBlock3D, self).__init__(drop_prob, block_size)

    def forward(self, x):
        # shape: (bsize, channels, depth, height, width)

        assert x.dim() == 5, \
            "Expected input with 5 dimensions (bsize, channels, depth, height, width)"

        if not self.training or self.drop_prob == 0.:
            return x
        else:
            mask_reduction = self.block_size // 2
            mask_depth = x.shape[-3] - mask_reduction
            mask_height = x.shape[-2] - mask_reduction
            mask_width = x.shape[-1] - mask_reduction
            mask_sizes = [mask_depth, mask_height, mask_width]

            if any([x <= 0 for x in mask_sizes]):
                raise ValueError('Input of shape {} is too small for block_size {}'
                                 .format(tuple(x.shape), self.block_size))

            # get gamma value
            gamma = self._compute_gamma(x, mask_sizes)

            # sample mask
            mask = Bernoulli(gamma).sample((x.shape[0], *mask_sizes))

            # place mask on input device
            mask = mask.to(x.device)

            # compute block mask
            block_mask = self._compute_block_mask(mask)

            # apply block mask
            out = x * block_mask[:, None, :, :, :]

            # scale output
            out = out * block_mask.numel() / block_mask.sum()

            return out

    def _compute_block_mask(self, mask):
        block_mask = F.conv3d(mask[:, None, :, :, :],
                              torch.ones((1, 1, self.block_size, self.block_size, self.block_size)).to(
                                  mask.device),
                              padding=int(np.ceil(self.block_size // 2) + 1))

        delta = self.block_size // 2
        input_depth = mask.shape[-3] + delta
        input_height = mask.shape[-2] + delta
        input_width = mask.shape[-1] + delta

        depth_to_crop = block_mask.shape[-3] - input_depth
        height_to_crop = block_mask.shape[-2] - input_height
        width_to_crop = block_mask.shape[-1] - input_width

        if depth_to_crop != 0:
            block_mask = block_mask[:, :, :-depth_to_crop, :, :]

        if height_to_crop != 0:
            block_mask = block_mask[:, :, :, :-height_to_crop, :]

        if width_to_crop != 0:
            block_mask = block_mask[:, :, :, :, :-width_to_crop]

        block_mask = (block_mask >= 1).to(device=block_mask.device, dtype=block_mask.dtype)
        block_mask = 1 - block_mask.squeeze(1)

        return block_mask

    def _compute_gamma(self, x, mask_sizes):
        feat_volume = x.shape[-3] * x.shape[-2] * x.shape[-1]
        mask_volume = mask_sizes[-3] * mask_sizes[-2] * mask_sizes[-1]
        return (self.drop_prob / (self.block_size ** 3)) * (feat_volume / mask_volume)