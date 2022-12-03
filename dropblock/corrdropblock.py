import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.distributions import Bernoulli
import sys
import math


class CorrDropBlockDB(nn.Module):

    def __init__(self, drop_prob, block_size):
        super(CorrDropBlockDB, self).__init__()

        self.drop_prob = drop_prob
        self.block_size = block_size
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

    def max(self,input):
        N,C,H,W = input.shape
        max_vec = torch.max(input, dim=1, keepdim=True)[0].view(N, H * W)
        return max_vec

    def mean(self,input):
        N, C, H, W = input.shape
        max_vec = torch.mean(input, dim=1, keepdim=True).view(N, H * W)
        return max_vec


    def ortho(self,input):
        N,C,H,W = input.shape
        vec = torch.transpose(input.view(N, C, H*W), 1, 2)
        vec = vec / torch.sqrt(torch.sum(torch.pow(vec, 2), dim=-1, keepdim=True)+1e-8)
        P = torch.abs(torch.matmul(vec, torch.transpose(vec, 1, 2)) -
                      torch.eye(H*W).to(input.device).view(1,H*W,H*W))
        rank = torch.sum(P, dim=-1)/(H*W)
        rank = rank.view(N, H*W)
        return rank

    def forward(self, x):
        # shape: (bsize, channels, height, width)

        assert x.dim() == 4, \
            "Expected input with 4 dimensions (bsize, channels, height, width)"

        if self.training:
            self.step()
        if not self.training or self.drop_prob == 0:
            return x
        else:
            # self.step()
            # sample from a mask
            # print (x.shape)
            N, C, H, W = x.shape
            # print(x.shape)
            # print(x)
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
            # x_norm = self.normalize(x)
            if H % self.block_size == 0:
                pad = 0
            else:
                pad = math.ceil((self.block_size - H % self.block_size) / 2)
            pool = nn.AvgPool2d(kernel_size=self.block_size, stride=self.block_size, padding=pad)
            x_gather = pool(x)
            # print (x_gather.shape)
            N1, C1, H1, W1 = x_gather.shape
            # self.inner_shape = (C1, H1, W1)
            # print ('inner shape')
            block_gamma = self._compute_gamma(x, [H1, W1])
            # print(x_gather)
            # ortho
            max_vec = self.ortho(x_gather)
            # print(max_vec)

            x_max = F.softmax(max_vec, dim=-1).view(N1, 1, H1, W1)
            # print(x_max)

            x_mask = 1 - Bernoulli(x_max).sample().to(x.device)
            gamma = block_gamma / (x_mask.sum() / x_mask.numel())
            gamma = torch.clamp(gamma, 0, 1)
            mask = Bernoulli(gamma).sample((x_gather.shape[0], 1, H1, W1)).float().to(x.device)

            gather_mask = 1 - (x_mask * mask)
            block_mask = F.interpolate(gather_mask, scale_factor=self.block_size, mode='nearest')
            delta = int((block_mask.shape[2] - H) / 2)
            block_mask = block_mask[:, :, delta:delta + H, delta:delta + W]
            assert block_mask.shape[2] == H

            out = x * block_mask
            out = out * block_mask.numel() / block_mask.sum()

            # channel attention
            # N, C, H, W = x.shape
            # out = out * channel_attention_map.view(N, C, 1, 1)
            return out
            # return out,orth_vis


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

        if height_to_crop != 0:
            block_mask = block_mask[:, :, :-height_to_crop, :]

        if width_to_crop != 0:
            block_mask = block_mask[:, :, :, :-width_to_crop]

        block_mask = (block_mask >= 1).to(device=block_mask.device, dtype=block_mask.dtype)
        block_mask = 1 - block_mask.squeeze(1)

        return block_mask

    def _compute_gamma(self, x, mask_sizes):
        feat_area = x.shape[-2] * x.shape[-1]
        mask_area = mask_sizes[-2] * mask_sizes[-1]
        return (self.drop_prob / (self.block_size ** 2)) * (feat_area / mask_area)

def count_your_model(model, x, y):
    print (x[0].shape)
    # assert False
    # print (y)
    flops = 6
    N, C, H, W = x[0].shape

    # pool_flops = self.block_size * self.block_size * C1 * H1 * W1 + 6
    H_1 = W_1 = np.ceil(H/model.block_size)
    C = x[0].shape[1]
    #pooling
    flops += 2 * C * H * W
    #orth
    flops += 2 * H_1 * W_1 * C + 3 * H_1 * W_1 + H_1 * W_1 * C * H_1 * W_1
    #sampling
    flops += 4 * H_1 * W_1
    #point-wise multiplication
    flops += H * W * C
    model.total_ops = torch.Tensor([int(flops)])
    # return flops

