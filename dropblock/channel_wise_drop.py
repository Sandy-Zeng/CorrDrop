import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.distributions import Bernoulli
import sys
import math


class ChannelDrop(nn.Module):

    def __init__(self, drop_prob=0.1):
        super(ChannelDrop, self).__init__()
        self.drop_prob = drop_prob
        self.i = 0
        self.drop_values = np.linspace(start=0, stop=drop_prob, num=200)

    def step(self):
        if self.i < len(self.drop_values):
            self.drop_prob = self.drop_values[self.i]
        self.i += 1

    def ortho_channel(self, input):
        N, C, H, W = input.shape
        vec = input.view(N, C, H * W)
        vec = vec / (torch.sqrt(torch.sum(torch.pow(vec, 2), dim=-1, keepdim=True))+1e-8)
        # print(vec)
        # assert False
        P = torch.abs(torch.matmul(vec, torch.transpose(vec, 1, 2)) - torch.eye(C).to(input.device).view(1, C, C))
        # print(torch.matmul(vec, torch.transpose(vec, 1, 2)))
        # print(P)
        rank = torch.sum(P, dim=-1) / (C)
        rank = rank.view(N, C)
        # print(rank)
        return rank

    def forward(self, x):
        # shape: (bsize, channels, height, width)

        assert x.dim() == 4, \
            "Expected input with 4 dimensions (bsize, channels, height, width)"

        if self.training:
            self.step()
        if not self.training or self.drop_prob == 0:
            N, C, H, W = x.shape
            # print(self.drop_prob)
            # print(self.i)
            return x
        else:
            # self.step()
            # print(self.drop_prob)
            # get gamma value
            N, C, H, W = x.shape
            # print(x.shape)

            max_vec = self.ortho_channel(x)  # N,C
            # x_max = self.normalize(max_vec)
            x_max = F.softmax(max_vec, -1).view(N, C)
            # print(x_max)
            # assert False
            x_mask = 1 - Bernoulli(x_max).sample().to(x.device)
            gamma = self.drop_prob / (x_mask.sum() / x_mask.numel() + 1e-12)
            # assert False
            gamma = gamma.cpu().data.item()
            gamma = np.clip(gamma, 0, 1)
            # print(gamma)
            mask = Bernoulli(gamma).sample((N, C)).float().to(x.device)
            block_mask = (1 - x_mask * mask).view(N, C, 1, 1)
            out = x * block_mask

            # scale output
            out = out *(block_mask.numel() / block_mask.sum())

            return out, x_max
            # return out,orth_vis


def count_your_model_channel(model, x, y):
    print(x[0].shape)
    # assert False
    # print (y)
    flops = 0
    N, C, H, W = x[0].shape
    flops += 2 * C * H * W
    # orth
    flops += 2 * H * W * C + 3 * C +  C * H * W * C
    # sampling
    flops += 4 * C
    # point-wise multiplication
    flops += H * W * C
    model.total_ops = torch.Tensor([int(flops)])

