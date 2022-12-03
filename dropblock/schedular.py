import numpy as np
from torch import nn
import random

class LinearScheduler(nn.Module):
    def __init__(self, dropblock, start_value, stop_value, nr_steps):
        super(LinearScheduler, self).__init__()
        self.dropblock = dropblock
        self.i = 0
        self.drop_values = np.linspace(start=start_value, stop=stop_value, num=nr_steps)

    def forward(self, x):
        return self.dropblock(x)

    def step(self):
        if self.i < len(self.drop_values):
            self.dropblock.drop_prob = self.drop_values[self.i]

        self.i += 1


class BlockSizeLinearScheduler(nn.Module):
    def __init__(self, dropblock, start_value, stop_value, drop_prob,nr_steps):
        super(BlockSizeLinearScheduler, self).__init__()
        self.dropblock = dropblock
        self.i = 0
        self.blocksize_values = np.linspace(start=start_value, stop=stop_value, num=nr_steps)
        self.drop_values = np.linspace(start=0., stop=drop_prob, num=nr_steps)

    def forward(self, x):
        return self.dropblock(x)

    def step(self):
        if self.i < len(self.drop_values):
            self.dropblock.block_size = int(self.blocksize_values[self.i])
            # self.dropblock.drop_prob = self.drop_values[self.i]

        self.i += 1


class RandomScheduler(nn.Module):
    def __init__(self, dropblock, start_value, stop_value):
        super(RandomScheduler, self).__init__()
        self.dropblock = dropblock
        self.start_value = start_value
        self.stop_value = stop_value
        # self.drop_values = np.linspace(start=start_value, stop=stop_value, num=nr_steps)

    def forward(self, x):
        return self.dropblock(x)

    def step(self):
        # if self.i < len(self.drop_values):
        #     self.dropblock.drop_prob = self.drop_values[self.i]
        # self.dropblock.drop_prob = random.uniform(self.start_value,self.stop_value)
        self.dropblock.block_size = int(random.uniform(self.start_value, self.stop_value))

