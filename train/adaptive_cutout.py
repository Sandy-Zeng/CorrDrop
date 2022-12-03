import torch
import numpy as np
from summaries import TensorboardSummary
from torch.distributions import Bernoulli

class AdaptiveCutout(object):
    """Randomly mask out one or more patches from an image.
    Args:
        n_holes (int): Number of patches to cut out of each image.
        length (int): The length (in pixels) of each square patch.
    """
    def __init__(self, p):
       self.p = p

    def ortho(self,input):
        C,H,W = input.shape
        input = torch.FloatTensor(input).cuda()
        vec = torch.transpose(input.view(C,H*W),0,1)
        vec = vec / torch.sqrt(torch.sum(torch.pow(vec,2),dim=-1,keepdim=True))
        P = torch.abs(torch.matmul(vec,torch.transpose(vec,0,1)) - torch.eye(H*W).cuda().view(H*W,H*W))
        rank = torch.sum(P,dim=-1)/(H*W)
        rank = rank.view(H*W)
        return rank

    def normalize(self, input):
        min_c, max_c = input.min(0, keepdim=True)[0], input.max(0, keepdim=True)[0]
        input_norm = (input - min_c) / (max_c - min_c + 1e-8)
        return input_norm

    def __call__(self, img):
        """
        Args:
            img (Tensor): Tensor image of size (C, H, W).
        Returns:
            Tensor: Image with n_holes of dimension length x length cut out of it.
        """
        h = img.size(1)
        w = img.size(2)

        mask = np.ones((h, w), np.float32)
        norm = self.normalize(img)
        orth_vec = self.ortho(img)
        orth_p = self.normalize(orth_vec)
        orth_vec = 1 - Bernoulli(orth_p).sample().view(h,w)
        # print (1-orth_vec)
        p_mask = Bernoulli(self.p).sample((h,w)).cuda()
        mask = 1 - (orth_vec*p_mask).view(1,h,w)
        # print (mask)

        # for n in range(self.n_holes):
        #     y = np.random.randint(h)
        #     x = np.random.randint(w)
        #
        #     y1 = np.clip(y - self.length // 2, 0, h)
        #     y2 = np.clip(y + self.length // 2, 0, h)
        #     x1 = np.clip(x - self.length // 2, 0, w)
        #     x2 = np.clip(x + self.length // 2, 0, w)
        #
        #     mask[y1: y2, x1: x2] = 0.

        # mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img = img * mask.data.cpu()

        return img