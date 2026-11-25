"""Image/c-only evidence heads; preserve inherited CNN and remove five meta inputs."""
import math
import torch
from torch import nn
from torch.nn import functional as F
from .constants import LATENT_DIM, VAR_EPS

from .original_heads import MaskToC as OriginalMean, VarianceHead as OriginalVariance


class SpatialMean6(nn.Module):
    """[B,C,H,W]→[B,C,6,6], exact adaptive-average bins with deterministic backward.

    At H=W=6 each bin is one pixel: identity. No parameters or input resizing.
    Explicit slices avoid CUDA adaptive_avg_pool2d's nondeterministic backward.
    """
    def forward(self, x):
        h, w = x.shape[-2:]
        cells = [x[..., (i*h)//6:math.ceil((i+1)*h/6),
                       (j*w)//6:math.ceil((j+1)*w/6)].mean(dim=(-2, -1))
                 for i in range(6) for j in range(6)]
        return torch.stack(cells, -1).reshape(*x.shape[:-2], 6, 6)


def remove_meta(head):
    head.encoder = nn.Sequential(*list(head.encoder.children())[:-1], SpatialMean6(), nn.Flatten())
    head.head[0] = nn.Linear(64*6*6 + LATENT_DIM, 128)


class ImageShapeMean(OriginalMean):
    def __init__(self):
        super().__init__()
        remove_meta(self)

    def forward(self, probability, calculated):
        return calculated + self.head(torch.cat([self.encoder(probability), calculated], -1))


class ImageShapeVariance(OriginalVariance):
    def __init__(self, initial_variance):
        super().__init__(initial_variance)
        remove_meta(self)

    def forward(self, probability, center):
        raw = self.rho0 + self.head(torch.cat([self.encoder(probability), center], -1))
        return F.softplus(raw) + VAR_EPS
