"""标准二维U-Net及概率头；任何图像卷积均逐帧，无时序交互。"""
import torch
from torch import nn
from torch.nn import functional as F


def block(a, b):
    """两层3×3卷积，GroupNorm不引入batch内帧交互。"""
    return nn.Sequential(nn.Conv2d(a, b, 3, padding=1), nn.GroupNorm(4, b), nn.SiLU(),
                         nn.Conv2d(b, b, 3, padding=1), nn.GroupNorm(4, b), nn.SiLU())


class SegmentationModel(nn.Module):
    """与概率实验逐层一致的U-Net backbone，仅保留分割头，无概率或时相头。"""

    def __init__(self):
        """同一1→16→32→64→128编码器与skip decoder，GN、SiLU。"""
        super().__init__()
        self.e1 = block(1, 16)
        self.e2 = block(16, 32)
        self.e3 = block(32, 64)
        self.e4 = block(64, 128)
        self.d3 = block(192, 64)
        self.d2 = block(96, 32)
        self.d1 = block(48, 16)
        self.seg = nn.Conv2d(16, 1, 1)

    def forward(self, x):
        """x[T,1,H,W] -> framewise segmentation logits[T,1,H,W]."""
        a = self.e1(x * 2 - 1)
        b = self.e2(F.avg_pool2d(a, 2))
        c = self.e3(F.avg_pool2d(b, 2))
        d = self.e4(F.avg_pool2d(c, 2))
        z = self.d3(
            torch.cat([
                F.interpolate(
                    d, size=c.shape[-2:], mode='bilinear', align_corners=False), c
            ], 1))
        z = self.d2(
            torch.cat([
                F.interpolate(
                    z, size=b.shape[-2:], mode='bilinear', align_corners=False), b
            ], 1))
        z = self.d1(
            torch.cat([
                F.interpolate(
                    z, size=a.shape[-2:], mode='bilinear', align_corners=False), a
            ], 1))
        return self.seg(z)
