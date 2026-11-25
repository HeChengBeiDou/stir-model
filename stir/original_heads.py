import torch
from torch import nn
from torch.nn import functional as F
from .constants import VAR_EPS
class MaskToC(nn.Module):
    """c_hat = c_calc + f(p_seg, metadata, c_calc)；60维均在train尺度下。"""

    def __init__(self):
        """常规小CNN，保留6×6空间位置；末层零初始化从确定性分支开始。"""
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.SiLU(),
            nn.AvgPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.SiLU(),
            nn.AvgPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.SiLU(),
            nn.AvgPool2d(2),
            nn.AvgPool2d(2),
            nn.Flatten(),
        )
        self.head = nn.Sequential(nn.Linear(64 * 6 * 6 + 5 + 60, 128), nn.SiLU(),
                                  nn.Linear(128, 60))
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)

    def forward(self, probability, metadata, calculated):
        """[B,1,96,96],[B,5],[B,60] → [B,60]；不访问GT或frame index。"""
        feature = self.encoder(probability)
        residual = self.head(torch.cat([feature, metadata, calculated], dim=-1))
        return calculated + residual

class VarianceHead(nn.Module):
    """μ固定；R_d(X)=softplus(rho0_d+f_d(p_seg,metadata,μ))+eps。

不接触GT phase、邻帧、RNN或时序attention，输出[B,60]。
"""

    def __init__(self, initial_variance):
        """从train经验残差方差开始；末层零初始化，不手动扫描方差倍率。"""
        super().__init__()
        target = initial_variance.float().clamp_min(1e-4) - VAR_EPS
        self.register_buffer('rho0', target + torch.log(-torch.expm1(-target)))
        self.encoder = nn.Sequential(nn.Conv2d(1, 16, 3, padding=1), nn.SiLU(),
                                     nn.AvgPool2d(2), nn.Conv2d(16, 32, 3, padding=1),
                                     nn.SiLU(), nn.AvgPool2d(2),
                                     nn.Conv2d(32, 64, 3, padding=1), nn.SiLU(),
                                     nn.AvgPool2d(2), nn.AvgPool2d(2), nn.Flatten())
        self.head = nn.Sequential(nn.Linear(64 * 6 * 6 + 5 + 60, 128), nn.SiLU(),
                                  nn.Linear(128, 60))
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)

    def forward(self, probability, metadata, center):
        """[B,1,96,96],[B,5],[B,60]→R[B,60]；全部帧独立计算。"""
        feature = self.encoder(probability)
        raw = self.rho0 + self.head(torch.cat([feature, metadata, center], -1))
        return F.softplus(raw) + VAR_EPS
