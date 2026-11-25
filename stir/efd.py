"""现有EFD逆变换的独立快照；保留DC，固定统一相位采样。"""
import math
import torch
from torch import Tensor


def require_finite(x: Tensor, name: str) -> None:
    """拒绝非浮点或非有限输入，避免悄悄将异常传播为 NaN。"""
    if not x.is_floating_point() or not torch.isfinite(x).all():
        raise ValueError(f'{name} 必须是有限浮点张量')


def inverse_efd(A: Tensor, points: int = 128, dc: Tensor | None = None) -> Tensor:
    """A [...,K,2,2] → 闭合轮廓采样点 [...,P,2]；不重复最后一个点。

    x(u)=Σ(a_k cos(2πku)+b_k sin(2πku))，y 同理由 c_k、d_k 得到。
    dc 为可选 [...,2] 平移；省略时中心为0，面积不受平移影响。
    """
    if A.shape[-2:] != (2, 2) or points < 3:
        raise ValueError('A 末维必须2×2且 points>=3')
    require_finite(A, 'A')
    phase = 2 * math.pi * torch.arange(points, device=A.device, dtype=A.dtype) / points
    harmonic = torch.arange(1, A.shape[-3] + 1, device=A.device, dtype=A.dtype)
    angles = phase[:, None] * harmonic[None]
    basis = torch.stack((angles.cos(), angles.sin()), -1)  # [P,K,2]
    contour = torch.einsum('...kij,pkj->...pi', A, basis)
    return contour if dc is None else contour + dc[..., None, :]


def polygon_area(contour: Tensor) -> Tensor:
    """鞋带公式：[...,P,2] → 非负面积 [...]；面积比得到 FAC，不是真实体积 EF。"""
    if contour.shape[-1] != 2 or contour.shape[-2] < 3:
        raise ValueError('轮廓应含至少3个二维点')
    following = contour.roll(-1, dims=-2)
    return .5 * (contour[..., 0] * following[..., 1] -
                 contour[..., 1] * following[..., 0]).sum(-1).abs()
