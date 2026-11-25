import numpy as np
import torch
from skimage.measure import find_contours
from .constants import HARMONICS
def mask_to_contour(mask, spacing, points=256, origin=None, direction=None):
    """Binary mask[y,x] -> native-pixel closed contour[P,2], inherited start/orientation."""
    array = mask.detach().cpu().numpy()
    if array.ndim != 2 or not np.isin(array, [0, 1]).all():
        raise ValueError('mask must be a binary2D EchoNet label')
    lv = array == 1
    contours = find_contours(np.pad(lv, 1).astype(float), .5)
    if not contours:
        raise ValueError('LV标签为空')
    s = np.asarray(spacing.detach().cpu(), dtype=float)
    if s.shape != (2, ) or not np.isfinite(s).all() or np.any(s <= 0):
        raise ValueError('spacing应为正[2]')
    D = np.eye(2) if direction is None else direction.detach().cpu().numpy()
    b = np.zeros(2) if origin is None else origin.detach().cpu().numpy()
    polygons = [((rc - 1)[:, ::-1] * s) @ D.T + b for rc in contours
                ]  #rc 是(row, col) 也就是(y, x) 前面pad 所以要减一 ::-1就是第二维调换 s是单位长度
    areas = [
        abs(np.sum(p[:-1, 0] * p[1:, 1] - p[1:, 0] * p[:-1, 1])) * .5 for p in polygons
    ]  #鞋带公式
    p = polygons[int(np.argmax(areas))][:-1]  #第一个点和最后一个重合 去掉
    signed = np.sum(p[:, 0] * np.roll(p[:, 1], -1) - p[:, 1] * np.roll(p[:, 0], -1))
    if signed > 0:
        p = p[::-1]  #利用鞋带公式 固定遍历方向
    p = np.roll(p, -np.lexsort((p[:, 0], -p[:, 1]))[0], axis=0)  #最右下的点为起点
    q = np.vstack([p, p[0]])
    distance = np.r_[0., np.cumsum(np.linalg.norm(np.diff(q, axis=0),
                                                  axis=1))]  #distance[-1]是总长度
    if distance[-1] <= 0:
        raise ValueError('退化轮廓')
    t = np.arange(points) * distance[-1] / points
    contour = np.stack([np.interp(t, distance, q[:, i]) for i in range(2)],
                       -1)  #插值得到(256,2)坐标点
    return contour, dict(boundary_count=len(contours),
                         largest_outer_boundary_used=True,
                         touches_border=bool(lv[0].any() or lv[-1].any()
                                             or lv[:, 0].any() or lv[:, -1].any()))

def contour_to_efd(contour, K):
    """闭合分段线性曲线的EFD积分：A_k=T/(2π²k²) Σ velocity·Δ(cos,sin)。"""
    delta = np.diff(np.vstack([contour, contour[0]]), axis=0)
    dt = np.linalg.norm(delta, axis=1)  #delta (256,2) dt (256,)
    delta, dt = delta[dt > 1e-12], dt[dt > 1e-12]
    cumulative = np.r_[0., np.cumsum(dt)]
    period = cumulative[-1]
    #轮廓长度
    harmonics = np.arange(1, K + 1)
    phi = 2 * np.pi * cumulative[:, None] * harmonics[None] / period
    #phi  2pai kt/T 其中t和T都是长度 （256，K）
    scale = period / (2 * np.pi**2 * harmonics**2)
    velocity = delta / dt[:, None]  #每段轮廓单位切向量 （256，2）
    coefficients = np.stack([
        scale * (velocity[:, i, None] * np.diff(fn(phi), axis=0)).sum(0)
        for i in range(2) for fn in (np.cos, np.sin)
    ], -1)  #（K，2*2）
    xi = np.cumsum(delta, axis=0) - velocity * cumulative[1:, None]  #要从固定P0开始 所以有这个项
    dc = contour[0] + (velocity * .5 * np.diff(cumulative**2)[:, None] +
                       xi * dt[:, None]).sum(0) / period
    A = torch.tensor(coefficients.reshape(K, 2, 2), dtype=torch.float64)
    return A, torch.tensor(dc)

def mask_to_efd(mask, spacing, K=HARMONICS, points=256, origin=None, direction=None):
    """mask -> C -> A[K,2,2] -> S=AAᵀ; spacing is native pixels/grid cell, not mm."""
    contour, qc = mask_to_contour(mask, spacing, points, origin, direction)
    A, dc = contour_to_efd(contour, K)
    return dict(A=A,
                S=A @ A.transpose(-1, -2),
                contour=torch.tensor(contour),
                dc=dc,
                **qc)
