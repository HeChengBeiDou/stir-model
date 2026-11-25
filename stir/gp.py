"""多输出GP：保留60维关联与完整时间协方差；不使用torch.inverse。"""
import math
import torch
from torch import nn
from torch.nn import functional as F
from .constants import GP_RANK


def chol(C):
    """对称化+jitter；只稳定概率协方差，不向几何S添加ridge。"""
    C = (C + C.transpose(-1, -2)) / 2
    eye = torch.eye(C.shape[-1], dtype=C.dtype, device=C.device)
    for jitter in (1e-8, 1e-7, 1e-6, 1e-5):
        L, info = torch.linalg.cholesky_ex(C + jitter * eye)
        if not torch.any(info):
            return L
    raise RuntimeError('GP covariance Cholesky failed')


class PopulationGP(nn.Module):
    """c(τ)的矩阵值Matérn核=k_time×H；H低秩+正对角而非60个独立GP。"""

    def __init__(self, dimension=60, *, n_views=1):
        """视图特异线性均值；时间核和跨坐标协方差共享。"""
        super().__init__()
        self.dimension=dimension
        self.mean = nn.Parameter(torch.zeros(n_views, 2, self.dimension, dtype=torch.float64))
        self.factor = nn.Parameter(torch.randn(self.dimension, GP_RANK, dtype=torch.float64) * .02)
        self.diag = nn.Parameter(torch.zeros(self.dimension, dtype=torch.float64))
        self.raw_length = nn.Parameter(torch.tensor(-1., dtype=torch.float64))
        self.raw_constant = nn.Parameter(torch.tensor(-1., dtype=torch.float64))

    def mean_at(self, t, view):
        """[F]→[F,60]，绝不依赖GT相位；t是原采集窗口进度。"""
        return self.mean[view,
                         0] + t[..., None] * (self.mean[view, 1] - self.mean[view, 0])

    def covariance(self, t, u):
        """[F],[O]→[60F,60O]，按时间后坐标顺序堆叠。"""
        distance = (t[:, None] - u[None, :]).abs()
        z = math.sqrt(3) * distance / (F.softplus(self.raw_length) + .02)
        kt = (1 + z) * torch.exp(-z) + F.softplus(self.raw_constant)
        H = self.factor @ self.factor.T + torch.diag(F.softplus(self.diag) + 1e-4)
        return (kt[:, None, :, None] * H[None, :, None, :]).reshape(
            len(t) * self.dimension,
            len(u) * self.dimension)

    def condition(self, t, mu, var, query, view):
        """解析p(c_query|图像证据)，mu/var[O,60]；返回[F,60]及[60F,60F]。"""
        prior = self.covariance(query, query)
        cross = self.covariance(query, t)
        L = chol(self.covariance(t, t) + torch.diag(var.double().flatten()))
        delta = (mu.double() - self.mean_at(t, view)).flatten()
        mean = self.mean_at(query, view).flatten() + cross @ torch.cholesky_solve(
            delta[:, None], L)[:, 0]
        covariance = prior - cross @ torch.cholesky_solve(cross.T, L)
        return mean.reshape(-1, self.dimension), (covariance + covariance.T) / 2



def joint_sample(mean, covariance, noise):
    """noise[M,F*D]→整条轨迹[M,F,D]，共享Cholesky保留时间和参数相关性。"""
    return (mean.flatten()[None] + noise @ chol(covariance).T).reshape(
        len(noise), *mean.shape)
