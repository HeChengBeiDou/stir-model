"""Starting-point-invariant harmonic shape coordinates."""

import torch

from .constants import HARMONICS


def encode_cholesky_factor(factor, epsilon):
    """Map a lower Cholesky factor of shape (..., 20, 2, 2) to (..., 60)."""
    if factor.shape[-2:] != (2, 2) or not torch.isfinite(factor).all():
        raise ValueError("Invalid Cholesky factor")
    if torch.any(factor[..., 0, 1] != 0):
        raise ValueError("Expected a lower-triangular factor")
    first = factor[..., 0, 0] - epsilon
    second = factor[..., 1, 1] - epsilon
    if torch.any(first <= 0) or torch.any(second <= 0):
        raise ValueError("Cholesky diagonals must exceed the coordinate epsilon")
    u = first + torch.log(-torch.expm1(-first))
    w = second + torch.log(-torch.expm1(-second))
    return torch.stack((u, factor[..., 1, 0], w), dim=-1).flatten(-2)


def encode(shape, epsilon):
    factor, info = torch.linalg.cholesky_ex(shape)
    if torch.any(info):
        raise ValueError("Shape matrix is not positive definite")
    return encode_cholesky_factor(factor, epsilon)


def decode(coordinates, epsilon):
    u, v, w = coordinates.reshape(*coordinates.shape[:-1], HARMONICS, 3).unbind(-1)
    first = torch.logaddexp(u, torch.zeros_like(u)) + epsilon
    second = torch.logaddexp(w, torch.zeros_like(w)) + epsilon
    factor = torch.stack((first, torch.zeros_like(v), v, second), dim=-1)
    factor = factor.reshape(*u.shape, 2, 2)
    return factor @ factor.transpose(-1, -2)
