"""STIR model for in-memory frame sequences or segmentation probabilities."""

import math

import numpy as np
import torch
from torch import nn

from .constants import (
    FRAME_CHUNK,
    HARMONICS,
    INITIAL_VARIANCE,
    LATENT_DIM,
    PREDICTION_DIAGONAL_FLOOR,
    REFINED_HARMONICS,
)
from .contours import mask_to_efd
from .coordinates import decode, encode, encode_cholesky_factor
from .evidence import ImageShapeMean, ImageShapeVariance
from .gp import PopulationGP, joint_sample
from .postprocessing import clean_mask
from .segmentation import SegmentationModel


def _initialise(offset, factory):
    with torch.random.fork_rng(devices=[]):
        generator = torch.Generator().manual_seed(20261026 + offset)
        torch.set_rng_state(generator.get_state())
        return factory()


class _ShapeResidual(nn.Module):
    """Map inferred shape matrices to updates of the first five harmonics."""

    def __init__(self, scale):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(LATENT_DIM, 128), nn.SiLU(), nn.Linear(128, 4 * HARMONICS)
        )
        nn.init.zeros_(self.layers[-1].weight)
        nn.init.zeros_(self.layers[-1].bias)
        self.register_buffer("a_scale", torch.as_tensor(scale, dtype=torch.float64))

    def forward(self, shape, mean, std, epsilon):
        coordinates = (encode(shape, epsilon) - mean) / std
        raw = self.layers(coordinates.float()).double()
        raw = raw.reshape(*coordinates.shape[:-1], HARMONICS, 2, 2)
        low = raw[..., :REFINED_HARMONICS, :, :]
        high = torch.zeros_like(raw[..., REFINED_HARMONICS:, :, :])
        return torch.cat((low, high), dim=-3) * self.a_scale


def _geometry(probability, step, shape_mean, epsilon):
    masks = probability.detach().cpu()[:, 0].numpy() >= 0.5
    step_xy = torch.tensor([step, step])
    coefficients = []
    centres = []
    coordinates = []
    valid = []
    for mask in masks:
        try:
            fitted = mask_to_efd(torch.from_numpy(clean_mask(mask).astype(np.uint8)), step_xy)
            a = fitted["A"]
            factor = torch.linalg.cholesky(a @ a.transpose(-1, -2))
            diagonal = factor.diagonal(dim1=-2, dim2=-1)
            floor = PREDICTION_DIAGONAL_FLOOR + 1e-12
            factor = factor + torch.diag_embed(diagonal.clamp_min(floor) - diagonal)
            projected = factor @ factor.transpose(-1, -2)
            try:
                c = encode(projected, epsilon)
            except ValueError:
                c = encode_cholesky_factor(factor, epsilon)
                torch.testing.assert_close(
                    decode(c, epsilon), projected, rtol=1e-12, atol=1e-12
                )
            dc = fitted["dc"]
        except (ValueError, torch.linalg.LinAlgError):
            a = torch.full((HARMONICS, 2, 2), torch.nan, dtype=torch.float64)
            dc = torch.full((2,), torch.nan, dtype=torch.float64)
            c = shape_mean.detach().cpu()
            valid.append(False)
        else:
            valid.append(True)
        coefficients.append(a)
        centres.append(dc)
        coordinates.append(c)
    return (
        torch.stack(coefficients),
        torch.stack(centres),
        torch.stack(coordinates),
        torch.tensor(valid, dtype=torch.bool),
    )


class STIR(nn.Module):
    """U-Net, invariant harmonic coordinates, temporal GP and contour refinement."""

    def __init__(self, shape_mean, shape_std, coefficient_scale, *, preset="echonet"):
        super().__init__()
        try:
            self.image_size, self.coordinate_epsilon, self.n_views = {
                "echonet": (112, 1e-9, 1),
                "camus": (224, 1e-6, 2),
            }[preset]
        except KeyError as error:
            raise ValueError("preset must be 'echonet' or 'camus'") from error

        self.segmentation = _initialise(10, SegmentationModel)
        self.mean = _initialise(20, ImageShapeMean)
        self.variance = _initialise(
            30, lambda: ImageShapeVariance(torch.full((LATENT_DIM,), INITIAL_VARIANCE))
        )
        self.gp = _initialise(40, lambda: PopulationGP(LATENT_DIM, n_views=self.n_views))
        self.residual = _initialise(50, lambda: _ShapeResidual(coefficient_scale))
        self.register_buffer(
            "c_mean", torch.as_tensor(shape_mean, dtype=torch.float64).reshape(LATENT_DIM)
        )
        self.register_buffer(
            "c_std", torch.as_tensor(shape_std, dtype=torch.float64).reshape(LATENT_DIM)
        )

    def coefficients(self, normalized_shape, baseline):
        """Add posterior-guided low-order corrections to the anchored EFDs."""
        shape = decode(normalized_shape * self.c_std + self.c_mean, self.coordinate_epsilon)
        delta = self.residual(shape, self.c_mean, self.c_std, self.coordinate_epsilon)
        return baseline + delta

    def forward(self, frames, times, query_indices, *, step=1.0, view=0, noise=None):
        """Refine contours from U-Net segmentation of frames shaped (T, 1, H, W)."""
        self._check_sequence(frames, "frames")
        frames = frames.to(device=self.c_mean.device, dtype=torch.float32)
        logits = torch.cat([self.segmentation(part) for part in frames.split(FRAME_CHUNK)])
        result = self.forward_probabilities(
            logits.sigmoid(), times, query_indices, step=step, view=view, noise=noise
        )
        result["logits"] = logits
        return result

    def forward_probabilities(
        self, probabilities, times, query_indices, *, step=1.0, view=0, noise=None
    ):
        """Refine contours from one frontend's sigmoid probability sequence."""
        self._check_sequence(probabilities, "probabilities")
        if not 0 <= view < self.n_views:
            raise ValueError("view is outside this model's configured views")
        step = float(step)
        if not math.isfinite(step) or step <= 0:
            raise ValueError("step must be a positive finite grid spacing")

        device = self.c_mean.device
        probabilities = probabilities.to(device=device, dtype=torch.float32)
        times = torch.as_tensor(times, dtype=torch.float64, device=device)
        indices = torch.as_tensor(query_indices, dtype=torch.long, device=device)
        if times.shape != (len(probabilities),) or not torch.isfinite(times).all():
            raise ValueError("times must contain one finite value per frame")
        if indices.ndim != 1 or not indices.numel() or not torch.all(
            (indices >= 0) & (indices < len(times))
        ):
            raise ValueError("query_indices must select one or more input frames")

        a, centres, coordinates, valid = _geometry(
            probabilities, step, self.c_mean, self.coordinate_epsilon
        )
        c = ((coordinates.to(device) - self.c_mean) / self.c_std).float()
        mu = self.mean(probabilities, c).double()
        variance = self.variance(probabilities, mu.float()).double()
        posterior, covariance = self.gp.condition(times, mu, variance, times[indices], view)
        baseline = a.to(device)[indices]
        result = {
            "baseline_coefficients": baseline,
            "coefficients": self.coefficients(posterior, baseline),
            "centres": centres.to(device)[indices],
            "valid_geometry": valid.to(device),
            "posterior_mean": posterior,
            "posterior_covariance": covariance,
        }
        if noise is not None:
            noise = torch.as_tensor(noise, device=device, dtype=posterior.dtype)
            if noise.ndim != 2 or noise.shape[1] != posterior.numel():
                raise ValueError("noise must have shape (draws, query_frames * 60)")
            result["sampled_coefficients"] = self.coefficients(
                joint_sample(posterior, covariance, noise), baseline
            )
        return result

    def _check_sequence(self, sequence, name):
        size = self.image_size
        if sequence.ndim != 4 or sequence.shape[1:] != (1, size, size) or not len(sequence):
            raise ValueError(f"{name} must have shape (T, 1, {size}, {size}) with T >= 1")
