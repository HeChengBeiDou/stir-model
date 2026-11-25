# STIR — Model Implementation

PyTorch implementation of **Shape-aware Temporal Inference and Refinement**.
STIR combines a framewise U-Net, starting-point-invariant harmonic shape
coordinates, a multioutput temporal Gaussian process, and anchored low-order
elliptic Fourier descriptor (EFD) refinement.

## Use

Install the dependencies with python -m pip install -r requirements.txt.

Import STIR and inverse_efd from stir. Instantiate STIR with the 60-element
training shape-coordinate mean and standard deviation and the harmonic
coefficient RMS scale. The presets echonet and camus configure the 112×112
single-view and 224×224 biplane models, respectively. For CAMUS, call the model
for each view with its view index and isotropic grid spacing. For EchoNet-Dynamic,
use view zero and native-pixel spacing.

Provide an in-memory float tensor of shape (T, 1, H, W) with values in [0, 1],
one normalized acquisition time in [0, 1] per frame, and the indices of the
frames to refine. Call forward_probabilities with per-frame probabilities to
use a different segmentation frontend. Both interfaces return anchored EFD
coefficients, contour centres, and the joint shape posterior. Use inverse_efd
to reconstruct any valid contour. Optional standard-normal noise of shape
(draws, query_frames * 60) produces joint posterior draws.

Empty or invalid mask geometry is flagged in valid_geometry. Such frames can
still contribute image-based observations, and a queried frame with invalid
geometry has nonfinite coefficients. Check the flags before reconstructing
its contours.

The repository contains the model architecture and its tensor operations.
Trained parameter values and normalization statistics are supplied by the caller.
