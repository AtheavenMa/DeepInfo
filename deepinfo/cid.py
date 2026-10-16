"""Computational information discarding (CID) through input-noise optimization.

The layer response is compared with the clean response while coordinate-wise
Gaussian noise scales are optimized to increase input entropy. This is a
Monte Carlo Lagrangian estimator, not a saliency-gradient attribution.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable
import torch
import torch.nn.functional as F

from ._common import (feature_reader, frozen, gaussian_entropy, get_grid,
                      input_image, validate_optimization)


@dataclass(frozen=True)
class CIDResult:
    cid_map: torch.Tensor              # (H,W) or (C,H,W), CPU, nats per coordinate
    sigma_map: torch.Tensor            # same shape, data-space noise std
    cid_mean: float                    # mean differential entropy (nats)
    distortion: float                  # MSE between clean and noisy layer outputs
    normalized_distortion: float       # divided by reference distortion
    reference_distortion: float
    selected_step: int
    layer: str | None
    grid: tuple[int, int]
    history: tuple[dict[str, float], ...]

    @property
    def relevance_map(self) -> torch.Tensor:
        """Negative CID, a relevance convention used by targeted applications."""
        return -self.cid_map


def explain(
    model: torch.nn.Module,
    image: torch.Tensor,
    *,
    layer: str | None = None,
    selector: Callable[[torch.Tensor], torch.Tensor] | None = None,
    channels: tuple[int, ...] | list[int] | None = None,
    grid: tuple[int, int] | None = (8, 8),
    per_channel: bool = False,
    steps: int = 40,
    noise_samples: int = 8,
    learning_rate: float = 0.08,
    initial_sigma: float = 0.08,
    entropy_weight: float = 2.0,
    distortion_ratio: float = 1.2,
    seed: int = 0,
) -> CIDResult:
    """Estimate input information discarded by one layer of a frozen model.

    Inputs are `(C,H,W)` or `(1,C,H,W)` floats and must already use the model's
    preprocessing. `layer=None` selects model output; otherwise pass an exact
    `named_modules()` name. `selector` can select a fixed class, spatial mask,
    or subrepresentation from every clean/noisy output. `channels` limits the
    input channels receiving noise. `grid` is the number of *noise cells* along
    the two image axes; the returned maps are nearest-neighbor upsampled.

    This implementation minimizes mean squared response change normalized by
    the distortion at the initial Gaussian scale, minus the entropy reward.
    Among steps it selects the one nearest the requested normalized distortion,
    matching the legacy implementation's result-selection principle. The
    finite-step estimate is not an exact solution of a hard constraint.
    """
    validate_optimization(steps=steps, samples=noise_samples, learning_rate=learning_rate,
                          initial_sigma=initial_sigma, entropy_weight=entropy_weight,
                          distortion_ratio=distortion_ratio)
    image = input_image(image, model)
    _, channel_count, height, width = image.shape
    rows, cols = get_grid(grid, (height, width))
    if channels is None:
        selected = tuple(range(channel_count))
    else:
        if (not isinstance(channels, (tuple, list)) or not channels or
                any(isinstance(c, bool) or not isinstance(c, int) or c < 0 or c >= channel_count
                    for c in channels) or len(set(channels)) != len(channels)):
            raise ValueError("channels must contain distinct valid input channel indices")
        selected = tuple(channels)
    channel_mask = image.new_zeros((1, channel_count, 1, 1))
    channel_mask[:, selected, :, :] = 1
    shape = (1, len(selected) if per_channel else 1, rows, cols)
    rng = torch.Generator(device=image.device)
    rng.manual_seed(seed)
    noise_shape = (noise_samples, channel_count, height, width)
    log_sigma = torch.nn.Parameter(image.new_full(shape, float(__import__('math').log(initial_sigma))))
    optimizer = torch.optim.Adam([log_sigma], lr=learning_rate)
    records: list[dict[str, float]] = []
    best: dict | None = None

    def expand_sigma() -> torch.Tensor:
        small = log_sigma.clamp(-9.0, 2.0).exp()
        large = F.interpolate(small, size=(height, width), mode="nearest")
        if per_channel:
            full = image.new_zeros((1, channel_count, height, width))
            full[:, selected] = large
            return full
        return large * channel_mask

    with frozen(model), feature_reader(model, layer, selector) as read:
        with torch.no_grad():
            original = read(image).detach()
            calibration_noise = torch.randn(noise_shape, generator=rng, device=image.device,
                                            dtype=image.dtype)
            calibrated = read(image + expand_sigma() * calibration_noise)
            reference = (calibrated - original).square().mean().item()
        if not (reference > 1e-12):
            raise ValueError("selected response is insensitive at initial_sigma; choose another layer, "
                             "selector, channel, or a larger initial_sigma")
        for iteration in range(steps):
            epsilon = torch.randn(noise_shape, generator=rng, device=image.device,
                                  dtype=image.dtype)
            perturbed = image + expand_sigma() * epsilon
            feature = read(perturbed)
            distortion = (feature - original).square().mean()
            normalized = distortion / reference
            entropy = gaussian_entropy(log_sigma.clamp(-9.0, 2.0)).mean()
            loss = normalized - entropy_weight * entropy
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            with torch.no_grad():
                candidate_log_sigma = log_sigma.detach().clone()
            optimizer.step()
            with torch.no_grad():
                log_sigma.clamp_(-9.0, 2.0)
                d, r, h = distortion.item(), normalized.item(), entropy.item()
                if not all(__import__('math').isfinite(v) for v in (d, r, h)):
                    raise FloatingPointError("CID optimization became non-finite")
                records.append({"step": float(iteration), "distortion": d,
                                "normalized_distortion": r, "mean_entropy": h})
                gap = abs(r - distortion_ratio)
                if best is None or gap < best["gap"]:
                    best = {"gap": gap, "step": iteration, "log_sigma": candidate_log_sigma,
                            "distortion": d, "ratio": r}

    assert best is not None
    cell_sigma = best["log_sigma"].exp()
    upsampled = F.interpolate(cell_sigma, size=(height, width), mode="nearest")
    if per_channel:
        full_sigma = image.new_zeros((channel_count, height, width))
        full_sigma[list(selected)] = upsampled[0]
        # Unselected channels have no CID estimate, not zero-valued CID.
        full_sigma[[i for i in range(channel_count) if i not in selected]] = float("nan")
    else:
        full_sigma = upsampled[0, 0]
    entropy_map = torch.log(full_sigma) + 0.5 * __import__('math').log(2.0 * __import__('math').pi * __import__('math').e)
    return CIDResult(entropy_map.detach().cpu(), full_sigma.detach().cpu(),
                     float(gaussian_entropy(best["log_sigma"]).mean().item()),
                     float(best["distortion"]), float(best["ratio"]),
                     float(reference), int(best["step"]), layer, (rows, cols), tuple(records))
