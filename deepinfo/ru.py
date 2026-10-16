"""Decoder-based reconstruction uncertainty under feature-preserving noise."""
from __future__ import annotations
from dataclasses import dataclass
import math
import torch
import torch.nn.functional as F
from ._common import feature_reader, frozen, get_grid, input_image, validate_optimization


@dataclass(frozen=True)
class RUResult:
    ru_map: torch.Tensor              # (C,H,W) reconstruction uncertainty in nats
    ru_mean: float
    distortion: float
    normalized_distortion: float
    selected_step: int
    layer: str
    history: tuple[dict[str, float], ...]


def explain_ru(
    model: torch.nn.Module,
    image: torch.Tensor,
    decoder: torch.nn.Module,
    *,
    layer: str,
    grid: tuple[int, int] | None = (8, 8),
    steps: int = 40,
    noise_samples: int = 8,
    learning_rate: float = 0.08,
    initial_sigma: float = 0.08,
    entropy_weight: float = 2.0,
    distortion_ratio: float = 1.2,
    seed: int = 0,
    eps: float = 1e-9,
) -> RUResult:
    """Estimate RU with a trained decoder mapping layer activations to input space.

    A decoder must already be trained for the layer and normalization scheme.
    RU optimizes input noise using a feature-invariance term and the entropy
    of reconstructed variation, following the original DeepInfo decoder path.
    The empirical variance is computed from MC decoder reconstructions.
    """
    validate_optimization(steps=steps, samples=noise_samples, learning_rate=learning_rate,
                          initial_sigma=initial_sigma, entropy_weight=entropy_weight,
                          distortion_ratio=distortion_ratio)
    x = input_image(image, model)
    _, channels, height, width = x.shape
    rows, cols = get_grid(grid, (height, width))
    generator = torch.Generator(device=x.device)
    generator.manual_seed(seed)
    log_sigma = torch.nn.Parameter(x.new_full((1, 1, rows, cols), math.log(initial_sigma)))
    opt = torch.optim.Adam([log_sigma], lr=learning_rate)
    history: list[dict[str, float]] = []
    best: dict | None = None

    def sigma():
        return F.interpolate(log_sigma.clamp(-9.0, 2.0).exp(), size=(height, width), mode="nearest")

    def noise():
        return torch.randn((noise_samples, channels, height, width), device=x.device,
                           dtype=x.dtype, generator=generator)

    with frozen(model, decoder), feature_reader(model, layer) as read:
        with torch.no_grad():
            f_clean = read(x).detach()
            reconstructed = decoder(f_clean).detach()
            if reconstructed.shape != x.shape:
                raise ValueError("decoder must map selected layer features back to input shape")
            f_initial = read(x + sigma() * noise())
            reference = (f_initial - f_clean).square().mean().item()
        if not reference > 1e-12:
            raise ValueError("selected layer is insensitive at initial_sigma")
        for t in range(steps):
            noisy_features = read(x + sigma() * noise())
            mse = (noisy_features - f_clean).square().mean()
            noisy_recon = decoder(noisy_features)
            if noisy_recon.shape != (noise_samples, *x.shape[1:]):
                raise ValueError("decoder output shape changed")
            variance = (noisy_recon - reconstructed).square().mean(dim=0)
            ru_map = 0.5 * torch.log(2.0 * math.pi * math.e * variance.clamp_min(eps))
            normalized = mse / reference
            objective = normalized - entropy_weight * ru_map.mean()
            opt.zero_grad(set_to_none=True)
            objective.backward()
            opt.step()
            with torch.no_grad():
                log_sigma.clamp_(-9.0, 2.0)
                d, r, h = mse.item(), normalized.item(), ru_map.mean().item()
                if not all(math.isfinite(v) for v in (d, r, h)):
                    raise FloatingPointError("RU optimization became non-finite")
                history.append({"step": float(t), "distortion": d,
                                "normalized_distortion": r, "mean_ru": h})
                gap = abs(r - distortion_ratio)
                if best is None or gap < best["gap"]:
                    best = {"gap": gap, "step": t, "map": ru_map.detach().clone(),
                            "distortion": d, "normalized": r}
    assert best is not None
    return RUResult(best["map"].detach().cpu(),
                    float(best["map"].mean().item()), float(best["distortion"]),
                    float(best["normalized"]), int(best["step"]), layer, tuple(history))


class ConvDecoder(torch.nn.Module):
    """Small trainable decoder for convolutional features and a fixed image size."""
    def __init__(self, feature_channels: int, output_channels: int,
                 output_size: tuple[int, int], hidden_channels: int = 32):
        super().__init__()
        self.output_size = output_size
        self.net = torch.nn.Sequential(
            torch.nn.Conv2d(feature_channels, hidden_channels, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(hidden_channels, output_channels, 3, padding=1),
        )

    def forward(self, features):
        if features.ndim != 4:
            raise ValueError("ConvDecoder expects convolutional feature maps (N,C,H,W)")
        return self.net(F.interpolate(features, size=self.output_size,
                                      mode="bilinear", align_corners=False))


def train_decoder(model: torch.nn.Module, decoder: torch.nn.Module,
                  images: torch.Tensor, *, layer: str,
                  epochs: int = 100, lr: float = 1e-3) -> list[float]:
    """Fit a decoder on a batch of preprocessed training images, leaving model unchanged."""
    if images.ndim != 4 or images.shape[0] < 2:
        raise ValueError("images must be a batch with shape (N,C,H,W), N >= 2")
    if epochs < 1 or lr <= 0:
        raise ValueError("epochs and lr must be positive")
    optimizer = torch.optim.Adam(decoder.parameters(), lr=lr)
    was_training = {module: module.training for module in model.modules()}
    decoder_was_training = {module: module.training for module in decoder.modules()}
    model.eval()
    decoder.train()
    try:
        with feature_reader(model, layer) as read, torch.no_grad():
            features = read(images).detach()
        if features.ndim != 4:
            raise ValueError("ConvDecoder training requires convolutional (N,C,H,W) features")
        history = []
        for _ in range(epochs):
            optimizer.zero_grad(set_to_none=True)
            decoded = decoder(features)
            if decoded.shape != images.shape:
                raise ValueError("decoder output must match image shape")
            loss = F.mse_loss(decoded, images)
            loss.backward()
            optimizer.step()
            history.append(float(loss.detach().item()))
        return history
    finally:
        for module, state in was_training.items():
            module.training = state
        for module, state in decoder_was_training.items():
            module.training = state
