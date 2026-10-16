"""Model feature capture and reversible evaluation contexts."""
from __future__ import annotations

from contextlib import contextmanager
from typing import Callable
import torch


def input_image(image: torch.Tensor, model: torch.nn.Module) -> torch.Tensor:
    if not isinstance(image, torch.Tensor):
        raise TypeError("image must be a torch.Tensor")
    if image.ndim == 3:
        image = image.unsqueeze(0)
    if image.ndim != 4 or image.shape[0] != 1:
        raise ValueError("image must have shape (C,H,W) or (1,C,H,W)")
    if not image.is_floating_point() or not torch.isfinite(image).all():
        raise ValueError("image must be a finite floating-point tensor")
    try:
        p = next(model.parameters())
        if p.device != image.device:
            raise ValueError(f"model and image must be on the same device ({p.device} vs {image.device})")
        if p.dtype != image.dtype:
            raise ValueError(f"model and image must have the same dtype ({p.dtype} vs {image.dtype})")
    except StopIteration:
        pass
    return image.detach()


def as_tensor(output: object) -> torch.Tensor:
    if isinstance(output, (tuple, list)) and output:
        output = output[0]
    if not isinstance(output, torch.Tensor):
        raise TypeError("selected model output must be a tensor or a tuple with a tensor first")
    return output


@contextmanager
def frozen(*modules: torch.nn.Module):
    """Keep all original mode/gradient states, including mixed training modes."""
    modes = [(sub, sub.training) for module in modules for sub in module.modules()]
    params = list(dict.fromkeys(p for m in modules for p in m.parameters()))
    requires_grad = [p.requires_grad for p in params]
    try:
        for module in modules:
            module.eval()
        for p in params:
            p.requires_grad_(False)
        yield
    finally:
        for p, state in zip(params, requires_grad):
            p.requires_grad_(state)
        for module, state in modes:
            module.training = state


@contextmanager
def feature_reader(model: torch.nn.Module, layer: str | None,
                   selector: Callable[[torch.Tensor], torch.Tensor] | None = None):
    """Capture a named layer without permanently registering hooks."""
    found = {}
    if layer is not None:
        modules = dict(model.named_modules())
        if not isinstance(layer, str) or layer not in modules or not layer:
            raise ValueError(f"unknown layer {layer!r}; select a name from model.named_modules()")
        handle = modules[layer].register_forward_hook(
            lambda _m, _args, output: found.__setitem__("output", as_tensor(output))
        )
    else:
        handle = None

    def read(x: torch.Tensor) -> torch.Tensor:
        found.clear()
        model_output = model(x)
        value = found.get("output") if layer is not None else as_tensor(model_output)
        if value is None:
            raise RuntimeError(f"layer {layer!r} did not run during the model forward pass")
        if selector is not None:
            value = selector(value)
        if not isinstance(value, torch.Tensor) or value.ndim < 1 or value.shape[0] != x.shape[0]:
            raise ValueError("selector must return a tensor retaining its batch dimension")
        return value

    try:
        yield read
    finally:
        if handle is not None:
            handle.remove()


def gaussian_entropy(log_sigma: torch.Tensor) -> torch.Tensor:
    """Differential entropy per coordinate, measured in nats."""
    import math
    return log_sigma + 0.5 * math.log(2.0 * math.pi * math.e)


def validate_optimization(*, steps: int, samples: int, learning_rate: float,
                          initial_sigma: float, entropy_weight: float,
                          distortion_ratio: float) -> None:
    if isinstance(steps, bool) or not isinstance(steps, int) or steps < 1:
        raise ValueError("steps must be a positive integer")
    if isinstance(samples, bool) or not isinstance(samples, int) or samples < 2:
        raise ValueError("samples must be an integer of at least 2")
    for name, value in (("learning_rate", learning_rate), ("initial_sigma", initial_sigma),
                        ("entropy_weight", entropy_weight), ("distortion_ratio", distortion_ratio)):
        if not isinstance(value, (int, float)) or not (0 < value < float("inf")):
            raise ValueError(f"{name} must be finite and positive")


def get_grid(grid: tuple[int, int] | None, shape: tuple[int, int]) -> tuple[int, int]:
    if grid is None:
        return shape
    if (not isinstance(grid, (tuple, list)) or len(grid) != 2 or
            any(isinstance(n, bool) or not isinstance(n, int) or n < 1 for n in grid) or
            any(g > s for g, s in zip(grid, shape))):
        raise ValueError("grid must be (rows, columns), each within the image resolution")
    return (grid[0], grid[1])
