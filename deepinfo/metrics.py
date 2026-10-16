"""Layer-wise and foreground/background summaries of CID and RU maps."""
from __future__ import annotations
import torch


def concentration(cid_map: torch.Tensor, foreground: torch.Tensor) -> dict[str, float]:
    """Background minus foreground CID means; positive means more background discarded.

    `foreground` is a boolean mask at image resolution. This is an entropy
    *contrast* as used by the historical DeepInfo result aggregation; it is
    not a ratio and may be negative.
    """
    values = torch.as_tensor(cid_map, dtype=torch.float64)
    mask = torch.as_tensor(foreground, dtype=torch.bool)
    if values.ndim == 3:
        values = values.mean(dim=0)
    if values.ndim != 2 or mask.shape != values.shape:
        raise ValueError("cid_map must be (H,W) or (C,H,W), mask must be (H,W)")
    if not mask.any() or mask.all() or not torch.isfinite(values).all():
        raise ValueError("both foreground and background must be nonempty; scores must be finite")
    fg = float(values[mask].mean().item())
    bg = float(values[~mask].mean().item())
    return {"foreground_cid": fg, "background_cid": bg, "concentration": bg - fg}


def layerwise(results: dict[str, object]) -> dict[str, float]:
    """Summarize fitted CIDResult/RUResult objects by layer name."""
    output = {}
    for name, result in results.items():
        if hasattr(result, "cid_mean"):
            output[name] = float(result.cid_mean)
        elif hasattr(result, "ru_mean"):
            output[name] = float(result.ru_mean)
        else:
            raise TypeError(f"unrecognized information discarding result for {name}")
    return output
