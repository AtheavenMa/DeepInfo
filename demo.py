"""Offline CID demonstration on a bundled trained CNN and synthetic image.

Run: python demo.py
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import platform
import time
import numpy as np
import torch

from deepinfo import explain, concentration, layerwise
from model import TinyCNN

ROOT = Path(__file__).resolve().parent


def hash_file(file):
    return hashlib.sha256(Path(file).read_bytes()).hexdigest()


def plot(image, foreground, results, path, curve_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize

    keys = list(results)
    maps = [results[k].cid_map.numpy() for k in keys]
    low = min(float(m.min()) for m in maps)
    high = max(float(m.max()) for m in maps)
    norm = Normalize(low, high)
    fig, axes = plt.subplots(1, 4, figsize=(15.2, 4.5), constrained_layout=True)
    fig.patch.set_facecolor("white")
    axes[0].imshow(np.moveaxis(image, 0, -1))
    axes[0].contour(foreground, levels=[0.5], linewidths=1.5, colors="white")
    axes[0].set_title("Input and foreground", fontsize=12, weight="bold")
    for j, key in enumerate(keys):
        im = axes[j+1].imshow(maps[j], cmap="viridis", norm=norm, interpolation="nearest")
        axes[j+1].set_title(f"CID  ·  {key}", fontsize=12, weight="bold")
    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
    fig.colorbar(im, ax=axes[1:], location="bottom", shrink=.78,
                 label="Gaussian input-noise entropy (nats); higher = more discarded", pad=.07)
    fig.savefig(path, dpi=180, facecolor="white")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.0, 4.6), constrained_layout=True)
    vals = [results[key].cid_mean for key in keys]
    ax.plot(range(len(keys)), vals, linewidth=2.3, color="#26799e", marker="o", markersize=9)
    ax.set_xticks(range(len(keys)), labels=keys)
    ax.set_ylabel("Mean CID (nats)")
    ax.set_title("Layer-wise information discarding · synthetic demo", weight="bold")
    ax.grid(alpha=.2)
    ax.spines[["top", "right"]].set_visible(False)
    fig.savefig(curve_path, dpi=180, facecolor="white")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample", type=Path, default=ROOT/"assets/examples/colored_disk.npz")
    parser.add_argument("--weights", type=Path, default=ROOT/"assets/checkpoints/tiny_cnn.pt")
    parser.add_argument("--output", type=Path, default=ROOT/"outputs/demo")
    parser.add_argument("--steps", type=int, default=32)
    parser.add_argument("--noise-samples", type=int, default=8)
    parser.add_argument("--grid", type=int, default=8, help="Number of noise cells along each axis")
    parser.add_argument("--seed", type=int, default=2022)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    args = parser.parse_args()
    torch.set_num_threads(2)
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is unavailable")
    if device == "mps" and not torch.backends.mps.is_available():
        parser.error("MPS is unavailable")
    sample = np.load(args.sample)
    image = torch.from_numpy(sample["image"]).to(device)
    mask = torch.from_numpy(sample["mask"].astype(np.bool_))
    model = TinyCNN().to(device)
    model.load_state_dict(torch.load(args.weights, map_location=device, weights_only=True))
    model.eval()
    names = [("Shallow", "stem.1"), ("Mid", "block.2"), ("Output", None)]
    results = {}
    started = time.perf_counter()
    for name, layer in names:
        selected = (lambda y: y[:, 0:1]) if layer is None else None
        r = explain(model, image, layer=layer, selector=selected,
                    grid=(args.grid, args.grid), steps=args.steps,
                    noise_samples=args.noise_samples, seed=args.seed)
        results[name] = r
    elapsed = time.perf_counter() - started
    with torch.inference_mode():
        probs = model(image.unsqueeze(0)).softmax(1)[0].cpu().tolist()
    args.output.mkdir(parents=True, exist_ok=True)
    plot(image.cpu().numpy(), mask.numpy(), results, args.output/"demo_output.png",
         args.output/"layerwise.png")
    np.savez_compressed(args.output/"attribution.npz",
                        image=image.cpu().numpy(), foreground=mask.numpy(),
                        **{f"{k.lower()}_cid":v.cid_map.numpy() for k,v in results.items()})
    payload = {
        "task": "red-vs-green synthetic disk classification", "dataset": "synthetic (not the ICML paper benchmark)",
        "device": device, "seed": args.seed, "steps": args.steps,
        "noise_samples": args.noise_samples, "grid": [args.grid, args.grid],
        "runtime_seconds": round(elapsed, 3), "class_probabilities": probs,
        "layerwise_cid_nats": layerwise(results),
        "concentration": {k:concentration(v.cid_map, mask) for k,v in results.items()},
        "selected_steps": {k:v.selected_step for k,v in results.items()},
        "normalized_distortions": {k:v.normalized_distortion for k,v in results.items()},
        "sample_sha256": hash_file(args.sample), "weights_sha256": hash_file(args.weights),
        "versions": {"python": platform.python_version(), "torch": torch.__version__, "numpy": np.__version__},
    }
    (args.output/"summary.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(f"DeepInfo | device={device} | {elapsed:.2f}s | {args.grid}x{args.grid} grid")
    print("Layer              Mean CID (nats)      Selected step      Contrast (BG−FG)")
    for name, value in results.items():
        print(f"{name:16s}   {value.cid_mean:+.5f}             {value.selected_step:3d}             "
              f"{payload['concentration'][name]['concentration']:+.5f}")
    print(f"Saved figures, arrays and metadata to {args.output}")
    return payload


if __name__ == "__main__":
    main()
