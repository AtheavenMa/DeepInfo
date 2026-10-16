"""Deterministically train the tiny demo CNN on generated two-color shapes."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from model import TinyCNN

ROOT = Path(__file__).resolve().parents[1]


def make_image(rng, label, size=32):
    yy, xx = np.mgrid[:size, :size]
    cy, cx = size//2 + rng.integers(-3, 4), size//2 + rng.integers(-3, 4)
    radius = rng.integers(5, 8)
    mask = (xx-cx)**2 + (yy-cy)**2 < radius**2
    bg = rng.normal(0.17, 0.045, (3, size, size)).astype(np.float32)
    bg += np.cos(xx[None] * .8 + yy[None] * .2).astype(np.float32) * .025
    image = bg.copy()
    image[label, mask] += .72
    image[1-label, mask] += .04
    image[2, mask] += .10
    return np.clip(image, 0, 1).astype(np.float32), mask.astype(np.uint8)


def main():
    torch.manual_seed(2022)
    torch.set_num_threads(2)
    rng = np.random.default_rng(2022)
    data, labels = [], []
    for i in range(160):
        cls = i % 2
        x, _ = make_image(rng, cls)
        data.append(x)
        labels.append(cls)
    images = torch.from_numpy(np.stack(data))
    targets = torch.tensor(labels, dtype=torch.long)
    network = TinyCNN().train()
    optim = torch.optim.Adam(network.parameters(), lr=0.022)
    for epoch in range(75):
        order = torch.randperm(len(images))
        for part in order.split(32):
            optim.zero_grad(set_to_none=True)
            loss = F.cross_entropy(network(images[part]), targets[part])
            loss.backward()
            optim.step()
    network.eval()
    with torch.inference_mode():
        accuracy = (network(images).argmax(1) == targets).float().mean().item()
    sample, mask = make_image(np.random.default_rng(1701), 0)
    root_weights = ROOT / "assets/checkpoints"
    root_examples = ROOT / "assets/examples"
    root_weights.mkdir(parents=True, exist_ok=True)
    root_examples.mkdir(parents=True, exist_ok=True)
    torch.save(network.cpu().state_dict(), root_weights / "tiny_cnn.pt")
    np.savez_compressed(root_examples / "colored_disk.npz", image=sample, mask=mask, label=0)
    (root_weights / "metadata.json").write_text(json.dumps({
        "architecture": "model.TinyCNN", "training_data": "160 generated red/green disk images",
        "generation_script": "scripts/make_assets.py", "epochs": 75, "seed": 2022,
        "training_accuracy": round(accuracy, 6), "sample_kind": "synthetic",
        "notes": "Illustrative checkpoint trained here, not an ICML 2022 experimental model."
    }, indent=2) + "\n")
    (root_examples / "metadata.json").write_text(json.dumps({
        "generation_script": "scripts/make_assets.py", "seed": 1701,
        "description": "Synthetic 32x32 RGB disk and binary foreground mask",
        "benchmark": False,
    }, indent=2) + "\n")
    print(f"Demo model training accuracy on generated training set: {accuracy:.3f}")


if __name__ == "__main__":
    main()
