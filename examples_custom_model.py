"""Minimal external-model integration without downloading any weights."""
import torch
from deepinfo import explain

# Replace this with your trained model and correctly preprocessed input.
model = torch.nn.Sequential(
    torch.nn.Conv2d(3, 8, 3, padding=1),
    torch.nn.ReLU(),
    torch.nn.AdaptiveAvgPool2d(1),
)
image = torch.rand(3, 32, 32)

result = explain(model, image, layer="1", grid=(8, 8), steps=20,
                 noise_samples=4, seed=2022)
print(f"Mean CID = {result.cid_mean:.5f} nats")
print(f"Map shape = {tuple(result.cid_map.shape)}")
