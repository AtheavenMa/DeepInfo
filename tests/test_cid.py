import math
import torch
import pytest
from deepinfo import explain, concentration, layerwise


def test_entropy_mean_matches_gaussian_formula():
    model = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(2*4*4, 3))
    image = torch.randn(2, 4, 4)
    r = explain(model, image, grid=(2, 2), steps=6, noise_samples=4, seed=11)
    assert r.cid_map.shape == (4, 4)
    assert math.isclose(r.cid_map.mean().item(), r.cid_mean, abs_tol=1e-6)
    assert torch.allclose(r.cid_map, torch.log(r.sigma_map) + .5*math.log(2*math.pi*math.e))
    assert len(r.history) == 6


def test_reproducibility_and_original_model_state():
    torch.manual_seed(4)
    model = torch.nn.Sequential(torch.nn.Conv2d(2, 3, 3, padding=1), torch.nn.BatchNorm2d(3),
                                torch.nn.ReLU(), torch.nn.Flatten(), torch.nn.Linear(3*6*6, 1))
    model.train()
    model[1].eval()
    original_modes = [m.training for m in model.modules()]
    model[0].weight.requires_grad_(False)
    flags = [p.requires_grad for p in model.parameters()]
    weights = [p.detach().clone() for p in model.parameters()]
    image = torch.rand(2, 6, 6)
    a = explain(model, image, layer="2", grid=(3,3), steps=5, noise_samples=3, seed=9)
    b = explain(model, image, layer="2", grid=(3,3), steps=5, noise_samples=3, seed=9)
    assert torch.equal(a.cid_map, b.cid_map)
    assert flags == [p.requires_grad for p in model.parameters()]
    assert original_modes == [m.training for m in model.modules()]
    assert all(torch.equal(old, new) for old, new in zip(weights, model.parameters()))
    assert all(len(mod._forward_hooks) == 0 for mod in model.modules())


def test_channel_specific_noise_and_invalid_requests():
    model = torch.nn.Sequential(torch.nn.Conv2d(3, 2, 1), torch.nn.ReLU())
    x = torch.rand(3, 6, 6)
    r = explain(model, x, channels=[0,2], per_channel=True, grid=(2,3), steps=4,
                noise_samples=3)
    assert r.cid_map.shape == (3,6,6)
    assert torch.isnan(r.cid_map[1]).all()
    assert torch.isfinite(r.cid_map[[0,2]]).all()
    with pytest.raises(ValueError, match="unknown layer"):
        explain(model, x, layer="missing", grid=(2,2), steps=2)
    with pytest.raises(ValueError, match="valid input channel"):
        explain(model, x, channels=[3], grid=(2,2), steps=2)
    with pytest.raises(ValueError, match="grid"):
        explain(model, x, grid=(10,10), steps=2)


def test_concentration_and_layerwise():
    cid = torch.tensor([[1.,2.],[3.,4.]])
    mask = torch.tensor([[True,False],[True,False]])
    r=concentration(cid, mask)
    assert r["foreground_cid"] == 2.
    assert r["background_cid"] == 3.
    assert r["concentration"] == 1.
    with pytest.raises(ValueError):
        concentration(cid, torch.ones_like(mask))
