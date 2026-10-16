import torch
import pytest
from deepinfo import ConvDecoder, explain_ru, train_decoder


def test_ru_zero_damage_and_train_decoder():
    torch.manual_seed(0)
    model = torch.nn.Sequential(torch.nn.Conv2d(2, 4, 3, padding=1), torch.nn.ReLU())
    decoder = ConvDecoder(4, 2, (8, 8), hidden_channels=6)
    data = torch.rand(6, 2, 8, 8)
    original = [p.detach().clone() for p in model.parameters()]
    history = train_decoder(model, decoder, data, layer="1", epochs=7)
    assert len(history) == 7 and history[-1] <= history[0]
    assert all(torch.equal(p,q) for p,q in zip(original, model.parameters()))
    r = explain_ru(model, data[0], decoder, layer="1", steps=5, noise_samples=3, grid=(2,2))
    assert r.ru_map.shape == (2,8,8)
    assert torch.isfinite(r.ru_map).all()
    assert len(r.history)==5


def test_ru_rejects_bad_decoder():
    model = torch.nn.Sequential(torch.nn.Conv2d(2,2,1), torch.nn.ReLU())
    decoder = torch.nn.Identity()
    x = torch.rand(2,8,8)
    explain_ru(model, x, decoder, layer="1", steps=2, noise_samples=2, grid=(2,2))
    with pytest.raises(ValueError, match="decoder must map"):
        explain_ru(model, x, torch.nn.Sequential(torch.nn.AdaptiveAvgPool2d((2,2))),
                   layer="1", steps=2, noise_samples=2)
