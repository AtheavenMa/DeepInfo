# The DeepInfo method

[← README](../README.md) | [Demo](DEMO.md) | [Paper results](PAPER_RESULTS.md)

DeepInfo operationalizes two complementary measurements from [Ma et al. (ICML 2022)](https://proceedings.mlr.press/v162/ma22b.html), computational information discarding (CID) and reconstruction uncertainty (RU), and derives foreground/background concentration from the CID map.

## Computational information discarding (CID)

Let `x` be one normalized input image and `h_l(x)` the network representation at a specified layer. We add Gaussian noise `ε` independently in input coordinates, scaling it by a trainable standard deviation `σ`, and measure the difference between the clean and perturbed representations. The information discarded by a coordinate is expressed as the differential entropy `log(σ) + 0.5 log(2πe)` in nats.

The optimization minimizes

```text
mean[ (h_l(x + sigma * epsilon) - h_l(x))**2 ] / reference_distortion
    - entropy_weight * mean[ log(sigma) + 0.5*log(2*pi*e) ]
```

In this implementation, `reference_distortion` is the response MSE for an independent Gaussian noise batch at the initial standard deviation. This follows the normalized feature-difference implementation of the historical DeepInfo repository. A candidate parameter snapshot is selected from all optimization iterations by minimizing `abs(normalized_distortion - distortion_ratio)`, paralleling the original `step_result.py` selection of a target feature distortion. These choices define a finite-step Monte Carlo approximation, not an exact constrained solution or a statistical confidence interval.

The Gaussian entropy is not an unsigned saliency score. Larger values indicate noise tolerance and therefore more discarded information. If an application uses negative CID as *relevance* (as in some gene prioritization settings), use `result.relevance_map` to make that sign choice explicit.

### Coordinate system and granularity

The input tensor has shape `(C,H,W)` or `(1,C,H,W)` and must already be preprocessed for the model. A grid `(G_h,G_w)` defines trainable noise scales, which are upsampled with nearest-neighbor interpolation to `(H,W)`. This is a coarse spatial approximation. Set `grid=None` to assign a separate scale to each pixel.

By default one learned scale is shared across input channels at each spatial grid position, producing `(H,W)` maps. `per_channel=True` optimizes each selected input channel independently and produces `(C,H,W)` maps. For unselected input channels, the result contains NaNs rather than misleading zero-entropy estimates. The `channels` argument controls which channels receive injected noise.

### Model and target selection

The model is evaluated without updating its parameters; original train/eval flags and requires-grad flags are restored. `layer` accepts an exact key from `dict(model.named_modules())` and `None` selects the final output. A `selector` can select a class logit or reduce a segmentation output to a fixed region; it must preserve the leading batch dimension. The selected target must be the **same** for the clean and noisy passes. `selector` is not allowed to depend on noisy model predictions, because doing so changes the question.

`noise_samples` stochastic forwards are used at each step. `seed` controls a local PyTorch random generator, leaving the global random generator untouched; exact reproducibility across different hardware and versions is not guaranteed.

### Limits and numerical stability

The implementation clamps `log_sigma` to `[-9,2]` to avoid degeneracy. Normalization requires an initially responsive feature: if clean and noisy representations have almost zero difference, the function raises a clear error rather than manufacturing a score. CID magnitude depends on the **input scale**; compare layers only after holding preprocessing, noise parameterization and optimization settings fixed. A nonconverged finite-step estimate should not be interpreted as the theoretical optimal entropy.

## Reconstruction uncertainty (RU)

RU approaches information discarding from an input reconstruction perspective. Given a feature `f=h_l(x)` and a decoder `g`, this implementation computes the reconstructed clean input `g(f)` and reconstructions `g(h_l(x+σ⊙ε))` from Gaussian-perturbed inputs. The empirical squared reconstruction variation estimates per-coordinate variance. RU is the Gaussian reconstruction entropy

```text
ru_i = 0.5 * log(2*pi*e * max( mean_k[(decoded_k - decoded_clean)_i**2], eps ))
```

The noise optimization balances a normalized feature distortion against mean reconstruction entropy and selects the step nearest the target distortion, mirroring the decoder path of the historical implementation. This is a practical decoder-based estimator: its output depends on decoder training quality and the Gaussian assumption, and is not an exact conditional entropy. It must not be equated to raw CID or converted to CID by negating a score.

A decoder must map the chosen layer's feature tensor back to the image dimensions. For 4D convolutional features, `ConvDecoder` and `train_decoder` supply a small standard baseline. Users can pass any compatible trained `torch.nn.Module`, including their own domain-specific reconstruction architecture. Do not train the decoder on the test image alone if the aim is out-of-sample evaluation.

```python
from deepinfo import ConvDecoder, train_decoder, explain_ru

# model: a pretrained network; training_images: (N,C,H,W) normalized tensor
# Check feature shape before constructing the decoder.
feature_channels = 8
image_size = (32, 32)
decoder = ConvDecoder(feature_channels, output_channels=3, output_size=image_size)
train_decoder(model, decoder, training_images, layer="stem.1", epochs=100)
result = explain_ru(model, training_images[0], decoder, layer="stem.1")
print(result.ru_map.shape, result.ru_mean)
```

## Concentration

For a CID map `H` and foreground mask `M`, calculate the background and foreground coordinate means and return `mean(H[~M]) - mean(H[M])`. A positive difference indicates more background than foreground information discarding. Masks must contain both regions. This matches the contrast calculation visible in the original DeepInfo result-analysis code; it is *not* a bounded proportion and can be negative.

```python
from deepinfo import concentration
stats = concentration(result.cid_map, foreground_mask)
print(stats["concentration"])
```

## What we borrowed from SegJointGene

The [SegJointGene CID implementation](https://github.com/daifengwanglab/segjointgene/blob/main/tasks/dynamic_segmentation_CID/compute_CID.py) informed chunkable noise sampling, positive standard deviation parameterization, and region/channel-targeted analysis. No specialized gene or self-training machinery is required here. SegJointGene's gene-celltype scoring routine optimizes a specified output and uses a negative-CID relevance convention; that application-specific objective should not be mistaken for the original layer-wise metric.

## Paper and code references

- [ICML 2022 paper](https://proceedings.mlr.press/v162/ma22b.html)
- [Original DeepInfo implementation](https://github.com/haotianSustc/deepinfo)
- [SegJointGene CID implementation](https://github.com/daifengwanglab/segjointgene)
