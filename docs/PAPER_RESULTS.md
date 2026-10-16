# Results from ICML 2022

[← README](../README.md) | [Method](METHOD.md) | [Demo](DEMO.md)

The statements here describe the **published paper and conference presentation**. They are not outputs of `python demo.py`, which uses a small, explicitly synthetic example.

## Publication

Haotian Ma, Hao Zhang, Fan Zhou, Yinqing Zhang, and Quanshi Zhang. *Quantification and Analysis of Layer-wise and Pixel-wise Information Discarding*. ICML 2022, Proceedings of Machine Learning Research 162:14664–14698.

- [Proceedings page and abstract](https://proceedings.mlr.press/v162/ma22b.html)
- [Camera-ready paper PDF](https://proceedings.mlr.press/v162/ma22b/ma22b.pdf)
- [Official ICML 2022 slides](https://icml.cc/media/icml-2022/Slides/17920.pdf)

## 1. Measurements across layers and models

The paper uses two complementary input-space entropy metrics, CID and RU, to investigate representation capacity. Its Figure 1 contrasts this objective with the magnitude of Grad-CAM, gradients, and CAM, which need not remain comparable across different layers. In the experiments, CID and RU enable quantitative cross-layer and cross-network comparisons under a shared definition of input information.

## 2. Pixel-wise CID and RU

[See original figures in the ICML 2022 conference presentation](https://icml.cc/media/icml-2022/Slides/17920.pdf)

The presentation (slide 7) compares AlexNet, VGG-16 and VGG-19 on image examples. It reports more information discarding in background than in foreground in many CID maps, and RU maps that retain distinctions in edge and color information. These observations concern the particular experimental settings, not a guarantee for every model and dataset.

## 3. Network architecture and depth

[See original figures in the ICML 2022 conference presentation](https://icml.cc/media/icml-2022/Slides/17920.pdf)

The presentation compares deeper and shallower networks and visualizes the progressive change of input information across selected convolutional layers. The main claim is that intermediate representations differ in how much and where information is discarded, and the degree of concentration can change with depth.

## 4. Compression and knowledge distillation

The paper analyzes how network compression and distillation alter the information retained by intermediate features. The reported studies show that network optimization choices can affect not only prediction performance but also representation-information behavior. For settings and quantitative comparisons, use the camera-ready paper and its appendix rather than treating the synthetic demo as an empirical replication.

## 5. Comparison with earlier attribution methods

[See original figures in the ICML 2022 conference presentation](https://icml.cc/media/icml-2022/Slides/17920.pdf)

The conference presentation (slide 10) reports comparative weakly supervised localization and descriptive-accuracy evaluations with CAM, Grad-CAM, LRP and related methods. Its localization table gives **.493** (ResNet-50) and **.578** (VGG-16) for CID, compared with Grad-CAM **.355** and **.507**, respectively. The published comparisons evaluate the paper's implementation and datasets; the new repository does not claim to have rerun those full experiments.

## Figures and permissions

Paper figures are linked to the official [ICML 2022 presentation](https://icml.cc/media/icml-2022/Slides/17920.pdf); no conference slide panels are bundled. The optional `scripts/extract_paper_figures.py` can render them locally if the source is accessible and the use is permitted.
