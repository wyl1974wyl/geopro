# GeoPro: Training-Free Class-Incremental Learning via Log-Space Geometric Statistics on DINOv3 Features

Official implementation of the paper:

> **GeoPro: Training-Free Class-Incremental Learning via Log-Space Geometric Statistics on DINOv3 Features**
> Luobin Wang, Ji Fan, Xiongxin Tang, Fanjiang Xu (ICASSP 2026)

GeoPro is a fully training-free class-incremental learning (CIL) framework built on geometric statistics of frozen DINOv3 features:

1. **SDGM prototype** — a sign-decomposed geometric-mean class center that votes per-dimension signs and averages magnitudes multiplicatively;
2. **Log-space covariance estimation** around the SDGM center's log-space image, shrunk towards a pooled Ledoit--Wolf target in closed form;
3. **Global between-class subspace projection** followed by Mahalanobis classification.

Every step is closed-form: no gradients, no stored exemplars, and only per-class statistics are maintained between tasks.

## Method Overview

```
frozen DINOv3 [CLS]  ->  L2 normalize  ->  signed log map  g = sign(f) * log(|f| + eps)
    |
    +-- SDGM prototype:  s_c = sign(mean(sign(f_i))),  m_c = exp(mean(log|f_i|)),  p_c = s_c * m_c
    +-- log-space center:  nu_c = s_c * mean(log|f_i|)
    +-- covariance:        Sigma_c = (1/n_c) sum (g_i - nu_c)(g_i - nu_c)^T
    +-- shrinkage:         Sigma_c^reg = (1 - a_c) Sigma_c + a_c Sigma_pool^LW
    |
    +-- between-class scatter S_B over all centers -> top-k eigenvectors V_k
    +-- Mahalanobis classification in the projected k-dim subspace
```

## Repository Layout

```
GeoPro/
├── configs/                # experiment configs (CIFAR-100, ImageNet-100, ImageNet-R, CUB-200)
├── geopro/
│   ├── backbones/          # frozen feature extractors (DINOv3, DINOv2, CLIP)
│   ├── heads/              # GeoPro head and frozen-feature baselines
│   ├── data/               # dataset loaders and incremental task schedulers
│   ├── stats/              # SDGM, log-space covariance, shrinkage, subspace
│   └── cil/                # incremental evaluation engine and metrics
├── scripts/                # entry points (feature extraction, CIL runs, analysis)
├── requirements.txt
└── README.md
```

## Installation

```bash
git clone https://github.com/your-org/GeoPro.git
cd GeoPro
pip install -r requirements.txt
```

## Pre-trained Weights

All experiments use **frozen** backbones (DINOv3-ViT-B/16 by default; DINOv2 and CLIP ViT-B/16 for cross-backbone controls). Pre-trained backbone weights will be released upon acceptance of the paper. Place them under `checkpoints/` (see `geopro/backbones/__init__.py` for expected paths). GeoPro itself has **no trainable parameters**.

## Data Preparation

| Dataset | Protocol | Notes |
|---|---|---|
| CIFAR-100 | B0-Inc10 / B50-Inc10 | automatic download |
| ImageNet-100 | B0-Inc10 / B50-Inc10 | 100-class subset of Zhou et al. (pre-trained-model CIL protocol) |
| ImageNet-R | B0-Inc10 | fixed-seed 100-class subset |
| CUB-200 | CEC/FACT few-shot (1/5-shot) | final-session Base/Novel/HM |

## Quick Start

```bash
# 1. Extract frozen features once (shared by all frozen-feature heads)
python scripts/extract_features.py --config configs/cifar100_b0_inc10.yaml

# 2. Run GeoPro (and optional baselines on identical features)
python scripts/run_cil.py --config configs/cifar100_b0_inc10.yaml --method geopro
python scripts/run_cil.py --config configs/cifar100_b0_inc10.yaml --method all

# 3. Feature analysis and ablations (Tables: stats, analysis)
python scripts/run_analysis.py --config configs/cifar100_b0_inc10.yaml --analysis all
```

## Main Results (average incremental accuracy, %)

| Method | CIFAR-100 B0-Inc10 | ImageNet-100 B0-Inc10 |
|---|---|---|
| SimpleCIL (AM-NCM) | 88.42 | 88.10 |
| FeCAM (full cov.) | 89.61 | 88.85 |
| RanPAC | 89.80 | 89.18 |
| **GeoPro (Ours)** | **90.96** | **90.62** |

GeoPro also leads under distribution shift (ImageNet-R: **72.88**), few-shot CIL (CUB-200 5-shot HM: **83.04**), and 30% symmetric label noise (**82.3** vs. RanPAC 75.5).

## Citation

```bibtex
@inproceedings{wang2026geopro,
  title     = {GeoPro: Training-Free Class-Incremental Learning via Log-Space Geometric Statistics on DINOv3 Features},
  author    = {Wang, Luobin and Fan, Ji and Tang, Xiongxin and Xu, Fanjiang},
  booktitle = {Proc. IEEE Int. Conf. Acoust. Speech Signal Process. (ICASSP)},
  year      = {2026}
}
```

## License

Apache License 2.0
