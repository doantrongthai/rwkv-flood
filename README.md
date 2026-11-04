# RWKV-Flood: Lightweight Linear-Attention Networks for Flood Segmentation

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/doantrongthai/rwkv-flood/blob/main/screening_rwkv_architectures.ipynb)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![RWKV Attention](https://img.shields.io/badge/Architecture-Linear%20Attention%20%7C%20O(N)-blue.svg)](#)

A high-performance benchmark suite featuring **RWKV (Receptance Weighted Key Value)** linear-attention segmentation architectures and real-time convolutional baselines for flood monitoring. Designed for linear $O(N)$ computational complexity, absolute numerical stability, and efficient evaluation at **256x256** resolution.

---

## ⚡ Master Google Colab Screening Notebook

Run the complete 21-architecture screening benchmark directly on Google Colab with hardware acceleration, automated NaN-divergence protection, and clean 1-line-per-epoch logging:

👉 **[Open `screening_rwkv_architectures.ipynb` in Google Colab](https://colab.research.google.com/github/doantrongthai/rwkv-flood/blob/main/screening_rwkv_architectures.ipynb)**

### Key Features of the Screening Pipeline:
- **Numerical Stability Shield:** Strictly enforces $T \le 4096$ token bounds across all multi-resolution stages, clamps key projections $k \in [-15.0, 15.0]$, bounds time-decay parameters $w \in [-8.0, 3.0]$, and protects denominators with $10^{-7}\epsilon$.
- **Automatic NaN Skip:** Any model encountering non-finite loss or gradients is immediately halted and skipped to the next architecture without crashing the execution pipeline.
- **Clean 1-Line-Per-Epoch Logging (`--quiet`):** Prevents output truncation from thousands of progress bar updates.
- **Paper-Ready Summary Table:** Automatically exports publication-ready LaTeX metrics and CSV leaderboards across datasets (`floodkaggle` and `aifloodsense`).

---

## 🌟 Supported Architectures & Provenance

All RWKV architectures in this repository are faithfully ported from their original publications and source repositories, adapted with input resolution $256 \times 256$ and binary flood segmentation heads ($1$ output channel). Pure PyTorch native fallbacks are included so models run out-of-the-box without requiring precompiled C++/CUDA WKV extensions.

| Model | Original Publication | Source Repository | Commit Hash | Key Architectural Focus |
| :--- | :--- | :--- | :--- | :--- |
| **RSRWKV** | Li et al. (*IEEE TGRS*, 2025; *arXiv:2503.20382*) | [Ling-yunchi/RSRWKV](https://github.com/Ling-yunchi/RSRWKV) | `ede06a56bc3768e93444f814025fbc29d2e7919e` | Remote Sensing Water Segmentation (GLH-Water: 85.06% IoU), MVCShift multi-scale dilated shift, ECA attention |
| **SCRWKV** | Zhang et al. (*ICML*, 2026; *arXiv:2605.14926*) | [zhxhzy/SCRWKV](https://github.com/zhxhzy/SCRWKV) | `f3a1063f6043c769ef828c2ae95edf1029d76c40` | Ultra-compact (1.22M params), Structure-Field Encoder (SFE) with AMCM multi-kernel convs, CSHF harmonic fusion decoder |
| **U-RWKV** | Ye et al. (*arXiv:2507.11415*, 2025) | [hbyecoding/U-RWKV](https://github.com/hbyecoding/U-RWKV) | `32f2d3c8bd52ef53b34cabcc8f879c1d39f183e0` | Direction-Adaptive RWKV Module (DARM), 2D spatial q-shift, SASE blocks |
| **VRWKV** | Duan et al. (*ICLR 2025 Spotlight*; *arXiv:2403.02308*) | [OpenGVLab/Vision-RWKV](https://github.com/OpenGVLab/Vision-RWKV) | `0e0e38a6fd4475ae3817e14420ab85fab08d9097` | Vision-RWKV backbone, 2D bi-directional WKV attention, squared-ReLU FFN |
| **SegRWKV** | *JCDE*, 2024 | [luminescentfrr/SegRWKV](https://github.com/luminescentfrr/SegRWKV) | `8dbd13217c4fb734a03ffdb020833e6e83152c53` | RWKV attention blocks at bottleneck and deep stages, residual skip connections |
| **SegFormer (B0)** | Xie et al. (*NeurIPS 2021*) | Official MiT-B0 | Standard Baseline | Lightweight hierarchical transformer |
| **STDC1** | Fan et al. (*CVPR 2021*) | Short-Term Dense Concatenate | Standard Baseline | Dual-path real-time detail preservation |
| **Fast-SCNN** | Poudel et al. (*BMVC 2019*) | Fast Segmentation ConvNet | Standard Baseline | Multi-branch real-time edge architecture |
| **UNet** | Ronneberger et al. (*MICCAI 2015*) | Standard U-Net | Standard Baseline | Encoder-decoder with direct skip connections |
| **ULite** | Lightweight ViT segmentation | Standard U-Lite | Standard Baseline | Ultra-compact axial attention baseline |
| **DABNet** | Depth-wise Asymmetric Bottleneck | Standard DABNet | Standard Baseline | Dilated asymmetric convolutional blocks |
| **DeepLabV3** | Chen et al. (*arXiv:1706.05587*) | ResNet50 ASPP | Standard Baseline | Heavy contextual baseline (pre-trained ImageNet) |
| **ENet** | Paszke et al. (*arXiv:1606.02147*) | Real-time ENet | Standard Baseline | Early downsampling for low latency |
| **ESPNetv2** | Mehta et al. (*CVPR 2019*) | ESPNetv2 | Standard Baseline | Extremely efficient spatial pyramid dilated convs |
| **EDANet** | Lo et al. (*arXiv:1809.06323*) | Efficient Dense Asymmetric Net | Standard Baseline | Dense connectivity with asymmetric convolutions |
| **SegNet** | Badrinarayanan et al. (*TPAMI 2017*) | SegNet | Standard Baseline | Max pooling indices decoder |

---

## 🚀 Quick Start

### Installation
```bash
git clone https://github.com/doantrongthai/rwkv-flood.git
cd rwkv-flood
pip install -r requirements.txt
```

### 1. Training an RWKV Model
Train `u_rwkv` (or any other architecture) on flood segmentation at $256 \times 256$:
```bash
# Train RSRWKV (Remote Sensing Water Segmentation)
python main.py --model rsrwkv --dataset floodkaggle --size 256 --batch_size 4 --epochs 50

# Train SCRWKV (Ultra-compact 1.22M Structure-Aware)
python main.py --model scrwkv --dataset floodkaggle --size 256 --batch_size 4 --epochs 50

# Train U-RWKV
python main.py --model u_rwkv --dataset floodkaggle --size 256 --batch_size 4 --epochs 50

# Train Vision-RWKV (VRWKV)
python main.py --model vrwkv --dataset floodkaggle --size 256 --batch_size 4 --epochs 50

# Train SegRWKV
python main.py --model seg_rwkv --dataset floodkaggle --size 256 --batch_size 4 --epochs 50
```

### 2. Multi-Seed Statistical Evaluation
Run rigorous 5-seed experiments ($[42, 123, 456, 789, 2024]$) to compute mean and standard deviations for research publications:
```bash
python benchmark.py --model u_rwkv --dataset floodkaggle --size 256 --multiseed
```

### 3. Cross-Dataset Zero-Shot Domain Transfer
Evaluate domain generalization by training on a source dataset and testing zero-shot on other domains:
```bash
python cross_dataset_eval.py --model u_rwkv --weights outputs/u_rwkv_best.pth --source_dataset floodkaggle --target_datasets floodkaggle aifloodsense floodvn --size 256
```

### 4. ONNX Export for Deployment
Export models directly to ONNX format for TensorRT, ONNX-Runtime, or edge accelerator deployment:
```bash
python export.py --model u_rwkv --size 256 --output outputs/u_rwkv.onnx
```

---

## 📁 Repository Structure

```
rwkv-flood/
├── models/
│   ├── __init__.py           # Model factory and deterministic initialization
│   ├── rwkv_block.py         # WKV linear attention operator with pure PyTorch fallback
│   ├── u_rwkv.py             # U-RWKV (Direction-Adaptive RWKV + SASE)
│   ├── vrwkv.py              # Vision-RWKV (VRWKV spatial-mix + channel-mix)
│   ├── seg_rwkv.py           # SegRWKV (Bottleneck RWKV + residual skips)
│   ├── segformer.py          # SegFormer (MiT-B0) baseline
│   ├── stdc1.py              # STDC1 baseline
│   ├── fast_scnn.py          # Fast-SCNN baseline
│   ├── unet.py               # UNet baseline
│   └── ...                   # Additional real-time baselines
├── losses/                   # BCE, Dice, Focal, Lovasz, Tversky loss functions
├── utils/
│   ├── dataloader.py         # Deterministic dataset loaders (default size=256)
│   ├── metrics.py            # mIoU, Dice, Boundary-IoU, FPS, latency, GMACs
│   └── trainer.py            # Unified training and validation engine
├── main.py                   # Main CLI entry point
├── benchmark.py              # Multi-seed and reproducibility test suite
├── export.py                 # ONNX deployment exporter
├── cross_dataset_eval.py     # Cross-domain generalization benchmark
├── requirements.txt
└── README.md
```

---

## 📄 License
This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

