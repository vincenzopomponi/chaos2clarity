# Diffusion Models — Lecture Notebooks
**Politecnico di Milano**

A self-contained series of Jupyter notebooks covering the theory and practice of diffusion models, from data representations to complete training pipelines and architectures.

---

## Notebooks

| # | Notebook | Topic |
|---|---|---|
| 1 | `01_data_modalities.ipynb` | Data types diffusion models operate on |
| 2 | `02_forward_reverse_diffusion.ipynb` | Forward & reverse process visualisation |
| 3 | `03_training_pipeline.ipynb` | Complete DDPM training from scratch |
| 4 | `04_architectures.ipynb` | U-Net, attention variants, 1D U-Net, DiT |

---

### 01 — Data Modalities

Introduces the three data types a diffusion model can operate on and builds the intuition that the diffusion algorithm is **agnostic to data semantics** — it always treats the input as a tensor $\mathbf{x} \in \mathbb{R}^d$.

**Covered topics:**
- **Images** — channel-first PyTorch format `(C, H, W)`, batching to `(B, C, H, W)`, RGB channel decomposition
- **Videos** — frame sequences `(T, C, H, W)`, kymograph (space-time slice) visualisation
- **Robot trajectories** — `(T_horizon, D_action)` format, 2D end-effector dataset (20 demos), 6-DOF joint angles
- **Normalisation** — why diffusion models require zero-mean unit-variance inputs; before/after histograms

**No external data required** — all content is generated synthetically.

---

### 02 — Forward & Reverse Diffusion

Deep-dive into the mathematics and visualisation of both DDPM processes.

**Covered topics:**
- **Noise schedules** — linear vs cosine $\beta_t$, derivation of $\bar\alpha_t$, mixing coefficient plots
- **Forward process** — reparameterisation trick $\mathbf{x}_t = \sqrt{\bar\alpha_t}\,\mathbf{x}_0 + \sqrt{1-\bar\alpha_t}\,\varepsilon$; 12-step noising grid
- **SNR analysis** — signal-to-noise ratio in dB across timesteps for both schedules
- **Oracle reverse process** — uses the stored true noise to demonstrate what a perfect denoiser would do
- **Effect of noise prediction error** — reconstruction quality vs prediction error $\sigma$
- **Side-by-side comparison** — forward (clean→noisy) and reverse (noisy→clean) in one figure

> **Configuration cell**: set `IMAGE_PATH = "path/to/your/image.jpg"` to use your own image instead of the synthetic default.

---

### 03 — Complete DDPM Training Pipeline

A full DDPM implementation trained on MNIST, built entirely from scratch.

**Covered topics:**
- **Dataset** — MNIST with `[-1, 1]` normalisation
- **Cosine noise schedule** — precomputed $\beta_t$, $\bar\alpha_t$, posterior variance
- **Forward process** — closed-form batch sampling at arbitrary $t$
- **U-Net architecture**:
  - Sinusoidal time embeddings
  - Residual blocks with AdaGN (Adaptive Group Norm) time conditioning
  - Self-attention at the bottleneck
  - Skip-connected encoder–decoder
- **Training loop** — MSE noise prediction loss, gradient clipping, cosine LR schedule, checkpoint saving
- **DDPM sampler** — full 1 000-step reverse process with denoising trajectory visualisation
- **Results** — generated samples grid + real vs generated comparison

> Increase `NUM_EPOCHS` (default 20) for publication-quality results.

---

### 04 — Architectures for Diffusion Models

Four complete, runnable neural network architectures for diffusion models, with forward-pass verification and benchmarks.

**Covered topics:**
- **Shared primitives** — sinusoidal embeddings, AdaGN, AdaLayerNorm
- **2D U-Net** (DDPM original) — ResBlocks + MaxPool + ConvTranspose, attention at bottleneck only
- **2D U-Net with full multi-scale attention** — self-attention at every resolution (Improved DDPM)
- **1D U-Net for sequences** — `(B, D, T)` format; backbone of Diffusion Policy for robot trajectories
- **Diffusion Transformer (DiT)** — patchify → AdaLayerNorm Transformer blocks → unpatchify; no U-Net
- **Parameter count & forward-pass time benchmark** — bar chart comparison across all four
- **Architecture schematics** — matplotlib diagrams of each architecture

---

## Requirements

```
python >= 3.10
torch
torchvision
torchaudio
numpy
matplotlib
Pillow
tqdm
jupyter
```

---

## Installation

### 1. Clone / download the notebooks

```bash
git clone <repo-url>
cd diffusion
```

### 2. Install PyTorch

Choose the command matching your CUDA version.
Check your version with `nvidia-smi`.

**CUDA 12.8** (latest)
```bash
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

**CUDA 12.6**
```bash
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
```

**CUDA 11.8**
```bash
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

**CPU only** (no GPU)
```bash
pip3 install torch torchvision torchaudio
```

**ROCm 6.3** (AMD GPU)
```bash
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.3
```

> Source: [https://pytorch.org/get-started/locally/](https://pytorch.org/get-started/locally/)  
> PyTorch stable release: **2.7.0** — requires Python ≥ 3.10

### 3. Install remaining dependencies

```bash
pip install numpy matplotlib Pillow tqdm jupyter
```

### 4. Launch Jupyter

```bash
jupyter notebook
```

---

## Quick check

Run this in a Python shell to verify your installation:

```python
import torch
print(f"PyTorch  : {torch.__version__}")
print(f"CUDA     : {torch.version.cuda}")
print(f"GPU      : {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'not available'}")
```

---

## References

- Ho et al., *Denoising Diffusion Probabilistic Models*, NeurIPS 2020
- Nichol & Dhariwal, *Improved Denoising Diffusion Probabilistic Models*, ICML 2021
- Song et al., *Denoising Diffusion Implicit Models*, ICLR 2021
- Peebles & Xie, *Scalable Diffusion Models with Transformers (DiT)*, ICCV 2023
- Chi et al., *Diffusion Policy: Visuomotor Policy Learning via Action Diffusion*, RSS 2023
