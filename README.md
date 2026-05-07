# chaos2clarity

Lecture materials for the sessions held on **May 6** and **May 13, 2026**, at Politecnico di Milano by PhD student **Vincenzo Pomponi**.

This repository collects the complete teaching material, mathematical derivations, theoretical notes, and practical Python implementations presented during two lectures on **Diffusion Models**, designed for master students interested in modern generative AI.

The title **"chaos2clarity"** reflects the central idea behind diffusion-based generative modeling:

> starting from pure random noise (chaos) and progressively transforming it into structured, meaningful data (clarity).

---

## 📚 Lecture Overview

The seminar series was divided into two complementary sessions:

| Lecture | Date | Main Topic |
|---------|------|------------|
| **Lecture 1** | May 6, 2026 | Mathematical Foundations of Diffusion Models |
| **Lecture 2** | May 13, 2026 | Sampling Algorithms, Architectures, and Practical Coding |

---

# 🧠 Lecture 1 — Mathematical Foundations of Diffusion Models

The first lecture provides a rigorous and intuitive mathematical derivation of the principles behind diffusion probabilistic models.

## Topics Covered

### 1. Forward Diffusion Process (Noise Injection)

We begin by formalizing how a clean datapoint \(x_0\) is gradually corrupted through a Markov chain of Gaussian perturbations:

\[
q(x_t|x_{t-1}) = \mathcal{N}(x_t;\sqrt{1-\beta_t}x_{t-1}, \beta_t I)
\]

Students are guided through:

- the meaning of the variance schedule \(\beta_t\),
- why repeated Gaussian perturbations converge toward isotropic noise,
- and the closed-form expression of sampling directly at timestep \(t\).

---

### 2. Reverse Diffusion Process (Denoising)

After understanding how information is destroyed, we mathematically construct the inverse process:

\[
p_\theta(x_{t-1}|x_t)
\]

which is learned by a neural network that predicts the noise component added at each timestep.

This section includes:

- probabilistic interpretation of reverse Markov transitions,
- why direct inversion is intractable,
- approximation through learned Gaussian conditionals.

---

### 3. Derivation of the Training Objective

A full derivation of the **Evidence Lower Bound (ELBO)** is presented and simplified step by step until obtaining the practical training loss used in modern diffusion models:

\[
L_{\text{simple}} = \mathbb{E}_{x_0,\epsilon,t}\left[\|\epsilon - \epsilon_\theta(x_t,t)\|^2\right]
\]

Special attention is dedicated to:

- why noise prediction is sufficient,
- why ELBO is used to approximate the log-likelihood of the data,
- computational simplifications that make training feasible.

---

# 🚀 Lecture 2 — From Theory to Practice

The second lecture moves from the mathematical framework toward modern implementations and accelerated generation techniques.

---

## Topics Covered

### 1. DDPM — Denoising Diffusion Probabilistic Models

We introduce the original sampling framework proposed by Ho et al., where generation is performed by iteratively reversing the noising process over many timesteps.

Main points discussed:

- ancestral sampling,
- stochastic denoising,
- computational cost of long diffusion chains,
- relation with the loss derived in Lecture 1.

---

### 2. DDIM — Denoising Diffusion Implicit Models

To overcome the slow sampling of DDPMs, we derive the deterministic DDIM formulation:

- non-Markovian reverse trajectories,
- reduced number of denoising steps,
- latent interpolation properties,
- faster inference with minimal quality degradation.

A direct DDPM vs DDIM comparison is included.

---

### 3. Diffusion Model Architectures

This section explores the most common neural backbones used to parameterize \(\epsilon_\theta\):

#### • U-Net based Diffusion Networks
The classical architecture used in image generation.

#### • Attention-Enhanced Diffusion Models
Integration of self-attention and cross-attention mechanisms.

#### • Latent Diffusion Architectures
Diffusion in compressed latent spaces for computational efficiency.

#### • Transformer-based Diffusion Backbones
Recent alternatives replacing convolutions with token-based processing.

Architectural trade-offs in terms of:

- speed,
- memory usage,
- generation fidelity,
- scalability

are analyzed in detail.

---

### 4. Practical Python Coding Session

The final part of the seminar includes hands-on Python implementation of:

- forward noising simulation,
- reverse denoising loop,
- beta schedule definition,
- simple diffusion training loop,
- DDPM sampling,
- DDIM accelerated sampling.

The coding notebooks are intentionally educational and minimal, focusing on clarity over framework complexity.

---

# 📂 Repository Structure

```bash
chaos2clarity/
│
├── lecture1/
│   ├── notes/
│   └── slides/
│
├── lecture2/
│   ├── notebooks/
│   ├── notes/
│   └── slides/
│
└── README.md
