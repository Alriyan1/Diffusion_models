# Diffusion Models Collection

Comprehensive repository of diffusion-model experiments, notebooks, and example code focused on Denoising Diffusion Probabilistic Models (DDPM), Stable Diffusion variants, ControlNet, diffusion transformers, and video generation. This workspace is arranged as a learning and research playground — suitable for experiments, reproducible training, and reference implementations.

---

## Key Goals
- Provide readable, well-documented reference implementations of diffusion methods.
- Include runnable notebooks and minimal scripts for training, sampling, and demos.
- Organize components so researchers and engineers can mix-and-match schedulers, model backbones, and datasets.

---

## Contents
- `01_DenoisingDiffusionProbabilisticModels(DDPM)/` — Reference DDPM implementations and examples.
	- `default.yaml` — Example configuration for training and sampling (hyperparameters, optimizer, scheduler settings).
	- `denoising-diffusion-probabilistic-model.ipynb` — Notebook walk-through of DDPM concepts and code.
	- `linearNoiseScheduler.py` — Simple linear noise scheduler implementation used by DDPM experiments.
	- `mnist_dataset.py` — Lightweight MNIST dataset loader and preprocessing utilities.
	- `unet.py` — U-Net backbone used as denoiser model in DDPM experiments.
	- `train_ddpm.py` — Minimal training script for DDPM-style models.
	- `sample_ddpm.py` — Script to generate samples from a trained DDPM model checkpoint.

- `02_StableDiffusion/` — Experimental notebooks exploring Stable Diffusion / latent diffusion ideas.
	- `ConditionalLDM.ipynb`, `unconditionalLDM.ipynb` — Notebooks for conditional and unconditional latent diffusion models.

- `03_ControlNet/` — ControlNet integration experiments and demo app.
	- `ControlNet.ipynb` — Research notebook for conditioning with spatial hints.
	- `app.py` — Small demo application (Flask/Streamlit-compatible) to showcase conditioned generation.

- `04_DiffusionTransformer/` — Transformer-based diffusion approaches and notes.

- `05_VideoGeneration/` — Notebook experiments on applying diffusion models to video data.

---

## Quickstart
Prerequisites: Python 3.8+ (3.10 recommended), CUDA-enabled GPU for training, and a virtual environment.

1. Create and activate a virtual environment (Windows example):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install --upgrade pip
```

2. Install core dependencies (example list):

```powershell
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install numpy matplotlib tqdm einops jupyter
```

Note: This repository does not include a single `requirements.txt`. Inspect notebooks and scripts for additional optional dependencies (e.g., `transformers`, `accelerate`, `diffusers`) when working with `02_StableDiffusion` or large pre-trained models.

---

## Running examples
- Notebook walkthroughs: launch Jupyter and open the notebooks in each folder:

```powershell
jupyter notebook
```

- Train a small DDPM (MNIST toy example):

```powershell
python 01_DenoisingDiffusionProbabilisticModels(DDPM)\train_ddpm.py --config 01_DenoisingDiffusionProbabilisticModels(DDPM)\default.yaml
```

- Sample from a trained checkpoint:

```powershell
python 01_DenoisingDiffusionProbabilisticModels(DDPM)\sample_ddpm.py --checkpoint path/to/checkpoint.pth --out_dir samples/
```

- Run the ControlNet demo app (if dependencies installed):

```powershell
python 03_ControlNet\app.py
```

---

## Design notes & pointers
- The `unet.py` implementation is intentionally minimal to clarify architecture and training loops — it's a great starting point for experimenting with attention layers, residual blocks, and conditional inputs.
- `linearNoiseScheduler.py` implements a simple beta schedule for forward diffusion; replace it with cosine or learned schedulers for state-of-the-art performance.
- Notebooks are pedagogical — they mix equations, visualizations, and runnable code to help you understand algorithmic choices.

---

## Recommended workflow
1. Start with the `01_DenoisingDiffusionProbabilisticModels(DDPM)` notebook to understand core concepts.
2. Use the toy training script to verify environment and GPU configuration.
3. Iterate on model/backbone and scheduler choices in `unet.py` and `linearNoiseScheduler.py`.
4. Move to `02_StableDiffusion` and `03_ControlNet` notebooks when you need latent-space or conditioned generation.

---

## Citations & References
- Ho et al., "Denoising Diffusion Probabilistic Models" (2020): https://arxiv.org/abs/2006.11239
- Rombach et al., "High-Resolution Image Synthesis with Latent Diffusion Models" (Stable Diffusion): https://arxiv.org/abs/2112.10752
- ControlNet: https://arxiv.org/abs/2302.05543

---

## Contributions
Contributions are welcome. Please open issues or PRs that add documentation, tests, or clearer examples. When submitting code, prefer focused commits and include runnable examples or notebook updates demonstrating the change.

---


If you'd like, I can also:
- generate a `requirements.txt` pinned to tested versions;
- add example checkpoints or CI to run a tiny smoke test;
- expand the `01_DenoisingDiffusionProbabilisticModels(DDPM)/` README with runnable CLI flags and examples.

