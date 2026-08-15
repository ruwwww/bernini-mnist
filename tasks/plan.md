# Technical Implementation Plan: Bernini-inspired MNIST Generator

## 1. Overview & Core Philosophy

This project implements a lightweight toy reproduction of ByteDance's **Bernini** architecture adapted for class-conditional MNIST image generation. It demonstrates the core two-stage paradigm:
1. **Semantic Planning (Stage 1)**: An MLLM (`Qwen/Qwen3-0.6B`) performs iterative, masked continuous token refinement via a Flow-Matching decoder to produce a structured continuous semantic representation $\mathbf{z} \in \mathbb{R}^{N \times d_{\text{semantic}}}$ from a discrete class condition (digits 0–9).
2. **Pixel-Space Rendering (Stage 2)**: A lightweight continuous-time Flow-Matching MLP/ResNet denoiser transforms the generated semantic tokens into $28 \times 28$ grayscale pixels.
3. **Joint Fine-Tuning (Stage 3)**: Co-training the planner decoder and the renderer to align semantic latent distributions with pixel synthesis.

---

## 2. Technology Stack & Framework Choices

| Layer / Component | Technology / Library | Version / Details | Rationale |
|---|---|---|---|
| **Deep Learning Framework** | `torch`, `torchvision` | PyTorch 2.13+ (CUDA enabled) | Core tensor ops, datasets, autograd |
| **MLLM Backbone** | `transformers` | HuggingFace Transformers (`Qwen/Qwen3-0.6B`) | Hidden size $d_{\text{model}} = 1024$, lightweight, fast forward pass |
| **Diffusion / Flow Matching** | Custom lightweight FM module (`diffloss_fm.py` / Euler ODE) | Flow Matching with Optimal Transport path ($x_t = (1-t)x_0 + t x_1$) | Replicates Bernini's exact flow matching math without heavy external dependencies |
| **Training & Acceleration** | `accelerate` | Hugging Face Accelerate | Clean multi-GPU/single-GPU execution, mixed precision (BF16/FP16) |
| **Evaluation & Metrics** | `torchmetrics`, `scikit-learn`, `matplotlib` | Accuracy, FID/L2 distance, variance, t-SNE | Quantitative fidelity & diversity assessment |
| **Project Structure** | Modular Python package (`src/bernini_mnist/`) | Clean CLI & script entry points | Modular, testable, reproducible |

---

## 3. Detailed Architecture Specifications

### 3.1. Target Semantic Token Space (MNIST ViT Encoder)
- **Role**: Serves as the ground-truth continuous semantic oracle for MNIST digits.
- **Architecture**:
  - Image size: $28 \times 28 \times 1$
  - Patch size: $7 \times 7 \implies (28/7) \times (28/7) = 4 \times 4 = 16$ spatial patch tokens ($N = 16$).
  - ViT layers: 4–6 Transformer encoder layers, hidden dimension $d_{\text{semantic}} = 256$ (or $512$).
  - Output: Sequence of $N=16$ tokens $\mathbf{z} \in \mathbb{R}^{16 \times d_{\text{semantic}}}$.
- **Pretraining**: Pretrained with a linear classification head on MNIST (target >99% test accuracy) and frozen.

### 3.2. Stage 1: Semantic Planner (Qwen3-0.6B + Flow-Matching Decoder)
- **MLLM Backbone**: `Qwen/Qwen3-0.6B` ($d_{\text{model}} = 1024$).
- **Input Sequence Formulation**:
  - Digit Class: Learned embedding lookup for digit $c \in \{0..9\}$ mapped to $\mathbb{R}^{1024}$, or tokenized prompt `"<|digit_c|>"` / text prompt `"Generate digit 3"`.
  - Semantic Sequence: $N=16$ token positions. During training, a subset of positions are masked (using learned `[MASK]` embeddings) while others receive projected ground-truth ViT embeddings.
- **Flow-Matching Decoder**:
  - Architecture: `SimpleMLPAdaLN` (ResBlocks with AdaLN modulation for timestep $t$ and conditioning $c$).
  - Takes Qwen hidden states $\mathbf{h} \in \mathbb{R}^{N \times 1024}$ at masked positions.
  - Predicts velocity $v_\theta(z_t, t, \mathbf{h}) = \frac{d z_t}{d t} \approx z_1 - z_0$, where $z_1 \sim \mathbf{z}_{\text{ViT}}$ and $z_0 \sim \mathcal{N}(0, \mathbf{I})$.
- **Iterative Refinement (Inference)**:
  - Step 1: Initialize $N=16$ mask tokens.
  - Forward Qwen $\implies$ decode $z_1^{(0)}$.
  - Mask fraction of lowest confidence or uncommitted tokens, re-inject partial predictions, repeat for $K = 4$ refinement iterations.

### 3.3. Stage 2: Pixel-Space Denoiser (Renderer)
- **Input**: Noisy image $x_t \in \mathbb{R}^{784}$ ($28 \times 28$ flattened), timestep $t \in [0, 1]$.
- **Conditioning**: Semantic tokens $\mathbf{s} \in \mathbb{R}^{16 \times d_{\text{semantic}}}$.
  - Conditioning projection: Flattened / pooled or Cross-Attention / AdaLN projector mapping $\mathbf{s} \to \mathbb{R}^{d_{\text{hidden}}}$.
- **Denoiser Architecture**:
  - 4-layer ResMLP with AdaLN conditioning and SiLU activations (width 512–1024).
  - Flow Matching vector field prediction $v_\phi(x_t, t, \mathbf{s}) \approx x_1 - x_0$.
- **Sampling**: Euler ODE solver from $t=0$ to $t=1$ in 20–50 steps.

### 3.4. Stage 3: Joint Fine-Tuning
- Joint loss: $\mathcal{L}_{\text{total}} = \lambda_1 \mathcal{L}_{\text{FM-planner}} + \lambda_2 \mathcal{L}_{\text{FM-pixel}}$.
- Refines planner decoder and renderer together to eliminate distribution shift between ground-truth ViT tokens and generated planner tokens.

---

## 4. Module & File Structure

```
bernini-mnist/
├── configs/
│   ├── vit_encoder.yaml
│   ├── planner.yaml
│   ├── renderer.yaml
│   └── joint.yaml
├── src/
│   └── bernini_mnist/
│       ├── __init__.py
│       ├── data/
│       │   ├── __init__.py
│       │   └── mnist_datamodule.py
│       ├── models/
│       │   ├── __init__.py
│       │   ├── vit_encoder.py        # 16-token MNIST ViT Feature Extractor
│       │   ├── flow_matching.py      # Flow Matching Scheduler & ODE Sampler
│       │   ├── planner.py            # Qwen3-0.6B + Flow-Matching Decoder Head
│       │   ├── renderer.py           # Pixel-Space MLP AdaLN Denoiser
│       │   └── pipeline.py           # End-to-End Generation Pipeline
│       ├── training/
│       │   ├── __init__.py
│       │   ├── train_vit.py          # Stage 0: Pretrain ViT Feature Oracle
│       │   ├── train_renderer.py     # Stage 2: Train Pixel Renderer
│       │   ├── train_planner.py      # Stage 1: Train Semantic Planner
│       │   └── train_joint.py        # Stage 3: Joint Fine-Tuning
│       ├── eval/
│       │   ├── __init__.py
│       │   ├── metrics.py            # Accuracy, Diversity, Nearest Neighbor
│       │   └── visualize.py          # Grid generation and plotting
│       └── utils/
│           ├── __init__.py
│           └── seed.py
├── scripts/
│   ├── 00_train_vit.sh
│   ├── 01_train_renderer.sh
│   ├── 02_train_planner.sh
│   ├── 03_train_joint.sh
│   └── 04_evaluate.sh
├── tests/
│   ├── test_vit.py
│   ├── test_flow_matching.py
│   ├── test_planner.py
│   ├── test_renderer.py
│   └── test_pipeline.py
├── tasks/
│   ├── plan.md
│   └── todo.md
└── project_brief_v1.md
```

---

## 5. Phased Implementation Roadmap & Verification Checkpoints

### Phase 1: Foundations & ViT Oracle (Stage 0)
- Setup dataset loaders, utilities, and training boilerplate.
- Build and train `MNISTViT` ($N=16$ patches) achieving >99% test accuracy.
- Save frozen checkpoint to extract $\mathbf{z} \in \mathbb{R}^{B \times 16 \times 256}$.
- **Checkpoint 1**: ViT classification accuracy > 99%, feature extractor outputs exact tensor shapes `(B, 16, 256)`.

### Phase 2: Flow Matching Engine & Pixel Denoiser (Stage 2)
- Implement `FlowMatchScheduler` & ODE Euler numerical integrator.
- Implement `MLPFlowRenderer` conditioned on semantic tokens.
- Train baseline class-conditioned renderer (Floor check).
- Train semantic-conditioned renderer on ground-truth ViT tokens.
- **Checkpoint 2**: Loss converges, sampler reconstructs crisp $28 \times 28$ digits from ground-truth ViT tokens with test classifier accuracy > 95%.

### Phase 3: Semantic Planner with Qwen3-0.6B (Stage 1)
- Wrap `Qwen/Qwen3-0.6B` with class input embedding and masked target tokens.
- Build `SimpleMLPAdaLN` flow matching head predicting semantic tokens $z_1 \in \mathbb{R}^{16 \times 256}$.
- Implement training loop with random masking (variable mask ratio).
- Implement $K$-step iterative refinement sampler.
- **Checkpoint 3**: Planner successfully generates semantic tokens from class label; iterative refinement improves velocity loss.

### Phase 4: End-to-End Pipeline & Joint Fine-Tuning (Stage 3)
- Connect Planner $\to$ Renderer into unified `BerniniMNISTPipeline`.
- Implement end-to-end sampling CLI: `generate(class_label=7, seed=42)`.
- Joint training of planner decoder + renderer.
- **Checkpoint 4**: Unassisted generation of all digits 0–9 from pure class IDs with diverse visual styles.

### Phase 5: Evaluation, Ablations & Documentation
- Measure quantitative metrics: Classification accuracy, intra-class variance, nearest-neighbor distance.
- Ablations: Compare single-step vs $K$-step refinement, stochastic noise injection vs deterministic.
- Generate final visualization galleries and packaging.
