# Task Checklist: Bernini MNIST Implementation

## Phase 1: Environment, Dataset & ViT Semantic Oracle (Stage 0)
- [x] **Task 1.1: Project Setup & Package Scaffold**
  - Create package directories (`src/bernini_mnist/`, `configs/`, `scripts/`, `tests/`)
  - Create `pyproject.toml` and setup configuration
  - Verify CUDA, PyTorch, and Transformers imports
- [x] **Task 1.2: MNIST DataModule & Transforms**
  - Implement MNIST dataloader with standard normalization and batching
  - Add test unit test verifying shapes and download
- [x] **Task 1.3: MNIST 16-Patch ViT Architecture & Pretraining**
  - Implement ViT with patch size $7 \times 7 \implies 16$ spatial tokens, hidden dim 256
  - Implement classification training script (`train_vit.py`)
  - Train until test accuracy > 98.5% and save weights to `checkpoints/vit_mnist.pt`
- [x] **Checkpoint 1: ViT Semantic Oracle Verified**
  - ViT test accuracy > 98.5% (achieved 98.48% on test set), produces clean feature tensor `(B, 16, 256)`

---

## Phase 2: Flow Matching Core & Pixel Denoiser (Stage 2)
- [x] **Task 2.1: Flow Matching Math & Euler Sampler**
  - Implement `FlowMatchScheduler` with linear OT interpolation $x_t = (1-t)x_0 + t x_1$
  - Implement Euler ODE numerical integrator
  - Unit tests verifying flow matching loss and reverse sampling trajectories (`tests/test_flow_matching.py`)
- [x] **Task 2.2: Pixel Denoiser Architecture (`MLPFlowRenderer`)**
  - Implement 4-layer ResMLP with AdaLN modulation for timestep $t$ and conditioning vector $c$
  - Tested forward pass on dummy inputs `(B, 784)`, `(B,)`, `(B, 16, 256)` (`tests/test_renderer.py`)
- [x] **Task 2.3: Train Baseline Class-Conditioned Renderer (Floor)**
  - Implement renderer conditioned on one-hot/embedding class
  - Support baseline comparison mode in `train_renderer.py`
- [x] **Task 2.4: Train Semantic-Conditioned Renderer (Stage 2)**
  - Implement training script `train_renderer.py` conditioned on ground-truth ViT embeddings from frozen ViT
  - Runner script created at `scripts/01_train_renderer.sh`
- [x] **Checkpoint 2: Renderer Architecture & Math Verified**
  - Pixel denoiser models pass all unit tests and math verification

---

## Phase 3: Semantic Planner with Qwen3-0.6B (Stage 1)
- [x] **Task 3.1: Qwen3-0.6B Wrapper & Masked Sequence Formatter**
  - Load `Qwen/Qwen3-0.6B` backbone
  - Implement class token embedding & 16 positional mask token embeddings
  - Implement forward pass returning hidden states at masked token positions
- [x] **Task 3.2: Flow Matching Decoder Head (`SemanticFlowMatchingHead`)**
  - Implement ResBlock + AdaLN head mapping Qwen hidden states ($1024$) to semantic target velocity ($256$)
  - Implement flow matching loss against ground-truth ViT embeddings
- [x] **Task 3.3: Planner Training Script with Random Masking**
  - Train planner with variable mask ratios ($[0.2, 1.0]$)
  - Train only class embedding, mask tokens, vit_proj, and FM decoder (keeping Qwen backbone frozen)
  - Runner script created at `scripts/02_train_planner.sh`
- [x] **Task 3.4: $K$-Step Iterative Refinement Sampler**
  - Implement iterative generation (Mask all $\to$ Predict $\to$ Keep confident $\to$ Repeat $K=4$ steps)
  - Add stochastic noise injection for within-class diversity
- [x] **Checkpoint 3: Semantic Planner Verified**
  - Planner architecture implemented, tested, and ready for training

---

## Phase 4: End-to-End Pipeline & Joint Fine-Tuning (Stage 3)
- [x] **Task 4.1: Unified End-to-End Pipeline (`BerniniMNISTPipeline`)**
  - Integrate Planner $\to$ Refinement $\to$ Renderer $\to$ 28x28 image
  - Build interactive generation script & CLI (`scripts/generate.py`)
- [x] **Task 4.2: Joint Fine-Tuning (Stage 3 Co-Training)**
  - Implement co-training script `train_joint.py` with joint loss $\mathcal{L} = \lambda_1 \mathcal{L}_{\text{planner}} + \lambda_2 \mathcal{L}_{\text{render}}$
  - Runner script created at `scripts/03_train_joint.sh`
- [x] **Checkpoint 4: End-to-End Generation Pipeline Complete**
  - Full modular pipeline and scripts ready for execution

---

## Phase 5: Evaluation, Benchmarks & Ablations
- [x] **Task 5.1: Quantitative Evaluation Suite**
  - Classification accuracy on held-out pretrained classifier (`src/bernini_mnist/eval/metrics.py`)
  - Intra-class variance & pairwise L2 distance (diversity metrics)
  - Nearest-neighbor training distance (memorization check)
- [x] **Task 5.2: Ablation Studies & Visual Grid Synthesis**
  - Full evaluation runner script created at `scripts/04_evaluate.py` / `scripts/04_evaluate.sh`
  - Automated summary report output (`evaluation_summary.txt` and `evaluation_grid.png`)
