# MRI Artifact Detection via Self-Supervised Anomaly Detection

A comparative study of **six anomaly-detection methods** for MRI motion-artifact detection across multiple anatomies. Four self-supervised methods (SimCLR, DINO, MAE, DAE) are compared against PatchCore (memory-bank) and a supervised ResNet-18 baseline, trained on **brain (IXI)** and **knee (FastMRI)** data.

All SSL/unsupervised models train exclusively on **clean** MRI slices and detect artifacts as out-of-distribution anomalies at inference time. Evaluation uses same-scanner test sets (MR-ART, KMAR) and simulated artifacts to control for dataset-identity confounds.

---

## Notebook Execution Order

The notebooks are designed to run on **Kaggle (T4 GPU, 12 h budget)**. Execute in this order:

### Phase 1 — Data Preprocessing

| # | Notebook | Purpose | Runtime | Output |
|---|----------|---------|---------|--------|
| 1 | `preprocessing-fastmri-knee.ipynb` | Convert raw FastMRI k-space → 192×192 `.npy` slices | ~30 min | `preprocessed-fastmri-knee/{train,val}/*.npy` |
| 2 | `preprocessing-ixi-brain (1).ipynb` | Convert IXI NIfTI T1+T2 → 192×192 `.npy` slices | ~30 min | `preprocessed-ixi-brain/{train,val}/{T1,T2}/*.npy` |
| 3 | `preprocessing-artifact-datasets.ipynb` | Process MR-ART, KMAR, simulated artifacts; build supervised manifests | ~1 h | `artifact_data/` (test sets + supervised manifests) |

### Phase 2 — Model Training (run independently, each ≤ 12 h)

| # | Notebook | Method | Architecture | Epochs | Output |
|---|----------|--------|-------------|--------|--------|
| 4 | `supervised-all-anatomies.ipynb` | Supervised | ResNet-18 (1-ch), BCEWithLogitsLoss | 20 | `checkpoints/sup_{knee,brain,combined}/best.pt` |
| 5 | `simclr-all-anatomies-new.ipynb` | SimCLR | ResNet-18 encoder + 128-d projection head | 20 | `checkpoints/simclr_{knee,brain,combined}/best.pt` |
| 6 | `vit-mae-all-anatomies.ipynb` | MAE | ViT-Small/16 encoder + lightweight decoder, 75% masking | 20 | `checkpoints/mae_{knee,brain,combined}/best.pt` |
| 7 | `dino-all-anatomies.ipynb` | DINO | ViT-Small/16 student-teacher (EMA) + 65536-d head | 20 | `checkpoints/dino_{knee,brain,combined}/best.pt` |
| 8 | `dae-all-anatomies.ipynb` | DAE | Conv encoder (512-d) + decoder, multi-corruption | 20 | `checkpoints/dae_{knee,brain,combined}/best.pt` |
| 9 | `patchcore-all-anatomies.ipynb` | PatchCore | WideResNet-50-2 (frozen, ImageNet) + coreset memory bank | N/A | `checkpoints/patchcore_{knee,brain,combined}/memory_bank.npy` |

### Phase 3 — Evaluation

| # | Notebook | Purpose | Output |
|---|----------|---------|--------|
| 10 | `evaluation-all-anatomies.ipynb` | Unified evaluation: 6 methods × 3 variants × 3 test sets | Tables 6–10, Figs 5–9, ROC/PR curves, t-SNE, confusion matrices |

### Supplementary Notebooks

| Notebook | Purpose |
|----------|---------|
| `dino-ablation-all-anatomies.ipynb` | DINO ablation study: reduced output dim (65536→4096) to address representational collapse on single-anatomy data |
| `simclr-brain-only.ipynb` | Early SimCLR prototype (brain-only, ResNet-18, τ=0.07) — superseded by `simclr-all-anatomies-new.ipynb` |
| `simclr-all-anatomies.ipynb` | Intermediate SimCLR version — superseded by `-new` variant |

---

## Notebook Descriptions

### 1. `preprocessing-fastmri-knee.ipynb`
Converts raw FastMRI knee k-space data into preprocessed 2D slices.
- Inverse FFT → magnitude images
- Center crop to 192×192
- Per-volume p1/p99 percentile normalization → [0, 1]
- Filter out empty slices (mean < 0.01)
- Output: ~12,864 train + ~3,216 val `.npy` files

### 2. `preprocessing-ixi-brain (1).ipynb`
Converts IXI brain NIfTI volumes (T1 and T2 weighted) into preprocessed 2D axial slices.
- Center crop to 192×192×32
- Per-volume p1/p99 normalization → [0, 1]
- Balanced T1/T2 output for equal representation
- Output: ~25,600 train + ~6,400 val `.npy` files

### 3. `preprocessing-artifact-datasets.ipynb`
Processes three artifact sources and constructs supervised training manifests:
- **MR-ART** (brain): Real motion artifacts — `headmotion1` (5,760 PNGs) + `headmotion2` (2,940 PNGs) + clean `Standard` (2,960 PNGs)
- **KMAR** (knee): Synthetically motion-corrupted .nii volumes — artifact (698 vols) + clean ground truth (904 vols)
- **Simulated**: Noisy/motion/blurred brain images from a brain-tumor dataset (test-only)
- Builds 3 supervised variants (knee/brain/combined) with 7.5% artifact ratio, 80/20 train/val split
- Includes 4-assertion leakage audit (manifest-level, on-disk, clean/artifact, train/val disjointness)

### 4. `supervised-all-anatomies.ipynb`
Trains 3 ResNet-18 binary classifiers (knee, brain, combined) as the **supervised baseline**.
- ResNet-18 with 1-channel input, `BCEWithLogitsLoss` with `pos_weight = N_neg/N_pos`
- 20 epochs, AdamW, cosine LR with 2-epoch warmup, effective batch size 256
- Best checkpoint by val AUROC
- Includes per-model analysis: training curves, ROC/PR, confusion matrix, threshold sweep

### 5. `simclr-all-anatomies-new.ipynb`
Trains 3 SimCLR models (contrastive learning) for anomaly detection.
- **ResNet-18 encoder** (1-channel input) → 512-d features
- 128-d projection head, NT-Xent loss with **τ = 0.5** (prevents feature collapse on grayscale MRI)
- Anomaly scoring: **kNN (k=5, cosine)** on L2-normalized encoder features
- Includes collapse check (feature STD + mean cosine similarity), t-SNE, NN retrieval

### 6. `vit-mae-all-anatomies.ipynb`
Trains 3 Masked Autoencoder models for anomaly detection.
- **ViT-Small/16 encoder** (384-d, 12 layers, 6 heads) + lightweight decoder (192-d, 4 layers, 3 heads)
- 75% random patch masking, normalized MSE loss on masked patches only
- **Two anomaly scores**: reconstruction error (averaged over 10 random masks) + kNN on encoder features
- Stochastic depth (drop path 0.1), AdamW with β=(0.9, 0.95), weight decay 0.05

### 7. `dino-all-anatomies.ipynb`
Trains 3 DINO student-teacher models for anomaly detection.
- **ViT-Small/16** student and teacher, 65536-d output head, EMA momentum 0.996→1.0
- Multi-crop: 2 global (192×192) + 6 local (96×96) views
- Centering + sharpening to prevent collapse, cross-entropy loss
- Anomaly scoring: **kNN (k=5, cosine)** on teacher backbone CLS features

### 8. `dae-all-anatomies.ipynb`
Trains 3 Denoising Autoencoder models for anomaly detection.
- 4-block convolutional encoder (1→64→128→256→512, 512-d bottleneck) + symmetric decoder
- Multi-corruption pipeline: Gaussian noise (always) + rectangular masking (p=0.3) + salt-and-pepper (p=0.2)
- **Two anomaly scores**: reconstruction MSE + kNN on encoder features (512-d)
- Includes all corruption function definitions (add_gaussian_noise, add_rect_mask, add_salt_pepper, corrupt_image, corrupt_tensor)

### 9. `patchcore-all-anatomies.ipynb`
Builds 3 PatchCore memory-bank models for anomaly detection.
- **WideResNet-50-2** (frozen, ImageNet pretrained), 1→3 channel replication
- Extracts layer2+layer3 patch features (1536-d) + layer4 global features (2048-d)
- Greedy coreset subsampling → `memory_bank.npy`
- Anomaly scoring: nearest-neighbor distance to memory bank (patch-level + global)

### 10. `evaluation-all-anatomies.ipynb`
Unified evaluation pipeline: 6 methods × 3 training variants × 3 test sets = **54 evaluations**.

**Test sets** (same-scanner clean+artifact pairs — no dataset-identity shortcut):
- `test_mrart.npz` — MR-ART clean + MR-ART artifact (brain, same scanner)
- `test_kmar.npz` — KMAR clean + KMAR artifact (knee, same scanner)
- `test_simulated.npz` — Brain-tumor images with synthetic noise/motion/blur

**Paper outputs**:
| Section | Output | Description |
|---------|--------|-------------|
| 4.2 | Table 6 | Overall quantitative results (AUROC, AUPRC, F1, Precision, Recall) |
| 4.3 | Fig 5 | Within-anatomy detection (brain→MR-ART, knee→KMAR) |
| 4.4 | Table 7, Fig 6 | Cross-anatomy generalization (brain→KMAR, knee→MR-ART) |
| 4.4b | Domain-shift diagnostic | Within vs cross-anatomy AUROC delta per method (shortcut test) |
| 4.5 | Table 8, Fig 7 | Multi-anatomy training effect (combined vs single-anatomy) |
| 4.6 | Table 9, Fig 8 | Synthetic artifact evaluation by type (noisy/motion/blurred) |
| 4.7 | Table 10, Fig 9 | Severity sensitivity on MR-ART (headmotion1 vs headmotion2) |

**Additional analyses**: ROC/PR curves, score distributions, confusion matrices, t-SNE feature visualizations, MAE/DAE reconstruction visualizations, comparative summary ranking.

### 11. `dino-ablation-all-anatomies.ipynb`
DINO ablation study addressing **representational collapse** on single-anatomy data.
- Baseline DINO with 65536-d output head collapsed on knee-only and brain-only (loss → ln(65536))
- Ablation reduces output dim to 4096: stabilizes training on smaller single-anatomy datasets
- Compares collapsed vs healthy feature distributions

---

## Models

| Method | Type | Architecture | Params | Anomaly Scoring |
|--------|------|-------------|--------|-----------------|
| **Supervised** | Supervised | ResNet-18, 1-ch input, binary FC head | 11.2M | Sigmoid probability |
| **SimCLR** | Self-supervised | ResNet-18, 1-ch input, 128-d projection | ~11.2M | kNN (k=5, cosine) |
| **MAE** | Self-supervised | ViT-Small/16 encoder (384-d) + decoder (192-d) | ~23M | Reconstruction MSE + kNN |
| **DINO** | Self-supervised | ViT-Small/16, teacher-student EMA, 65536-d head | ~22M | kNN (k=5, cosine) on teacher |
| **DAE** | Self-supervised | Conv encoder (512-d bottleneck) + decoder | ~7M | Reconstruction MSE + kNN |
| **PatchCore** | Memory bank | WideResNet-50-2 (frozen, ImageNet) + coreset | ~69M (frozen) | NN distance to memory bank |

All models accept **single-channel 192×192** MRI slices. All training uses SEED=42.

---

## Data Pipeline

### Preprocessing

| Step | Notebook | Input | Output |
|------|----------|-------|--------|
| Knee slices | `preprocessing-fastmri-knee.ipynb` | FastMRI k-space (`.h5`) | `{train,val}/*.npy` (192×192, [0,1]) |
| Brain slices | `preprocessing-ixi-brain (1).ipynb` | IXI NIfTI T1/T2 (`.nii.gz`) | `{train,val}/{T1,T2}/*.npy` (192×192, [0,1]) |
| Artifacts | `preprocessing-artifact-datasets.ipynb` | MR-ART (PNG), KMAR (NIfTI), Simulated (JPG) | Manifests + test sets |

### Balanced Brain Sampling
Brain has ~25,600 slices vs knee's ~12,864. All notebooks use the **same deterministic** `get_balanced_paths()` function to subsample brain T1/T2 equally to match knee count. This ensures identical training data across all methods.

### Training Variants

| Variant | Clean data | Artifact data (supervised only) |
|---------|-----------|-------------------------------|
| `knee` | FastMRI (~12,864 slices) | KMAR (7.5% ratio) |
| `brain` | IXI T1+T2 balanced (~12,864 slices) | MR-ART (7.5% ratio) |
| `combined` | FastMRI + IXI (~25,728 slices) | MR-ART + KMAR (7.5% ratio) |

### Augmentations (shared across all methods)
- Horizontal flip (50%)
- Rotation ±10° (50%)
- Intensity jitter: gain ∈ [0.9, 1.1], bias ∈ [−0.05, 0.05] (50%)
- **SimCLR/DINO only**: additional Gaussian blur for contrastive views

---

## Shared Training Configuration

| Parameter | Value | Notes |
|-----------|-------|-------|
| Seed | 42 | All notebooks |
| Image size | 192×192 | Single-channel grayscale |
| Batch size | 64 (×4 grad accum = 256 effective) | All methods |
| Epochs | 20 | All methods (fits 3 models in 12h) |
| Optimizer | AdamW | Per-method LR and weight decay |
| LR schedule | Cosine with 2-epoch linear warmup | All methods |
| Gradient clipping | max_norm = 1.0 | All methods |
| Mixed precision | AMP (float16 on CUDA) | All methods |
| kNN scoring | k=5, cosine metric | SimCLR, MAE, DINO, DAE |

---

## Checkpoint Format

| Method | `best.pt` keys | Selection criterion |
|--------|----------------|-------------------|
| Supervised | `epoch`, `model`, `val_auroc`, `val_metrics` | Best val AUROC |
| SimCLR | `epoch`, `encoder`, `projector`, `val_loss` | Best val loss |
| MAE | `epoch`, `model`, `optimizer`, `val_loss` | Best val loss |
| DINO | `epoch`, `student`, `teacher`, `loss` | Best val loss |
| DAE | `epoch`, `model`, `val_loss` | Best val loss |
| PatchCore | `memory_bank.npy` (not a `.pt` file) | N/A (no training) |

---

## Project Structure

```
mri_artifact/
├── README.md
├── requirements.txt
├── train.py                              # CLI entry point (optional, YAML-based)
├── configs/                              # 12 YAML configs (4 SSL methods × 3 anatomies)
│
├── ── Preprocessing ──
├── preprocessing-fastmri-knee.ipynb      # FastMRI k-space → .npy slices
├── preprocessing-ixi-brain (1).ipynb     # IXI NIfTI → .npy slices
├── preprocessing-artifact-datasets.ipynb # MR-ART, KMAR, simulated → test sets + manifests
│
├── ── Training (one per method) ──
├── supervised-all-anatomies.ipynb        # Supervised ResNet-18 (3 variants)
├── simclr-all-anatomies-new.ipynb        # SimCLR ResNet-18 (3 variants) ← USE THIS
├── vit-mae-all-anatomies.ipynb           # MAE ViT-Small/16 (3 variants)
├── dino-all-anatomies.ipynb              # DINO ViT-Small/16 (3 variants)
├── dae-all-anatomies.ipynb               # DAE Conv encoder-decoder (3 variants)
├── patchcore-all-anatomies.ipynb         # PatchCore WideResNet-50-2 (3 variants)
│
├── ── Evaluation ──
├── evaluation-all-anatomies.ipynb        # Unified eval: 54 experiments + paper figures
│
├── ── Ablation / Legacy ──
├── dino-ablation-all-anatomies.ipynb     # DINO ablation (65536→4096 output dim)
├── simclr-brain-only.ipynb               # Early prototype (brain-only)
├── simclr-all-anatomies.ipynb            # Intermediate version (superseded)
├── simclr-all-anatomies-old.ipynb        # Old version (superseded)
│
├── ── Python modules (optional CLI interface) ──
├── models/                               # Model definitions
├── trainers/                             # Training loops
├── evaluation/                           # Scoring and metrics
├── data/                                 # Dataset classes and transforms
└── utils/                                # Visualization helpers
```

---

## Kaggle Datasets

All experiments run on **Kaggle T4 GPU** (15.6 GB VRAM, 12 h session limit).

| Dataset | Kaggle Path | Contents |
|---------|-------------|----------|
| FastMRI Knee (preprocessed) | `kaustubhratna/fast-mri-preprocessed-kaust` | `{train,val}/*.npy` |
| IXI Brain (preprocessed) | `kaustubhratna/preprocessed-ixi-brain` | `{train,val}/{T1,T2}/*.npy` |
| Artifact Data | (output of preprocessing notebook 3) | Test sets + supervised manifests |
| MR-ART (raw) | `mdfaisalkarim/ranged-mrart` | Brain motion artifact PNGs |
| KMAR (raw) | `derbyfahim/438datast` | Knee motion artifact NIfTI volumes |
| Simulated (raw) | `mohamadabouali1/mri-brain-tumor-dataset-4-class-7023-images` | Noisy/motion/blurred JPGs |

---

## Domain-Shift Safeguards

Training uses cross-dataset pairs (e.g., FastMRI clean + KMAR artifact), which could introduce a dataset-identity shortcut. This is controlled by:

1. **Same-scanner test sets**: `test_mrart.npz` pairs MR-ART clean with MR-ART artifact (same scanner); `test_kmar.npz` pairs KMAR clean with KMAR artifact. No cross-dataset shortcut is possible at test time.
2. **Cross-anatomy generalization** (Table 7): Tests whether brain-trained models detect knee artifacts and vice versa.
3. **Domain-shift diagnostic** (Section 10b): Computes AUROC delta between within-anatomy and cross-anatomy evaluation. Small Δ = genuine artifact detection; large Δ = shortcut reliance.
4. **Simulated artifacts** (Table 9): Entirely unseen source dataset with synthetic corruptions — removes all domain cues.

---

## Installation

```bash
pip install -r requirements.txt
```

**Requirements:** Python 3.8+, PyTorch 2.0+, torchvision, numpy, scipy, scikit-learn, matplotlib, tqdm, pyyaml

For Kaggle notebooks, dependencies are pre-installed except `nibabel` (installed inline via `!pip install -q nibabel`).

---

## Reproducibility

- All notebooks use `SEED = 42` with `random.seed()`, `np.random.seed()`, `torch.manual_seed()`, `torch.cuda.manual_seed_all()`
- Brain subsampling is deterministic via evenly-spaced index selection on sorted file lists
- The same `get_balanced_paths()` function is copied identically across all training notebooks
- Preprocessing notebook includes a 4-assertion leakage audit
- Evaluation notebook loads all 18 model checkpoints and evaluates on identical test loaders
