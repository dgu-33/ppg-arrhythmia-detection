# H-AI : PPG Arrhythmia Detection

A deep learning system for arrhythmia detection from PPG (Photoplethysmography) signals, combining a hybrid ResNet1D + U-Net architecture with CLIP-based contrastive pre-training and a Gemini LLM-powered clinical report generator.

---

## Overview

This project detects four cardiac rhythm classes from raw PPG waveforms:

| Label | Condition |
|-------|-----------|
| Normal | Sinus rhythm, HR 60–100 bpm |
| AF | Atrial Fibrillation |
| B | Bradycardia (HR < 60 bpm) |
| T | Tachycardia (HR > 100 bpm) |

The system also produces a **binary anomaly segmentation mask** over the waveform, highlighting the regions that contributed to the classification.

---

## Architecture

### 1. CLIP Pre-training
The ResNet1D encoder is pre-trained with contrastive learning (CLIP-style) to align PPG signal representations with 13-dimensional clinical tabular features (age, BMI, lab values, comorbidities, etc.).

### 2. Hybrid ResNet1D + U-Net (Main Model)
- **Encoder**: CLIP pre-trained ResNet1D with Squeeze-and-Excitation (SE) blocks
- **Classification head**: Global average pooling → 4-class softmax
- **Decoder**: U-Net skip-connection decoder → binary segmentation mask

### 3. Multi-modal Inputs
| Input | Details |
|-------|---------|
| PPG waveform | Resampled to 286 samples, sample-wise Z-score normalised |
| HRV features | HR (bpm), SDNN, RMSSD, pNN50 — 4 features extracted from PPG peaks |
| Clinical features | 13 features: age, BMI, operation duration, lab values (Na, BUN, Cr, K, ephedrine, phenylephrine dose), sex, emergency operation flag, DM, hypertension |

### 4. LLM Report Generation (RAG + Vision)
- **RAG**: Similar patient retrieval via CLIP clinical embeddings + hardcoded ESC/ACC/AHA medical guidelines
- **Vision**: Gemini multimodal analysis of the raw PPG waveform image
- **Model**: `gemini-flash-latest`

---

## Project Structure

```
ppg_arrhythmia_detection/
├── model_architecture.py   # Model classes, HRV extraction, preprocessing (shared)
├── train.py                # Training pipeline with argparse CLI
├── app.py                  # Streamlit web application
├── llm_utils.py            # Gemini LLM setup, RAG encoders, medical guidelines
├── requirements.txt
└── .gitignore
```

### File Responsibilities

| File | Role |
|------|------|
| `model_architecture.py` | Defines `UNet1D_ResNet_Combined`, `ResNet1D_Backbone`, HRV feature extraction, inference preprocessing. Shared by both training and inference. |
| `train.py` | Full training loop: data loading, CLIP weight injection, focal + dice loss, cosine LR warmup, early stopping, evaluation metrics. |
| `app.py` | Streamlit UI: patient login/register (SQLite), PPG file upload, model inference, LLM report generation. |
| `llm_utils.py` | CLIP-compatible clinical encoders for RAG retrieval, Gemini model setup, ESC/ACC/AHA guideline text. |

> **Note**: Two versions of `UNet1D_ResNet_Combined` exist intentionally.
> - `model_architecture.py`: inference-only (no `load_clip_weights`)
> - `train.py`: training version (adds `load_clip_weights` method)
>
> Their `state_dict` keys are identical, so checkpoints are cross-compatible.

---

## Installation

```bash
pip install -r requirements.txt
```

Requires Python 3.9+ and PyTorch (CPU or CUDA).

---

## Training

```bash
python train.py \
  --base_path /path/to/project \
  --dataset_path /path/to/PPG_AD_Dataset \
  --clip_weights /path/to/clip_checkpoint.pth \
  --save_path /path/to/save/model.pth
```

### Key Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--epochs` | 200 | Total training epochs |
| `--batch_size` | 32 | Batch size |
| `--lr` | 5e-5 | Base learning rate |
| `--enc_lr_ratio` | 0.1 | Encoder LR multiplier (fine-tuning) |
| `--unfreeze_epoch` | 50 | Epoch at which encoder is unfrozen |
| `--patience` | 50 | Early stopping patience |
| `--alpha` | 2.0 | Focal loss gamma |
| `--beta` | 0.8 | BCE/Dice loss blend weight |
| `--seed` | 42 | Random seed |

### Dataset Format

```
PPG_AD_Dataset/
├── Normal/          # .npy files — raw PPG signal arrays
├── AF/              # .npz files — keys: signal, mask, class
├── B/               # .npz files
└── T/               # .npz files
```

Training expects a merged CSV (`*_combined.csv`) containing clinical features per sample. Pre-computed normalisation statistics are saved to `data_stats.json` and `clinical_stats.json`.

---

## Web Application

```bash
streamlit run app.py
```

1. Enter your Gemini API key in `app.py` (`MY_API_KEY = ""`).
2. Register or log in with patient clinical data.
3. Upload a `.npy` or `.npz` PPG file.
4. The app scans the signal in sliding windows, classifies each segment, and generates an LLM clinical report.

Place trained model weights (`*.pth`) in the project root and update `MODEL_PATH` in `app.py` if needed.

---

## Loss Functions

| Loss | Usage |
|------|-------|
| `FocalLoss` (γ=2.0) | 4-class classification |
| `BCEDiceLoss` | Binary segmentation mask |
| Combined | `L = L_cls + β · L_seg` |

---

## References

- **2020 ESC Guidelines** for the diagnosis and management of Atrial Fibrillation
- **ACC/AHA Guidelines** for Bradycardia and Tachycardia management
- **CLIP**: Radford et al., "Learning Transferable Visual Models From Natural Language Supervision", 2021
