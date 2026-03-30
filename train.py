import os
import json
import math
import random
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from datetime import datetime

import matplotlib
matplotlib.use('Agg')  # non-interactive backend for script usage
import matplotlib.pyplot as plt

from scipy import stats
from sklearn.metrics import (
    f1_score, accuracy_score,
    confusion_matrix, classification_report,
)
from sklearn.preprocessing import StandardScaler

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

# ---------------------------------------------------------------------------
# Imports from model_architecture
# ---------------------------------------------------------------------------
from model_architecture import (
    # Model building blocks (used directly in UNet1D_ResNet_Combined below)
    ResNet1D_Backbone,
    Up,
    # HRV feature extraction (used in PPGDataset_Combined)
    get_hrv_features,
    # Shared constants
    N_HRV_FEATURES,
    CLASS_MAP,
    INV_CLASS_MAP,
    CLINICAL_FEATURES_LIST,
    N_CLINICAL_FEATURES,
    TARGET_LENGTH,
)

# ---------------------------------------------------------------------------
# Path configuration — override via CLI arguments or edit these defaults
# ---------------------------------------------------------------------------
DEFAULT_BASE_PROJECT_PATH = "."
DEFAULT_DATASET_PATH      = os.path.join(DEFAULT_BASE_PROJECT_PATH, "PPG_AD_Dataset")
DEFAULT_CLIP_WEIGHTS_PATH = os.path.join(DEFAULT_DATASET_PATH, "checkpoints", "clip_biobert_hyh_xai.pth")
DEFAULT_MODEL_SAVE_PATH   = os.path.join(DEFAULT_BASE_PROJECT_PATH, "unet_clip_biobert_hyh_xai.pth")


# =============================================================================
# 1. Utilities
# =============================================================================

def set_seed(seed: int = 42) -> None:
    """Fix all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# =============================================================================
# 2. Clinical data preprocessing (training version — fits scaler on the CSV)
# =============================================================================

# Column definitions used by preprocess_clinical_data
_CONTINUOUS_COLS = [
    "age", "bmi", "opdur", "preop_na", "preop_bun",
    "preop_cr", "preop_k", "intraop_eph", "intraop_phe",
]
_CATEGORICAL_COLS_OHE = ["sex", "emop", "preop_dm", "preop_htn"]


def preprocess_clinical_data(csv_path: str) -> dict:
    """
    Load a clinical CSV, apply one-hot encoding, fill NaNs, run StandardScaler
    then min-max normalisation (replicating the CLIP training pipeline), and
    return a {caseid -> {feature: value}} dict ready for PPGDataset_Combined.

    This is the training-time version that *fits* the scaler on the given split.
    The inference-time counterpart (using pre-computed stats) lives in
    model_architecture.get_clinical_feature_vector().
    """
    if not os.path.exists(csv_path):
        print(f"Warning: clinical data file not found: {csv_path}")
        return {}

    df = pd.read_csv(csv_path)

    # Set caseid as index
    df = df.set_index("caseid")

    # One-Hot Encoding (drop_first=True to match CLIP training)
    df = pd.get_dummies(df, columns=_CATEGORICAL_COLS_OHE, drop_first=True)

    # Select / synthesise exactly the 13 expected features
    df_selected = df.copy()
    for col in CLINICAL_FEATURES_LIST:
        if col not in df_selected.columns:
            if col.endswith("_M") or col.endswith("_Y"):
                df_selected[col] = 0
            else:
                raise ValueError(f"Missing clinical feature column: {col}")

    df_selected = df_selected[CLINICAL_FEATURES_LIST]

    # Fill NaN with column mean
    df_selected = df_selected.fillna(df_selected.mean())

    # 1st stage: StandardScaler (Z-score)
    scaler = StandardScaler()
    scaled = scaler.fit_transform(df_selected)

    # 2nd stage: min-max normalisation across the fitted distribution
    col_min = np.min(scaled, axis=0)
    col_max = np.max(scaled, axis=0)
    diff = col_max - col_min
    diff[diff == 0] = 1.0  # avoid division by zero for constant columns

    normalised = (scaled - col_min) / diff

    df_norm = pd.DataFrame(normalised, columns=CLINICAL_FEATURES_LIST, index=df_selected.index)

    clinical_map = df_norm.to_dict("index")
    return clinical_map


def prepare_merged_csv(main_path: str, normal_path: str,
                       output_path: str) -> str:
    """
    Concatenate two CSV files (main split + extra normal samples).
    Returns the path that should be used for the dataset.
    If normal_path does not exist, returns main_path unchanged.
    """
    if os.path.exists(normal_path):
        try:
            df_combined = pd.concat(
                [pd.read_csv(main_path), pd.read_csv(normal_path)],
                ignore_index=True,
            )
            df_combined.to_csv(output_path, index=False)
            print(f"Merged CSV saved: {output_path} ({len(df_combined)} rows)")
            return output_path
        except Exception as exc:
            print(f"CSV merge failed ({exc}); using {main_path} only.")
            return main_path
    else:
        return main_path


# =============================================================================
# 3. Dataset
# =============================================================================

class PPGDataset_Combined(Dataset):
    """
    Loads windowed PPG recordings (.npz / .npy), resamples them to
    TARGET_LENGTH, applies sample-wise z-score normalisation, and pairs each
    window with its clinical feature vector and HRV features.

    Expected .npz fields:
        'signal'  — 1-D float array
        'mask'    — 1-D binary array (anomaly region), same length as signal
        'class'   — scalar or array of class codes
                    (0/1 = normal, 2 = af, 3 = b, 4 = t)
    """

    _LABEL_MAP = {0: "normal", 1: "normal", 2: "af", 3: "b", 4: "t"}

    def __init__(
        self,
        data_dirs,
        clinical_csv_path: str,
        target_length: int,
        fs: float = 100.0,
        is_test: bool = False,
    ):
        self.data_dirs    = data_dirs if isinstance(data_dirs, list) else [data_dirs]
        self.target_length = int(target_length)
        self.fs           = float(fs)
        self.is_test      = is_test

        self.clinical_map = preprocess_clinical_data(clinical_csv_path)
        self.file_list: list[dict] = []
        self._scan_and_match_files()
        self.labels = self._get_all_labels()

    # ------------------------------------------------------------------
    def _scan_and_match_files(self):
        seen = set()
        for d in self.data_dirs:
            if not os.path.isdir(d):
                continue
            for root, _, files in os.walk(d):
                for f in files:
                    if not f.endswith((".npy", ".npz")):
                        continue
                    full_path = os.path.join(root, f)
                    if full_path in seen:
                        continue
                    seen.add(full_path)
                    try:
                        caseid = int(f.split("_")[0])
                    except Exception:
                        caseid = -1
                    self.file_list.append(
                        {"path": full_path, "filename": f, "caseid": caseid}
                    )

    def _get_all_labels(self) -> list:
        labels = []
        for item in self.file_list:
            fname  = item["filename"]
            fpath  = item["path"]
            label_name = "normal"
            if fname.endswith(".npz"):
                try:
                    with np.load(fpath, allow_pickle=True) as data:
                        cls_raw = data.get("class")
                        if cls_raw is not None:
                            if isinstance(cls_raw, np.ndarray) and cls_raw.size > 1:
                                nz = cls_raw[cls_raw != 0]
                                if nz.size > 0:
                                    mode_val = stats.mode(nz, keepdims=True)[0][0]
                                    label_name = self._LABEL_MAP.get(int(mode_val), "normal")
                            else:
                                label_name = self._LABEL_MAP.get(
                                    int(np.array(cls_raw).item()), "normal"
                                )
                except Exception:
                    pass
            labels.append(CLASS_MAP.get(label_name, 0))
        return labels

    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.file_list)

    def __getitem__(self, idx):
        item   = self.file_list[idx]
        fpath  = item["path"]
        fname  = item["filename"]
        caseid = item["caseid"]

        label_name = "normal"
        try:
            if fname.endswith(".npy"):
                sig  = np.load(fpath).astype(np.float32)
                mask = np.zeros_like(sig, dtype=np.float32)
            else:
                with np.load(fpath, allow_pickle=True) as data:
                    sig  = data["signal"].astype(np.float32)
                    mask = data["mask"].astype(np.float32)
                    cls_raw = data.get("class")
                    if cls_raw is not None:
                        if isinstance(cls_raw, np.ndarray) and cls_raw.size > 1:
                            nz = cls_raw[cls_raw != 0]
                            if nz.size > 0:
                                mode_val = stats.mode(nz, keepdims=True)[0][0]
                                label_name = self._LABEL_MAP.get(int(mode_val), "normal")
                        else:
                            label_name = self._LABEL_MAP.get(
                                int(np.array(cls_raw).item()), "normal"
                            )
        except Exception:
            return self.__getitem__(random.randint(0, len(self) - 1))

        # HRV features (uses global stats from model_architecture)
        hrv_feat = get_hrv_features(sig, fs=self.fs)
        hrv_t    = torch.from_numpy(hrv_feat).float()

        # Clinical features
        if caseid in self.clinical_map:
            clin_dict = self.clinical_map[caseid]
            clin_list = [clin_dict[f] for f in CLINICAL_FEATURES_LIST]
        else:
            clin_list = [0.0] * N_CLINICAL_FEATURES
        clinical_t = torch.tensor(clin_list, dtype=torch.float32)

        # Resample signal and mask to target length
        sig_t  = torch.from_numpy(sig).unsqueeze(0).unsqueeze(0)   # (1,1,L)
        mask_t = torch.from_numpy(mask).unsqueeze(0).unsqueeze(0)

        sig_res  = F.interpolate(sig_t,  size=self.target_length, mode="linear",
                                  align_corners=False).squeeze(0)   # (1,L)
        mask_res = F.interpolate(mask_t, size=self.target_length, mode="nearest").squeeze(0)

        # Sample-wise z-score normalisation (matches CLIP training pipeline)
        x = sig_res.float()
        mu  = x.mean()
        std = x.std()
        if std > 1e-8:
            x = (x - mu) / std
        else:
            x = x - mu

        y_mask  = mask_res.float()
        y_label = torch.tensor(CLASS_MAP.get(label_name, 0), dtype=torch.long)

        return x, y_mask, y_label, clinical_t, hrv_t


# =============================================================================
# 4. Model — UNet1D_ResNet_Combined (training version with load_clip_weights)
# =============================================================================
# NOTE: model_architecture.py's UNet1D_ResNet_Combined does NOT include
# load_clip_weights(), so we define the full training version here and do NOT
# import it from model_architecture.

class UNet1D_ResNet_Combined(nn.Module):
    """
    Hybrid ResNet-1D encoder (initialised from CLIP checkpoint) + U-Net decoder.

    Inputs  : signal (B,1,L), clinical_features (B,13), hrv_features (B,4)
    Outputs : clf_logits (B, n_classes),  seg_logits (B, L)  — raw (pre-sigmoid)
    """

    def __init__(
        self,
        n_channels: int,
        n_classes:  int,
        n_clinical_features: int = N_CLINICAL_FEATURES,
        n_hrv_features:      int = N_HRV_FEATURES,
    ):
        super().__init__()
        self.n_classes           = n_classes
        self.n_clinical_features = n_clinical_features
        self.n_hrv_features      = n_hrv_features
        self.encoder_feature_dim = 256
        self.clip_feature_dim    = 128

        # Encoder
        self.encoder        = ResNet1D_Backbone(in_channels=n_channels, base_filters=32)
        self.clip_projection = nn.Linear(self.encoder_feature_dim, self.clip_feature_dim)

        # Decoder
        self.up1 = Up(256 + 128, 128)
        self.up2 = Up(128 + 64,  64)
        self.up3 = Up(64  + 32,  32)
        self.final_up = nn.Sequential(
            nn.Upsample(scale_factor=4, mode="linear", align_corners=False),
            nn.Conv1d(32, 16, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(4, 16),
            nn.ReLU(inplace=True),
        )

        # Heads
        self.seg_head = nn.Conv1d(16, 1, kernel_size=1)

        clf_in_dim = self.clip_feature_dim + n_clinical_features + n_hrv_features  # 145
        self.clf_head = nn.Sequential(
            nn.Linear(clf_in_dim, 128),
            nn.LayerNorm(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.6),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.6),
            nn.Linear(64, n_classes),
        )

    def forward(self, signal_input, clinical_features, hrv_features):
        features = self.encoder(signal_input)
        x_stem, x1, x2, x3 = features[0], features[1], features[2], features[3]

        # Classification path
        gap          = F.adaptive_avg_pool1d(x3, 1).squeeze(-1)   # (B,256)
        clip_feature = self.clip_projection(gap)                    # (B,128)
        combined     = torch.cat([clip_feature, clinical_features, hrv_features], dim=1)
        clf_logits   = self.clf_head(combined)

        # Segmentation path (U-Net decoder)
        d1      = self.up1(x3, x2)
        d2      = self.up2(d1, x1)
        d3      = self.up3(d2, x_stem)
        d_final = self.final_up(d3)
        seg_logits = self.seg_head(d_final)

        # Align output length to input length if necessary
        if seg_logits.size(2) != signal_input.size(2):
            seg_logits = F.interpolate(
                seg_logits, size=signal_input.size(2),
                mode="linear", align_corners=False,
            )

        return clf_logits, seg_logits.squeeze(1)

    def load_clip_weights(self, weights_path: str, device) -> None:
        """
        Load encoder + projection weights from a CLIP checkpoint.

        Supports checkpoints that store the wave encoder under the key
        'wave_enc' or 'wave_encoder_state_dict'; falls back to using the
        top-level state dict directly.
        """
        if not os.path.exists(weights_path):
            print(f"Warning: CLIP weights not found at {weights_path}; "
                  "training from random initialisation.")
            return

        checkpoint   = torch.load(weights_path, map_location=device, weights_only=False)
        source_state = checkpoint
        if "wave_enc" in checkpoint:
            source_state = checkpoint["wave_enc"]
        elif "wave_encoder_state_dict" in checkpoint:
            source_state = checkpoint["wave_encoder_state_dict"]

        model_state  = self.state_dict()
        loaded_state = {}
        match_count  = 0

        for k, v in source_state.items():
            k = k.replace("module.", "")

            # ResNet Backbone: initial.* / layer*.* -> encoder.*
            if k.startswith("initial.") or k.startswith("layer"):
                target_key = f"encoder.{k}"
                if target_key in model_state and model_state[target_key].shape == v.shape:
                    loaded_state[target_key] = v
                    match_count += 1

            # Projection layer: fc.* -> clip_projection.*
            elif k.startswith("fc."):
                target_key = f"clip_projection.{k[len('fc.'):]}"
                if target_key in model_state and model_state[target_key].shape == v.shape:
                    loaded_state[target_key] = v
                    match_count += 1

        if match_count > 0:
            self.load_state_dict(loaded_state, strict=False)
            print(f"CLIP weights loaded: {match_count} encoder/projection layers matched.")
        else:
            print("Warning: no matching layers found in CLIP checkpoint.")


# =============================================================================
# 5. Loss functions
# =============================================================================

class FocalLoss(nn.Module):
    """Multi-class focal loss for classification."""

    def __init__(self, gamma: float = 2.2, alpha=None, reduction: str = "mean"):
        super().__init__()
        self.gamma     = gamma
        self.reduction = reduction
        if isinstance(alpha, (float, int)):
            self.alpha = torch.Tensor([alpha, 1 - alpha])
        elif isinstance(alpha, (list, torch.Tensor)):
            self.alpha = torch.as_tensor(alpha)
        else:
            self.alpha = None

    def forward(self, inputs, targets):
        logpt = F.log_softmax(inputs, dim=1)
        logpt = logpt.gather(1, targets.view(-1, 1)).view(-1)
        pt    = logpt.exp()
        if self.alpha is not None:
            if self.alpha.type() != inputs.data.type():
                self.alpha = self.alpha.type_as(inputs.data)
            at    = self.alpha.gather(0, targets.data.view(-1))
            logpt = logpt * at
        loss = -1.0 * (1 - pt) ** self.gamma * logpt
        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss


class DiceLoss(nn.Module):
    def __init__(self, smooth: float = 1e-6):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        probs   = torch.sigmoid(logits).contiguous().view(-1)
        targets = targets.contiguous().view(-1)
        inter   = (probs * targets).sum()
        denom   = probs.sum() + targets.sum() + self.smooth
        return 1 - (2 * inter + self.smooth) / denom


class BCEDiceLoss(nn.Module):
    def __init__(self, smooth: float = 1e-6, weight_bce: float = 0.5, weight_dice: float = 0.5):
        super().__init__()
        self.bce  = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss(smooth)
        self.wb   = weight_bce
        self.wd   = weight_dice

    def forward(self, logits, targets):
        return self.wb * self.bce(logits, targets) + self.wd * self.dice(logits, targets)


# =============================================================================
# 6. EarlyStopping
# =============================================================================

class EarlyStopping:
    """Save best model and stop training when the monitored metric stagnates."""

    def __init__(
        self,
        patience: int    = 50,
        verbose:  bool   = False,
        delta:    float  = 1e-3,
        path:     str    = "best_model.pth",
        mode:     str    = "min",
        monitor_name: str = "val_total",
    ):
        self.patience     = patience
        self.verbose      = verbose
        self.counter      = 0
        self.best_score   = None
        self.early_stop   = False
        self.delta        = float(delta)
        self.path         = path
        self.mode         = mode
        self.monitor_name = monitor_name

    def _is_improved(self, score: float) -> bool:
        if self.best_score is None:
            return True
        if self.mode == "max":
            return (score - self.best_score) > self.delta
        return (self.best_score - score) > self.delta

    def __call__(self, score, model, ref_metric_for_log=None, ref_name: str = "ref"):
        if hasattr(score, "item"):
            score = float(score.item())

        if self._is_improved(score):
            self.best_score = score
            if self.verbose:
                msg = f"{self.monitor_name} improved to {score:.6f}."
                if ref_metric_for_log is not None:
                    msg += f" ({ref_name}: {float(ref_metric_for_log):.6f})"
                print(f"  [{msg}] Saving model...")
            torch.save(model.state_dict(), self.path)
            self.counter = 0
        else:
            self.counter += 1
            if self.verbose:
                print(
                    f"  EarlyStopping counter: {self.counter}/{self.patience} "
                    f"(best {self.monitor_name}: {self.best_score:.6f})"
                )
            if self.counter >= self.patience:
                self.early_stop = True


# =============================================================================
# 7. Training and evaluation functions
# =============================================================================

def _dice_coefficient_bin(
    pred_bin: torch.Tensor, target_bin: torch.Tensor, eps: float = 1e-7
) -> torch.Tensor:
    pred_bin   = pred_bin.float()
    target_bin = target_bin.float()
    intersect  = (pred_bin * target_bin).sum()
    denom      = pred_bin.sum() + target_bin.sum() + eps
    return (2.0 * intersect + eps) / denom


def train_one_epoch(
    model,
    dataloader,
    criterion_clf,
    criterion_seg,
    optimizer,
    device,
    alpha: float = 1.0,
    beta:  float = 1.0,
    max_norm: float = 1.0,
):
    model.train()
    total_loss = total_clf_loss = total_seg_loss = 0.0
    all_labels: list = []
    all_preds:  list = []

    for signals, masks, labels, clinical_features, hrv_features in tqdm(
        dataloader, desc="Training", leave=False
    ):
        signals           = signals.to(device)
        masks             = masks.to(device)
        labels            = labels.to(device)
        clinical_features = clinical_features.to(device)
        hrv_features      = hrv_features.to(device)

        optimizer.zero_grad()
        clf_logits, seg_logits = model(signals, clinical_features, hrv_features)

        target_mask = masks.squeeze(1) if (masks.dim() == 3 and masks.size(1) == 1) else masks

        clf_loss = criterion_clf(clf_logits, labels)
        seg_loss = criterion_seg(seg_logits, target_mask)
        loss     = clf_loss * alpha + seg_loss * beta

        loss.backward()
        if max_norm is not None and max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        optimizer.step()

        total_loss     += float(loss.item())
        total_clf_loss += float(clf_loss.item())
        total_seg_loss += float(seg_loss.item())

        preds = torch.argmax(clf_logits, dim=1)
        all_preds.extend(preds.detach().cpu().numpy())
        all_labels.extend(labels.detach().cpu().numpy())

    n = max(1, len(dataloader))
    return (
        total_loss / n,
        total_clf_loss / n,
        total_seg_loss / n,
        f1_score(all_labels, all_preds, average="macro", zero_division=0),
        accuracy_score(all_labels, all_preds),
    )


def evaluate(
    model,
    dataloader,
    criterion_clf,
    criterion_seg,
    device,
    alpha: float = 1.0,
    beta:  float = 1.0,
    threshold: float = 0.5,
):
    """
    Returns:
        total_loss, clf_loss, seg_loss,
        macro_f1, accuracy,
        all_preds, all_labels, all_confidences,
        pixel_accuracy, dice_score
    """
    model.eval()
    total_loss = total_clf_loss = total_seg_loss = 0.0
    total_pixel_acc = total_dice = 0.0
    all_labels:      list = []
    all_preds:       list = []
    all_confidences: list = []

    with torch.no_grad():
        for signals, masks, labels, clinical_features, hrv_features in tqdm(
            dataloader, desc="Evaluating", leave=False
        ):
            signals           = signals.to(device)
            masks             = masks.to(device)
            labels            = labels.to(device)
            clinical_features = clinical_features.to(device)
            hrv_features      = hrv_features.to(device)

            clf_logits, seg_logits = model(signals, clinical_features, hrv_features)

            target_mask = (
                masks.squeeze(1) if (masks.dim() == 3 and masks.size(1) == 1) else masks
            )

            clf_loss = criterion_clf(clf_logits, labels)
            seg_loss = criterion_seg(seg_logits, target_mask)
            loss     = clf_loss * alpha + seg_loss * beta

            total_loss     += float(loss.item())
            total_clf_loss += float(clf_loss.item())
            total_seg_loss += float(seg_loss.item())

            probs = F.softmax(clf_logits, dim=1)
            preds = torch.argmax(probs, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_confidences.extend(torch.max(probs, dim=1)[0].cpu().numpy())

            seg_probs = torch.sigmoid(seg_logits)
            seg_bin   = (seg_probs > threshold).to(target_mask.dtype)

            # Post-processing: zero out segmentation mask for normal predictions
            normal_mask = preds == 0
            if normal_mask.any():
                seg_bin[normal_mask] = 0.0

            total_pixel_acc += float((seg_bin == target_mask).float().mean().item())
            total_dice      += float(_dice_coefficient_bin(seg_bin, target_mask).item())

    n = max(1, len(dataloader))
    return (
        total_loss / n,
        total_clf_loss / n,
        total_seg_loss / n,
        f1_score(all_labels, all_preds, average="macro", zero_division=0),
        accuracy_score(all_labels, all_preds),
        all_preds,
        all_labels,
        all_confidences,
        total_pixel_acc / n,
        total_dice / n,
    )


# =============================================================================
# 8. Training history visualisation
# =============================================================================

def plot_history(history: dict, save_path: str) -> None:
    """Save a 3×2 grid of training curves to disk."""
    epochs_ran = len(history.get("train_total", []))
    if epochs_ran == 0:
        return
    x = range(1, epochs_ran + 1)
    best_epoch = int(np.argmin(history["val_total"])) + 1

    fig, axes = plt.subplots(3, 2, figsize=(14, 12))
    fig.suptitle("Training & Validation History", fontsize=16)

    pairs = [
        ((0, 0), "train_total",       "val_total",         "Total Loss"),
        ((0, 1), "train_clf",         "val_clf",           "Classification Loss"),
        ((1, 0), "train_seg",         "val_seg",           "Segmentation Loss"),
        ((1, 1), None,                None,                "Validation Classification Metrics"),
        ((2, 0), "val_seg_pixel_acc", None,                "Validation Seg Pixel Accuracy"),
        ((2, 1), "val_seg_dice",      None,                "Validation Seg Dice"),
    ]
    for (r, c), tr_key, va_key, title in pairs:
        ax = axes[r][c]
        if tr_key and tr_key in history:
            ax.plot(x, history[tr_key], label=tr_key, marker=".", markersize=4)
        if va_key and va_key in history:
            ax.plot(x, history[va_key], label=va_key, marker=".", markersize=4)
        ax.axvline(best_epoch, color="r", linestyle="--", label=f"Best: {best_epoch}")
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend(fontsize=7)

    # Dual-axis for accuracy + F1
    ax = axes[1][1]
    ax.cla()
    ax.plot(x, history["val_acc"], label="Val Acc", color="C0")
    ax.set_ylabel("Accuracy", color="C0")
    ax.tick_params(axis="y", labelcolor="C0")
    ax2 = ax.twinx()
    ax2.plot(x, history["val_f1"], label="Val F1", linestyle="--", color="C1")
    ax2.set_ylabel("F1 (Macro)", color="C1")
    ax2.tick_params(axis="y", labelcolor="C1")
    ax.set_title("Validation Classification Metrics")
    ax.axvline(best_epoch, color="r", linestyle="--")
    ax.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    fig.savefig(save_path, dpi=100)
    plt.close(fig)
    print(f"Training curves saved to: {save_path}")


# =============================================================================
# 9. Main
# =============================================================================

def parse_args():
    p = argparse.ArgumentParser(description="Train PPG arrhythmia detection model.")
    p.add_argument("--base_path",    default=DEFAULT_BASE_PROJECT_PATH,
                   help="Root project directory")
    p.add_argument("--dataset_path", default=None,
                   help="Dataset directory (default: <base_path>/PPG_AD_Dataset)")
    p.add_argument("--clip_weights", default=None,
                   help="Path to CLIP checkpoint .pth file")
    p.add_argument("--save_path",    default=None,
                   help="Output path for the best model checkpoint")
    p.add_argument("--epochs",       type=int,   default=200)
    p.add_argument("--batch_size",   type=int,   default=32)
    p.add_argument("--lr",           type=float, default=5e-5)
    p.add_argument("--enc_lr_ratio", type=float, default=0.1,
                   help="Encoder LR = lr * enc_lr_ratio")
    p.add_argument("--alpha",        type=float, default=2.0,
                   help="Classification loss weight")
    p.add_argument("--beta",         type=float, default=0.8,
                   help="Segmentation loss weight")
    p.add_argument("--weight_decay", type=float, default=5e-4)
    p.add_argument("--max_norm",     type=float, default=1.0,
                   help="Gradient clipping max norm")
    p.add_argument("--unfreeze_epoch", type=int, default=50,
                   help="Epoch at which to unfreeze the encoder for fine-tuning")
    p.add_argument("--patience",     type=int,   default=50,
                   help="EarlyStopping patience")
    p.add_argument("--num_workers",  type=int,   default=0)
    p.add_argument("--seed",         type=int,   default=42)
    return p.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    # --- Resolve paths ---
    dataset_path = args.dataset_path or os.path.join(args.base_path, "PPG_AD_Dataset")
    clip_weights = args.clip_weights or os.path.join(
        dataset_path, "checkpoints", "clip_biobert_hyh_xai.pth"
    )
    model_save_path = args.save_path or os.path.join(
        args.base_path, "unet_clip_biobert_hyh_xai.pth"
    )

    print(f"Dataset path : {dataset_path}")
    print(f"CLIP weights : {clip_weights}")
    print(f"Model output : {model_save_path}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device       : {device}")

    # --- CSV paths ---
    train_csv_main   = os.path.join(dataset_path, "train.csv")
    train_csv_normal = os.path.join(dataset_path, "train_normal.csv")
    valid_csv_main   = os.path.join(dataset_path, "valid.csv")
    valid_csv_normal = os.path.join(dataset_path, "valid_normal.csv")
    test_csv         = os.path.join(dataset_path, "test.csv")

    final_train_csv = prepare_merged_csv(
        train_csv_main, train_csv_normal,
        os.path.join(dataset_path, "train_combined.csv"),
    )
    final_valid_csv = prepare_merged_csv(
        valid_csv_main, valid_csv_normal,
        os.path.join(dataset_path, "valid_combined.csv"),
    )

    # --- Wave directories ---
    train_wave_dirs = [
        os.path.join(dataset_path, "train"),
        os.path.join(dataset_path, "train_normal"),
    ]
    valid_wave_dirs = [
        os.path.join(dataset_path, "valid"),
        os.path.join(dataset_path, "val_normal"),
    ]
    test_wave_dir = os.path.join(dataset_path, "test")

    # --- Datasets ---
    print("\n[Building datasets]")
    train_dataset = PPGDataset_Combined(
        data_dirs=train_wave_dirs,
        clinical_csv_path=final_train_csv,
        target_length=TARGET_LENGTH,
        fs=100.0,
        is_test=False,
    )
    valid_dataset = PPGDataset_Combined(
        data_dirs=valid_wave_dirs,
        clinical_csv_path=final_valid_csv,
        target_length=TARGET_LENGTH,
        fs=100.0,
        is_test=False,
    )

    # Weighted sampler to handle class imbalance
    class_counts = np.bincount(train_dataset.labels, minlength=len(CLASS_MAP))
    print(f"Train class distribution: {dict(zip(INV_CLASS_MAP.values(), class_counts))}")
    safe_counts           = np.clip(class_counts, 1, None)
    sample_weights_tensor = torch.tensor(1.0 / safe_counts, dtype=torch.float32)
    sample_weights        = sample_weights_tensor[train_dataset.labels]
    sampler = WeightedRandomSampler(
        weights=sample_weights, num_samples=len(sample_weights), replacement=True
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=bool(args.num_workers),
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=bool(args.num_workers),
    )

    # --- Model ---
    print("\n[Building model]")
    model = UNet1D_ResNet_Combined(
        n_channels=1,
        n_classes=len(CLASS_MAP),
        n_clinical_features=N_CLINICAL_FEATURES,
        n_hrv_features=N_HRV_FEATURES,
    ).to(device)

    model.load_clip_weights(clip_weights, device)

    # Freeze encoder initially
    freeze_encoder = True
    for name, param in model.named_parameters():
        if name.startswith("encoder.") or name.startswith("clip_projection."):
            param.requires_grad = False
    print(f"Encoder frozen until epoch {args.unfreeze_epoch}.")

    # --- Loss functions ---
    custom_weights  = torch.tensor([0.2, 0.6, 0.6, 0.2], dtype=torch.float32).to(device)
    criterion_clf   = FocalLoss(gamma=2.2, alpha=custom_weights)
    criterion_seg   = BCEDiceLoss()

    # --- Optimiser (layer-wise LR) ---
    encoder_params = [
        p for n, p in model.named_parameters()
        if p.requires_grad and (n.startswith("encoder.") or n.startswith("clip_projection."))
    ]
    other_params = [
        p for n, p in model.named_parameters()
        if p.requires_grad and not (n.startswith("encoder.") or n.startswith("clip_projection."))
    ]
    optimizer = optim.AdamW([
        {"params": encoder_params, "lr": args.lr * args.enc_lr_ratio,
         "weight_decay": args.weight_decay},
        {"params": other_params,   "lr": args.lr,
         "weight_decay": args.weight_decay},
    ])

    # Warmup + cosine annealing LR schedule
    warmup_epochs = max(1, int(0.05 * args.epochs))

    def lr_lambda(epoch_idx):
        if epoch_idx < warmup_epochs:
            return (epoch_idx + 1) / warmup_epochs
        progress = (epoch_idx - warmup_epochs + 1) / max(1, args.epochs - warmup_epochs)
        return 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # --- EarlyStopping ---
    early_stopping = EarlyStopping(
        patience=args.patience,
        verbose=True,
        path=model_save_path,
        mode="min",
        monitor_name="val_total",
        delta=1e-3,
    )

    # --- Training loop ---
    history = {k: [] for k in [
        "train_total", "val_total",
        "train_clf",   "val_clf",
        "train_seg",   "val_seg",
        "train_f1",    "val_f1",
        "train_acc",   "val_acc",
        "val_seg_pixel_acc", "val_seg_dice",
    ]}

    print(f"\n[Starting training — {args.epochs} epochs]")
    start_time = datetime.now()

    for epoch in range(args.epochs):
        epoch_start = datetime.now()
        print(f"\nEpoch {epoch + 1}/{args.epochs}")

        # Unfreeze encoder at the scheduled epoch
        if freeze_encoder and (epoch + 1) == args.unfreeze_epoch:
            print(f"  Unfreezing encoder at epoch {args.unfreeze_epoch} (fine-tuning start).")
            for name, param in model.named_parameters():
                if name.startswith("encoder.") or name.startswith("clip_projection."):
                    param.requires_grad = True
            freeze_encoder = False

        tr_total, tr_clf, tr_seg, tr_f1, tr_acc = train_one_epoch(
            model, train_loader, criterion_clf, criterion_seg,
            optimizer, device,
            alpha=args.alpha, beta=args.beta, max_norm=args.max_norm,
        )
        scheduler.step()

        (va_total, va_clf, va_seg, va_f1, va_acc,
         _, _, _, va_pixel_acc, va_dice) = evaluate(
            model, valid_loader, criterion_clf, criterion_seg,
            device, alpha=args.alpha, beta=args.beta,
        )

        duration     = datetime.now() - epoch_start
        current_lr   = optimizer.param_groups[0]["lr"]

        print(f"  [Train] total={tr_total:.4f}  clf={tr_clf:.4f}  seg={tr_seg:.4f}"
              f"  F1={tr_f1:.4f}  Acc={tr_acc:.4f}")
        print(f"  [Valid] total={va_total:.4f}  clf={va_clf:.4f}  seg={va_seg:.4f}"
              f"  F1={va_f1:.4f}  Acc={va_acc:.4f}")
        print(f"  [Seg]   PixelAcc={va_pixel_acc:.4f}  Dice={va_dice:.4f}"
              f"  |  enc_lr={current_lr:.2e}  |  time={duration}")

        history["train_total"].append(tr_total);  history["val_total"].append(va_total)
        history["train_clf"].append(tr_clf);      history["val_clf"].append(va_clf)
        history["train_seg"].append(tr_seg);      history["val_seg"].append(va_seg)
        history["train_f1"].append(tr_f1);        history["val_f1"].append(va_f1)
        history["train_acc"].append(tr_acc);      history["val_acc"].append(va_acc)
        history["val_seg_pixel_acc"].append(va_pixel_acc)
        history["val_seg_dice"].append(va_dice)

        early_stopping(
            va_total, model,
            ref_metric_for_log=va_pixel_acc, ref_name="seg_pixel_acc",
        )
        if early_stopping.early_stop:
            print(f"\nEarly stopping triggered at epoch {epoch + 1}.")
            break

    total_time = datetime.now() - start_time
    print(f"\nTotal training time: {total_time}")

    # Save training curves
    curve_path = os.path.join(args.base_path, "training_history.png")
    plot_history(history, curve_path)

    # Load best weights
    if os.path.exists(model_save_path):
        model.load_state_dict(torch.load(model_save_path, map_location=device, weights_only=True))
        print(f"Best checkpoint loaded from {model_save_path}"
              f" (best score: {early_stopping.best_score:.6f})")

    # --- Final evaluation on test set ---
    print("\n[Test set evaluation]")
    test_dataset = PPGDataset_Combined(
        data_dirs=[test_wave_dir],
        clinical_csv_path=test_csv,
        target_length=TARGET_LENGTH,
        fs=100.0,
        is_test=False,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=bool(args.num_workers),
    )

    (test_loss, test_clf_loss, test_seg_loss, test_f1, test_acc,
     all_preds, all_labels, _, test_pixel_acc, test_dice) = evaluate(
        model, test_loader, criterion_clf, criterion_seg, device,
        alpha=args.alpha, beta=args.beta,
    )

    print(f"\n--- Test Results ---")
    print(f"  Total loss      : {test_loss:.4f}")
    print(f"  Accuracy        : {test_acc:.4f}  ({test_acc*100:.2f}%)")
    print(f"  F1 (Macro)      : {test_f1:.4f}")
    print(f"  Clf loss        : {test_clf_loss:.4f}")
    print(f"  Pixel accuracy  : {test_pixel_acc:.4f}")
    print(f"  Dice            : {test_dice:.4f}")
    print(f"  Seg loss        : {test_seg_loss:.4f}")

    labels_order = list(range(len(CLASS_MAP)))
    target_names = [INV_CLASS_MAP[i] for i in labels_order]
    print("\n[Classification Report]\n")
    print(classification_report(
        all_labels, all_preds,
        labels=labels_order, target_names=target_names,
        digits=4, zero_division=0,
    ))

    # Confusion matrix figure
    cm = confusion_matrix(all_labels, all_preds, labels=labels_order)
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(target_names))); ax.set_xticklabels(target_names)
    ax.set_yticks(range(len(target_names))); ax.set_yticklabels(target_names)
    for i in range(len(target_names)):
        for j in range(len(target_names)):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center")
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title("Confusion Matrix (Test)")
    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    cm_path = os.path.join(args.base_path, "confusion_matrix.png")
    fig.savefig(cm_path, dpi=100)
    plt.close(fig)
    print(f"Confusion matrix saved to: {cm_path}")


if __name__ == "__main__":
    main()
