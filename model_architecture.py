import os
import json
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.signal import find_peaks
from scipy import stats
import pandas as pd


# =============================================================================
# 1. Constants & Stats
# =============================================================================

# Paths to pre-computed normalisation statistics
# data_stats.json    — PPG/HRV stats
# clinical_stats.json — clinical feature scaling stats
CONFIG_PATH = "data_stats.json"
CLINICAL_STATS_PATH = "clinical_stats.json"


N_HRV_FEATURES = 4  # HR, SDNN, RMSSD, pNN50
eps_hrv = 1e-6      # small epsilon to prevent division by zero during HRV normalisation

# Clinical feature column definitions
CLINICAL_FEATURES_LIST = [
    'age', 'bmi', 'opdur', 'preop_na', 'preop_bun', 'preop_cr', 'preop_k',
    'intraop_eph', 'intraop_phe',
    'sex_M', 'emop_Y', 'preop_dm_Y', 'preop_htn_Y'
]
N_CLINICAL_FEATURES = len(CLINICAL_FEATURES_LIST)  # 13

# Continuous columns (StandardScaler targets)
CONTINUOUS_COLS = CLINICAL_FEATURES_LIST[:9]

# Binary string columns (one-hot encoding targets)
BINARY_STR_COLS = {
    "sex": ["M", "F"], "emop": ["N", "Y"],
    "preop_dm": ["N", "Y"], "preop_htn": ["N", "Y"]
}

CLASS_MAP = {'normal': 0, 'af': 1, 'b': 2, 't': 3}
INV_CLASS_MAP = {0: 'normal', 1: 'af', 2: 'b', 3: 't'}
N_CLASSES = 4

# -----------------------------------------------------------------------------
# Stats loading
# -----------------------------------------------------------------------------
# [A] Load data_stats.json (PPG/HRV stats)
stats_cfg = {}
try:
    with open(CONFIG_PATH, "r") as f:
        stats_cfg = json.load(f)
except FileNotFoundError:
    pass

def _as_vec(x, default_val):
    default_vec = [default_val] * N_HRV_FEATURES
    arr = np.array(x if x is not None else default_vec, dtype=np.float32).reshape(-1)
    if arr.size != N_HRV_FEATURES:
        arr = np.array(default_vec, dtype=np.float32)
    return arr

HRV_MEAN   = _as_vec(stats_cfg.get("hrv_mean"),   0.0)
HRV_STD    = _as_vec(stats_cfg.get("hrv_std"),    1.0)
HRV_MEDIAN = _as_vec(stats_cfg.get("hrv_median"), 0.0)

CONT_MEAN = {c: 0.0 for c in CONTINUOUS_COLS}
CONT_STD = {c: 1.0 for c in CONTINUOUS_COLS}
for col in CONTINUOUS_COLS:
    CONT_MEAN[col] = float(stats_cfg.get(f"cont_mean_{col}", 0.0))
    CONT_STD[col] = float(stats_cfg.get(f"cont_std_{col}", 1.0))
    if CONT_STD[col] == 0: CONT_STD[col] = 1.0

TARGET_LENGTH = int(stats_cfg.get("target_length", 286))

# [B] Load clinical_stats.json (clinical scaling stats)
clinical_stats_cfg = {}
try:
    with open(CLINICAL_STATS_PATH, "r") as f:
        clinical_stats_cfg = json.load(f)
except FileNotFoundError:
    pass

# StandardScaler stats — used for stage-1 Z-score normalisation
STD_SCALER_MEAN = np.array(clinical_stats_cfg.get("std_scaler_mean", [0.0] * N_CLINICAL_FEATURES), dtype=np.float32)
STD_SCALER_STD  = np.array(clinical_stats_cfg.get("std_scaler_std",  [1.0] * N_CLINICAL_FEATURES), dtype=np.float32)
if np.any(STD_SCALER_STD == 0):
    STD_SCALER_STD[STD_SCALER_STD == 0] = 1.0

# Min-Max scaling stats — used for stage-2 normalisation
STDS_MIN = np.array(clinical_stats_cfg.get("scaled_features_min", [0.0] * N_CLINICAL_FEATURES), dtype=np.float32)
STDS_MAX = np.array(clinical_stats_cfg.get("scaled_features_max", [1.0] * N_CLINICAL_FEATURES), dtype=np.float32)


# =============================================================================
# 2. HRV and clinical feature preprocessing
# =============================================================================

def _compute_hrv_raw(signal: np.ndarray, fs: float = 100.0) -> np.ndarray | None:
    """Compute raw HRV features (pre-normalisation): [HR (bpm), SDNN, RMSSD, pNN50]."""
    if signal is None or len(signal) < 10 or not np.any(np.isfinite(signal)):
        return None
    try:
        min_height = np.nanquantile(signal, 0.6)
    except Exception:
        return None

    peaks, _ = find_peaks(signal, height=min_height, distance=int(fs * 0.3))
    if len(peaks) < 3: return None

    rri_ms = np.diff(peaks) * (1000.0 / fs)
    if rri_ms.size < 2 or not np.all(np.isfinite(rri_ms)): return None

    mean_rri = np.mean(rri_ms)
    hr_bpm   = 60000.0 / mean_rri if mean_rri > 0 else np.nan
    sdnn     = np.std(rri_ms)
    diff_rri = np.diff(rri_ms)
    if diff_rri.size == 0: return None
    rmssd = np.sqrt(np.mean(diff_rri ** 2))
    pnn50 = (np.sum(np.abs(diff_rri) > 50.0) / diff_rri.size) * 100.0

    feats = np.array([hr_bpm, sdnn, rmssd, pnn50], dtype=np.float32)
    if not np.any(np.isfinite(feats)): return None
    return feats

def get_hrv_features(signal: np.ndarray, fs: float = 100.0) -> np.ndarray:
    """Compute HRV features, impute missing values with global median, then Z-score standardise."""
    raw = _compute_hrv_raw(signal, fs=fs)
    if raw is None:
        raw = HRV_MEDIAN.copy()
    else:
        bad_mask = ~np.isfinite(raw)
        if np.any(bad_mask):
            raw[bad_mask] = HRV_MEDIAN[bad_mask]  # impute with global median

    # Global Z-score standardisation
    standardized = (raw - HRV_MEAN) / (HRV_STD + eps_hrv)
    return standardized.astype(np.float32)

def get_clinical_feature_vector(patient_data: dict, fill_value: float = 0.0) -> np.ndarray:
    """
    Convert a raw clinical record (from the SQLite DB) into a 13-dim feature vector
    by replicating the 2-stage scaling used during training:
        Stage 1: StandardScaler (Z-score)
        Stage 2: Min-Max normalisation
    """
    
    df_raw = pd.DataFrame([patient_data])

    
    for col_raw in list(BINARY_STR_COLS.keys()):
        if col_raw not in df_raw.columns:
            df_raw[col_raw] = BINARY_STR_COLS[col_raw][0]

    # One-hot encoding (drop_first=True to produce 13-dim vector)
    categorical_cols = list(BINARY_STR_COLS.keys())
    df_ohe = pd.get_dummies(df_raw, columns=categorical_cols, drop_first=True)

    raw_features_list = []

    # Insert continuous features; impute missing values with training mean
    for col in CONTINUOUS_COLS:
        val = df_ohe.get(col, np.nan).iloc[0] if col in df_ohe.columns else np.nan
        if np.isnan(val) or val is None:
            val = STD_SCALER_MEAN[CLINICAL_FEATURES_LIST.index(col)]
        raw_features_list.append(val)

    # Insert binary features (missing OHE columns default to 0)
    for col in CLINICAL_FEATURES_LIST[len(CONTINUOUS_COLS):]:
        val = df_ohe.get(col, 0.0).iloc[0] if col in df_ohe.columns else 0.0
        raw_features_list.append(val)

    feature_vector = np.array(raw_features_list, dtype=np.float32)


    scaled_features = (feature_vector - STD_SCALER_MEAN) / STD_SCALER_STD

    # Stage 2: Min-Max normalisation using training distribution bounds
    diff = STDS_MAX - STDS_MIN
    diff[diff == 0] = 1  # constant columns: keep at 0 rather than dividing by zero

    normalized_features = (scaled_features - STDS_MIN) / diff

    return normalized_features.astype(np.float32)


def preprocess_for_inference(signal_raw, patient_clinical_data: dict):
    """
    Convert a raw PPG signal and patient clinical dict into model-ready tensors.
    Also returns the resampled signal array for plotting.

    Returns:
        model_input          : (1, 1, TARGET_LENGTH) tensor
        hrv_tensor           : (1, 4) tensor
        clinical_vector_tensor: (1, 13) tensor
        plot_signal          : (TARGET_LENGTH,) numpy array (original amplitude scale)
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 1. Extract HRV features using global training stats
    hrv = get_hrv_features(signal_raw)
    hrv_tensor = torch.from_numpy(hrv).float().unsqueeze(0).to(device)  # (1, 4)

    # 2. Extract clinical feature vector (2-stage scaling)
    clinical_feat = get_clinical_feature_vector(patient_clinical_data)
    clinical_vector_tensor = torch.from_numpy(clinical_feat).float().unsqueeze(0).to(device)  # (1, 13)

    # 3. Resample signal to TARGET_LENGTH via linear interpolation
    sig_tensor = torch.from_numpy(signal_raw).float().view(1, 1, -1)  # (1, 1, L_raw)
    sig_resampled = F.interpolate(sig_tensor, size=TARGET_LENGTH, mode='linear', align_corners=False)

    # 4. Sample-wise Z-score normalisation
    x = sig_resampled.squeeze(0).float() 
    current_mean = x.mean()
    current_std  = x.std()
    if current_std > 1e-8:
        x = (x - current_mean) / current_std
    else:
        x = x - current_mean  

    model_input = x.unsqueeze(0)  

    return model_input, hrv_tensor, clinical_vector_tensor, sig_resampled.squeeze().cpu().numpy()


# =============================================================================
# 3. Model component definitions
# =============================================================================

class ResidualBlock(nn.Module):
    """Basic ResNet block: Conv -> BN -> ReLU -> Conv -> BN -> (+skip) -> ReLU."""
    def __init__(self, in_channels, out_channels, kernel_size=5, stride=1, padding=2):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, stride=1, padding=padding, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels)
            )

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.relu(out)
        return out


class ResNet1D_Backbone(nn.Module):
    """
    Shares the same architecture as contrastive_learning.ResNet1DEncoder so that
    CLIP checkpoint weights can be loaded directly.

    Unlike ResNet1DEncoder (which outputs a single embedding), this returns a list
    of intermediate feature maps for U-Net skip connections.

    IMPORTANT: layer names (self.initial, self.layer1-3) must stay in sync with
    the CLIP training code so that load_clip_weights() can map keys correctly.
    """
    def __init__(self, in_channels=1, base_filters=32):
        super().__init__()

        # Entry block: stride 2 → L/2, MaxPool → L/4
        self.initial = nn.Sequential(
            nn.Conv1d(in_channels, base_filters, kernel_size=15, stride=2, padding=7, bias=False),
            nn.BatchNorm1d(base_filters),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        )

        # Residual blocks: channels double, length halves at each stage
        self.layer1 = ResidualBlock(base_filters,     base_filters * 2, stride=2)  # 32→64,  L/8
        self.layer2 = ResidualBlock(base_filters * 2, base_filters * 4, stride=2)  # 64→128, L/16
        self.layer3 = ResidualBlock(base_filters * 4, base_filters * 8, stride=2)  # 128→256,L/32

    def forward(self, x):
        # Return intermediate feature maps for U-Net skip connections
        features = []

        x = self.initial(x)
        features.append(x)  

        x = self.layer1(x)
        features.append(x) 

        x = self.layer2(x)
        features.append(x)

        x = self.layer3(x)
        features.append(x) 

        return features


class DoubleConv(nn.Module):
    """Two consecutive Conv1d-GroupNorm-ReLU layers used in the U-Net decoder."""
    def __init__(self, in_channels, out_channels, mid_channels=None, gn_groups=8):
        super().__init__()
        if mid_channels is None:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv1d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(gn_groups, mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(gn_groups, out_channels),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.double_conv(x)


class Up(nn.Module):
    """U-Net upsampling block: upsample x1, concat with skip x2, then DoubleConv."""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up   = nn.Upsample(scale_factor=2, mode='linear', align_corners=False)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)

        # Pad x1 if its length does not exactly match x2
        diff = x2.size(2) - x1.size(2)
        if diff != 0:
            x1 = F.pad(x1, [diff // 2, diff - diff // 2])

        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class SE1D(nn.Module):
    """1-D Squeeze-and-Excitation block."""
    def __init__(self, C, r=16):
        super().__init__()
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(C, C // r), nn.ReLU(inplace=True),
            nn.Linear(C // r, C), nn.Sigmoid()
        )
    def forward(self, x):
        b, c, _ = x.shape
        w = self.gap(x).view(b, c)
        w = self.fc(w).view(b, c, 1)
        return x * w


class UNet1D_ResNet_Combined(nn.Module):
    """
    Inference-only hybrid model: CLIP pre-trained ResNet1D encoder + U-Net decoder.

    Inputs:
        signal_input      : (B, 1, L)
        clinical_features : (B, 13)
        hrv_features      : (B, 4)
    Outputs:
        clf_logits : (B, n_classes)  — raw classification logits
        y_seg      : (B, L)          — raw segmentation logits (pre-sigmoid)

    """
    def __init__(self, n_channels, n_classes, n_clinical_features=N_CLINICAL_FEATURES, n_hrv_features=N_HRV_FEATURES):
        super().__init__()
        self.n_classes = n_classes
        self.n_clinical_features = n_clinical_features
        self.n_hrv_features = n_hrv_features

        self.encoder_feature_dim = 256
        self.clip_feature_dim    = 128  

        # Encoder
        self.encoder = ResNet1D_Backbone(in_channels=n_channels, base_filters=32)

        # Project bottleneck (256-d) into CLIP feature space (128-d)
        self.clip_projection = nn.Linear(self.encoder_feature_dim, self.clip_feature_dim)

        # Decoder
        self.up1 = Up(256 + 128, 128)
        self.up2 = Up(128 + 64,  64)
        self.up3 = Up(64  + 32,  32)

        self.final_up = nn.Sequential(
            nn.Upsample(scale_factor=4, mode='linear', align_corners=False),
            nn.Conv1d(32, 16, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(4, 16),
            nn.ReLU(inplace=True),
        )

        # Segmentation head
        self.seg_head = nn.Conv1d(16, 1, kernel_size=1)

        # Classification head 
        clf_in_dim = self.clip_feature_dim + self.n_clinical_features + self.n_hrv_features

        self.clf_head = nn.Sequential(
            nn.Linear(clf_in_dim, 128),
            nn.LayerNorm(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.6),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.6),
            nn.Linear(64, n_classes)
        )

    def forward(self, signal_input, clinical_features, hrv_features):
        # Encoder
        features = self.encoder(signal_input)
        x_stem = features[0]
        x1     = features[1]
        x2     = features[2]
        x3     = features[3] 

        # Classification path
        gap          = F.adaptive_avg_pool1d(x3, 1).squeeze(-1)  
        clip_feature = self.clip_projection(gap)                  
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
                seg_logits,
                size=signal_input.size(2),
                mode='linear',
                align_corners=False
            )

        y_seg = seg_logits.squeeze(1)

        return clf_logits, y_seg


def find_mask_indices(mask_array, offset=0, threshold=0.5):
    """
    Convert a 1-D binary/probability mask array into a human-readable string
    describing contiguous anomaly segments, e.g. "Anomaly mask detected from
    index 12 to 45, index 80 to 95."

    Args:
        mask_array : 1-D numpy array of mask values
        offset     : index offset added to all reported positions
        threshold  : binarisation threshold
    """
    binary_mask = (mask_array > threshold).astype(np.int32)

    if not np.any(binary_mask):
        return "No anomaly mask detected."

    indices = np.where(binary_mask > 0)[0]
    if len(indices) == 0:
        return "No anomaly mask detected."

    # Find contiguous segments
    segments = []
    start_index = indices[0]
    for i in range(1, len(indices)):
        if indices[i] != indices[i - 1] + 1:  
            segments.append((start_index, indices[i - 1]))
            start_index = indices[i]
    segments.append((start_index, indices[-1])) 

    segment_texts = [f"index {s + offset} to {e + offset}" for s, e in segments]

    if len(segment_texts) == 0:
        return "No anomaly mask detected."

    return "Anomaly mask detected from " + ", ".join(segment_texts) + "."
