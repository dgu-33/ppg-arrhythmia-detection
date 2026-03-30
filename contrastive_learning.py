import os
import random
import argparse

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
from sklearn.manifold import TSNE

from model_architecture import ResidualBlock
from llm_utils import ClinicalMLP, Projection


# =============================================================================
# 1. Constants
# =============================================================================

CONTINUOUS_COLS = [
    "age", "bmi", "opdur", "preop_na", "preop_bun",
    "preop_cr", "preop_k", "intraop_eph", "intraop_phe",
]
CATEGORICAL_COLS  = {"sex": ["M", "F"], "emop": ["N", "Y"]}
BINARY_STR_COLS   = {"preop_dm": ["N", "Y"], "preop_htn": ["N", "Y"]}

STATE_LIST   = ["Normal", "AF", "B", "T"]
STATE_TO_IDX = {s: i for i, s in enumerate(STATE_LIST)}

# Medical text descriptions used as contrastive anchors (BioBERT input)
LABEL_TEXT_MAP = {
    "Normal": (
        "Normal Sinus Rhythm (NSR). Heart rate is within normal range (60-100 bpm). "
        "Regular and consistent beats with stable R-R intervals."
    ),
    "AF": (
        "Atrial Fibrillation. The hallmark is an 'irregularly irregular' rhythm. "
        "P waves are completely absent and replaced by chaotic fibrillatory waves. "
        "R-R intervals vary unpredictably."
    ),
    "B": (
        "Sinus Bradycardia. Heart rate is strictly less than 60 bpm. "
        "Rhythm is regular but significantly slow. Long R-R intervals."
    ),
    "T": (
        "Sinus Tachycardia. Heart rate exceeds 100 bpm but the rhythm remains regular. "
        "Fast, steady beating with consistent but shortened R-R intervals. "
        "Distinct from chaotic rhythms."
    ),
}

TARGET_SEG_LEN = 286


# =============================================================================
# 2. Clinical data preprocessing
# =============================================================================

# Global scaler stats — populated by fit_clinical_scaler()
CONT_MEAN = {c: 0.0 for c in CONTINUOUS_COLS}
CONT_STD  = {c: 1.0 for c in CONTINUOUS_COLS}


def preprocess_clinical_data(csv_path: str) -> pd.DataFrame:
    """Load a clinical CSV, drop unused columns, and fill missing values."""
    df = pd.read_csv(csv_path)
    drop_cols = ["preop_ph", "preop_hco3", "preop_pao2", "preop_paco2"]
    df = df.drop(columns=drop_cols, errors="ignore")

    all_features = (
        CONTINUOUS_COLS
        + list(CATEGORICAL_COLS.keys())
        + list(BINARY_STR_COLS.keys())
    )
    for col in all_features:
        if col not in df.columns:
            continue
        if col in CONTINUOUS_COLS:
            df[col] = df[col].fillna(df[col].mean())
        else:
            mode_val = df[col].mode()
            df[col] = df[col].fillna(mode_val[0] if not mode_val.empty else "Unknown")
    return df


def fit_clinical_scaler(pairs: list) -> None:
    """Compute per-column mean/std from matched pairs and update global dicts."""
    df = pd.DataFrame([p[0] for p in pairs])
    for c in CONTINUOUS_COLS:
        vals = pd.to_numeric(df[c], errors="coerce").dropna()
        if not vals.empty:
            CONT_MEAN[c] = float(vals.mean())
            std = float(vals.std(ddof=0))
            CONT_STD[c] = std if std > 0 else 1.0


def clinical_to_vector(clin_dict: dict) -> np.ndarray:
    """
    Convert a clinical record dict to a normalised 17-dim float vector.
    Layout: 9 continuous (z-scored) + 2+2+2+2 full one-hot categorical = 17
    (Note: downstream inference uses 13-dim drop_first encoding;
     llm_utils.transform_13_to_17 bridges the two representations for RAG.)
    """
    vec = []
    for c in CONTINUOUS_COLS:
        v = float(clin_dict.get(c, CONT_MEAN[c]))
        vec.append((v - CONT_MEAN[c]) / (CONT_STD[c] + 1e-6))
    for col_map in [CATEGORICAL_COLS, BINARY_STR_COLS]:
        for col, cats in col_map.items():
            val = str(clin_dict.get(col, "")).upper()
            vec.extend([1.0 if val == str(c).upper() else 0.0 for c in cats])
    return np.array(vec, dtype=np.float32)


# =============================================================================
# 3. File matching
# =============================================================================

def match_files_with_clinical(
    root_dirs: list,
    clinical_dfs: list,
    file_exts: list,
    exclude_keywords: list = [],
) -> list:
    """
    Walk root_dirs and pair each .npy/.npz file with its clinical record
    (matched by the integer prefix of the filename, i.e. caseid).

    Returns a list of (clinical_dict, file_path, caseid) tuples.
    Only files whose caseid appears in one of the clinical_dfs are included.
    """
    combined_clinical = {}
    for df in clinical_dfs:
        combined_clinical.update(df.set_index("caseid").to_dict("index"))

    pairs = []
    for root_dir in root_dirs:
        for root, _, files in os.walk(root_dir):
            for fname in files:
                path_parts = os.path.join(root, fname).lower().split(os.sep)
                if any(k in path_parts for k in exclude_keywords):
                    continue
                if not any(fname.endswith(ext) for ext in file_exts):
                    continue
                file_path = os.path.join(root, fname)
                try:
                    caseid = int(os.path.basename(fname).split("_")[0])
                    if caseid in combined_clinical:
                        pairs.append((combined_clinical[caseid], file_path, caseid))
                except ValueError:
                    continue
    return pairs


# =============================================================================
# 4. Dataset
# =============================================================================

def _load_npz_sample(path: str):
    """Extract (wave, mask, state_str) from a .npz file."""
    d = np.load(path, allow_pickle=True)
    wave = d["signal"].astype(np.float32)
    mask = d["mask"].astype(np.int32)

    state_str = "Unknown"
    if d["meta"].size > 0 and isinstance(d["meta"][0], dict):
        state_str = str(d["meta"][0].get("arr_type", "Unknown"))
    if state_str == "Unknown":
        masked_class = d["class"][mask.astype(bool)]
        if masked_class.size > 0:
            vals, counts = np.unique(masked_class, return_counts=True)
            mapper = {0: "Normal", 1: "Normal", 2: "AF", 3: "B", 4: "T"}
            state_str = mapper.get(int(vals[np.argmax(counts)]), str(vals[np.argmax(counts)]))
    return wave, mask, state_str


def _augment_ppg(wave: np.ndarray) -> np.ndarray:
    if np.random.rand() < 0.5:
        wave = wave + np.random.normal(0, 0.01, wave.shape)
    if np.random.rand() < 0.5:
        wave = wave * np.random.uniform(0.9, 1.1)
    return wave.astype(np.float32)


class WaveClinTextDataset(Dataset):
    def __init__(self, pairs: list, tokenizer, target_len: int = 286, is_train: bool = False):
        self.pairs      = pairs
        self.tokenizer  = tokenizer
        self.target_len = target_len
        self.is_train   = is_train

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        clin, path, caseid = self.pairs[idx]

        try:
            if path.endswith(".npz"):
                wave, _, state = _load_npz_sample(path)
            else:
                wave  = np.load(path).astype(np.float32)
                state = "Normal"
        except Exception:
            return self.__getitem__((idx + 1) % len(self))

        if state not in STATE_TO_IDX:
            return self.__getitem__((idx + 1) % len(self))

        # Resample to target length
        wave_t = torch.from_numpy(wave).unsqueeze(0).unsqueeze(0) 
        if wave.shape[-1] > self.target_len:
            seg = F.interpolate(
                wave_t, size=self.target_len, mode="linear", align_corners=True
            ).squeeze().numpy()
        else:
            seg = wave
            if len(seg) < self.target_len:
                seg = np.pad(seg, (0, self.target_len - len(seg)), mode="constant")

        seg = seg.astype(np.float32)
        if self.is_train:
            seg = _augment_ppg(seg)
        seg = (seg - seg.mean()) / (seg.std() + 1e-6)

        text_desc = LABEL_TEXT_MAP.get(state, "Unknown heart rhythm condition.")
        encoding  = self.tokenizer(
            text_desc,
            padding="max_length",
            max_length=32,
            truncation=True,
            return_tensors="pt",
        )

        return {
            "wave":           torch.from_numpy(seg).unsqueeze(0),
            "clin":           torch.from_numpy(clinical_to_vector(clin)),
            "caseid":         torch.tensor(caseid, dtype=torch.long),
            "state":          state,
            "input_ids":      encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
        }


# =============================================================================
# 5. Model definitions
# =============================================================================

class ResNet1DEncoder(nn.Module):
    """
    CLIP wave encoder: produces a single L2-normalised 128-dim embedding.

    Distinct from model_architecture.ResNet1D_Backbone which returns
    intermediate feature maps for U-Net skip connections.
    Layer names (initial, layer1-3) intentionally match ResNet1D_Backbone
    so that train.py's load_clip_weights() can transfer encoder weights.
    """
    def __init__(self, in_channels: int = 1, base_filters: int = 32, out_dim: int = 128):
        super().__init__()
        self.initial = nn.Sequential(
            nn.Conv1d(in_channels, base_filters, kernel_size=15, stride=2, padding=7, bias=False),
            nn.BatchNorm1d(base_filters),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1),
        )
        self.layer1 = ResidualBlock(base_filters,   base_filters * 2, stride=2)
        self.layer2 = ResidualBlock(base_filters * 2, base_filters * 4, stride=2)
        self.layer3 = ResidualBlock(base_filters * 4, base_filters * 8, stride=2)
        self.gap     = nn.AdaptiveAvgPool1d(1)
        self.flatten = nn.Flatten()
        self.fc      = nn.Linear(base_filters * 8, out_dim)

    def forward(self, x):
        x = self.initial(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.flatten(self.gap(x))
        return F.normalize(self.fc(x), dim=-1)


class BioBERTTextEncoder(nn.Module):
    """Bio_ClinicalBERT → 128-dim L2-normalised embedding."""
    def __init__(self, out_dim: int = 128, freeze_bert: bool = True):
        super().__init__()
        self.bert = AutoModel.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False
        self.dropout = nn.Dropout(0.1)
        self.fc      = nn.Linear(768, out_dim)

    def forward(self, input_ids, attention_mask):
        cls = self.bert(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state[:, 0, :]
        return F.normalize(self.fc(self.dropout(cls)), dim=-1)


# =============================================================================
# 6. Loss
# =============================================================================

def masked_info_nce(
    sim_matrix: torch.Tensor,
    same_case_mask: torch.Tensor,
    tau: float = 0.07,
) -> torch.Tensor:
    """
    InfoNCE loss where samples from the same patient are excluded from the
    negative set (to avoid penalising truly similar pairs).

    sim_matrix     : (B, B) cosine similarity matrix
    same_case_mask : (B, B) bool — True where samples share the same ID
    """
    B       = sim_matrix.size(0)
    pos_sim = torch.diag(sim_matrix) / tau
    neg_mask = (
        ~torch.eye(B, dtype=torch.bool, device=sim_matrix.device)
        & ~same_case_mask
    )
    nll_list = []
    for i in range(B):
        pos_exp  = torch.exp(pos_sim[i])
        neg_exps = torch.exp(sim_matrix[i][neg_mask[i]] / tau)
        nll_list.append(-torch.log(pos_exp / (pos_exp + neg_exps.sum() + 1e-8)))
    return torch.stack(nll_list).mean()


# =============================================================================
# 7. Training and validation
# =============================================================================

def train_one_epoch(
    train_loader, wave_enc, clin_enc, text_enc, proj_w, proj_c, optimizer, device
) -> float:
    wave_enc.train(); clin_enc.train(); text_enc.train()
    proj_w.train();   proj_c.train()

    running_loss = 0.0
    for batch in train_loader:
        optimizer.zero_grad()

        wave           = batch["wave"].to(device)
        clin           = batch["clin"].to(device)
        input_ids      = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        caseids        = batch["caseid"].to(device)
        state_indices  = torch.tensor(
            [STATE_TO_IDX[s] for s in batch["state"]], device=device
        )

        z_w = proj_w(wave_enc(wave))
        z_c = proj_c(clin_enc(clin))
        z_t = text_enc(input_ids, attention_mask)

        same_case_mask  = caseids[:, None] == caseids[None, :]
        same_state_mask = state_indices[:, None] == state_indices[None, :]

        loss_clin = masked_info_nce(torch.matmul(z_w, z_c.T), same_case_mask)
        loss_text = masked_info_nce(torch.matmul(z_w, z_t.T), same_state_mask, tau=0.07)
        loss      = 0.2 * loss_clin + 2.0 * loss_text

        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    return running_loss / len(train_loader)


def validate(
    valid_loader, wave_enc, clin_enc, text_enc, proj_w, proj_c, device
) -> float:
    wave_enc.eval(); clin_enc.eval(); text_enc.eval()
    proj_w.eval();   proj_c.eval()

    total_loss = 0.0
    with torch.no_grad():
        for batch in valid_loader:
            wave           = batch["wave"].to(device)
            clin           = batch["clin"].to(device)
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            caseids        = batch["caseid"].to(device)
            state_indices  = torch.tensor(
                [STATE_TO_IDX[s] for s in batch["state"]], device=device
            )

            z_w = proj_w(wave_enc(wave))
            z_c = proj_c(clin_enc(clin))
            z_t = text_enc(input_ids, attention_mask)

            same_case_mask  = caseids[:, None] == caseids[None, :]
            same_state_mask = state_indices[:, None] == state_indices[None, :]

            loss_clin = masked_info_nce(torch.matmul(z_w, z_c.T), same_case_mask)
            loss_text = masked_info_nce(torch.matmul(z_w, z_t.T), same_state_mask, tau=0.07)
            total_loss += (0.2 * loss_clin + 1.2 * loss_text).item()

    return total_loss / max(1, len(valid_loader))


# =============================================================================
# 8. Knowledge base construction (for RAG retrieval in app.py)
# =============================================================================

def build_knowledge_base(
    dataloader, wave_enc, clin_enc, proj_w, proj_c, device, save_path: str
) -> None:
    """
    Embed every training sample and save as a retrieval database for RAG.
    Loaded at app startup via knowledge_base = torch.load(KB_PATH).
    """
    print(f"\n[Knowledge Base] Building from {len(dataloader.dataset)} samples...")
    wave_enc.eval(); clin_enc.eval(); proj_w.eval(); proj_c.eval()

    kb = {
        "clinical_vectors": [],
        "wave_vectors":     [],
        "raw_clinical":     [],
        "labels":           [],
        "caseids":          [],
    }

    with torch.no_grad():
        for batch in dataloader:
            wave    = batch["wave"].to(device)
            clin    = batch["clin"].to(device)
            caseids = batch["caseid"]
            states  = batch["state"]

            kb["clinical_vectors"].append(proj_c(clin_enc(clin)).cpu())
            kb["wave_vectors"].append(proj_w(wave_enc(wave)).cpu())
            kb["raw_clinical"].append(clin.cpu())
            kb["labels"].extend(states)
            kb["caseids"].extend(caseids.tolist())

    kb["clinical_vectors"] = torch.cat(kb["clinical_vectors"])
    kb["wave_vectors"]     = torch.cat(kb["wave_vectors"])
    kb["raw_clinical"]     = torch.cat(kb["raw_clinical"])

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(kb, save_path)
    print(f"Knowledge base saved: {save_path} ({len(kb['labels'])} entries, "
          f"dim={kb['clinical_vectors'].shape[1]})")


# =============================================================================
# 9. Visualisation
# =============================================================================

def plot_loss_curves(train_losses, val_losses, save_path: str) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(train_losses, label="Train Loss", color="blue")
    ax.plot(val_losses,   label="Val Loss",   color="red")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Loss")
    ax.set_title("Training vs Validation Loss")
    ax.legend(); ax.grid(True)
    plt.tight_layout()
    fig.savefig(save_path, dpi=100)
    plt.close(fig)
    print(f"Loss curves saved: {save_path}")


def plot_tsne(
    valid_loader, wave_enc, proj_w, text_enc, tokenizer, device, save_path: str,
    sample_limit: int = 1000,
) -> None:
    wave_enc.eval(); proj_w.eval(); text_enc.eval()
    all_embeds, all_labels = [], []

    with torch.no_grad():
        for batch in valid_loader:
            z_w = proj_w(wave_enc(batch["wave"].to(device)))
            all_embeds.append(z_w.cpu().numpy())
            all_labels.extend(batch["state"])
            if sum(len(e) for e in all_embeds) >= sample_limit:
                break

    all_embeds = np.concatenate(all_embeds)[:sample_limit]
    all_labels = np.array(all_labels)[:sample_limit]

    # Text anchor embeddings (one per class)
    text_anchors = []
    with torch.no_grad():
        for state in STATE_LIST:
            enc = tokenizer(
                LABEL_TEXT_MAP[state], return_tensors="pt",
                padding="max_length", max_length=32, truncation=True,
            )
            z_t = text_enc(enc["input_ids"].to(device), enc["attention_mask"].to(device))
            text_anchors.append(z_t.cpu().numpy())
    text_anchors = np.concatenate(text_anchors)

    combined = np.vstack([all_embeds, text_anchors])
    print(f"Running t-SNE on {len(combined)} points...")
    tsne   = TSNE(n_components=2, random_state=42, perplexity=30,
                  init="pca", learning_rate="auto")
    z_2d   = tsne.fit_transform(combined)
    z_wave = z_2d[:len(all_embeds)]
    z_text = z_2d[len(all_embeds):]

    fig, ax = plt.subplots(figsize=(12, 10))
    sns.scatterplot(x=z_wave[:, 0], y=z_wave[:, 1], hue=all_labels,
                    palette="viridis", alpha=0.6, s=60, ax=ax)
    for i, label in enumerate(STATE_LIST):
        ax.scatter(z_text[i, 0], z_text[i, 1],
                   marker="*", s=500, color="red", edgecolors="black", linewidth=1.5,
                   label=f"Text Anchor: {label}")
        ax.text(z_text[i, 0], z_text[i, 1] + 0.5, label,
                fontsize=12, fontweight="bold",
                bbox=dict(facecolor="white", alpha=0.7, edgecolor="none"))
    ax.set_title("BioBERT-CLIP: PPG Signal vs Medical Text Alignment", fontsize=15)
    ax.set_xlabel("t-SNE Dim 1"); ax.set_ylabel("t-SNE Dim 2")
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(save_path, dpi=100)
    plt.close(fig)
    print(f"t-SNE plot saved: {save_path}")


# =============================================================================
# 10. CLI
# =============================================================================

def parse_args():
    p = argparse.ArgumentParser(description="CLIP-style contrastive pre-training for PPG.")
    p.add_argument("--dataset_path", default="PPG_AD_Dataset",
                   help="Root of the PPG dataset directory")
    p.add_argument("--save_dir",     default=None,
                   help="Directory for checkpoints and plots (default: <dataset_path>/checkpoints)")
    p.add_argument("--epochs",       type=int,   default=100)
    p.add_argument("--batch_size",   type=int,   default=64)
    p.add_argument("--lr",           type=float, default=1e-3)
    p.add_argument("--target_len",   type=int,   default=286)
    p.add_argument("--num_workers",  type=int,   default=2)
    p.add_argument("--freeze_bert",  action="store_true", default=True,
                   help="Freeze BioBERT weights (only train projection head)")
    p.add_argument("--seed",         type=int,   default=42)
    return p.parse_args()


# =============================================================================
# 11. Main
# =============================================================================

def main():
    args = parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    save_dir = args.save_dir or os.path.join(args.dataset_path, "checkpoints")
    os.makedirs(save_dir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device     : {device}")
    print(f"Dataset    : {args.dataset_path}")
    print(f"Save dir   : {save_dir}")

    # -------------------------------------------------------------------------
    # Data preparation
    # -------------------------------------------------------------------------
    train_clinical        = preprocess_clinical_data(os.path.join(args.dataset_path, "train.csv"))
    train_normal_clinical = preprocess_clinical_data(os.path.join(args.dataset_path, "train_normal.csv"))
    valid_clinical        = preprocess_clinical_data(os.path.join(args.dataset_path, "valid.csv"))
    valid_normal_clinical = preprocess_clinical_data(os.path.join(args.dataset_path, "valid_normal.csv"))

    train_pairs = match_files_with_clinical(
        root_dirs=[args.dataset_path],
        clinical_dfs=[train_clinical, train_normal_clinical],
        file_exts=[".npy", ".npz"],
        exclude_keywords=["val", "test"],
    )
    valid_pairs = match_files_with_clinical(
        root_dirs=[args.dataset_path],
        clinical_dfs=[valid_clinical, valid_normal_clinical],
        file_exts=[".npy", ".npz"],
        exclude_keywords=[],
    )
    print(f"Train pairs: {len(train_pairs)}  |  Valid pairs: {len(valid_pairs)}")

    fit_clinical_scaler(train_pairs)

    print("Loading BioBERT tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")

    train_ds     = WaveClinTextDataset(train_pairs, tokenizer, target_len=args.target_len, is_train=True)
    valid_ds     = WaveClinTextDataset(valid_pairs, tokenizer, target_len=args.target_len, is_train=False)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, drop_last=True)
    valid_loader = DataLoader(valid_ds, batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers)
    print(f"Train batches: {len(train_loader)}  |  Valid batches: {len(valid_loader)}")

    # -------------------------------------------------------------------------
    # Model
    # -------------------------------------------------------------------------
    d_clin = train_ds[0]["clin"].shape[0]  # 17

    wave_enc = ResNet1DEncoder(in_channels=1, base_filters=32, out_dim=128).to(device)
    clin_enc = ClinicalMLP(in_dim=d_clin).to(device)
    text_enc = BioBERTTextEncoder(out_dim=128, freeze_bert=args.freeze_bert).to(device)
    proj_w   = Projection(128).to(device)
    proj_c   = Projection(128).to(device)

    optimizer = torch.optim.Adam(
        list(wave_enc.parameters()) + list(clin_enc.parameters())
        + list(text_enc.parameters()) + list(proj_w.parameters()) + list(proj_c.parameters()),
        lr=args.lr,
    )

    # -------------------------------------------------------------------------
    # Training loop
    # -------------------------------------------------------------------------
    train_losses, val_losses = [], []
    best_val_loss = float("inf")
    ckpt_path = os.path.join(save_dir, "clip_biobert_hyh_xai.pth")

    print(f"\nStarting training ({args.epochs} epochs)...")
    for epoch in range(args.epochs):
        avg_train = train_one_epoch(
            train_loader, wave_enc, clin_enc, text_enc, proj_w, proj_c, optimizer, device
        )
        avg_val = validate(
            valid_loader, wave_enc, clin_enc, text_enc, proj_w, proj_c, device
        )
        train_losses.append(avg_train)
        val_losses.append(avg_val)

        print(f"Epoch [{epoch+1}/{args.epochs}]  "
              f"Train: {avg_train:.4f}  Val: {avg_val:.4f}")

        if avg_val < best_val_loss:
            best_val_loss = avg_val
            torch.save({
                "epoch":     epoch,
                "wave_enc":  wave_enc.state_dict(),
                "clin_enc":  clin_enc.state_dict(),
                "text_enc":  text_enc.state_dict(),
                "proj_w":    proj_w.state_dict(),
                "proj_c":    proj_c.state_dict(),
                "optimizer": optimizer.state_dict(),
                "loss":      best_val_loss,
            }, ckpt_path)
            print(f"  --> Best model saved (val loss: {best_val_loss:.4f})")

    print(f"\nTraining complete. Best val loss: {best_val_loss:.4f}")

    # -------------------------------------------------------------------------
    # Visualisation
    # -------------------------------------------------------------------------
    plot_loss_curves(
        train_losses, val_losses,
        save_path=os.path.join(save_dir, "clip_loss_curve.png"),
    )

    # Load best weights before t-SNE and KB construction
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    wave_enc.load_state_dict(ckpt["wave_enc"])
    clin_enc.load_state_dict(ckpt["clin_enc"])
    text_enc.load_state_dict(ckpt["text_enc"])
    proj_w.load_state_dict(ckpt["proj_w"])
    proj_c.load_state_dict(ckpt["proj_c"])

    plot_tsne(
        valid_loader, wave_enc, proj_w, text_enc, tokenizer, device,
        save_path=os.path.join(save_dir, "clip_tsne.png"),
    )

    # -------------------------------------------------------------------------
    # Build RAG knowledge base
    # -------------------------------------------------------------------------
    build_knowledge_base(
        dataloader=train_loader,
        wave_enc=wave_enc, clin_enc=clin_enc,
        proj_w=proj_w,     proj_c=proj_c,
        device=device,
        save_path=os.path.join(save_dir, "knowledge_base.pt"),
    )


if __name__ == "__main__":
    main()
