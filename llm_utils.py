import os
import torch.nn as nn
import torch
import torch.nn.functional as F
import google.generativeai as genai
from model_architecture import INV_CLASS_MAP


LABEL_DISPLAY_MAP = {'normal': 'Normal', 'af': 'AF', 'b': 'B', 't': 'T'}


# --- Encoder classes for RAG retrieval ---

class ClinicalMLP(nn.Module):
    def __init__(self, in_dim, out_dim=128):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, 256), nn.ReLU(), nn.Linear(256, out_dim))
    def forward(self, x): return F.normalize(self.net(x), dim=-1)

class Projection(nn.Module):
    def __init__(self, in_dim, proj_dim=128):
        super().__init__()
        self.fc = nn.Linear(in_dim, proj_dim)
    def forward(self, x): return F.normalize(self.fc(x), dim=-1)

def transform_13_to_17(vec_13):  
    if vec_13.dim() == 1: vec_13 = vec_13.unsqueeze(0)
    cont = vec_13[:, :9]
    sex_m = vec_13[:, 9:10]; sex_f = 1.0 - sex_m
    emop_y = vec_13[:, 10:11]; emop_n = 1.0 - emop_y
    dm_y = vec_13[:, 11:12]; dm_n = 1.0 - dm_y
    htn_y = vec_13[:, 12:13]; htn_n = 1.0 - htn_y
    return torch.cat([cont, sex_m, sex_f, emop_n, emop_y, dm_n, dm_y, htn_n, htn_y], dim=1)



# --- LLM model setup ---
llm_model = genai.GenerativeModel('gemini-flash-latest')



# --- RAG guidelines (hardcoded knowledge base from ESC/ACC/AHA) ---
MEDICAL_GUIDELINES = {

    "AF": """
    
    [Reference: 2020 ESC Guidelines for Atrial Fibrillation (Adapted for PPG)]

    - **Diagnostic Criteria**: Suspicion arises if irregularity in pulse intervals on PPG persists for ≥ 30 seconds (ECG required for confirmation).

    - **Characteristics**: 'Irregularly Irregular' R-R intervals, absence of P-waves (ECG).

    - **Recommendations (Management)**:
      1. Opportunistic screening is recommended for patients aged ≥ 65 years.
      2. Consider oral anticoagulant (NOAC) administration for males with a CHA2DS2-VASc score ≥ 2.
      3. Consider beta-blockers for rate control.
    """,
    "T": """
    
    [Reference: ACC/AHA Guidelines for Tachycardia]

    - **Definition**: ventricular rate above 100 bpm with consistent morphology.

    - **Evaluation**: Differentiation between sinus tachycardia and arrhythmias via 12-lead ECG is mandatory.
    
    - **Management**:
      1. Correction of secondary causes (e.g., fever, anemia, anxiety, dehydration).
      2. Attempt vagal maneuvers for symptomatic Supraventricular Tachycardia (SVT).
    """,

    "B": """
    
    [Reference: ACC/AHA Guidelines for Bradycardia]

    - **Definition**: sustained ventricular rate below 60 bpm

    - **Management**:
      1. Observation if asymptomatic.
      2. Consider discontinuation of causative agents or administration of atropine/pacing if accompanied by symptoms (e.g., syncope, dizziness).
    """,
    
    "Normal": """

    [Reference: General Health Checkup Guidelines]

    - **Normal Findings**: Regular sinus rhythm, heart rate 60–100 bpm.

    - **Recommendations**: No specific findings. Maintain regular health monitoring.
    """
}


# --- RAG component loading ---

def load_rag_components(device):
    """Load knowledge base and RAG clinical encoders from disk.

    Returns:
        (knowledge_base, rag_clin_enc, rag_proj_c)
        knowledge_base is None if the file does not exist.
    """
    kb_path = "knowledge_base.pt"
    clip_ckpt_path = "clip_biobert_hyh_xai.pth"

    knowledge_base = None
    if os.path.exists(kb_path):
        knowledge_base = torch.load(kb_path, weights_only=False)
        knowledge_base["clinical_vectors"] = knowledge_base["clinical_vectors"].to(device)

    rag_clin_enc = ClinicalMLP(in_dim=17).to(device)
    rag_proj_c = Projection(in_dim=128).to(device)

    if os.path.exists(clip_ckpt_path):
        ckpt = torch.load(clip_ckpt_path, map_location=device, weights_only=False)
        rag_clin_enc.load_state_dict(ckpt['clin_enc'])
        rag_proj_c.load_state_dict(ckpt['proj_c'])
        rag_clin_enc.eval()
        rag_proj_c.eval()

    return knowledge_base, rag_clin_enc, rag_proj_c


# --- RAG retrieval functions ---

def get_kb_patient_stats(knowledge_base, target_caseid):
    """Return label distribution stats for a given caseid in the knowledge base."""
    if knowledge_base is None:
        return None
    kb_ids = knowledge_base["caseids"]
    if isinstance(kb_ids, torch.Tensor):
        kb_ids = kb_ids.tolist()

    indices = [i for i, x in enumerate(kb_ids) if x == target_caseid]
    if not indices:
        return None

    counts = {"Normal": 0, "AF": 0, "B": 0, "T": 0}
    kb_labels = knowledge_base["labels"]

    for idx in indices:
        lbl = kb_labels[idx]
        if isinstance(lbl, int):
            lbl = INV_CLASS_MAP[lbl]
        display_lbl = LABEL_DISPLAY_MAP.get(lbl, lbl)
        counts[display_lbl] = counts.get(display_lbl, 0) + 1

    total = len(indices)
    ratios = {k: v / total * 100 for k, v in counts.items()}

    arrs = {k: v for k, v in counts.items() if k != "Normal"}
    dom = max(arrs, key=arrs.get) if sum(arrs.values()) > 0 else "Normal"
    return {"counts": counts, "ratios": ratios, "dominant": dom, "total": total}


def find_similar_patients_with_stats(query_vec, knowledge_base, rag_clin_enc, rag_proj_c, device, k=3):
    """Retrieve the k most similar patients from the knowledge base by clinical embedding."""
    if knowledge_base is None:
        return []
    with torch.no_grad():
        q_17 = transform_13_to_17(query_vec.to(device))
        q_emb = rag_proj_c(rag_clin_enc(q_17))
        db_embs = knowledge_base["clinical_vectors"]
        sim = torch.mm(
            F.normalize(q_emb, p=2, dim=1),
            F.normalize(db_embs, p=2, dim=1).T
        ).squeeze(0)
        top_vals, top_idxs = torch.topk(sim, k=k * 10)

    results = []
    seen = set()
    for i, idx in enumerate(top_idxs):
        idx = idx.item()
        caseid = knowledge_base["caseids"][idx]
        if isinstance(caseid, torch.Tensor):
            caseid = caseid.item()
        if caseid in seen:
            continue
        seen.add(caseid)
        patient_stats = get_kb_patient_stats(knowledge_base, caseid)
        if patient_stats:
            results.append({"caseid": caseid, "similarity": top_vals[i].item(), "stats": patient_stats})
        if len(results) >= k:
            break
    return results