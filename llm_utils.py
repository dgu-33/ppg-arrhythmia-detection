import torch.nn as nn
import torch 
import torch.nn.functional as F
import google.generativeai as genai


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

def transform_13_to_17(vec_13):  # expand drop_first OHE (13-dim) to full OHE (17-dim)
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