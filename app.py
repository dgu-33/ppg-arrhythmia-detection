import streamlit as st
import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
import sqlite3
import pickle
from datetime import date
import torch
import os
from collections import defaultdict
import google.generativeai as genai
import torch.nn as nn
import torch.nn.functional as F
import PIL.Image
import io
from dotenv import load_dotenv

load_dotenv()

# --- Settings ---
device = "cuda" if torch.cuda.is_available() else "cpu"

try:
    genai.configure(api_key=os.environ.get("GEMINI_API_KEY", ""))
except Exception as e:
    print(f"Gemini API error: {e}")

LABEL_DISPLAY_MAP = {
    'normal': 'Normal', 'af': 'AF', 'b': 'B', 't': 'T'
}


def create_full_clinical_info_text(patient_data):
    """ 
    Returns 13 clinical features as a formatted string.
    (age, bmi, opdur, preop_na, preop_bun, preop_cr, preop_k, intraop_eph, intraop_phe) + (sex, emop, preop_dm, preop_htn)
    """
    cont_parts = []
    for col in CONTINUOUS_COLS:
        try:
            val = patient_data[col]
            if col in ["bmi", "preop_na", "preop_bun", "preop_cr", "preop_k", "intraop_eph", "intraop_phe"]:
                cont_parts.append(f"{col.upper()}:{val:.1f}")
            else:
                cont_parts.append(f"{col.upper()}:{val}")
        except KeyError:
            cont_parts.append(f"{col.upper()}:N/A")

    bin_parts = []
    for col in list(BINARY_STR_COLS.keys()):
        try:
            bin_parts.append(f"{col.upper()}:{patient_data[col]}")
        except KeyError:
            bin_parts.append(f"{col.upper()}:N/A")

    all_parts = cont_parts + bin_parts
    return ", ".join(all_parts)


# --- Import model_architecture and Preprocessing ---
try:
    from model_architecture import (
        UNet1D_ResNet_Combined, preprocess_for_inference, find_mask_indices,
        TARGET_LENGTH, INV_CLASS_MAP,
        CONTINUOUS_COLS, BINARY_STR_COLS, 
        N_CLINICAL_FEATURES 
    )
except ImportError:
    st.error("Error: 'model_architecture.py' nonfound.")
    st.stop()
except Exception as e:
    st.error(f"model_architecture.py Import Error: {e}")
    st.stop()

# --- Import llm_utils ---
try:
    from llm_utils import (
        ClinicalMLP, Projection, transform_13_to_17, llm_model, MEDICAL_GUIDELINES 
    )
except ImportError:
    st.error("Error: 'llm_utils.py' nonfound.")
    st.stop()
except Exception as e:
    st.error(f"llm_utils.py Import Error: {e}")
    st.stop()


# --- Load knowledge base ---
KB_PATH = "knowledge_base.pt"
knowledge_base = None
if os.path.exists(KB_PATH):
    knowledge_base = torch.load(KB_PATH)
    knowledge_base["clinical_vectors"] = knowledge_base["clinical_vectors"].to(device)

# --- Load RAG model weights ---
rag_clin_enc = ClinicalMLP(in_dim=17).to(device)
rag_proj_c = Projection(in_dim=128).to(device)


CLIP_CKPT_PATH = "clip_biobert_hyh_xai.pth"
if os.path.exists(CLIP_CKPT_PATH):
    ckpt = torch.load(CLIP_CKPT_PATH, map_location=device, weights_only=False)
    rag_clin_enc.load_state_dict(ckpt['clin_enc'])
    rag_proj_c.load_state_dict(ckpt['proj_c'])
    rag_clin_enc.eval(); rag_proj_c.eval()


# --- Knowledge base patient stats lookup ---
def get_kb_patient_stats(target_caseid):
    if knowledge_base is None: return None
    kb_ids = knowledge_base["caseids"]
    if isinstance(kb_ids, torch.Tensor): kb_ids = kb_ids.tolist()

    indices = [i for i, x in enumerate(kb_ids) if x == target_caseid]
    if not indices: return None

    counts = {"Normal": 0, "AF": 0, "B": 0, "T": 0}
    kb_labels = knowledge_base["labels"]

    for idx in indices:
        lbl = kb_labels[idx]
        if isinstance(lbl, int): lbl = INV_CLASS_MAP[lbl]  # convert integer label to string
        display_lbl = LABEL_DISPLAY_MAP.get(lbl, lbl)       # map lowercase to display label
        counts[display_lbl] = counts.get(display_lbl, 0) + 1

    total = len(indices)
    ratios = {k: v/total*100 for k, v in counts.items()}

    # Find dominant arrhythmia class (excluding Normal)
    arrs = {k: v for k, v in counts.items() if k != "Normal"}
    if sum(arrs.values()) > 0:
        dom = max(arrs, key=arrs.get)
    else:
        dom = "Normal"
    return {"counts": counts, "ratios": ratios, "dominant": dom, "total": total}

# --- Similar patient retrieval ---
def find_similar_patients_with_stats(query_vec_13, k=3):
    if knowledge_base is None: return []
    with torch.no_grad():
        q_17 = transform_13_to_17(query_vec_13.to(device))
        q_emb = rag_proj_c(rag_clin_enc(q_17))
        db_embs = knowledge_base["clinical_vectors"]
        sim = torch.mm(F.normalize(q_emb, p=2, dim=1), F.normalize(db_embs, p=2, dim=1).T).squeeze(0)
        top_vals, top_idxs = torch.topk(sim, k=k*10)

    results = []
    seen = set()
    for i, idx in enumerate(top_idxs):
        idx = idx.item()
        caseid = knowledge_base["caseids"][idx]
        if isinstance(caseid, torch.Tensor): caseid = caseid.item()
        if caseid in seen: continue
        seen.add(caseid)

        stats = get_kb_patient_stats(caseid)
        if stats:
            results.append({"caseid": caseid, "similarity": top_vals[i].item(), "stats": stats})
        if len(results) >= k: break
    return results


def create_pure_waveform_plot_to_image(signal_resampled, patient_id, window_id):
    """Render the PPG waveform (no mask overlay) and return it as a PIL.Image in memory."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 3), dpi=120)
    indices = np.arange(len(signal_resampled))
    ax.plot(indices, signal_resampled, color='blue')
    ax.set_title(f"PPG Waveform | Patient {patient_id}, Window {window_id}")
    ax.set_xlabel("Sample Index"); ax.set_ylabel("Amplitude")
    ax.grid(True)
    plt.tight_layout()

    # Write to an in-memory buffer instead of disk
    buf = io.BytesIO()
    fig.savefig(buf, format='png')
    plt.close(fig)

    buf.seek(0)
    return PIL.Image.open(buf)




# --- UI and chart helpers ---
def create_arrhythmia_plot(signal_resampled, pred_mask, title):
    """Plot PPG waveform with anomaly mask overlay."""
    fig, ax = plt.subplots(figsize=(6, 3)) 
    ax.set_title(title, fontsize=12)
    ax.plot(signal_resampled, color='blue', label='Signal', zorder=2)
    y_min, y_max = ax.get_ylim()
    ax.fill_between(range(len(pred_mask)), y_min, y_max,
                     where=pred_mask > 0.5,
                     color='red', alpha=0.3, label='Mask (p>0.5)', zorder=1)
    ax.set_xlabel(f"Samples ({TARGET_LENGTH})"); ax.set_ylabel("Amplitude")
    ax.grid(True)
    plt.tight_layout(pad=0.5)
    return fig


# --- Database utilities ---
def setup_db_schema(conn):
    cursor = conn.cursor()
    
    clinical_cols_sql = []
    for col in CONTINUOUS_COLS: clinical_cols_sql.append(f"{col} REAL")
    for col in list(BINARY_STR_COLS.keys()): clinical_cols_sql.append(f"{col} TEXT")
    
    clinical_schema = ", ".join(clinical_cols_sql)
    
    cursor.execute(f"""
    CREATE TABLE IF NOT EXISTS patients (
        patient_id TEXT PRIMARY KEY,
        {clinical_schema},
        info_text TEXT 
    );
    """)
    
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS analysis_history (
        id INTEGER PRIMARY KEY AUTOINCREMENT, patient_id TEXT, date TEXT,
        file_name TEXT, prediction TEXT, signal BLOB, mask BLOB,
        llm_report_text TEXT,
        FOREIGN KEY (patient_id) REFERENCES patients (patient_id)
    );
                  
    """)

    try:
        cursor.execute("ALTER TABLE analysis_history ADD COLUMN llm_report_text TEXT")
    except sqlite3.OperationalError as e:
        if "duplicate column name" in str(e):
            pass 
        elif "no such column" in str(e):
            pass
        else:
            print(f"ALTER TABLE Error: {e}")

    conn.commit()


def save_analysis_history(db_conn, patient_id, date_str, file_name, prediction, signal, mask):
    """Save an analysis record (file name, prediction, raw signal, mask) to the DB."""
    try:
        cursor = db_conn.cursor()

        # Serialise signal and mask as BLOBs
        signal_blob = pickle.dumps(signal)
        mask_blob = pickle.dumps(mask)
        
        cursor.execute("""
            INSERT INTO analysis_history 
            (patient_id, date, file_name, prediction, signal, mask)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (patient_id, date_str, file_name, prediction, signal_blob, mask_blob))
        
        db_conn.commit()
    except Exception as e:
        st.error(f"Failed to save: {e}")
    
def delete_analysis_record(db_conn, patient_id, file_name):
    """Delete the analysis record for a specific patient and file."""
    try:
        cursor = db_conn.cursor()
        cursor.execute("DELETE FROM analysis_history WHERE patient_id = ? AND file_name = ?", (patient_id, file_name))
        db_conn.commit()
        return True
    except Exception as e:
        st.error(f"Error deleting record: {e}")
        return False

def delete_user_account(db_conn, patient_id):
    """Delete a patient account along with all associated analysis records."""
    try:
        cursor = db_conn.cursor()
        cursor.execute("DELETE FROM analysis_history WHERE patient_id = ?", (patient_id,))
        cursor.execute("DELETE FROM patients WHERE patient_id = ?", (patient_id,))
        db_conn.commit()
        return True
    except Exception as e:
        st.error(f"Error deleting account: {e}")
        return False
    

@st.cache_resource
def init_sqlite_db():
    DB_FILE_NAME = "ppg_app.db"
    try:
        conn = sqlite3.connect(DB_FILE_NAME, check_same_thread=False)
        setup_db_schema(conn)
        return conn
    except Exception as e:
        st.error(f"DB connection failed: {e}"); st.stop()


@st.cache_resource
def load_unet_model():
    model_path = "best_combined_model.pth" 
    try:
        model = UNet1D_ResNet_Combined(
            n_channels=1, 
            n_classes=4, 
            n_hrv_features=4, 
            n_clinical_features=N_CLINICAL_FEATURES
        )
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu'), weights_only=True))
        model.eval()
        
        return model
    except FileNotFoundError:
        st.error(f"Model file ({model_path}) nonfound.")
        return None
    except Exception as e:
        st.error(f"Model file Import error: {e}")
        return None
 

# --- Segment naming utility ---
def get_segment_base_name(patient_id, file_name, idx):
    """Derive a unique base name for a segment from its file name."""
    file_parts = file_name.replace('.npz', '').replace('.npy', '').split('_')
    patient_id_file = file_parts[0] if file_parts[0].isdigit() else patient_id
    window_id = file_parts[1].replace('win', '') if len(file_parts) > 1 else str(idx + 1)
    return f"{patient_id_file}_{window_id}"


# --- Page renderers: registration / login / data management ---

def show_registration_page(db_conn):
    CLINICAL_COLS_FULL = ['patient_id'] + CONTINUOUS_COLS + list(BINARY_STR_COLS.keys()) + ['info_text']
    insert_cols = ['patient_id'] + CONTINUOUS_COLS + list(BINARY_STR_COLS.keys()) + ['info_text']

    st.subheader("User Registration (13 Clinical Features)")
    
    with st.form(key="registration_form"):
        st.markdown("##### Required Clinical Information")
        
        col_bin = st.columns(4)
        sex = col_bin[0].radio("Sex", BINARY_STR_COLS['sex'], index=0, horizontal=True)
        emop = col_bin[1].radio("Emergency Operation (emop)", BINARY_STR_COLS['emop'], index=0, horizontal=True)
        preop_dm = col_bin[2].radio("Diabetes (preop_dm)", BINARY_STR_COLS['preop_dm'], index=0, horizontal=True)
        preop_htn = col_bin[3].radio("Hypertension (preop_htn)", BINARY_STR_COLS['preop_htn'], index=0, horizontal=True)
        
        
        data = {}
        
        col_cont1 = st.columns(3)
        data['patient_id'] = col_cont1[0].text_input("User ID", key="reg_pid")
        data['age'] = col_cont1[1].number_input("Age", 0, 120, value=50, key="reg_age")
        data['bmi'] = col_cont1[2].number_input("BMI", min_value=10.0, max_value=50.0, value=25.0, step=0.1, key="reg_bmi")
        
        col_cont2 = st.columns(3)
        data['opdur'] = col_cont2[0].number_input("Operation Duration (opdur)", min_value=0, max_value=500, value=120, key="reg_opdur")
        data['preop_na'] = col_cont2[1].number_input("Preop Na (preop_na)", min_value=120.0, max_value=160.0, value=140.0, step=0.1, key="reg_na")
        data['preop_k'] = col_cont2[2].number_input("Preop K (preop_k)", min_value=3.0, max_value=6.0, value=4.0, step=0.1, key="reg_k")

        col_cont3 = st.columns(3)
        data['preop_bun'] = col_cont3[0].number_input("Preop BUN (preop_bun)", value=15.0, key="reg_bun")
        data['preop_cr'] = col_cont3[1].number_input("Preop Cr (preop_cr)", value=0.8, key="reg_cr")
        data['intraop_eph'] = col_cont3[2].number_input("Ephedrine (intraop_eph)", value=0.0, key="reg_eph")
        
        data['intraop_phe'] = st.number_input("Phenylephrine (intraop_phe)", value=0.0, key="reg_phe")
        
        final_data = {
        'patient_id': data['patient_id'], 
        'sex': sex, 'emop': emop, 'preop_dm': preop_dm, 'preop_htn': preop_htn,
        'age': data['age'], 'bmi': data['bmi'], 'opdur': data['opdur'],
        'preop_na': data['preop_na'], 'preop_bun': data['preop_bun'], 
        'preop_cr': data['preop_cr'], 'preop_k': data['preop_k'],
        'intraop_eph': data['intraop_eph'], 'intraop_phe': data['intraop_phe'],
        # Stored in the DB 'info_text' column for sidebar display
        'info_text': create_full_clinical_info_text({
            'age': data['age'], 'sex': sex, 'bmi': data['bmi'], 'preop_htn': preop_htn,
            'emop': emop, 'preop_dm': preop_dm, 'opdur': data['opdur'],
            'preop_na': data['preop_na'], 'preop_bun': data['preop_bun'], 'preop_cr': data['preop_cr'], 'preop_k': data['preop_k'],
            'intraop_eph': data['intraop_eph'], 'intraop_phe': data['intraop_phe'],
            'patient_id': data['patient_id']
        })
    }

        if st.form_submit_button("Register & Log In", type="primary"):
            if not final_data['patient_id']: st.error("User ID is required."); st.stop()
            
            try:
                cur = db_conn.cursor()
                cur.execute("SELECT 1 FROM patients WHERE patient_id=?", (final_data['patient_id'],))
                if cur.fetchone(): st.error("ID already exists. Please use the login page."); st.stop()
                else:
                    insert_values = [final_data['patient_id']]
                    for col in CONTINUOUS_COLS: insert_values.append(final_data[col])
                    for col in list(BINARY_STR_COLS.keys()): insert_values.append(final_data[col])
                    insert_values.append(final_data['info_text'])
                    
                    placeholders = ', '.join(['?'] * len(insert_values))
                    col_names = ', '.join(insert_cols)
                    
                    cur.execute(f"INSERT INTO patients ({col_names}) VALUES ({placeholders})", tuple(insert_values))
                    db_conn.commit()
                    
                    st.success("Registration completed. Logging in..."); time.sleep(1); 
                    
                    st.session_state.logged_in = True
                    st.session_state.patient_id = final_data['patient_id']
                    st.session_state.patient_info = final_data 
                    st.session_state.patient_info_text = final_data['info_text']
                    
                    st.query_params['pid'] = final_data['patient_id']; st.rerun() 
            except Exception as e: st.error(f"DB Error: {e}")
            
    st.markdown("---")
    if st.button("Go to Log In page"):
            st.query_params["page"] = "login"
            st.rerun()


def show_login_page(db_conn):
    CLINICAL_COLS_FULL = ['patient_id'] + CONTINUOUS_COLS + list(BINARY_STR_COLS.keys()) + ['info_text']
    st.subheader("User Login")
    pid_input = st.text_input("User ID", key="login_pid")
    if st.button("Log In", type="primary"):
        col_names = ', '.join(CLINICAL_COLS_FULL)
        
        try:
            row = db_conn.execute(f"SELECT {col_names} FROM patients WHERE patient_id=?", (pid_input,)).fetchone()
        except sqlite3.OperationalError as e:
            st.error(f"DB Error: {e}. The DB schema may conflict with an older version. Please contact the administrator.")
            return
        
        if row:
            patient_data = dict(zip(CLINICAL_COLS_FULL, row))
            
            info_text_full = create_full_clinical_info_text(patient_data)
            
            st.query_params['pid'] = pid_input 
            st.session_state.logged_in = True
            st.session_state.patient_id = pid_input
            st.session_state.patient_info = patient_data 
            st.session_state.patient_info_text = info_text_full
            st.rerun()
        else: st.error("No registered patient found with this ID. Please use the registration page.")
    st.markdown("---")
    if st.button("Register"):
            st.query_params["page"] = "register"
            st.rerun()


def show_data_management_page(db_conn):
    st.subheader("Database Management")
    st.error("Warning: This action cannot be UNDONE")
    
    st.markdown("##### Delete All Patient Data and Analysis Records")
    st.info("This will permanently delete all patient registration information "
        "from the 'patients' table and all analysis records from the 'analysis_history' table. "
        "**Proceed with caution.**")
    
    confirm_expander = st.expander("Click to proceed with deletion", expanded=False)
    with confirm_expander:
        st.warning("**Are you sure you want to delete all data?**")
        
        col1, col2 = st.columns([1, 3])
        if col1.button("Confirm & Delete All", type="primary", key="delete_all_confirm_btn"):
            try:
                db_conn.execute("DELETE FROM analysis_history")
                db_conn.execute("DELETE FROM patients")
                db_conn.commit()
                
                st.session_state.logged_in = False
                if 'pid' in st.query_params: del st.query_params['pid']
                
                st.success("All patient data and analysis records have been successfully deleted.")
                st.info("Redirecting to the login page in 3 seconds...")
                time.sleep(3)
                st.rerun()
                
            except Exception as e:
                st.error(f"Error occurred while deleting data: {e}")


def update_llm_report_in_db(db_conn, patient_id, file_name, llm_report_content):
    """Persist the generated LLM report text into the analysis_history table."""
    cursor = db_conn.cursor()
    cursor.execute("""
        UPDATE analysis_history 
        SET llm_report_text = ? 
        WHERE patient_id = ? AND file_name = ?
    """, (llm_report_content, patient_id, file_name))
    db_conn.commit()


# --- Tab 2: Detailed arrhythmia report archive with delete support ---
def show_past_reports_page(db_conn, patient_id):
    st.subheader(f"Detailed Arrhythmia Reports for Patient {patient_id}")
    
    try:
        # Fetch only the latest analysis record per file (de-duplicate re-runs)
        query = """
            WITH RankedHistory AS (
                SELECT
                    date, file_name, prediction, signal, mask, llm_report_text,
                    ROW_NUMBER() OVER (PARTITION BY file_name ORDER BY id DESC) as rn
                FROM analysis_history
                WHERE patient_id=?
                    AND llm_report_text IS NOT NULL
                    AND llm_report_text != ''
            )
            SELECT date, file_name, prediction, signal, mask, llm_report_text
            FROM RankedHistory
            WHERE rn = 1
            ORDER BY date DESC, file_name DESC
        """
        history_df = pd.read_sql_query(query, db_conn, params=(patient_id,))
    except Exception as e:
        st.error(f"Error loading past records: {e}")
        return

    st.markdown("### Waveforms Requested for AI Report Analysis")

    if history_df.empty:
        st.info("No arrhythmia segments with completed LLM analysis were found.")
        return
    
    for i, row in history_df.iterrows():
        file_name = row['file_name']
        pred_type = row['prediction']
        llm_report_content = row['llm_report_text']

        col_header, col_btn = st.columns([0.85, 0.15])

        with col_header:
            st.markdown(f"#### Segment: `{file_name}` ({pred_type.upper()})")

        with col_btn:
            if st.button("Delete", key=f"del_btn_{file_name}_{i}", type="secondary"):
                if delete_analysis_record(db_conn, patient_id, file_name):
                    st.toast(f"Record '{file_name}' deleted successfully!")
                    time.sleep(1)
                    st.rerun()

        try:
            # Deserialise BLOB fields
            signal = pickle.loads(row['signal'])
            mask = pickle.loads(row['mask'])
            
            fig = create_arrhythmia_plot(signal, mask, f"{file_name} | Pred: {pred_type.upper()}")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.pyplot(fig)
                plt.close(fig)

            with col2:
                mask_info_text = find_mask_indices(mask) 
                st.markdown(f"**Mask Indices:** `{mask_info_text}`")
                
                st.markdown("##### LLM Detailed View")
                if llm_report_content:
                    st.expander(f"Read Full Report").markdown(llm_report_content)
                else:
                    st.error("Report text not found.")
            
            st.markdown("---") 
            
        except Exception as e:
            st.warning(f"Visualization error ({file_name}): {e}")
                    

def show_main_app(db_conn, unet_model):
    patient_id = st.session_state.patient_id
    patient_data = st.session_state.patient_info
    
    st.sidebar.success(f"User ID: **{patient_id}**")
    st.sidebar.info(f"Clinical Information: {st.session_state.patient_info_text}")
    if st.sidebar.button("Log Out"):
        
        try:
            cursor = db_conn.cursor()
            # Clear cached LLM reports for this patient so they are regenerated fresh on next login
            cursor.execute("""
                UPDATE analysis_history
                SET llm_report_text = NULL
                WHERE patient_id = ?
            """, (patient_id,))
            db_conn.commit()
            st.sidebar.success("LLM assessment for the current patient has been reset.")
        except Exception as e:
            st.sidebar.error(f"Error resetting LLM report: {e}")
        keys_to_clear = [
            'analysis_results', 'files_to_process', 'process_stage', 
            'all_segments_data', 'llm_review_segments', 'current_segment_idx',
            'llm_analysis_log', 'db_saved'
        ]
        for key in keys_to_clear:
            if key in st.session_state:
                del st.session_state[key]

        st.session_state.logged_in = False
        if 'pid' in st.query_params: del st.query_params['pid']
        st.rerun()

    st.sidebar.markdown("<br>", unsafe_allow_html=True)

    with st.sidebar.expander("Account Settings"):
        st.warning("**Danger Zone**")
        st.markdown("Deleting your account will permanently remove all personal data and analysis history.")

        confirm_delete = st.checkbox("I understand the consequences.")
        
        if st.button("Delete Account", type="primary", disabled=not confirm_delete):
            if delete_user_account(db_conn, patient_id):
                st.session_state.logged_in = False
                if 'pid' in st.query_params: del st.query_params['pid']
                
                st.success("Account deleted successfully.")
                time.sleep(2)
                st.rerun()
                

    tab1, tab2 = st.tabs(["1. Summary Report", "2. LLM Report Archive"])
    
    with tab1:
        st.subheader("Arrhythmia Summary Report")
        
        # Initialise session state on first load
        if 'files_to_process' not in st.session_state:
            st.session_state.files_to_process = None
            st.session_state.current_segment_idx = 0
            st.session_state.analysis_results = defaultdict(int)
            st.session_state.llm_analysis_log = {}
            st.session_state.llm_review_segments = []
            st.session_state.llm_review_idx = 0
            st.session_state.current_llm_cache_file = None
            st.session_state.current_seg_data = None
            st.session_state.total_files = 0
            st.session_state.process_stage = 'UPLOAD'  # stages: UPLOAD → SCANNING → MODEL_SUMMARY → LLM_REVIEW → FINAL_REPORT
            st.session_state.all_segments_data = []
        if st.session_state.process_stage == 'UPLOAD':
            st.info("Upload your PPG files to begin sequential analysis.")
            m_date = st.date_input("Measurement Date", date.today())
            uploaded_files = st.file_uploader("Upload PPG files (.npy, .npz)", accept_multiple_files=True)
            
            if uploaded_files and st.button("Analysis ... ", type="primary"):
                st.session_state.files_to_process = uploaded_files
                st.session_state.total_files = len(uploaded_files)
                st.session_state.measurement_date = m_date.strftime('%Y-%m-%d')
                
                st.session_state.analysis_results = defaultdict(int) 
                st.session_state.llm_review_segments = []
                st.session_state.current_segment_idx = 0 
                st.session_state.all_segments_data = [] 
                st.session_state.process_stage = 'SCANNING'
                if 'db_saved' in st.session_state: del st.session_state.db_saved
                st.rerun()
            st.markdown("---")
            return

        # --- SCANNING: run model inference on all uploaded files ---
        files = st.session_state.files_to_process
        idx = st.session_state.current_segment_idx

        if st.session_state.process_stage == 'SCANNING':
            st.subheader("Inferencing Model...")
            progress_bar = st.progress(idx / st.session_state.total_files, text=f"Arrhythmia Classification: {idx}/{st.session_state.total_files}")
            
            while idx < st.session_state.total_files:
                current_file = files[idx]
                try:
                    # Load file and run model inference
                    dat = np.load(current_file, allow_pickle=True)
                    sig_raw = dat['signal'] if current_file.name.endswith('.npz') else dat
                    sig_raw = sig_raw.astype(np.float32)

                    model_input, hrv_input, clinical_input, plot_sig = preprocess_for_inference(sig_raw, patient_data)
                    
                    with torch.no_grad():
                        clf, seg = unet_model(model_input, clinical_input, hrv_input)
                    
                    # Classification prediction and mask post-processing
                    clf_pred_index = clf.argmax(1).item()
                    pred = INV_CLASS_MAP.get(clf_pred_index, 'normal')

                    mask_tensor = torch.sigmoid(seg).squeeze()

                    # Zero out the segmentation mask when the prediction is Normal
                    if clf_pred_index == 0:
                        mask_tensor = torch.zeros_like(mask_tensor)

                    mask = mask_tensor.cpu().numpy()
                    confidence = torch.softmax(clf, dim=1).max().item()

                    # Store all segment data for DB saving later
                    st.session_state.all_segments_data.append({
                        'file_name': current_file.name,
                        'pred': pred,
                        'sig_raw': sig_raw,
                        'mask': mask,
                        'plot_sig': plot_sig,
                    })

                    # Update result counters
                    st.session_state.analysis_results[pred] += 1

                    # Queue segment for LLM review
                    st.session_state.llm_review_segments.append({
                            'idx': idx,
                            'file_name': current_file.name,
                            'pred': pred,
                            'confidence': confidence,
                            'mask': mask,
                            'plot_sig': plot_sig,
                            'file': current_file 
                        })
                    
                    idx += 1
                    st.session_state.current_segment_idx = idx

                    # Refresh progress bar every 10 files
                    if idx % 10 == 0 or idx == st.session_state.total_files:
                        progress_bar.progress(idx / st.session_state.total_files, text=f"Arrhythmia Classification: {idx}/{st.session_state.total_files}")
                        time.sleep(0.1)
                        
                except Exception as e:
                    st.error(f"`{current_file.name}` SCANNING Error: {e}")
                    st.session_state.current_segment_idx = st.session_state.total_files  # abort scan on error
                    break
            
            progress_bar.empty()
            st.session_state.process_stage = 'MODEL_SUMMARY'
            st.rerun()
        


        # --- MODEL_SUMMARY: display results table and save to DB ---
        elif st.session_state.process_stage == 'MODEL_SUMMARY':
            
            if 'db_saved' not in st.session_state or st.session_state.db_saved == False:
                with st.spinner("Saving classification results and masks into the database..."):
                    try:
                        date_str = st.session_state.measurement_date
                        all_segments = st.session_state.all_segments_data
                        
                        for seg_data in all_segments:
                            save_analysis_history(
                                db_conn, patient_id, date_str, 
                                seg_data['file_name'], seg_data['pred'], 
                                seg_data['sig_raw'], seg_data['mask']
                            )
                        st.session_state.db_saved = True
                    except Exception as e:
                        st.error(f"Critical error while saving records to DB: {e}")
                        st.session_state.db_saved = False
            
            st.markdown("---")
            
            st.subheader("1. Overall Analysis Summary & Management Guideline")
            
            # Summary statistics (computed during the SCANNING stage)
            stats = st.session_state.analysis_results
            arrs = {LABEL_DISPLAY_MAP[k]: v for k, v in stats.items() if k != 'normal'}
            dominant = max(arrs, key=arrs.get) if sum(arrs.values()) > 0 else "Normal"
            guide = MEDICAL_GUIDELINES.get(dominant, "No Information.")
            
            st.markdown(f"""
            <div style='line-height:1.4; border: 2px solid #A9D0F5; padding: 10px; margin-bottom: 15px;'>
                <b>Patient {patient_id} Summary (Primary arrhythmia: <span style='color:red'>{dominant}</span>)</b>
                <ul><li>Distribution: Normal({stats['normal']}), AF({stats['af']}), B({stats['b']}), T({stats['t']})</li></ul>
                <summary><b>{dominant} Management Guideline (RAG): <pre style='background:#f4f4f4; padding:5px; white-space: pre-wrap;'>{guide}</pre></summary>
            </div>
            """, unsafe_allow_html=True)

            non_normal_count = len(st.session_state.llm_review_segments)

            non_normal_review_needed = len(st.session_state.llm_review_segments) > 0
                
            if non_normal_review_needed:
                if st.button("Start LLM Review", type="primary"):
                    st.session_state.llm_review_idx = 0
                    st.session_state.llm_state = 'ask_llm' 
                    st.session_state.process_stage = 'LLM_REVIEW'
                    st.rerun()
            else:
                st.success("No abnormal segments detected. Proceeding to final report generation.")
                st.session_state.process_stage = 'FINAL_REPORT'
                st.rerun()

            st.markdown("---")
            st.subheader("2. PPG Segment Visualization")
            
            st.warning(f"{non_normal_count} abnormal segments detected. Review all segments using visual inspection. ")
        
            st.info("Check each tab to review waveform segments classified by the model.")
            
            all_segments = st.session_state.all_segments_data 
        
            grouped_segments = defaultdict(list)
            for seg in all_segments:
                grouped_segments[seg['pred']].append(seg)

            all_types = list(INV_CLASS_MAP.values())  # ['normal', 'af', 'b', 't']

            if all_types:
                tabs = st.tabs([t.upper() for t in all_types])
                
                for tab_idx, pred_type in enumerate(all_types):
                    with tabs[tab_idx]:
                        segments_list = grouped_segments.get(pred_type, [])
                        st.markdown(f"#### {pred_type.upper()} ({len(segments_list)})")
                        
                        if not segments_list:
                            st.info(f"No segments classified as '{pred_type.upper()}'")
                            continue

                        cols = st.columns(2)
                        for i, data in enumerate(segments_list):
                            fig = create_arrhythmia_plot(
                                data['plot_sig'],
                                data['mask'], 
                                f"{data['file_name']} | Pred: {data['pred'].upper()}"
                            )
                            
                            with cols[i % 2]:
                                st.pyplot(fig)
                                plt.close(fig)

                                mask_info_text = find_mask_indices(data['mask']) 
                                
                                st.markdown(f"**Segment:** `{data['file_name']}`")
                                st.markdown(f"**Mask Indices:** `{mask_info_text}`")
                                st.markdown("---")
                                            
            else:
                st.error("No classified segments found")
                st.session_state.process_stage = 'FINAL_REPORT'
                st.rerun()
            
            return
        
            
        # --- LLM_REVIEW: interactive per-segment analysis ---
        elif st.session_state.process_stage == 'LLM_REVIEW':
            
            review_list = st.session_state.llm_review_segments
            review_idx = st.session_state.llm_review_idx
            
            if review_idx >= len(review_list):
                st.success("LLM Review Success!")
                st.session_state.process_stage = 'FINAL_REPORT'
                st.rerun()
                return

            current_seg_data = review_list[review_idx]
            current_file = current_seg_data['file']
            
            
            st.markdown(f"### **LLM Review Segment: {review_idx + 1}/{len(review_list)}** (`{current_file.name}`)")
            
            # State 1: ask whether to run LLM analysis
            if st.session_state.llm_state == 'ask_llm':
                data = current_seg_data
                
                st.error(f"**Abnormal Rhythm Detected!** Model Prediction: **{data['pred'].upper()}** (Confidence: {data['confidence']:.4f})")
                
                col1, col2 = st.columns(2)
                with col1:
                    st.subheader("Waveform Visualization")
                    fig = create_arrhythmia_plot(data['plot_sig'], data['mask'], f"{current_file.name} - {data['pred'].upper()}")
                    st.pyplot(fig); plt.close(fig)
                
                with col2:
                    st.subheader("Request LLM Report")
                    st.markdown("Ask LLM for a deeper clinical opinion to improve reliability.")
                    st.warning("Proceed with LLM analysis? \nThis may take 5~10 seconds.))")

                    if st.button("Start LLM Review", type="primary"):
                        st.session_state.llm_state = 'rag_analysis'

                        try:
                            window_id_base = get_segment_base_name(patient_id, current_file.name, review_idx)

                            # Generate waveform image in memory for Gemini vision analysis
                            image_obj = create_pure_waveform_plot_to_image(
                                data['plot_sig'],
                                patient_id,
                                window_id_base
                            )
                            st.session_state.current_img_buffer = image_obj
                            st.session_state.llm_state = 'rag_analysis'
                            st.rerun()

                        except Exception as e:
                            st.error(f"Image Error: {e}")

                        st.rerun()
            
                    if st.button("No, skip to next segment"):
                        st.session_state.llm_state = 'next_review'
                        st.rerun()

            # State 2: RAG-based clinical and similar patient analysis
            elif st.session_state.llm_state == 'rag_analysis':
                st.subheader("1. RAG-Based Clinical Report")
                
                with st.spinner("Analyzing similar patient data (RAG)..."):

                    # Build 13-dim clinical feature vector:
                    # 9 continuous + 4 binary ('M'/'F', 'N'/'Y' → 1.0/0.0)
                    clin_13_values = []
                    for col in CONTINUOUS_COLS:
                        clin_13_values.append(float(st.session_state.patient_info[col]))

                    for col in list(BINARY_STR_COLS.keys()):
                        col_val = st.session_state.patient_info[col]
                        # First category in each list maps to 1.0
                        encoded_val = 1.0 if col_val == BINARY_STR_COLS[col][0] else 0.0
                        clin_13_values.append(encoded_val)

                    clin_13_tensor = torch.tensor(clin_13_values).float()

                    current_cache = {
                    'patient_id': patient_id,
                    'pred_class': current_seg_data['pred'].upper(),
                    'pred_conf': current_seg_data['confidence'] * 100,
                    'mask_detected': np.sum(current_seg_data['mask']) > 0,
                    'clin_vec': transform_13_to_17(clin_13_tensor.unsqueeze(0))[0],
                    }
                
                    try:
                        sims = find_similar_patients_with_stats(current_cache['clin_vec'].unsqueeze(0), k=3)
                        evidence = ""
                        if sims:
                            for i, s in enumerate(sims):
                                dom = s['stats']['dominant']
                                ratio = s['stats']['ratios'][dom]
                                evidence += f"- [ID:{s['caseid']}] ({s['similarity']:.2f}): **{dom}** ({ratio:.1f}%)\n"
                        else: evidence = "No Similar Patient Case"

                        prompt = f"""

                        You are a board-certified cardiologist at a university hospital. 
                        Write a medical note based on the information below.
                        Follow the markdown structure in [Output Format] exactly and avoid unnecessary introductions.

                        [Patient Data]
                        - Patient ID: : {current_cache['patient_id']}
                        - AI Diagnosis: {current_cache['pred_class']} (confidence {current_cache['pred_conf']:.1f}%)
                        - Signal Anomaly: {'Detected' if current_cache['mask_detected'] else 'None'}
                        - Similar Patient Statistics (Evidence): \n{evidence}

                        [Output Format]
                        ### 1. Diagnostic Summary
                        - **AI Prediction:** (write result here)
                        - **Interpretation:** (interpret based on confidence score and presence/absence of waveform anomaly)

                        ### 2. Evidence Analysis
                        - **Similar Patient Analysis:** (cite statistics from similar patients. Example: “Among the most similar cases, XX% had the same diagnosis…”)
                        - **Clinical Impression:** (clinical reasoning based on provided data)

                        ### 3. Conclusion & Recommendation
                        - (one-sentence final impression or recommendation)
                        """

                        rag_resp = llm_model.generate_content(prompt, generation_config=genai.types.GenerationConfig(temperature=0.1))
                        st.markdown(rag_resp.text)
                        st.session_state.rag_report_text = rag_resp.text  # stored for use in the vision analysis stage

                    except Exception as e: 
                        st.error(f"RAG Analysis Error: {e}")
                        st.session_state.rag_report_text = f"RAG Analysis Error: {e}"

                st.markdown("---")
                if st.button("Pure Waveform Analysis", type="primary"):
                    st.session_state.llm_state = 'vision_analysis'
                    st.rerun()
        
            # State 3: Gemini vision analysis of the pure waveform image
            elif st.session_state.llm_state == 'vision_analysis':
                st.subheader("2. Pure Waveform Analysis Report")
        
                with st.spinner("Analyzing Pure Waveform Image..."):
                    
                    img = st.session_state.current_img_buffer
                    
                    try:
                        prompt = """
                            You are an expert in PPG waveform morphology analysis. You are analyzing this image purely based on visual patterns without raw data.

                            **OBJECTIVE:** Classify the waveform into one of four categories: **Normal, Tachycardia, Bradycardia, or Atrial Fibrillation.**

                            **CRITICAL INSTRUCTION:** Do not be misled by a single short interval. Look at the **overall density** of peaks across the entire visible window.

                            Please analyze the image following this step-by-step visual inspection:

                            ### 1. Visual Density & Sparsity Check (Rate Estimation)
                            - **Count the Peaks:** How many distinct systolic peaks are visible in the entire window?
                            - **Gap vs. Pulse Ratio:** Compare the width of the "hump" (the wave itself) to the width of the "flat gap" between waves.
                                - **Visual Check:** Is the flat gap significantly *wider/longer* than the wave itself?
                                - [ ] **Sparse (Bradycardia):** Few peaks, wide gaps between them. The "flat" line dominates the image.
                                - [ ] **Crowded (Tachycardia):** Many peaks packed closely. The waves are touching or very close.
                                - [ ] **Moderate (Normal):** Balanced spacing.

                            ### 2. Rhythm Regularity (X-axis Spacing)
                            - Look at the horizontal distance between consecutive peak tips.
                            - **Question:** Is the rhythm chaotic or generally stable?
                                - *Note: A slow rhythm with one missed beat or pause is still fundamentally "Bradycardia/Slow", not necessarily AFib.*
                                - [ ] **Regular:** Intervals are roughly constant.
                                - [ ] **Irregularly Irregular:** Intervals are completely random and chaotic (Characteristic of AFib).

                            ### 3. Classification Logic
                            Use this priority logic to determine the finding:

                            1.  **If patterns are chaotic & random:** $\rightarrow$ **Atrial Fibrillation**
                            2.  **If peaks are sparse (wide gaps) & count is low:** $\rightarrow$ **Bradycardia**
                            3.  **If peaks are crowded (narrow gaps) & count is high:** $\rightarrow$ **Tachycardia**
                            4.  **If neither sparse nor crowded + regular:** $\rightarrow$ **Normal**

                            ### 4. Final Conclusion
                            - **Observed Feature:** (e.g., "The waveform shows wide gaps between peaks, indicating a slow rate.")
                            - **Suspected Finding:** (Choose ONE: Normal / Tachycardia / Bradycardia / Atrial Fibrillation)
                            """

                        vision_resp = llm_model.generate_content([prompt, img], generation_config=genai.types.GenerationConfig(temperature=0.1))
                        
                        st.image(img, caption="Pure Waveform Image", use_container_width=True)
                        st.markdown(vision_resp.text)
                        
                        st.session_state.llm_analysis_log[f"{current_file.name}-Vision"] = {
                            'pred': current_seg_data['pred'],
                            'report_text': vision_resp.text
                        }

                        if st.button("Next segment", type="primary"):
                            st.session_state.llm_state = 'next_review'
                            st.rerun()

                    except Exception as e: 
                        st.error(f"Vision Analysis Error: {e}")
                        st.session_state.llm_analysis_log[f"{current_file.name}-Vision"] = {
                            'pred': current_seg_data['pred'], 
                            'report_text': f"Vision Analysis Error: {e}"
                        }

                # Combine RAG report and vision report into a single string
                final_llm_report = (
                    "### RAG-Based Clinical Report\n\n"
                    + st.session_state.rag_report_text + "\n\n\n"
                    + "### Pure Waveform Analysis Report\n\n"
                    + vision_resp.text
                )
                
                # Save combined report to DB
                try:
                    update_llm_report_in_db(
                        db_conn,
                        patient_id,
                        current_file.name,
                        final_llm_report
                    )
                    st.session_state.llm_analysis_log[current_file.name] = {
                        'pred': current_seg_data['pred'], 'report_text': final_llm_report
                    }
                except Exception as e:
                    st.error(f"LLM Report DB Save Error: {e}")
                     

            # Advance to the next segment
            elif st.session_state.llm_state == 'next_review':
                st.session_state.llm_review_idx += 1
                st.session_state.llm_state = 'ask_llm'

                if 'current_img_buffer' in st.session_state:
                    del st.session_state.current_img_buffer

                if 'rag_report_text' in st.session_state:
                    del st.session_state.rag_report_text

                st.rerun()
        
            else:
                st.error("Unknown analysis state. Please restart.")
                

        # --- FINAL_REPORT: summary display ---
        elif st.session_state.process_stage == 'FINAL_REPORT':
            st.success("Arrhythmia analysis and LLM review completed.")

            # DB saving was already handled in MODEL_SUMMARY; nothing to do here
            if 'db_saved' not in st.session_state or st.session_state.db_saved == False:
                pass
            
            st.subheader("Classification Summary")
            stats_df = pd.DataFrame(list(st.session_state.analysis_results.items()), columns=['Class', 'Count'])
            st.dataframe(stats_df, hide_index=True)
            
            
            if st.button("Start New Analysis", type="primary"):
                st.session_state.process_stage = 'UPLOAD'
                if 'files_to_process' in st.session_state: del st.session_state.files_to_process
                if 'all_segments_data' in st.session_state: del st.session_state.all_segments_data
                if 'db_saved' in st.session_state: del st.session_state.db_saved
                st.rerun()

        

    with tab2:
        show_past_reports_page(db_conn, patient_id)


# --- App entry point ---
st.set_page_config(layout="centered", page_title="H-AI")
st.title("H-AI : PPG Analysis")

# Restore login state from URL param and initialise DB / model
pid_from_url = st.query_params.get("pid")
db_conn = init_sqlite_db()

CLINICAL_COLS_FULL = ['patient_id'] + CONTINUOUS_COLS + list(BINARY_STR_COLS.keys()) + ['info_text']

if pid_from_url:
    col_names = ', '.join(CLINICAL_COLS_FULL)
    row = db_conn.execute(f"SELECT {col_names} FROM patients WHERE patient_id=?", (pid_from_url,)).fetchone()
    
    if row:
        patient_data = dict(zip(CLINICAL_COLS_FULL, row))

        info_text_full = create_full_clinical_info_text(patient_data)

        st.session_state.logged_in = True
        st.session_state.patient_id = pid_from_url
        st.session_state.patient_info = patient_data 
        st.session_state.patient_info_text = info_text_full
    else:
        st.session_state.logged_in = False

if 'logged_in' not in st.session_state: st.session_state.logged_in = False

unet_model = load_unet_model()


page = st.query_params.get("page")
if page == "register": show_registration_page(db_conn)
elif not st.session_state.logged_in or not pid_from_url: show_login_page(db_conn)
else:
    if unet_model: show_main_app(db_conn, unet_model)
    else: st.error("Model Loading Failure")