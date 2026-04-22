"""
fe_client.py — Python client that talks to the Hospital node (Port 8080).

HOW THE OFFSET WORKS:
  Damgård's discrete log solver requires ALL vector components to be
  strictly positive integers. Our Lasso weights are floats between roughly
  -1.0 and +1.0. We apply two transforms:
    1. Scale by SCALE_FACTOR (e.g. 10000) to convert floats → integers
    2. Add OFFSET to guarantee all values > 0

  Both Patient RNA and Weights are sent as int64 after this transform.

USAGE (Jupyter / plain Python):
  pip install numpy requests

  Then either:
    - Run this script directly (it uses demo vectors), OR
    - Import the send_to_hospital() function and pass your real arrays.
"""

import numpy as np
import pandas as pd
import requests
import json
import pickle
# ── Configuration ─────────────────────────────────────────────────────────────

HOSPITAL_URL = "http://localhost:8080/process"
SCALE_FACTOR = 1000   # Multiply floats by this to make them integers
OFFSET       = 2000  # Add this to guarantee every value > 0
# The 33 TCGA Pan-Cancer Types
CANCER_NAMES = [
    "ACC (Adrenocortical Carcinoma)", "BLCA (Bladder Urothelial Carcinoma)", 
    "BRCA (Breast Invasive Carcinoma)", "CESC (Cervical Squamous Cell)", 
    "CHOL (Cholangiocarcinoma)", "COAD (Colon Adenocarcinoma)", 
    "DLBC (Lymphoid Neoplasm)", "ESCA (Esophageal Carcinoma)", 
    "GBM (Glioblastoma Multiforme)", "HNSC (Head and Neck Squamous)", 
    "KICH (Kidney Chromophobe)", "KIRC (Kidney Renal Clear Cell)", 
    "KIRP (Kidney Renal Papillary)", "LAML (Acute Myeloid Leukemia)", 
    "LGG (Brain Lower Grade Glioma)", "LIHC (Liver Hepatocellular)", 
    "LUAD (Lung Adenocarcinoma)", "LUSC (Lung Squamous Cell)", 
    "MESO (Mesothelioma)", "OV (Ovarian Serous Cystadenocarcinoma)", 
    "PAAD (Pancreatic Adenocarcinoma)", "PCPG (Pheochromocytoma)", 
    "PRAD (Prostate Adenocarcinoma)", "READ (Rectum Adenocarcinoma)", 
    "SARC (Sarcoma)", "SKCM (Skin Cutaneous Melanoma)", 
    "STAD (Stomach Adenocarcinoma)", "TGCT (Testicular Germ Cell Tumors)", 
    "THCA (Thyroid Carcinoma)", "THYM (Thymoma)", 
    "UCEC (Uterine Corpus Endometrial)", "UCS (Uterine Carcinosarcoma)", 
    "UVM (Uveal Melanoma)"
]

# ── Core function ─────────────────────────────────────────────────────────────

def prepare_vector(float_array: np.ndarray) -> list[int]:
    """
    Convert a float numpy array to a list of strictly positive int64 values.

    Steps:
      1. Scale by SCALE_FACTOR  (e.g. 0.0659  → 659)
      2. Round to nearest int
      3. Add OFFSET              (e.g. 659 + 100000 = 100659)

    All values are now guaranteed > 0. The Hospital node validates this.
    """
    scaled = np.round(float_array * SCALE_FACTOR).astype(np.int64)
    positive = scaled + OFFSET
    assert np.all(positive > 0), "Offset failed — some values still ≤ 0!"
    return positive.tolist()


def send_to_hospital(patient_rna: np.ndarray, ml_weights: np.ndarray) -> str:
    """
    Send patient RNA + model weights to the Hospital node.

    Args:
        patient_rna:  1-D numpy array of RNA expression floats
                      (only the ACTIVE genes selected by Lasso)
        ml_weights:   1-D numpy array of Lasso weights for those same genes
                      (must be same length as patient_rna)

    Returns:
        Risk score as a decimal string (the inner product ⟨x, w⟩).
    """
    assert len(patient_rna) == len(ml_weights), (
        f"Length mismatch: patient_rna={len(patient_rna)}, "
        f"ml_weights={len(ml_weights)}"
    )

    payload = {
        "PatientRNA": prepare_vector(patient_rna),
        "MLWeights":  prepare_vector(ml_weights),
    }

    print(f"[Client] Sending vector of length {len(patient_rna)} to Hospital…")
    print(f"[Client] Example transformed value: "
          f"{patient_rna[0]:.4f} → {payload['PatientRNA'][0]}")

    try:
        resp = requests.post(HOSPITAL_URL, json=payload, timeout=120)
        resp.raise_for_status()
    except requests.exceptions.ConnectionError:
        raise RuntimeError(
            "Could not connect to Hospital node at localhost:8080. "
            "Is hospital_node running?"
        )

    result = resp.json()

    if result.get("Error"):
        raise RuntimeError(f"Hospital returned error: {result['Error']}")

    return result["RiskScore"]


# ── Demo: Load real weights from your saved .npy file ────────────────────────

def demo_with_saved_weights(weights_path: str, cancer_index: int = 0):
    """
    Load the sparse_weights.npy saved by the notebook and run a demo inference.

    weights_path:  path to sparse_weights.npy (or master_33_cancer_weights.npy)
    cancer_index:  which cancer row to use (0=first cancer in model.classes_)
    """
    weights_matrix = np.load(weights_path)
    w = weights_matrix[cancer_index]

    # Find the active (non-zero) gene indices
    active_idx = np.where(w != 0)[0]
    print(f"[Demo] Cancer index {cancer_index}: {len(active_idx)} active genes")

    if len(active_idx) == 0:
        print("[Demo] No active genes for this cancer index — try a different one.")
        return

    # Use the actual weights as the "patient RNA" for a sanity-check demo.
    # In production you would load a real patient's RNA values here.
    active_weights = w[active_idx]

    # Simulate patient RNA as abs(weights) * some random scale
    rng = np.random.default_rng(42)
    simulated_rna = np.abs(active_weights) * rng.uniform(0.8, 1.2, size=len(active_weights))

    risk_score = send_to_hospital(simulated_rna, active_weights)
    print(f"\n[Demo]  Risk score (inner product): {risk_score}")
    print("[Demo] Divide by (SCALE_FACTOR² × length) to get approximate Lasso score.")

def demo_real_patient(weights_path: str, dataset_path: str, patient_row: int, cancer_index: int = 0):
    """
    Loads a real patient from a massive Xena TSV/CSV, extracts only the 
    Lasso-selected genes, and securely computes their risk score.
    """
    # 1. Load the ML "Rules"
    weights_matrix = np.load(weights_path)
    w = weights_matrix[cancer_index]
    
    active_idx = np.where(w != 0)[0]
    print(f"[Demo] Cancer index {cancer_index}: Model requires {len(active_idx)} active genes.")
    
    if len(active_idx) == 0:
        print("[Demo] Error: No active genes found in this model.")
        return
    active_weights = w[active_idx]

   # 2. Load the Massive Xena Data safely
    print(f"[Client] Opening Massive Medical Database: {dataset_path}")
    print("[Client] (Please wait, loading Pan-Cancer genomic data into memory...)")
    
    try:
        df = pd.read_csv(dataset_path, sep='\t')
    except Exception as e:
        df = pd.read_csv(dataset_path)

    # ==========================================
    # FIX: THE XENA TRANSPOSE
    # Medical data often has Genes as Rows and Patients as Columns.
    # ML needs Patients as Rows and Genes as Columns.
    # ==========================================
    if 'sample' in df.columns:
        print("[Client] Transposing Xena matrix (swapping rows/columns to match ML)...")
        # Make the 'sample' column (gene names) the index, then flip the table
        df = df.set_index('sample').T

    # Clean up the labels so we don't accidentally encrypt the answer
    df_features = df.drop(columns=['target', 'label', 'cancer_type'], errors='ignore')
    
    # 3. Extract the Patient
    # Now iloc[patient_row] correctly grabs ONE patient's entire gene sequence!
    patient_data = df_features.iloc[patient_row].values
    print(f"[Client] Extracted Patient #{patient_row}. Total recorded genes: {len(patient_data)}")
    
    # 3. Extract the Patient
    patient_data = df_features.iloc[patient_row].values
    print(f"[Client] Extracted Patient #{patient_row}. Total recorded genes: {len(patient_data)}")

    # 4. Filter and Transmit
    patient_rna = patient_data[active_idx]
    
    print(f"[Client] Transmitting {len(patient_rna)} encrypted biomarkers to Hospital...")
    risk_score = send_to_hospital(patient_rna, active_weights)
    # --- MANUAL MATH VALIDATION ---
    python_rna_ints = prepare_vector(patient_rna)
    python_weight_ints = prepare_vector(active_weights)
    expected_score = np.dot(python_rna_ints, python_weight_ints)
    
    print(f"\n[Demo] Expected Score (Plaintext Math): {expected_score}")
    print(f"[Demo]  REAL PATIENT RISK SCORE (GoFE):  {risk_score}")
    # --- CONVERT TO HUMAN-READABLE PERCENTAGE ---
    # Calculate the true raw float score 
    raw_float_score = np.dot(patient_rna, active_weights)
    
    # Pass it through the Sigmoid function: 1 / (1 + e^-x)
    probability = 1 / (1 + np.exp(-raw_float_score))
    risk_percentage = probability * 100
    
    print("=======================================================")
    print(f"[Demo] CLINICAL DIAGNOSIS RISK: {risk_percentage:.2f}%")
    print("=======================================================")
# ── Quick smoke-test with tiny demo vectors ───────────────────────────────────
# ── Demo: Autonomous Pan-Cancer Auto-Detect ───────────────────────────────────



# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sysdef demo_auto_detect(weights_path: str, dataset_path: str, patient_row: int, model_pkl_path: str):
    """
    Scans a patient against all 33 cancer types, auto-detects the highest risk, 
    and sends the flagged profile to the Go Enclave for secure verification.
    """
    # 1. Hardcoded TCGA Pan-Cancer Labels (Bypassing the PKL file)
    print("[Client] Loading Clinical Labels...")
    cancer_labels = [
        "ACC", "BLCA", "BRCA", "CESC", "CHOL", "COAD", "DLBC", "ESCA", "GBM", 
        "HNSC", "KICH", "KIRC", "KIRP", "LAML", "LGG", "LIHC", "LUAD", "LUSC", 
        "MESO", "OV", "PAAD", "PCPG", "PRAD", "READ", "SARC", "SKCM", "STAD", 
        "TGCT", "THCA", "THYM", "UCEC", "UCS", "UVM"
    ]

    # 2. Load the Weights & Massive Xena Data
    weights_matrix = np.load(weights_path)
    
    print(f"[Client] Opening Massive Medical Database: {dataset_path}")
    try:
        df = pd.read_csv(dataset_path, sep='\t')
    except Exception as e:
        df = pd.read_csv(dataset_path)

    if 'sample' in df.columns:
        df = df.set_index('sample').T

    df_features = df.drop(columns=['target', 'label', 'cancer_type'], errors='ignore')
    patient_data = df_features.iloc[patient_row].values
    
    print(f"\n[Client] 🧬 Scanning Patient #{patient_row} across all 33 Cancer Profiles...")

    best_cancer_name = "Unknown"
    best_raw_score = -float('inf')  # Start with the lowest possible number
    best_active_weights = None
    best_patient_rna = None

    # 3. Stage 1: Local Prescreen (Find the highest raw score)
    for i in range(len(cancer_labels)):
        w = weights_matrix[i]
        active_idx = np.where(w != 0)[0]
        if len(active_idx) == 0: continue

        active_weights = w[active_idx]
        patient_rna = patient_data[active_idx]

        # Calculate raw float score
        raw_float_score = np.dot(patient_rna, active_weights)

        # Compare the RAW score, not the squashed percentage!
        if raw_float_score > best_raw_score:
            best_raw_score = raw_float_score
            best_cancer_name = cancer_labels[i]
            best_active_weights = active_weights
            best_patient_rna = patient_rna

    # Now that we found the true winner, calculate its percentage
    probability = 1 / (1 + np.exp(-best_raw_score))
    best_risk_percentage = probability * 100

    print("\n=======================================================")
    print(f" AUTO-DETECT FLAG: Highest match is {best_cancer_name}")
    print(f" PRESUMPTIVE RISK: {best_risk_percentage:.2f}%")
    print("=======================================================")

    # 4. Stage 2: Transmit to the Go Enclave for Cryptographic Verification
    print(f"\n[Client] Transmitting {best_cancer_name} biomarkers to Secure Cloud for Zero-Knowledge Verification...")
    
    encrypted_risk_score = send_to_hospital(best_patient_rna, best_active_weights)

    print("\n=======================================================")
    print(f" CLOUD VERIFICATION COMPLETE")
    print(f" SECURE CRYPTO-SCORE:  {encrypted_risk_score}")
    print(f"  OFFICIAL DIAGNOSIS:   {best_cancer_name} ({best_risk_percentage:.2f}% Risk)")
    print("=======================================================")

    # Usage 1: python fe_client.py
    if len(sys.argv) == 1:
        print("Running tiny 3-element smoke test…\n")
        demo_tiny()
    
    # Usage: python fe_client.py weights.npy dataset.xena [patient_row] model.pkl
    else:
        weights_file = sys.argv[1]
        dataset_file = sys.argv[2]
        patient_row  = int(sys.argv[3])
        model_pkl    = sys.argv[4] if len(sys.argv) > 4 else "master_33_cancer_model.pkl"
        
        demo_auto_detect(weights_file, dataset_file, patient_row, model_pkl)

    # # Usage 2: python fe_client.py sparse_weights.npy [cancer_idx]
    # elif len(sys.argv) <= 3:
    #     weights_file = sys.argv[1]
    #     cancer_idx   = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    #     demo_with_saved_weights(weights_file, cancer_idx)

    # # Usage 3: python fe_client.py sparse_weights.npy uci_data.csv [patient_row] [cancer_idx]
    # else:
    #     weights_file = sys.argv[1]
    #     dataset_file = sys.argv[2]
    #     patient_row  = int(sys.argv[3])
    #     cancer_idx   = int(sys.argv[4]) if len(sys.argv) > 4 else 0
    #     demo_real_patient(weights_file, dataset_file, patient_row, cancer_idx)