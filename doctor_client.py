import pandas as pd
import numpy as np
import joblib
import requests
import base64
import sys
from mife.single.damgard import FeDamgard
from crypto_utils import secure_serialize, secure_deserialize
import uuid
import json
import urllib3

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

HOSPITAL_URL = "https://localhost:8081"
CLOUD_URL = "https://localhost:8082"
QUANTIZATION_FACTOR = 100

# --- Helper: Safe Serialization (No Pickle) ---
def safe_serialize(obj):
    return base64.b64encode(str(obj).encode('utf-8')).decode('utf-8')

def run_automated_diagnosis(patient_id: str):
    print("\n[DOCTOR CLIENT] Booting Secure Medical Terminal...")
    
    # --- PHASE 1: DATA EXTRACTION (NO ML AI INVOLVED) ---
    scaler = joblib.load('master_33_cancer_scaler.pkl')
    
    try:
        # 1. Load the raw patient genomic sequence
        df = pd.read_csv('EB++AdjustPANCAN_IlluminaHiSeq_RNASeqV2.geneExp.xena', sep='\t')
        if 'sample' in df.columns:
            df = df.set_index('sample').T
        df_features = df.drop(columns=['target', 'label', 'cancer_type'], errors='ignore')
        
        raw_patient_data = df_features.loc[[patient_id]].copy()
        aligned_patient_data = raw_patient_data.reindex(columns=scaler.feature_names_in_, fill_value=0)
        
    except KeyError:
        print("ERROR: Patient ID not found in dataset.")
        return
    except FileNotFoundError:
        print("ERROR: Xena dataset file not found.")
        return

    # 2. Scale the data so the cryptography engine can process it
    scaled_patient_data = scaler.transform(aligned_patient_data)

    # --- PHASE 2: CLINICAL TEST SELECTION (MANUAL UI) ---
    TCGA_CANCERS = {
        0: 'ACC (Adrenocortical)', 1: 'BLCA (Bladder)', 2: 'BRCA (Breast)', 
        3: 'CESC (Cervical)', 4: 'CHOL (Bile Duct)', 5: 'COAD (Colon)',
        6: 'DLBC (Lymphoma)', 7: 'ESCA (Esophageal)', 8: 'GBM (Glioblastoma)', 
        9: 'HNSC (Head & Neck)', 10: 'KICH (Kidney Chromophobe)', 11: 'KIRC (Kidney Clear Cell)',
        12: 'KIRP (Kidney Papillary)', 13: 'LAML (Leukemia)', 14: 'LGG (Lower Grade Glioma)', 
        15: 'LIHC (Liver)', 16: 'LUAD (Lung Adenocarcinoma)', 17: 'LUSC (Lung Squamous)',
        18: 'MESO (Mesothelioma)', 19: 'OV (Ovarian)', 20: 'PAAD (Pancreatic)', 
        21: 'PCPG (Pheochromocytoma)', 22: 'PRAD (Prostate)', 23: 'READ (Rectal)',
        24: 'SARC (Sarcoma)', 25: 'SKCM (Melanoma)', 26: 'STAD (Stomach)', 
        27: 'TGCT (Testicular)', 28: 'THCA (Thyroid)', 29: 'THYM (Thymoma)',
        30: 'UCEC (Endometrial)', 31: 'UCS (Uterine Carcinosarcoma)', 32: 'UVM (Uveal Melanoma)'
    }

    print("\n" + "="*60)
    print("      SECURE CLINICAL VERIFICATION TERMINAL")
    print("="*60)
    
    # Print the 33 options in a clean 3-column grid
    for i in range(0, 33, 3):
        col1 = f"[{i:2d}] {TCGA_CANCERS[i].split(' ')[0]:<6}" if i < 33 else ""
        col2 = f"[{i+1:2d}] {TCGA_CANCERS[i+1].split(' ')[0]:<6}" if i+1 < 33 else ""
        col3 = f"[{i+2:2d}] {TCGA_CANCERS[i+2].split(' ')[0]:<6}" if i+2 < 33 else ""
        print(f"{col1:<18} | {col2:<18} | {col3}")
    
    print("-" * 60)
    
    # Doctor manually orders the test
    while True:
        try:
            selection = input("\nEnter the Target Cancer Index [0-32] to verify: ")
            predicted_index = int(selection)
            if 0 <= predicted_index <= 32:
                break
            print("Invalid index. Please select between 0 and 32.")
        except ValueError:
            print("Please enter a valid number.")
            
    print(f"\n[CLINICAL UI] Authorized verification for: {TCGA_CANCERS[predicted_index]}")

    # =====================================================================
    # FROM HERE DOWN, KEEP YOUR EXISTING NETWORK AND ENCRYPTION CODE
    # e.g., Requesting Active Gene Metadata, Fetching Public Key, etc.
    # =====================================================================

    # --- PHASE 2: OBFUSCATED VERIFICATION ---
    print(f"[NETWORK] Requesting Active Gene Metadata from Hospital...")
    meta_resp = requests.post(f"{HOSPITAL_URL}/model_metadata", json={"model_index": predicted_index},verify=False)
    active_idx = meta_resp.json()['active_idx']
    
    # Filter and Quantize DNA
    scaled_rna = scaled_patient_data[0][active_idx]
    patient_data = (scaled_rna * QUANTIZATION_FACTOR).astype(int).tolist()
    vector_len = len(patient_data)

    print(f"[NETWORK] Fetching Master Public Key for {vector_len} genes...")
    resp = requests.post(f"{HOSPITAL_URL}/public_key", json={"vector_len": vector_len}, verify=False)
    mpk_str = resp.json()['mpk'] 
    
    # THE FIX: Safely unpack the string back into a PyMIFE Object
    mpk = secure_deserialize(mpk_str)

    print(f"[CRYPTO] Encrypting DNA locally...")
    ciphertext = FeDamgard.encrypt(patient_data, mpk)

    # THE FIX: Generate a random, single-use Session ID
    session_id = str(uuid.uuid4())

    print(f"[NETWORK] Authorizing Session {session_id} with Hospital...")
    requests.post(f"{HOSPITAL_URL}/prep_key", json={"model_index": predicted_index, "session_id": session_id}, verify=False)

    # We send the Session ID to the Cloud, NOT the model index
    payload = {
        "ciphertext": secure_serialize(ciphertext),
        "mpk": secure_serialize(mpk),
        "session_id": session_id # <--- Update here
    }
    
    print(f"[NETWORK] Transmitting Ciphertext to Cloud...")
    cloud_resp = requests.post(f"{CLOUD_URL}/compute", json=payload, verify=False)
    
    if "risk_score" in cloud_resp.json():
        score = cloud_resp.json()['risk_score']
        
        # Reverse the math using the Universal Constant (squared because Weights * DNA)
        true_dot_product = score / (QUANTIZATION_FACTOR ** 2)
        with open('clinical_intercepts.json', 'r') as f:
            intercept_db = json.load(f)
        intercept = intercept_db[str(predicted_index)]
        x = true_dot_product + intercept
        final_percentage = (1 / (1 + np.exp(-x))) * 100
        
        print(f" CLOUD VERIFIED RISK LEVEL: {final_percentage:.2f}%")
    else:
        # THE FIX: Don't swallow errors!
        print(f"\n[ERROR] Cloud Verification Failed!")
        print(f"Cloud Response: {cloud_resp.json()}")
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 doctor_client.py <patient_id>")
    else:
        run_automated_diagnosis(sys.argv[1])