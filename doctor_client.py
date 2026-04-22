import pandas as pd
import numpy as np
import joblib
import requests
import base64
import sys
from mife.single.damgard import FeDamgard
from crypto_utils import secure_serialize, secure_deserialize
import uuid


HOSPITAL_URL = "http://localhost:8081"
CLOUD_URL = "http://localhost:8082"
QUANTIZATION_FACTOR = 100

# --- Helper: Safe Serialization (No Pickle) ---
def safe_serialize(obj):
    return base64.b64encode(str(obj).encode('utf-8')).decode('utf-8')

def run_automated_diagnosis(patient_id: str):
    print("\n[DOCTOR CLIENT] Booting Secure Medical Terminal...")
    
    # --- PHASE 1: LOCAL TRIAGE ---
    model = joblib.load('master_33_cancer_model.pkl')
    scaler = joblib.load('master_33_cancer_scaler.pkl')

    try:
        df = pd.read_csv('EB++AdjustPANCAN_IlluminaHiSeq_RNASeqV2.geneExp.xena', sep='\t')
        if 'sample' in df.columns:
            df = df.set_index('sample').T
        df_features = df.drop(columns=['target', 'label', 'cancer_type'], errors='ignore')
        
        raw_patient_data = df_features.loc[[patient_id]].copy()
        aligned_patient_data = raw_patient_data.reindex(columns=scaler.feature_names_in_, fill_value=0)
    except KeyError:
        print("ERROR: Patient ID not found.")
        return

    scaled_patient_data = scaler.transform(aligned_patient_data)
    prediction = model.predict(scaled_patient_data)[0]
    predicted_index = int(np.where(model.classes_ == prediction)[0][0])
    
    print(f"Local Triage Diagnosed: {prediction} (Index: {predicted_index})")

    # --- PHASE 2: OBFUSCATED VERIFICATION ---
    print(f"[NETWORK] Requesting Active Gene Metadata from Hospital...")
    meta_resp = requests.post(f"{HOSPITAL_URL}/model_metadata", json={"model_index": predicted_index})
    active_idx = meta_resp.json()['active_idx']
    
    # Filter and Quantize DNA
    scaled_rna = scaled_patient_data[0][active_idx]
    patient_data = (scaled_rna * QUANTIZATION_FACTOR).astype(int).tolist()
    vector_len = len(patient_data)

    print(f"[NETWORK] Fetching Master Public Key for {vector_len} genes...")
    resp = requests.post(f"{HOSPITAL_URL}/public_key", json={"vector_len": vector_len})
    mpk_str = resp.json()['mpk'] 
    
    # THE FIX: Safely unpack the string back into a PyMIFE Object
    mpk = secure_deserialize(mpk_str)

    print(f"[CRYPTO] Encrypting DNA locally...")
    ciphertext = FeDamgard.encrypt(patient_data, mpk)

    # THE FIX: Generate a random, single-use Session ID
    session_id = str(uuid.uuid4())

    print(f"[NETWORK] Authorizing Session {session_id} with Hospital...")
    requests.post(f"{HOSPITAL_URL}/prep_key", json={"model_index": predicted_index, "session_id": session_id})

    # We send the Session ID to the Cloud, NOT the model index
    payload = {
        "ciphertext": secure_serialize(ciphertext),
        "mpk": secure_serialize(mpk),
        "session_id": session_id # <--- Update here
    }
    
    print(f"[NETWORK] Transmitting Ciphertext to Cloud...")
    cloud_resp = requests.post(f"{CLOUD_URL}/compute", json=payload)
    
    if "risk_score" in cloud_resp.json():
        score = cloud_resp.json()['risk_score']
        
        # Reverse the math using the Universal Constant (squared because Weights * DNA)
        true_dot_product = score / (QUANTIZATION_FACTOR ** 2)
        intercept = model.estimators_[predicted_index].intercept_[0]
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