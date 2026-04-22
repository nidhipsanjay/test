import pandas as pd
import numpy as np
import joblib
import requests
import pickle
import base64
import sys
from mife.single.damgard import FeDamgard

HOSPITAL_URL = "http://localhost:8081"
CLOUD_URL = "http://localhost:8082"

def serialize(obj):
    return base64.b64encode(pickle.dumps(obj)).decode('utf-8')

def deserialize(obj_str):
    return pickle.loads(base64.b64decode(obj_str.encode('utf-8')))

def run_automated_diagnosis(patient_id: str):
    print("\n[DOCTOR CLIENT] Booting Secure Medical Terminal...")
    
    # ---------------------------------------------------------
    # PHASE 1: LOCAL TRIAGE (The Autodetect)
    # ---------------------------------------------------------
    print(" Loading Diagnostic Model & Scaler...")
    model = joblib.load('master_33_cancer_model.pkl')
    scaler = joblib.load('master_33_cancer_scaler.pkl')

    print(f"Extracting DNA for Patient: {patient_id}...")
    try:
        # Load Xena safely
        try:
            df = pd.read_csv('EB++AdjustPANCAN_IlluminaHiSeq_RNASeqV2.geneExp.xena', sep='\t')
        except Exception:
            df = pd.read_csv('EB++AdjustPANCAN_IlluminaHiSeq_RNASeqV2.geneExp.xena')

        if 'sample' in df.columns:
            df = df.set_index('sample').T

        df_features = df.drop(columns=['target', 'label', 'cancer_type'], errors='ignore')
        
        # Extract and Align!
        raw_patient_data = df_features.loc[[patient_id]].copy()
        aligned_patient_data = raw_patient_data.reindex(columns=scaler.feature_names_in_, fill_value=0)
        
    except KeyError:
        print(f"ERROR: Patient ID '{patient_id}' not found in database.")
        return

    print("[LOCAL MODEL] Scaling DNA parameters...")
    scaled_patient_data = scaler.transform(aligned_patient_data)

    print("\n=======================================================")
    print("LOCAL TRIAGE RESULT:")
    prediction = model.predict(scaled_patient_data)[0]
    probabilities = model.predict_proba(scaled_patient_data)[0]
    
    # DYNAMIC INDEXING: Find the exact index of the predicted cancer!
    predicted_index = np.where(model.classes_ == prediction)[0][0]
    confidence = probabilities[predicted_index] * 100
    
    print(f"Diagnosed: {prediction} (Index: {predicted_index})")
    print(f"Confidence: {confidence:.2f}%")
    print("=======================================================\n")

    # If confidence is too low, the doctor might not want to verify.
    if confidence < 50:
        print(" Warning: Local model confidence is low. Proceeding with verification anyway...")

    # ---------------------------------------------------------
    # PHASE 2: CLOUD VERIFICATION (The 3-Node Crypto)
    # ---------------------------------------------------------
    print(f"[CRYPTO] Preparing Zero-Trust Cloud Verification for {prediction}...")
    
    print(f"[CRYPTO] Extracting Cloud Weights from .npy payload...")
    weights_matrix = np.load('master_33_cancer_weights.npy')
    w = weights_matrix[predicted_index] # <--- Using the dynamic index!
    
    # Find the active genes for this specific cancer
    active_idx = np.where(w != 0)[0]
    raw_weights = w[active_idx]
    
    # Quantize for PyMIFE
    ml_weights = (raw_weights * 100).astype(int).tolist()
    vector_len = len(ml_weights)
    print(f"[CRYPTO] Model requires {vector_len} active genes for verification.")

    # Filter the patient's SCALED DNA down to just the active genes
    scaled_rna = scaled_patient_data[0][active_idx]
    patient_data = (scaled_rna * 100).astype(int).tolist()

    # Network Transmission
    print(f"[NETWORK] Fetching Master Public Key from Hospital...")
    resp = requests.post(f"{HOSPITAL_URL}/public_key", json={"vector_len": vector_len})
    mpk = deserialize(resp.json()['mpk'])

    print(f"[NETWORK] Encrypting Patient DNA locally...")
    ciphertext = FeDamgard.encrypt(patient_data, mpk)

    print(f"[NETWORK] Transmitting Ciphertext to Untrusted Cloud...")
    payload = {
        "ciphertext": serialize(ciphertext),
        "mpk": serialize(mpk),
        "weights": ml_weights
    }
    
    try:
        cloud_resp = requests.post(f"{CLOUD_URL}/compute", json=payload)
        
        if "risk_score" in cloud_resp.json():
            score = cloud_resp.json()['risk_score']
            print("\n SECURE CLOUD VERIFICATION COMPLETE")
            print(f" Encrypted Inner Product Score: {score}")
            
            # ---------------------------------------------------------
            # THE CLINICAL TRANSLATION
            # ---------------------------------------------------------
            # 1. Shrink the score back down (since we multiplied both DNA and Weights by 100)
            true_dot_product = score / 10000.0
            
            # 2. Grab the model's baseline intercept for this specific cancer
            # Dig into the OneVsRest estimator array to find the specific intercept
            intercept = model.estimators_[predicted_index].intercept_[0]
            
            # 3. Calculate the final Logit (x)
            x = true_dot_product + intercept
            
            # 4. Push it through the Sigmoid Function
            probability = 1 / (1 + np.exp(-x))
            final_percentage = probability * 100
            
            print("=======================================================")
            print(f" CLOUD VERIFIED RISK LEVEL: {final_percentage:.2f}%")
            print("=======================================================")
            
        else:
            print(f"Error from Cloud: {cloud_resp.json()}")
    except Exception as e:
        print(f"Network Error. Is the Cloud running? Error: {e}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 doctor_client.py <patient_id>")
    else:
        run_automated_diagnosis(sys.argv[1])