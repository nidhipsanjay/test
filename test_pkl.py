import pandas as pd
import numpy as np
import joblib
import sys

def autodetect_cancer(pkl_path: str, scaler_path: str, dataset_path: str, patient_id: str):
    print("🧠 [AI] Booting Full Diagnostic Brain...")
    model = joblib.load(pkl_path)
    scaler = joblib.load(scaler_path) # LOAD YOUR SEPARATE SCALER!

    print("🧠 [AI] Loading Patient Database...")
    try:
        df = pd.read_csv(dataset_path, sep='\t')
    except Exception:
        df = pd.read_csv(dataset_path)

    if 'sample' in df.columns:
        df = df.set_index('sample').T

    df_features = df.drop(columns=['target', 'label', 'cancer_type'], errors='ignore')
    
    # Extract Raw Data by actual Patient ID instead of a random row number!
    try:
        raw_patient_data = df_features.loc[[patient_id]].copy()
    except KeyError:
        print(f"❌ ERROR: Patient ID '{patient_id}' does not exist in the Xena DNA file!")
        return
    
    # ---------------------------------------------------------
    # CRITICAL FIX 2: ALIGN THE COLUMNS!
    # Force the patient's genes into the exact order the AI memorized.
    # If a gene is missing in Xena, it safely fills it with 0.
    expected_genes = scaler.feature_names_in_
    aligned_patient_data = raw_patient_data.reindex(columns=expected_genes, fill_value=0)
    # ---------------------------------------------------------
    
    # Scale the newly aligned data
    scaled_patient_data = scaler.transform(aligned_patient_data)
# DEBUGGING: Let's look inside the patient's DNA
    print(f"🔍 [DIAGNOSTICS] Total Genes Checked: {aligned_patient_data.shape[1]}")
    active_genes = (aligned_patient_data.values != 0).sum()
    print(f"🔍 [DIAGNOSTICS] Non-Zero Genes Found: {active_genes}")

    print(f"🏥 [HOSPITAL] Extracted Patient {patient_id}. Running full body scan...")

    prediction = model.predict(scaled_patient_data)[0]
    probabilities = model.predict_proba(scaled_patient_data)[0]
    
    top_5_indices = np.argsort(probabilities)[-5:][::-1]

    print("\n=======================================================")
    print("🚨 AI DIFFERENTIAL DIAGNOSIS (TOP 5):")
    print("-------------------------------------------------------")
    for i in top_5_indices:
        cancer_type = model.classes_[i]
        confidence = probabilities[i] * 100
        print(f"[{cancer_type}] Confidence: {confidence:.2f}%")
    print("=======================================================")

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 5:
        print("Usage: python3 test_pkl.py <model.pkl> <scaler.pkl> <dataset.xena> <patient_id>")
    else:
        autodetect_cancer(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])