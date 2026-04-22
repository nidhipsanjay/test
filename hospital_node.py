from flask import Flask, request, jsonify
from mife.single.damgard import FeDamgard
import numpy as np
import base64
import json
from crypto_utils import secure_serialize, secure_deserialize
import uuid

app = Flask(__name__)

# Universal System Constants
QUANTIZATION_FACTOR = 100
hospital_vault = {}
active_sessions = {} # <--- Added this to hold keys temporarily

print("[HOSPITAL] Booting Secure Vault...")
print("[HOSPITAL] Loading Proprietary AI Weights...")
weights_matrix = np.load('master_33_cancer_weights.npy')

@app.route('/model_metadata', methods=['POST'])
def get_model_metadata():
    """Tells the Doctor which genes to filter without revealing the weights."""
    model_index = request.json['model_index']
    w = weights_matrix[model_index]
    active_idx = np.where(w != 0)[0].tolist()
    
    print(f"[API] Doctor requested metadata for Model Index {model_index}. ({len(active_idx)} active genes)")
    return jsonify({"active_idx": active_idx})

@app.route('/public_key', methods=['POST'])
def get_public_key():
    vector_len = request.json['vector_len']
    if vector_len not in hospital_vault:
        print(f"[HOSPITAL] Generating new 1024-bit universe for length {vector_len}...")
        hospital_vault[vector_len] = FeDamgard.generate(vector_len)
        
    mpk = hospital_vault[vector_len].get_public_key()
    return jsonify({"mpk": secure_serialize(mpk)})

# --- NEW SESSION ARCHITECTURE REPLACES /functional_key ---

@app.route('/prep_key', methods=['POST'])
def prep_key():
    """Doctor calls this to authorize a computation without telling the Cloud."""
    model_index = request.json['model_index']
    session_id = request.json['session_id']
    
    # 1. Retrieve internal weights
    w = weights_matrix[model_index]
    active_idx = np.where(w != 0)[0]
    raw_weights = w[active_idx]
    
    # 2. Quantize
    ml_weights = (raw_weights * QUANTIZATION_FACTOR).astype(int).tolist()
    vector_len = len(ml_weights)
    
    print(f"[HOSPITAL] Prepping Key for Session {session_id} (Hidden Model Index: {model_index})")
    
    master_key = hospital_vault[vector_len]
    sk_y = FeDamgard.keygen(ml_weights, master_key)
    
    # Store the serialized key under the random session ID
    active_sessions[session_id] = secure_serialize(sk_y)
    
    return jsonify({"status": "Authorized"})

@app.route('/fetch_key', methods=['POST'])
def fetch_key():
    """Cloud calls this. It only knows the Session ID."""
    session_id = request.json['session_id']
    
    if session_id not in active_sessions:
        return jsonify({"error": "Invalid or expired Session ID"}), 403
        
    # .pop() ensures the key is deleted from the Hospital's memory the moment it is fetched
    sk_y_serialized = active_sessions.pop(session_id)
    
    print(f"[API] Issued Functional Key for Session {session_id} to Cloud. Key deleted from local cache.")
    return jsonify({"sk_y": sk_y_serialized})

if __name__ == '__main__':
    # Wrap in HTTPS in production: ssl_context='adhoc'
    app.run(port=8081)