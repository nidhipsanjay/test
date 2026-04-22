from flask import Flask, request, jsonify
from mife.single.damgard import FeDamgard
import requests
import base64
import json
from crypto_utils import secure_deserialize

app = Flask(__name__)
HOSPITAL_URL = "http://localhost:8081"

# In-Memory Rainbow Table (Simulated pre-computation on boot)
# In production, iterate through the required bounds and store elliptic curve point strings mapped to their integers.
print("[CLOUD] Generating In-Memory Rainbow Table for rapid decryption...")
rainbow_table = {} # Pre-populate this mapping curve points -> integers

def safe_deserialize(obj_str):
    # Placeholder: Replace with Charm-Crypto / PyMIFE native deserialization
    return obj_str

@app.route('/compute', methods=['POST'])
def compute():
    print("\n[CLOUD] Received Encrypted Ciphertext from Doctor.")
    
    ciphertext = secure_deserialize(request.json['ciphertext'])
    mpk = secure_deserialize(request.json['mpk'])
    
    # THE FIX: Cloud only extracts the random string
    session_id = request.json['session_id'] 
    
    print(f"[CLOUD] Requesting Functional Key for Session {session_id} from Hospital...")
    resp = requests.post(f"{HOSPITAL_URL}/fetch_key", json={"session_id": session_id})
    
    if "error" in resp.json():
        return jsonify({"error": "Hospital refused to issue key for this session."})
        
    sk_y = secure_deserialize(resp.json()['sk_y'])

    print("[CLOUD] Computing Inner Product blindly...")
    search_space = (-5000000, 5000000) 
    
    try:
        result = FeDamgard.decrypt(ciphertext, mpk, sk_y, search_space)
        print(f"[CLOUD] Math complete. Sending score ({result}) back to Doctor.")
        return jsonify({"risk_score": result})
    except Exception as e:
        return jsonify({"error": "Compute failed", "details": str(e)})


if __name__ == '__main__':
    app.run(port=8082)