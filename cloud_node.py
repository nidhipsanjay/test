from flask import Flask, request, jsonify
from mife.single.damgard import FeDamgard
import requests
import pickle
import base64

app = Flask(__name__)
HOSPITAL_URL = "http://localhost:8081"

def deserialize(obj_str):
    return pickle.loads(base64.b64decode(obj_str.encode('utf-8')))

@app.route('/compute', methods=['POST'])
def compute():
    print("\n[CLOUD] Received Encrypted Ciphertext from Doctor.")
    
    ciphertext = deserialize(request.json['ciphertext'])
    mpk = deserialize(request.json['mpk'])
    weights = request.json['weights'] 
    
    print(f"[CLOUD] Requesting Functional Key for {len(weights)} weights...")
    resp = requests.post(f"{HOSPITAL_URL}/functional_key", json={"weights": weights})
    sk_y = deserialize(resp.json()['sk_y'])

    print("[CLOUD] Computing Inner Product blindly...")
    # Expanded search space for real data (searching from -500000 to +500000)
    # The larger this is, the longer it takes, but it accommodates real genomic variance.
    search_space = (-500000, 500000) 
    
    try:
        result = FeDamgard.decrypt(ciphertext, mpk, sk_y, search_space)
        print(f"[CLOUD] Math complete. Sending score ({result}) back to Doctor.")
        return jsonify({"risk_score": result})
    except Exception as e:
        print("[CLOUD] Error: Score exceeded PyMIFE search bounds.")
        return jsonify({"error": "Score out of bounds", "details": str(e)})

if __name__ == '__main__':
    app.run(port=8082)