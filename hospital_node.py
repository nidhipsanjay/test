from flask import Flask, request, jsonify
from mife.single.damgard import FeDamgard
import pickle
import base64

app = Flask(__name__)

# Dictionary to store keys based on vector length
hospital_vault = {}

def serialize(obj):
    return base64.b64encode(pickle.dumps(obj)).decode('utf-8')

@app.route('/public_key', methods=['POST'])
def get_public_key():
    vector_len = request.json['vector_len']
    print(f"[API] Doctor requested Master Public Key for {vector_len} genes.")
    
    if vector_len not in hospital_vault:
        print(f"[HOSPITAL] Generating new 1024-bit universe for length {vector_len}...")
        hospital_vault[vector_len] = FeDamgard.generate(vector_len)
        
    mpk = hospital_vault[vector_len].get_public_key()
    return jsonify({"mpk": serialize(mpk)})

@app.route('/functional_key', methods=['POST'])
def get_functional_key():
    weights = request.json['weights']
    vector_len = len(weights)
    print(f"[API] Cloud requested Functional Key for AI model (Length: {vector_len})")
    
    master_key = hospital_vault[vector_len]
    sk_y = FeDamgard.keygen(weights, master_key)
    return jsonify({"sk_y": serialize(sk_y)})

if __name__ == '__main__':
    app.run(port=8081)