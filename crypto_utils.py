import pickle
import base64
import hmac
import hashlib

# In a real hospital, this is injected via environment variables.
# For the demo, we define the shared internal network secret here.
NETWORK_SECRET = b"super_secure_capstone_network_key_2026"

def secure_serialize(obj):
    """Serializes the object and attaches a mathematically verifiable signature."""
    serialized_bytes = pickle.dumps(obj)
    # Generate a 256-bit signature of the bytes
    signature = hmac.new(NETWORK_SECRET, serialized_bytes, hashlib.sha256).digest()
    
    # Prepend the 32-byte signature to the payload and base64 encode
    payload = base64.b64encode(signature + serialized_bytes).decode('utf-8')
    return payload

def secure_deserialize(payload_str):
    """Verifies the signature to prevent RCE before allowing unpickling."""
    raw_bytes = base64.b64decode(payload_str.encode('utf-8'))
    
    # SHA-256 signatures are exactly 32 bytes
    signature = raw_bytes[:32]
    serialized_bytes = raw_bytes[32:]
    
    # Recompute the signature locally to ensure it matches
    expected_signature = hmac.new(NETWORK_SECRET, serialized_bytes, hashlib.sha256).digest()
    
    if not hmac.compare_digest(signature, expected_signature):
        raise ValueError("SECURITY ALERT: Signature mismatch. Payload tampered with!")
        
    return pickle.loads(serialized_bytes)