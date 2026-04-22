from mife.single.damgard import FeDamgard

def test_crypto():
    print("[1] Initializing PyMIFE Engine (Damgård)...")
    n = 3 # Vector length

    print("[2] Generating Master Keys...")
    key = FeDamgard.generate(n)

    print("[3] Simulating Doctor: Encrypting Patient Data [4, 5, 6]")
    ciphertext = FeDamgard.encrypt([4, 5, 6], key)

    print("[4] Simulating Hospital: Generating Functional Key for ML Weights [1, 2, 3]")
    sk_y = FeDamgard.keygen([1, 2, 3], key)

    print("[5] Simulating Cloud: Computing the Inner Product blindly")
    # Math: (4*1) + (5*2) + (6*3) = 4 + 10 + 18 = 32
    # The (0, 1000) is the search space for the discrete logarithm
    result = FeDamgard.decrypt(ciphertext, key.get_public_key(), sk_y, (0, 1000))
    
    print("-------------------------------------------------")
    print(f"✅ Decrypted Score: {result}")
    print(f"🎯 Expected Score:  32")
    print("-------------------------------------------------")

if __name__ == "__main__":
    test_crypto()