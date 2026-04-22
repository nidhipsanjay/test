package main

import (
	"encoding/json"
	"fmt"
	"math/big"
	"net/http"

	"github.com/fentec-project/gofe/data"
	"github.com/fentec-project/gofe/innerprod/fullysec"
)

// Configuration
const modBits = 1024
const boundInt = 1_000_000_000

var (
	engine     *fullysec.Damgard
	msk        *fullysec.DamgardSecKey
	mpk        data.Vector
	currentLen int
)

// HTTP Payloads
type ProcessRequest struct {
	PatientRNA []int64
	MLWeights  []int64
}

type ProcessResponse struct {
	RiskScore string
	Error     string
}

func processHandler(w http.ResponseWriter, r *http.Request) {
	var req ProcessRequest
	json.NewDecoder(r.Body).Decode(&req)
	l := len(req.PatientRNA)

	// Re-initialize the Mathematical Universe only if the vector length changes
	if engine == nil || currentLen != l {
		fmt.Printf("\n[SYSTEM] Generating 1024-bit Safe Prime for l=%d. Please wait...\n", l)
		engine, _ = fullysec.NewDamgard(l, modBits, big.NewInt(boundInt))
		msk, mpk, _ = engine.GenerateMasterKeys()
		currentLen = l
	}

	rnaVec := int64SliceToVector(req.PatientRNA)
	wVec := int64SliceToVector(req.MLWeights)

	// ---------------------------------------------------------
	// 1. HOSPITAL ENCLAVE (Trusted Area)
	// ---------------------------------------------------------
	fmt.Println("\n=======================================================")
	fmt.Println("   [HOSPITAL ENCLAVE]  (Trusted Key Generation)        ")
	fmt.Println("=======================================================")
	fmt.Println(" 🔒 Locking Patient RNA with Master Public Key...")
	ciphertext, _ := engine.Encrypt(rnaVec, mpk)

	fmt.Println(" 🔑 Fusing ML Weights with Master Secret Key (MSK)...")
	feKey, _ := engine.DeriveKey(msk, wVec)

	fmt.Println(" 🚀 Transmitting Ciphertext & FE Key over secure channel...")

	// ---------------------------------------------------------
	// 2. CLOUD ENCLAVE (Untrusted Area)
	// ---------------------------------------------------------
	fmt.Println("\n=======================================================")
	fmt.Println("   [CLOUD ENCLAVE]  (Untrusted Compute Node)           ")
	fmt.Println("=======================================================")
	fmt.Println(" 📥 Received payload. NO ACCESS to raw patient data.")
	fmt.Println(" ⚙️  Computing secure inner product via Damgård solver...")

	// The cloud uses its own wrapper for the weights
	cloudWVec := int64SliceToVector(req.MLWeights)

	result, err := engine.Decrypt(ciphertext, feKey, cloudWVec)
	if err != nil {
		fmt.Printf(" ❌ Decrypt error: %v\n", err)
		json.NewEncoder(w).Encode(ProcessResponse{Error: err.Error()})
		return
	}

	fmt.Printf(" ✅ Decryption Success! Raw Risk Score: %s\n", result.String())
	fmt.Println("=======================================================\n")

	// Return result to Python client
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(ProcessResponse{RiskScore: result.String()})
}

func int64SliceToVector(vals []int64) data.Vector {
	bigInts := make([]*big.Int, len(vals))
	for i, v := range vals {
		bigInts[i] = big.NewInt(v)
	}
	return data.NewVector(bigInts)
}

func main() {
	http.HandleFunc("/process", processHandler)
	fmt.Println("Healthcare 5.0 Unified Enclave running on :8080")
	http.ListenAndServe(":8080", nil)
}