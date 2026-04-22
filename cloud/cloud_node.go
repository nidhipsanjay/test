package main

// -----------------------------------------------------------------------
// cloud_node.go — The Zero-Knowledge Cloud (Port 8081)
//
// Responsibilities:
//   1. On startup: initialise the SAME Damgård engine parameters as the
//      Hospital (same l, modBits, bound). NO master keys are generated here.
//   2. POST /compute — receives gob-encoded ComputeRequest, calls
//      engine.Decrypt(ciphertext, feKey, mlWeights), returns the
//      inner product as a decimal string.
//
// What the Cloud CANNOT see:
//   - The raw patient RNA values (it only has the ciphertext)
//   - The Master Secret Key (it only has the derived feKey for this query)
//
// CONSTRAINTS MIRRORED FROM BRIEF:
//   - Same engine initialisation as Hospital (parameters must match).
//   - gob deserialisation (not JSON).
//   - No negative numbers — validated before calling Decrypt.
// -----------------------------------------------------------------------

import (
	"bytes"
	"encoding/gob"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"math/big"
	"net/http"

	"github.com/fentec-project/gofe/data"
	"github.com/fentec-project/gofe/innerprod/fullysec"
)

// ── Global state ─────────────────────────────────────────────────────────────

var cloudEngine *fullysec.Damgard

const (
	cloudModBits = 1024
	cloudBound   = int64(1_000_000_000)
)

// ── Startup ───────────────────────────────────────────────────────────────────

func initCloud(l int) {
	RegisterGobTypes()

	var err error
	// ⚠️  These parameters MUST be identical to the Hospital node.
	//     If they differ, Decrypt will return garbage or panic.
	cloudEngine, err = fullysec.NewDamgard(l, cloudModBits, big.NewInt(cloudBound))
	if err != nil {
		log.Fatalf("Cloud: failed to create Damgård engine: %v", err)
	}

	log.Printf("Cloud: Damgård engine ready (l=%d, modBits=%d). No master keys here.", l, cloudModBits)
}

// ── /compute handler ──────────────────────────────────────────────────────────

func computeHandler(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "POST only", http.StatusMethodNotAllowed)
		return
	}

	// 1. Read raw body (gob bytes sent by Hospital)
	body, err := io.ReadAll(r.Body)
	if err != nil {
		respondComputeError(w, fmt.Sprintf("failed to read body: %v", err))
		return
	}

	// 2. gob-decode into ComputeRequest
	var req ComputeRequest
	dec := gob.NewDecoder(bytes.NewReader(body))
	if err := dec.Decode(&req); err != nil {
		respondComputeError(w, fmt.Sprintf("gob decode failed: %v", err))
		return
	}

	if len(req.Ciphertext) == 0 {
		respondComputeError(w, "empty ciphertext received")
		return
	}
	if req.FeKey == nil {
		respondComputeError(w, "nil FE key received")
		return
	}

	l := len(req.MLWeights)
	log.Printf("Cloud: received /compute request, vector length l=%d", l)

	// 3. Re-initialise engine if vector length doesn't match
	if cloudEngine == nil || cloudCurrentLength != l {
		log.Printf("Cloud: (re)initialising engine for l=%d", l)
		var err error
		cloudEngine, err = fullysec.NewDamgard(l, cloudModBits, big.NewInt(cloudBound))
		if err != nil {
			respondComputeError(w, fmt.Sprintf("engine re-init failed: %v", err))
			return
		}
		cloudCurrentLength = l
	}

	// 4. Validate MLWeights — must all be strictly positive
	for i, v := range req.MLWeights {
		if v.Sign() <= 0 {
			respondComputeError(w, fmt.Sprintf(
				"MLWeights[%d]=%s is not strictly positive", i, v.String()))
			return
		}
	}

	// 5. Wrap MLWeights in data.Vector
	//    RULE: Always use data.NewVector — never pass raw []*big.Int
	wVec := data.NewVector(req.MLWeights)

	// 6. Run the Functional Decryption (the core privacy-preserving operation)
	//    Decrypt computes ⟨x, w⟩ = inner product WITHOUT decrypting x itself.
	innerProduct, err := cloudEngine.Decrypt(req.Ciphertext, req.FeKey, wVec)
	if err != nil {
		respondComputeError(w, fmt.Sprintf("Decrypt failed: %v", err))
		return
	}

	log.Printf("Cloud: inner product computed: %s", innerProduct.String())

	// 7. Return result as JSON (just a decimal string — safe cross-platform)
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(ComputeResponse{
		InnerProduct: innerProduct.String(),
	})
}

// ── Helpers ───────────────────────────────────────────────────────────────────

var cloudCurrentLength int

func respondComputeError(w http.ResponseWriter, msg string) {
	log.Printf("Cloud ERROR: %s", msg)
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(http.StatusInternalServerError)
	json.NewEncoder(w).Encode(ComputeResponse{Error: msg})
}

// ── main ──────────────────────────────────────────────────────────────────────

func main() {
	initCloud(1) // placeholder l=1; lazily re-initialised per request

	http.HandleFunc("/compute", computeHandler)
	http.HandleFunc("/health", func(w http.ResponseWriter, r *http.Request) {
		w.Write([]byte(`{"status":"cloud node running"}`))
	})

	log.Println("Cloud node listening on :8081")
	if err := http.ListenAndServe(":8081", nil); err != nil {
		log.Fatalf("Cloud: server error: %v", err)
	}
}