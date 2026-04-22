package main

// -----------------------------------------------------------------------
// hospital_node.go — The Trusted Enclave (Port 8080)
//
// Responsibilities:
//   1. On startup: initialise Damgård engine, generate MSK + MPK.
//      MSK never leaves this process.
//   2. POST /process  — receives PatientRNA + MLWeights from Python,
//      encrypts the RNA, derives the FE key from the weights,
//      forwards both to the Cloud node, and returns the risk score.
//
// KEY CONSTRAINTS WE HANDLE HERE:
//   - data.NewVector() wrapping (never pass raw []*big.Int to fullysec).
//   - DeriveKey(msk, weights) — MSK must be the FIRST argument.
//   - gob serialisation over HTTP to the cloud (not JSON).
//   - All integers must be strictly positive (Python must send offset values).
// -----------------------------------------------------------------------

import (
	"bytes"
	"encoding/gob"
	"encoding/json"
	"fmt"
	"log"
	"math/big"
	"net/http"

	"github.com/fentec-project/gofe/data"
	"github.com/fentec-project/gofe/innerprod/fullysec"
)

// ── Global state ─────────────────────────────────────────────────────────────

var (
	engine *fullysec.Damgard
	msk    *fullysec.DamgardSecKey
	mpk    data.Vector
)

const (
	cloudURL  = "http://localhost:8081/compute"
	modBits   = 1024                 // Bit length of the safe prime modulus
	bound     = int64(1_000_000_000) // Upper bound on vector component values
)

// ── Startup ───────────────────────────────────────────────────────────────────

func initHospital(l int) {
	// l = vector length (number of active genes).
	// We initialise with a placeholder length at startup; the actual
	// length comes from the first /process request.  For production you
	// would fix this at deploy time.
	//
	// ⚠️  The engine must be initialised with the EXACT same parameters
	//     as the Cloud node, otherwise Decrypt will produce wrong results.

	RegisterGobTypes()

	var err error
	engine, err = fullysec.NewDamgard(l, modBits, big.NewInt(bound))
	if err != nil {
		log.Fatalf("Hospital: failed to create Damgård engine: %v", err)
	}

	msk, mpk, err = engine.GenerateMasterKeys()
	if err != nil {
		log.Fatalf("Hospital: failed to generate master keys: %v", err)
	}

	currentEngineLength = l
	log.Printf("Hospital: Damgård engine ready (l=%d, modBits=%d)", l, modBits)
	log.Println("Hospital: MSK generated and stored locally. MPK ready for encryption.")
}

// ── /process handler ─────────────────────────────────────────────────────────

func processHandler(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "POST only", http.StatusMethodNotAllowed)
		return
	}

	// 1. Decode the Python client's JSON request
	var req ProcessRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		respondProcessError(w, fmt.Sprintf("bad request body: %v", err))
		return
	}

	if len(req.PatientRNA) == 0 || len(req.MLWeights) == 0 {
		respondProcessError(w, "PatientRNA and MLWeights must not be empty")
		return
	}
	if len(req.PatientRNA) != len(req.MLWeights) {
		respondProcessError(w, fmt.Sprintf(
			"vector length mismatch: PatientRNA=%d, MLWeights=%d",
			len(req.PatientRNA), len(req.MLWeights),
		))
		return
	}

	l := len(req.PatientRNA)
	log.Printf("Hospital: received /process request, vector length l=%d", l)

	// 2. Validate: ALL values must be strictly positive (Damgård DL solver
	//    will hang / panic on negatives or zero).
	for i, v := range req.PatientRNA {
		if v <= 0 {
			respondProcessError(w, fmt.Sprintf(
				"PatientRNA[%d]=%d is not strictly positive — apply offset before sending", i, v))
			return
		}
	}
	for i, v := range req.MLWeights {
		if v <= 0 {
			respondProcessError(w, fmt.Sprintf(
				"MLWeights[%d]=%d is not strictly positive — apply offset before sending", i, v))
			return
		}
	}

	// 3. Reinitialise engine if vector length changed
	//    (handles the first call correctly too)
	if engine == nil || engineLength(engine) != l {
		log.Printf("Hospital: (re)initialising engine for l=%d", l)
		var err error
		engine, err = fullysec.NewDamgard(l, modBits, big.NewInt(bound))
		if err != nil {
			respondProcessError(w, fmt.Sprintf("engine init failed: %v", err))
			return
		}
		msk, mpk, err = engine.GenerateMasterKeys()
		if err != nil {
			respondProcessError(w, fmt.Sprintf("key generation failed: %v", err))
			return
		}
		currentEngineLength = l
	}

	// 4. Convert int64 slices → []*big.Int → data.Vector
	//    RULE: Always wrap with data.NewVector — never pass raw []*big.Int
	rnaVec := int64SliceToVector(req.PatientRNA)
	wVec := int64SliceToVector(req.MLWeights)

	// 5. Encrypt patient RNA under the Master Public Key
	//    ciphertext is []data.Vector
	ciphertext, err := engine.Encrypt(rnaVec, mpk)
	if err != nil {
		respondProcessError(w, fmt.Sprintf("encryption failed: %v", err))
		return
	}
	log.Println("Hospital: RNA encrypted successfully.")

	// 6. Derive the Functional Encryption key
	//    ⚠️  CRITICAL: DeriveKey(msk, weights) — MSK is FIRST
	feKey, err := engine.DeriveKey(msk, wVec)
	if err != nil {
		respondProcessError(w, fmt.Sprintf("DeriveKey failed: %v", err))
		return
	}
	log.Println("Hospital: Functional key derived. MSK stays local.")

	// 7. Build the payload for the Cloud node and gob-encode it.
	//    wVec is a data.Vector (which is []big.Int under the hood).
	//    We convert back to []*big.Int so the cloud can wrap it with
	//    data.NewVector on its side.
	weightsBigInt := make([]*big.Int, l)
	for i := 0; i < l; i++ {
		weightsBigInt[i] = new(big.Int).Set(wVec[i])
	}

	computeReq := ComputeRequest{
		Ciphertext: ciphertext,
		FeKey:      feKey,
		MLWeights:  weightsBigInt,
	}

	// Use gob — JSON cannot handle GoFE's unexported big.Int fields
	var buf bytes.Buffer
	enc := gob.NewEncoder(&buf)
	if err := enc.Encode(computeReq); err != nil {
		respondProcessError(w, fmt.Sprintf("gob encode failed: %v", err))
		return
	}

	// 8. POST to Cloud node
	resp, err := http.Post(cloudURL, "application/octet-stream", &buf)
	if err != nil {
		respondProcessError(w, fmt.Sprintf("cloud POST failed: %v", err))
		return
	}
	defer resp.Body.Close()

	// 9. Decode cloud response (plain JSON — just a string and an error field)
	var cloudResp ComputeResponse
	if err := json.NewDecoder(resp.Body).Decode(&cloudResp); err != nil {
		respondProcessError(w, fmt.Sprintf("bad cloud response: %v", err))
		return
	}
	if cloudResp.Error != "" {
		respondProcessError(w, fmt.Sprintf("cloud error: %s", cloudResp.Error))
		return
	}

	log.Printf("Hospital: risk score received from cloud: %s", cloudResp.InnerProduct)

	// 10. Return result to Python client
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(ProcessResponse{RiskScore: cloudResp.InnerProduct})
}

// ── Helpers ───────────────────────────────────────────────────────────────────

// int64SliceToVector converts []int64 to a data.Vector (which is []big.Int
// wrapped by data.NewVector). This is the ONLY correct way to build vectors
// for fullysec functions.
func int64SliceToVector(vals []int64) data.Vector {
	bigInts := make([]*big.Int, len(vals))
	for i, v := range vals {
		bigInts[i] = big.NewInt(v)
	}
	return data.NewVector(bigInts)
}

// currentEngineLength tracks the vector length the engine was last
// initialised with, so we can lazily reinitialise when a different-length
// request arrives.
var currentEngineLength int

func engineLength(_ *fullysec.Damgard) int {
	return currentEngineLength
}

func respondProcessError(w http.ResponseWriter, msg string) {
	log.Printf("Hospital ERROR: %s", msg)
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(http.StatusInternalServerError)
	json.NewEncoder(w).Encode(ProcessResponse{Error: msg})
}

// ── main ──────────────────────────────────────────────────────────────────────

func main() {
	// We pass l=1 as a placeholder; the engine is lazily re-initialised
	// per request if the vector length changes (see processHandler step 3).
	initHospital(1)

	http.HandleFunc("/process", processHandler)
	http.HandleFunc("/health", func(w http.ResponseWriter, r *http.Request) {
		w.Write([]byte(`{"status":"hospital node running"}`))
	})

	log.Println("Hospital node listening on :8080")
	if err := http.ListenAndServe(":8080", nil); err != nil {
		log.Fatalf("Hospital: server error: %v", err)
	}
}