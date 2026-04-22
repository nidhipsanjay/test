package main

// -----------------------------------------------------------------------
// types.go — Shared structs for HTTP payloads between hospital and cloud.
//
// WHY GOB AND NOT JSON?
// GoFE's cryptographic structs contain unexported fields and *big.Int 
// slices that encoding/json cannot handle. encoding/gob can serialize 
// them as long as we register the concrete types at startup.
// -----------------------------------------------------------------------

import (
	"encoding/gob"
	"math/big"

	"github.com/fentec-project/gofe/data"
	"github.com/fentec-project/gofe/innerprod/fullysec"
)

// ComputeRequest is what the Hospital node POSTs to the Cloud node.
// FIX: Ciphertext is a single vector, FeKey is a DerivedKey.
type ComputeRequest struct {
	Ciphertext data.Vector                 
	FeKey      *fullysec.DamgardDerivedKey 
	MLWeights  []*big.Int                  
}

// ComputeResponse is what the Cloud node returns to the Hospital node.
type ComputeResponse struct {
	InnerProduct string 
	Error        string
}

// ProcessRequest is what the Python client POSTs to /process on the Hospital.
type ProcessRequest struct {
	PatientRNA []int64 
	MLWeights  []int64 
}

// ProcessResponse is what the Hospital returns to the Python client.
type ProcessResponse struct {
	RiskScore string 
	Error     string
}

// RegisterGobTypes must be called once at startup in BOTH servers.
func RegisterGobTypes() {
	// FIX: We must register the DamgardDerivedKey so gob knows how to send it
	gob.Register(&fullysec.DamgardDerivedKey{})
	gob.Register(data.Vector{})
	gob.Register([]*big.Int{})
	gob.Register(new(big.Int))
}