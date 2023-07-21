// Package ANN contains Artificial Neural Networks common operations.

package ann

import (
	"gonum.org/v1/gonum/mat"
)

// type NeuronLayers[T m.PossibleMatrixTypes] struct {
// 	Weights m.Matrix[T]
// 	Biases  m.Matrix[T]
// }

type NeuronLayer struct {
	Weights mat.Dense
	Biases  mat.Dense
}

// Simple two layer ANN.
type SimpleANN struct {
	NL1 NeuronLayer
	NL2 NeuronLayer
	// Deviation values for N1 and N2 weights and biases correction in
	// backpropagation. It is deactivated for now as it doesn't seem to be
	// necessary to be included in the base struct.
	// dNL1 NeuronLayers[T]
	// dNL2 NeuronLayers[T]
}
