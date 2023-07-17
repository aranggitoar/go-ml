// Package ANN contains Artificial Neural Networks common operations.

package ann

import m "aranggitoar/go-ml/matrix"

type NeuronLayers[T m.PossibleMatrixTypes] struct {
	Weights m.Matrix[T]
	Biases  m.Matrix[T]
}

// Simple two layer ANN.
type SimpleANN[T m.PossibleMatrixTypes] struct {
	NL1 NeuronLayers[T]
	NL2 NeuronLayers[T]
	// Deviation values for N1 and N2 weights and biases correction in
	// backpropagation. It is deactivated for now as it doesn't seem to be
	// necessary to be included in the base struct.
	// dNL1 NeuronLayers[T]
	// dNL2 NeuronLayers[T]
}
