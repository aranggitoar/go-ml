// Package ANN contains Artificial Neural Networks common operations.

package ann

import m "aranggitoar/go-ml/matrix"

type Neuron[T m.PossibleMatrixTypes] struct {
	Weights m.Matrix[T]
	Biases  m.Matrix[T]
}

// Simple two layer ANN.
type SimpleANN[T m.PossibleMatrixTypes] struct {
	N1 Neuron[T]
	N2 Neuron[T]
	// Deviation values for N1 and N2 weights and biases correction in
	// backpropagation. It is deactivated for now as it doesn't seem to be
	// necessary to be included in the base struct.
	// dN1 Neuron[T]
	// dN2 Neuron[T]
}
