// DE: Data Encoding

package de

import (
	"gonum.org/v1/gonum/mat"
)

// Encode an array of labels into a 2D array with each row representing
// each label.
// TODO: Find out if the type constraint should only accept matrix of int.
func OneHot(data []float64) mat.Matrix {
	var maxVal float64
	for i := 0; i < len(data); i++ {
		if data[i] > maxVal {
			maxVal = data[i]
		}
	}

	oh := mat.NewDense(len(data), int(maxVal+1), nil)

	// Go through each row and write 1 in every column with the number that
	// the original data has.
	for i := 0; i < len(data); i++ {
		oh.Set(i, int(data[i]), 1)
	}

	return oh.T()
}
