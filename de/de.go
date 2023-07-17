// DE: Data Encoding

package de

import m "aranggitoar/go-ml/matrix"

// Encode an array of labels into a 2D array with each row representing
// each label.
// TODO: Find out if the type constraint should only accept matrix of int.
func OneHot[T m.PossibleMatrixTypes](data []T) m.Matrix[T] {
	var oh m.Matrix[T]

	var maxVal T
	for i := 0; i < len(data); i++ {
		if data[i] > maxVal {
			maxVal = data[i]
		}
	}

	oh.ZerosMatrix(len(data), int(maxVal+1))

	// Go through each row and write 1 in every column with the number that
	// the original data has.
	for i := 0; i < len(data); i++ {
		oh.Value[i][int(data[i])] = 1
	}

	return oh.Transpose()
}
