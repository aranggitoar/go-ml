// AF: Activation Functions

package ann

import (
	"math"

	"gonum.org/v1/gonum/mat"
)

// Computes the Rectified Linear Unit of each element in the matrix,
// choose 0 if number is lower than 0 and the number itself if its more
// than 0.
func ReLU(matrix mat.Dense) *mat.Dense {
	fn := func(r, c int, v float64) float64 { return float64(math.Max(0, v)) }
	matrix.Apply(fn, &matrix)
	return &matrix
}

// Computes the derivation of ReLU of a matrix and returns a new matrix.
func ReLUDerivation(matrix mat.Dense) *mat.Dense {
	fn := func(r, c int, v float64) float64 {
		moreThanZero := v > 0
		if moreThanZero {
			return 1
		} else {
			return 0
		}
	}
	matrix.Apply(fn, &matrix)
	return &matrix
}

func Softmax(matrix mat.Dense) mat.Dense {
	maxRow, _ := matrix.Dims()
	maxRow = maxRow - 1 // As rows are indexed from 0 in the Apply method

	var divisorsByRow []float64

	row := 0
	var divisorTemp float64
	fn := func(r, c int, v float64) float64 {
		v = math.Exp(v)
		if row != r || maxRow == r {
			divisorsByRow = append(divisorsByRow, divisorTemp)
			divisorTemp = 0
		}
		divisorTemp = divisorTemp + v
		row = r

		return v
	}

	matrix.Apply(fn, &matrix)

	fn = func(r, c int, v float64) float64 {
		return v / divisorsByRow[r]
	}

	matrix.Apply(fn, &matrix)

	return matrix
}
