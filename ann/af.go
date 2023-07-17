// AF: Activation Functions

package ann

import (
	m "aranggitoar/go-ml/matrix"
	"math"
)

// Computes the Rectified Linear Unit of each element in a 2D matrix,
// choose 0 if number is lower than 0 and the number itself if its more
// than 0.
func ReLU[T m.PossibleMatrixTypes](matrix m.Matrix[T]) m.Matrix[T] {
	var newMatrix m.Matrix[T]
	row, column := matrix.Shape()

	for i := 0; i < row; i++ {
		newRow := []T{}
		for j := 0; j < column; j++ {
			// math.Max doesn't accept generic types, conversion to float64 of
			// matrix.Value is a hack to get around that and keep the values.
			newRow = append(newRow, T(math.Max(0, float64(matrix.Value[i][j]))))
		}
		newMatrix.Value = append(newMatrix.Value, newRow)
	}

	return newMatrix
}

// Computes the derivation of ReLU of a matrix and returns a new matrix.
func ReLUDerivation[T m.PossibleMatrixTypes](matrix m.Matrix[T]) m.Matrix[T] {
	var newMatrix m.Matrix[T]
	row, column := matrix.Shape()

	for i := 0; i < row; i++ {
		newRow := []T{}
		for j := 0; j < column; j++ {

			moreThanZero := matrix.Value[i][j] > 0
			var newValue T
			if moreThanZero {
				newValue = 1
			} else {
				newValue = 0
			}

			newRow = append(newRow, newValue)
		}
		newMatrix.Value = append(newMatrix.Value, newRow)
	}

	return newMatrix
}

func Softmax[T m.PossibleMatrixTypes](matrix m.Matrix[T]) m.Matrix[T] {
	var newMatrix m.Matrix[T]
	row, column := matrix.Shape()

	for i := 0; i < row; i++ {
		newRow := []T{}
		dividends := []T{}
		var divisor T = 0.0

		for j := 0; j < column; j++ {
			// math.Exp doesn't accept generic types, conversion to float64 of
			// matrix.Value is a hack to get around that and keep the values.
			currentItem := T(math.Exp(float64(matrix.Value[i][j])))
			// Summing all of the individual exponents.
			divisor = divisor + currentItem

			// Gathering all of the individual exponents.
			dividends = append(dividends, (currentItem))
		}

		for j := 0; j < len(dividends); j++ {
			// Divide each individual exponents by the sum of the exponents.
			newRow = append(newRow, dividends[j]/divisor)
		}

		newMatrix.Value = append(newMatrix.Value, newRow)
	}

	return newMatrix
}
