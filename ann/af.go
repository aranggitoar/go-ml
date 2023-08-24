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

// Computes the Softmax of each element, which is the result of exponent
// of an element divided by the sum of exponents of all elements of the
// original elements' column.
func Softmax(matrix mat.Dense) mat.Dense {
	// matrix = *LSE(matrix)
	// fmt.Println(matrix)
	maxRow, maxCol := matrix.Dims()

	// As indexes starts from 0
	maxRow = maxRow - 1
	maxCol = maxCol - 1

	minVal := mat.Min(&matrix)
	fn := func(r, c int, v float64) float64 {
		return math.Exp(v / minVal)
	}

	matrix.Apply(fn, &matrix)

	// lse := LSE(matrix).RawRowView(0)

	// Will be as many as 41000
	var divisorsByRow []float64

	for i := 0; i < maxCol+1; i++ {
		elems := matrix.ColView(i)
		divisorsByRow = append(divisorsByRow, mat.Sum(elems))
	}

	// fmt.Println(len(divisorsByRow))
	// fmt.Println(divisorsByRow)

	fn = func(r, c int, v float64) float64 {
		// return v / lse[c]
		return (v / divisorsByRow[c]) * minVal
	}

	matrix.Apply(fn, &matrix)

	return matrix
}

// Computes the Log-Sum-Exp or RealSoftMax of a matrix.
func LSE(matrix mat.Dense) *mat.Dense {
	// Get the number of rows and columns in the matrix
	rows, cols := matrix.Dims()

	// Create a new matrix to store the adjusted matrix
	adjusted := mat.NewDense(rows, cols, nil)

	// Find the maximum value in each column
	for j := 0; j < cols; j++ {
		maxValue := matrix.At(0, j)
		for i := 1; i < rows; i++ {
			value := matrix.At(i, j)
			if value > maxValue {
				maxValue = value
			}
		}

		// Subtract the maximum value from each element in the corresponding column
		for i := 0; i < rows; i++ {
			adjusted.Set(i, j, matrix.At(i, j)-maxValue)
			// adjusted.Set(i, j, matrix.At(i, j))
		}
	}

	// Exponentiate the adjusted matrix
	// exp := mat.NewDense(rows, cols, nil)
	// exp.Apply(func(i, j int, v float64) float64 {
	// 	return math.Exp(v)
	// }, adjusted)

	// Compute the column-wise sum of the exponentiated matrix
	colSum := mat.NewDense(1, cols, nil)
	colSum.Apply(func(i, j int, v float64) float64 {
		// return mat.Sum(exp.ColView(j))
		return mat.Sum(adjusted.ColView(j))
	}, colSum)

	// Take the logarithm of the column-wise sum
	// logColSum := mat.NewDense(1, cols, nil)
	// logColSum.Apply(func(i, j int, v float64) float64 {
	// 	return math.Log(v)
	// }, colSum)

	// fmt.Println(logColSum)

	// Create a new matrix to store the final result
	result := mat.NewDense(1, cols, nil)

	// Add the maximum value back to the result
	for j := 0; j < cols; j++ {
		maxValue := matrix.At(0, j)
		for i := 1; i < rows; i++ {
			value := matrix.At(i, j)
			if value > maxValue {
				maxValue = value
			}
		}
		// result.Set(0, j, logColSum.At(0, j)+maxValue)
		result.Set(0, j, colSum.At(0, j)+maxValue)
	}

	return result
}
