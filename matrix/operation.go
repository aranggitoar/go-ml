// TODO: Simplify matrix initialization functions and basic arithmetic
// operation functions as they have repeating codes.

package matrix

import (
	"fmt"
	"math/rand"
)

// Initialize a 2D matrix with zero values.
func (m *Matrix[T]) ZerosMatrix(row int, column int) {
	for i := 0; i < row; i++ {
		newRow := []T{}
		for j := 0; j < column; j++ {
			newRow = append(newRow, 0)
		}
		m.Value = append(m.Value, newRow)
	}
}

// Initialize a 2D matrix with random values in a certain range.
func (m *Matrix[T]) RandMatrix(row int, column int, min T, max T) {
	for i := 0; i < row; i++ {
		newRow := []T{}
		for j := 0; j < column; j++ {

			var randElement T

			switch fmt.Sprintf("%T", min) {
			case "int64":
				// Which one would be the correct implementation?
				randElement = min + T(rand.Int63())*(max-min)
				// randElement = min + T(rand.Intn(int(max-min+1)))
			case "float64":
				randElement = min + T(rand.Float64())*(max-min)
			}

			newRow = append(newRow, randElement)
		}

		m.Value = append(m.Value, newRow)
	}
}

// Find the highest value inside the matrix.
func (m *Matrix[T]) Max() T {
	row, column := m.Shape()
	var highestVal T

	for i := 0; i < row; i++ {
		for j := 0; j < column; j++ {
			if m.Value[i][j] > highestVal {
				highestVal = m.Value[i][j]
			}
		}
	}

	return highestVal
}

// Find the lowest value inside the matrix.
func (m *Matrix[T]) Min() T {
	row, column := m.Shape()
	var lowestVal T

	for i := 0; i < row; i++ {
		for j := 0; j < column; j++ {
			if m.Value[i][j] < lowestVal {
				lowestVal = m.Value[i][j]
			}
		}
	}

	return lowestVal
}

// As we are dealing with array of arrays, "row" is basically the length
// of the outer array and "column" is basically the length of the inner
// array. Because this is a matrix operation, we assume the length
// similarity of each "row".
func (m *Matrix[T]) Transpose() Matrix[T] {
	var transposedMatrix Matrix[T]
	originalRowLen := len(m.Value)
	originalColumnLen := 0

	for i := 0; i < 1; i++ {
		originalColumnLen = len(m.Value[i])
	}

	// The transpose of an array of array is basically every j-th row of
	// i-th column appended as a new item of an array with the original row
	// length iterated as many as the original column length.
	for i := 0; i < originalColumnLen; i++ {
		newColumn := []T{}

		for j := 0; j < originalRowLen; j++ {
			newColumn = append(newColumn, m.Value[j][i])
		}

		transposedMatrix.Value = append(transposedMatrix.Value, newColumn)
	}

	return transposedMatrix
}

// Element-wise operation of a matrix with a certain number.
// Operation codes (op):
// 0. Addition
// 1. Subtraction
// 2. Multiplication
// 3. Division
func (m *Matrix[T]) ScaOp(num T, op int) Matrix[T] {
	row, column := m.Shape()
	var newMatrix Matrix[T]

	for i := 0; i < row; i++ {
		var newRow []T

		newRow = m.Value[i]

		for j := 0; j < column; j++ {
			switch op {
			case 0:
				newRow[j] = newRow[j] + num
			case 1:
				newRow[j] = newRow[j] - num
			case 2:
				newRow[j] = newRow[j] * num
			case 3:
				newRow[j] = newRow[j] / num
			}
		}

		newMatrix.Value = append(newMatrix.Value, newRow)
	}

	return newMatrix
}

// As there can be only one output, failed matrix dot product operation
// would result in a matrix of [[0.0]].
func (leftMatrix *Matrix[T]) Dot(rightMatrix Matrix[T]) Matrix[T] {
	var dotProduct Matrix[T]
	leftRow, leftColumn := leftMatrix.Shape()
	rightRow, rightColumn := rightMatrix.Shape()

	// If dot product can be done, which means:
	// - Row of right matrix is equal to column of left matrix.
	if rightRow == leftColumn {
		for i := 0; i < leftRow; i++ {
			newRow := []T{}
			for j := 0; j < rightColumn; j++ {
				var newItem T = 0.0

				// For each x[i][:] and y[:][i]
				for k := 0; k < leftColumn; k++ {
					newItem = newItem + (leftMatrix.Value[i][k] * rightMatrix.Value[k][i])

				}

				newRow = append(newRow, newItem)
			}
			dotProduct.Value = append(dotProduct.Value, newRow)
		}
	}

	// If dot product cannot be done.
	if rightRow != leftColumn {
		newRow := []T{}
		var newItem T = 0.0
		newRow = append(newRow, newItem)
		dotProduct.Value = append(dotProduct.Value, newRow)
	}

	return dotProduct
}

// Add the original matrix with the one given in the arguments. Broadcast
// the columns if possible.
// TODO: Create a test to see if column of y is 1 really is broadcasted to
// other columns of x. Example:
// [[0, 0, 0],
//
//	[0, 0, 0]]
//
// [[1],
//
//	[1]]
//
// ___________ +
// [[1, 1, 1],
//
//	[1, 1, 1]]
func (leftMatrix *Matrix[T]) Add(rightMatrix Matrix[T]) Matrix[T] {
	var result Matrix[T]
	leftRow, leftColumn := leftMatrix.Shape()
	rightRow, rightColumn := rightMatrix.Shape()

	// If addition can be done, which means:
	// - Row of x is equal to row of y, AND
	// - Column of y is one OR equal to column of x
	if leftRow == rightRow && (rightColumn == 1 || rightColumn == leftColumn) {
		if rightColumn == 1 {
			for i := 0; i < leftRow; i++ {
				newRow := []T{}
				for j := 0; j < leftColumn; j++ {
					newRow = append(newRow, leftMatrix.Value[i][j]+rightMatrix.Value[i][0])
				}
				result.Value = append(result.Value, newRow)
			}
		}

		if rightColumn == leftColumn {
			for i := 0; i < leftRow; i++ {
				newRow := []T{}
				for j := 0; j < leftColumn; j++ {
					newRow = append(newRow, leftMatrix.Value[i][j]+rightMatrix.Value[i][j])
				}
				result.Value = append(result.Value, newRow)
			}
		}
	}

	// If addition cannot be done.
	if leftRow != rightRow || (leftRow == rightRow && (rightColumn > 1 && rightColumn < leftColumn)) {
		newRow := []T{}
		var newItem T = 0.0
		newRow = append(newRow, newItem)
		result.Value = append(result.Value, newRow)
	}

	return result
}

// Subtract the original matrix with the one given in the arguments.
// Broadcast the columns if possible.
// TODO: Create a test if the broadcasting really happens too.
func (leftMatrix *Matrix[T]) Subtract(rightMatrix Matrix[T]) Matrix[T] {
	var result Matrix[T]
	leftRow, leftColumn := leftMatrix.Shape()
	rightRow, rightColumn := rightMatrix.Shape()

	// If subtraction can be done, which means:
	// - Row of x is equal to row of y, AND
	// - Column of y is one OR equal to column of x
	if leftRow == rightRow && (rightColumn == 1 || rightColumn == leftColumn) {
		if rightColumn == 1 {
			for i := 0; i < leftRow; i++ {
				newRow := []T{}
				for j := 0; j < leftColumn; j++ {
					newRow = append(newRow, leftMatrix.Value[i][j]-rightMatrix.Value[i][0])
				}
				result.Value = append(result.Value, newRow)
			}
		}

		if rightColumn == leftColumn {
			for i := 0; i < leftRow; i++ {
				newRow := []T{}
				for j := 0; j < leftColumn; j++ {
					newRow = append(newRow, leftMatrix.Value[i][j]-rightMatrix.Value[i][j])
				}
				result.Value = append(result.Value, newRow)
			}
		}
	}

	// If subtraction cannot be done.
	if leftRow != rightRow || (leftRow == rightRow && (rightColumn > 1 && rightColumn < leftColumn)) {
		newRow := []T{}
		var newItem T = 0.0
		newRow = append(newRow, newItem)
		result.Value = append(result.Value, newRow)
	}

	return result
}

// Sum the matrix element-wise.
func (m *Matrix[T]) Sum() T {
	var sum T
	row, column := m.Shape()

	for i := 0; i < row; i++ {
		for j := 0; j < column; j++ {
			sum = sum + m.Value[i][j]
		}
	}

	return sum
}

// Shuffle the data matrix with the labels in place.
func (m *Matrix[T]) Shuffle(labels []T) {
	for i := range m.Value {
		j := rand.Intn(i + 1)
		m.Value[i], m.Value[j] = m.Value[j], m.Value[i]

		labels[i], labels[j] = labels[j], labels[i]

		for k := range m.Value[i] {
			l := rand.Intn(k + 1)
			m.Value[i][k], m.Value[i][l] = m.Value[i][l], m.Value[i][k]
		}

		for k := range m.Value[j] {
			l := rand.Intn(k + 1)
			m.Value[j][k], m.Value[j][l] = m.Value[j][l], m.Value[j][k]
		}

	}
}
