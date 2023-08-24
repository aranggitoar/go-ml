// TODO: Simplify matrix initialization functions and basic arithmetic
// operation functions as they have repeating codes.

package matrix

import (
	"fmt"
	"math/rand"

	"gonum.org/v1/gonum/mat"
)

// Initialize matrix of dim/shape r x c with minimum and maximum
// values of min and max respectively.
func RandMatrix(r, c int, min, max float64) *mat.Dense {
	data := make([]float64, r*c)

	for i := range data {
		data[i] = min + rand.Float64()*(max-min)
	}

	return mat.NewDense(r, c, data)
}

// Broadcast a single column src matrix into the trg matrix.
// The use case are element-wise operations of trg matrix with single
// column and src matrix with many columns.
func Broadcast(src, trg mat.Dense) *mat.Dense {
	trgRow, trgColumn := trg.Dims()
	broadcastedMatrix := *mat.NewDense(trgRow, trgColumn, nil)
	for i := 0; i < trgRow; i++ {
		elem := src.At(i, 0)
		var rowContent []float64
		for j := 0; j < trgColumn; j++ {
			rowContent = append(rowContent, elem)
		}
		broadcastedMatrix.SetRow(i, rowContent)
	}

	return &broadcastedMatrix
}

// Shuffle the data matrix with the labels in place.
func Shuffle(matrix mat.Dense, labels []float64) {
	row, _ := matrix.Dims()

	var tmpMatRow [][]float64
	for i := 0; i < row; i++ {
		tmpMatRow = append(tmpMatRow, matrix.RawRowView(i))
	}

	for i := 0; i < row; i++ {
		j := rand.Intn(i + 1)
		tmpMatRow[i], tmpMatRow[j] = tmpMatRow[j], tmpMatRow[i]
		labels[i], labels[j] = labels[j], labels[i]
	}

	for i := 0; i < row; i++ {
		matrix.SetRow(i, tmpMatRow[i])
	}
}

/*
STANDARD LIBRARY ONLY
*/

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
func (m *Matrix[T]) scaOp(num T, op int) Matrix[T] {
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

func (m *Matrix[T]) ScalarAdd(num T) Matrix[T] {
	return m.scaOp(num, 0)
}

func (m *Matrix[T]) ScalarSub(num T) Matrix[T] {
	return m.scaOp(num, 1)
}

func (m *Matrix[T]) ScalarMul(num T) Matrix[T] {
	return m.scaOp(num, 2)
}

func (m *Matrix[T]) ScalarDiv(num T) Matrix[T] {
	return m.scaOp(num, 3)
}

// As there can be only one output, failed matrix dot product operation
// would result in a matrix of [[0.0]].
func (leftMatrix *Matrix[T]) Dot(rightMatrix Matrix[T]) Matrix[T] {
	var dotProduct Matrix[T]
	leftRow, leftColumn := leftMatrix.Shape()
	rightRow, rightColumn := rightMatrix.Shape()
	fmt.Println(leftMatrix.SprintfShape())
	fmt.Println(rightMatrix.SprintfShape())

	// If dot product can be done, which means:
	// - Row of right matrix is equal to column of left matrix.
	if rightRow == leftColumn {
		for i := 0; i < leftRow; i++ {
			newRow := []T{}
			for j := 0; j < rightColumn; j++ {
				var newItem T = 0.0

				// For each x[i][:] and y[:][i]
				for k := 0; k < leftColumn; k++ {
					// fmt.Println(leftMatrix.Value[i][k] * rightMatrix.Value[k][i])
					newItem = newItem + (leftMatrix.Value[i][k] * rightMatrix.Value[k][i])

				}
				// fmt.Println(newItem)

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
func (leftMatrix *Matrix[T]) opElem(rightMatrix Matrix[T], op int) Matrix[T] {
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
					switch op {
					case 0:
						newRow = append(newRow, leftMatrix.Value[i][j]+rightMatrix.Value[i][0])
					case 1:
						newRow = append(newRow, leftMatrix.Value[i][j]-rightMatrix.Value[i][0])
					case 2:
						newRow = append(newRow, leftMatrix.Value[i][j]*rightMatrix.Value[i][0])
					case 3:
						newRow = append(newRow, leftMatrix.Value[i][j]/rightMatrix.Value[i][0])
					}
				}
				result.Value = append(result.Value, newRow)
			}
		}

		if rightColumn == leftColumn {
			for i := 0; i < leftRow; i++ {
				newRow := []T{}
				for j := 0; j < leftColumn; j++ {
					switch op {
					case 0:
						newRow = append(newRow, leftMatrix.Value[i][j]+rightMatrix.Value[i][j])
					case 1:
						newRow = append(newRow, leftMatrix.Value[i][j]-rightMatrix.Value[i][j])
					case 2:
						newRow = append(newRow, leftMatrix.Value[i][j]*rightMatrix.Value[i][j])
					case 3:
						newRow = append(newRow, leftMatrix.Value[i][j]/rightMatrix.Value[i][j])

					}
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

func (m *Matrix[T]) AddElem(matrix Matrix[T]) Matrix[T] {
	return m.opElem(matrix, 0)
}

func (m *Matrix[T]) SubElem(matrix Matrix[T]) Matrix[T] {
	return m.opElem(matrix, 1)
}

func (m *Matrix[T]) MulElem(matrix Matrix[T]) Matrix[T] {
	return m.opElem(matrix, 2)
}

func (m *Matrix[T]) DivElem(matrix Matrix[T]) Matrix[T] {
	return m.opElem(matrix, 3)
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
