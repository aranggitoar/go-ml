// Package matrix contains matrix interface and common operations.
//
// TODO: Find a way to represent higher dimension matrices the same with
// 2D matrices. Maybe interface{} or generic type?
// TODO: Find a way to change the bitsize of integers and floats depending
// on the architecture. Ref:
// https://pkg.go.dev/internal/goarch
// https://pkg.go.dev/runtime
// TODO: Find a way to do the matrix operations concurrently, as we're
// usually dealing with tens of thousands of row/columns

package matrix

type PossibleMatrixTypes interface {
	int64 | float64
}

type MatrixInterface interface {
	Shape() []int
	SprintfShape() string
	Transpose() [][]interface{}
}

// type Matrices[T PossibleMatrixTypes] interface {
// 	[]T
// }

type Matrix[T PossibleMatrixTypes] struct {
	Value [][]T
}
