// Package pprint contains matrix pretty printing codes.

package pprint

import (
	"fmt"

	"gonum.org/v1/gonum/mat"
)

func MatPrint(X mat.Matrix) {
	fa := mat.Formatted(X, mat.Prefix(""), mat.Squeeze())
	fmt.Printf("%v\n", fa)
}
