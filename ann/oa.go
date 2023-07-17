// OA: Optimization Algorithm

package ann

import (
	"aranggitoar/go-ml/de"
	m "aranggitoar/go-ml/matrix"
	"fmt"
)

func (sann *SimpleANN[T]) Initialize(inputUnits int, outputUnits int,
	minInitValue float64, maxInitValue float64) {

	sann.NL1.Weights.RandMatrix(outputUnits, inputUnits, T(minInitValue),
		T(maxInitValue))
	// W1Row is the row of b1 as the shape of the result of W1 X is row of
	// W1 and column of X.
	sann.NL1.Biases.RandMatrix(outputUnits, 1, T(minInitValue),
		T(maxInitValue))
	sann.NL2.Weights.RandMatrix(outputUnits, outputUnits, T(minInitValue),
		T(maxInitValue))
	sann.NL2.Biases.RandMatrix(outputUnits, 1, T(minInitValue),
		T(maxInitValue))

	// sann.dNL1.Weights.ZerosMatrix(outputUnits, inputUnits)
	// sann.dNL1.Biases.ZerosMatrix(outputUnits, 1)
	// sann.dNL2.Weights.ZerosMatrix(outputUnits, outputUnits)
	// sann.dNL2.Biases.ZerosMatrix(outputUnits, 1)
}

func (sann *SimpleANN[T]) Update(dW1 m.Matrix[T], db1 T, dW2 m.Matrix[T], db2 T, alpha T) {
	sann.NL1.Weights = sann.NL1.Weights.ScaOp(alpha, 1)
	sann.NL1.Weights = sann.NL1.Weights.Dot(dW1)

	sann.NL1.Biases = sann.NL1.Biases.ScaOp(alpha, 1)
	sann.NL1.Biases = sann.NL1.Biases.ScaOp(db1, 2)

	sann.NL2.Weights = sann.NL2.Weights.ScaOp(alpha, 1)
	sann.NL2.Weights = sann.NL2.Weights.Dot(dW2)

	sann.NL2.Biases = sann.NL2.Biases.ScaOp(alpha, 1)
	sann.NL2.Biases = sann.NL2.Biases.ScaOp(db1, 2)
}

func (sann *SimpleANN[T]) ForwardPropagation(X m.Matrix[T]) (m.Matrix[T],
	m.Matrix[T], m.Matrix[T], m.Matrix[T]) {
	_, W1Column := sann.NL1.Weights.Shape()
	XRow, _ := X.Shape()

	if W1Column != XRow {
		return X, X, X, X
	}

	Z1 := sann.NL1.Weights.Dot(X)
	Z1 = Z1.Add(sann.NL1.Biases)
	A1 := ReLU(Z1)
	Z2 := sann.NL2.Weights.Dot(A1)
	Z2 = Z2.Add(sann.NL2.Biases)
	A2 := Softmax(A1)

	return Z1, A1, Z2, A2
}

func (sann *SimpleANN[T]) BackPropagation(Z1 m.Matrix[T], A1 m.Matrix[T],
	Z2 m.Matrix[T], A2 m.Matrix[T], X m.Matrix[T], Y []T) (m.Matrix[T], T, m.Matrix[T], T) {
	oneHotY := de.OneHot(Y)
	Y_len := T(len(Y))

	dZ2 := A2.Subtract(oneHotY)

	dW2 := dZ2.Dot(A1.Transpose())
	dW2 = dW2.ScaOp(1/Y_len, 2)
	db2 := 1 / T(len(Y)) * dZ2.Sum()

	dZ1 := sann.NL2.Weights.Transpose()
	dZ1 = dZ1.Dot(dZ2)
	dZ1 = dZ1.Dot(ReLUDerivation(Z1))

	dW1 := dZ1.Dot(X.Transpose())
	dW1 = dW1.ScaOp(1/Y_len, 2)
	db1 := 1 / Y_len * dZ1.Sum()

	return dW1, db1, dW2, db2
}

// Compute Gradient Descent for X (data) of Y (labels).
func (sann *SimpleANN[T]) GradientDescent(X m.Matrix[T], Y []T, alpha T,
	iterations int) SimpleANN[T] {
	Xrow, _ := X.Shape()

	uniqueFound := map[T]bool{}
	unique := []T{}

	for e := range Y {
		if uniqueFound[Y[e]] != true {
			uniqueFound[Y[e]] = true
			unique = append(unique, Y[e])
		}
	}

	sann.Initialize(Xrow, len(unique), 0.5, -0.5)

	fmt.Printf("\nGradient Descent training starts ...\n")
	for i := 0; i < iterations; i++ {
		Z1, A1, Z2, A2 := sann.ForwardPropagation(X)
		dW1, db1, dW2, db2 := sann.BackPropagation(Z1, A1, Z2, A2, X, Y)
		sann.Update(dW1, db1, dW2, db2, alpha)
		if i%10 == 0 {
			fmt.Println("Iteration: ", i)
			// predictions := GetPredictions(A2)
			// fmt.Println(GetAccuracy(predictions, Y))
		}
	}

	return *sann
}
