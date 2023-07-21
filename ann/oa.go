// OA: Optimization Algorithm

package ann

import (
	"aranggitoar/go-ml/de"
	"aranggitoar/go-ml/eval"
	m "aranggitoar/go-ml/matrix"
	"fmt"

	"gonum.org/v1/gonum/mat"
)

func (sann *SimpleANN) Initialize(inputUnits int, outputUnits int,
	minInitValue float64, maxInitValue float64) {

	sann.NL1.Weights = *m.RandMatrix(outputUnits, inputUnits, minInitValue, maxInitValue)
	// NL1 Weights row is the row of NL1 Biases as the shape of the product
	// of NL1 Weights and X is row of NL1 Weights and column of X.
	sann.NL1.Biases = *m.RandMatrix(outputUnits, 1, minInitValue, maxInitValue)

	sann.NL2.Weights = *m.RandMatrix(outputUnits, outputUnits, minInitValue, maxInitValue)
	sann.NL2.Biases = *m.RandMatrix(outputUnits, 1, minInitValue, maxInitValue)
}

func (sann *SimpleANN) Update(dW1 mat.Dense, db1 float64, dW2 mat.Dense,
	db2, alpha float64) {
	sann.NL1.Weights.Apply(func(i, j int, v float64) float64 {
		return v - alpha
	}, &sann.NL1.Weights)
	sann.NL1.Weights.MulElem(&sann.NL1.Weights, &dW1)

	sann.NL1.Biases.Apply(func(i, j int, v float64) float64 {
		return v - alpha*db1
	}, &sann.NL1.Biases)

	sann.NL2.Weights.Apply(func(i, j int, v float64) float64 {
		return v - alpha
	}, &sann.NL2.Weights)
	sann.NL2.Weights.MulElem(&sann.NL2.Weights, &dW2)

	sann.NL2.Biases.Apply(func(i, j int, v float64) float64 {
		return v - alpha*db2
	}, &sann.NL2.Biases)
}

func (sann *SimpleANN) ForwardPropagation(X mat.Dense) (mat.Dense, mat.Dense, mat.Dense, mat.Dense) {
	var Z1 mat.Dense
	var Z2 mat.Dense

	Z1.Mul(&sann.NL1.Weights, &X)
	Z1.Add(&Z1, m.Broadcast(sann.NL1.Weights, Z1))

	A1 := *ReLU(Z1)

	Z2.Mul(&sann.NL2.Weights, &A1)
	Z2.Add(&Z2, m.Broadcast(sann.NL2.Biases, Z2))

	A2 := Softmax(A1)

	return Z1, A1, Z2, A2
}

func (sann *SimpleANN) BackPropagation(Z1, A1, Z2, A2, X mat.Dense,
	Y []float64) (mat.Dense, float64, mat.Dense, float64) {
	var dZ1 mat.Dense
	var dZ2 mat.Dense
	var dW1 mat.Dense
	var dW2 mat.Dense
	oneHotY := de.OneHot(Y)
	mult := 1 / float64(len(Y))

	dZ2.Sub(&A2, oneHotY)

	dW2.Mul(&dZ2, A1.T())
	dW2.Apply(
		func(i int, j int, v float64) float64 {
			return mult * v
		}, &dW2)

	db2 := mult * mat.Sum(&dZ2)

	dZ1.Mul(sann.NL2.Weights.T(), &dZ2)
	dZ1.MulElem(&dZ1, ReLUDerivation(Z1))

	dW1.Mul(&dZ1, X.T())
	dW1.Apply(
		func(i int, j int, v float64) float64 {
			return mult * v
		}, &dW1)

	db1 := mult * mat.Sum(&dZ1)

	return dW1, db1, dW2, db2
}

// Compute Gradient Descent for X (data) of Y (labels).
func (sann *SimpleANN) GradientDescent(X mat.Dense, Y []float64,
	alpha float64, iterations int) SimpleANN {
	XRow, _ := X.Dims()

	uniqueFound := map[float64]bool{}
	unique := []float64{}

	for e := range Y {
		if uniqueFound[Y[e]] != true {
			uniqueFound[Y[e]] = true
			unique = append(unique, Y[e])
		}
	}

	sann.Initialize(XRow, len(unique), 0.5, -0.5)

	fmt.Printf("\nGradient Descent training starts ...\n")
	for i := 0; i < iterations; i++ {
		Z1, A1, Z2, A2 := sann.ForwardPropagation(X)
		if i == 0 {
			// After the first index, its all NaN values.
			// Even the first index contains +Inf values, maybe something wrong
			// with the Activation Functions?
			fmt.Println(A2)
		}
		dW1, db1, dW2, db2 := sann.BackPropagation(Z1, A1, Z2, A2, X, Y)
		sann.Update(dW1, db1, dW2, db2, alpha)
		if i%2 == 0 {
			fmt.Println("Iteration: ", i)
			predictions := eval.GetPredictions(*mat.DenseCopyOf(A2.T()))
			result := eval.GetAccuracy(predictions, Y)
			fmt.Println("Score: ", result)
		}
	}

	return *sann
}
