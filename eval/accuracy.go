package eval

import (
	"gonum.org/v1/gonum/mat"
)

// Iterate rows and columns of a matrix and get the highest value column
// of each row.
func GetPredictions(matrix mat.Dense) []int {
	maxRow, _ := matrix.Dims()
	var arrayOfIndexes []int

	row := 0
	var highest float64
	columnIndexOfHighest := 0
	fn := func(r, c int, v float64) float64 {
		if v > highest {
			highest = v
			columnIndexOfHighest = c
		}
		if row != r || maxRow == r {
			arrayOfIndexes = append(arrayOfIndexes, columnIndexOfHighest)
		}
		row = r

		return v
	}

	matrix.Apply(fn, &matrix)

	return arrayOfIndexes
}

func GetAccuracy(predictions []int, Y []float64) float64 {
	var score float64

	for i := 0; i < len(predictions); i++ {
		// fmt.Println(predictions[i], Y[i])
		if predictions[i] == int(Y[i]) {
			score = score + 1
		}
	}

	// fmt.Println(score)
	// fmt.Println(len(Y))
	return score / float64(len(Y))
}
