package main

import (
	"aranggitoar/go-ml/ann"
	m "aranggitoar/go-ml/matrix"
	"bufio"
	"flag"
	"fmt"
	"os"
	"strconv"
	"strings"
	"time"

	"gonum.org/v1/gonum/mat"
)

var (
	altMainFlag bool
)

func CsvToArrayofArrays(pathToFile string, labelIndex int) (mat.Dense, []float64) {
	var data mat.Dense
	var labels []float64

	// Get file
	file, err := os.Open(pathToFile)
	if err != nil {
		fmt.Println(err)
	}
	defer file.Close()

	// Scan the file line by line
	var rowArray [][]float64
	scanner := bufio.NewScanner(file)
	for scanner.Scan() {
		line := scanner.Text()

		// If line is a header, skip the line
		isHeader := strings.Contains(line, "pixel0") // Substring depends on CSV headers
		if isHeader {
			continue
		}

		// Split the line by CSV's separator (automatically converting it into
		// an array), then go through each item and convert it to the intended
		// value data type
		splitLine := strings.Split(line, ",")
		var convertedSplitLine []float64
		for i, v := range splitLine {
			j, err := strconv.ParseFloat(v, 8)
			if err != nil {
				panic(err)
			}

			if i == labelIndex {
				labels = append(labels, j)
			} else {
				convertedSplitLine = append(convertedSplitLine, j)
			}
		}

		// Append the array into the main data array
		rowArray = append(rowArray, convertedSplitLine)
	}

	if err := scanner.Err(); err != nil {
		fmt.Println(err)
	}

	data = *mat.NewDense(len(rowArray), len(rowArray[0]), nil)

	data.Apply(func(i, j int, v float64) float64 {
		return rowArray[i][j]
	}, &data)

	return data, labels
}

func Main() {
	fmt.Println("Loading data ...")
	data, labels := CsvToArrayofArrays("data/digit-recognizer/train.csv", 0)

	// Length of development
	devLen := 1000

	originalRow, originalColumn := data.Dims()

	// fmt.Println("data shape:", data.SprintfShape())

	// Shuffle the data and labels in place.
	fmt.Println("Shuffling data ...")
	m.Shuffle(data, labels)

	fmt.Println("Splitting data ...")
	// var dataDev m.Matrix[float64]
	dataDev := data.Slice(0, devLen, 0, originalColumn)
	// dataDev.Value = data.Value[0:devLen]
	dataDev = dataDev.T()
	// dataDev = dataDev.Transpose()

	YDev := labels[0:devLen]
	// var XDev m.Matrix[float64]
	XDev := dataDev

	dataDevRow, dataDevColumn := dataDev.Dims()
	fmt.Println("dataDev shape:", dataDevRow, dataDevColumn)
	fmt.Println("YDev (labels) shape:", len(YDev))
	XDevRow, XDevColumn := XDev.Dims()
	fmt.Println("XDev shape:", XDevRow, XDevColumn)

	// var dataTrain m.Matrix[float64]
	dataTrain := data.Slice(devLen, originalRow, 0, originalColumn)
	// dataTrain.Value = data.Value[devLen:originalRow]
	dataTrain = dataTrain.T()
	// dataTrain = dataTrain.Transpose()

	YTrain := labels[devLen:originalRow]
	// var XTrain m.Matrix[float64]
	XTrain := dataTrain

	dataTrainRow, dataTrainColumn := dataTrain.Dims()
	fmt.Println("dataTrain shape:", dataTrainRow, dataTrainColumn)
	fmt.Println("YTrain (labels) shape:", len(YTrain))
	XTrainRow, XTrainColumn := XTrain.Dims()
	fmt.Println("XDev shape:", XTrainRow, XTrainColumn)

	var simpleANN ann.SimpleANN

	start := time.Now()

	alpha := 0.10
	iterations := 50
	simpleANN = simpleANN.GradientDescent(*mat.DenseCopyOf(XTrain), YTrain, alpha, iterations)

	elapsed := time.Since(start)
	fmt.Printf("%v iterations of Gradient Descent took %s.\n", iterations, elapsed)
}

func AltMain() {
	x := m.RandMatrix(4, 6, -0.5, 0.5)
	// y := m.RandMatrix(10, 15, -0.5, 0.5)

	// var z mat.Dense

	// z.Mul(x, y)
	fmt.Println(x)
	fmt.Println()
	// ann.Softmax(*x)
	xRow, _ := x.Dims()
	var y []float64
	for i := 0; i < xRow; i++ {
		y = append(y, x.At(i, 0))
	}
	fmt.Println(y)
	fmt.Println()
	z := mat.NewDense(4, 6, nil)
	_, zCol := z.Dims()
	for i := 0; i < zCol; i++ {
		z.SetCol(i, y)
	}
	fmt.Println(z)
	fmt.Println()
	x.Sub(x, z)

	fmt.Println(x)
	fmt.Println()
	fmt.Println(y)
	// fmt.Println()
	// fmt.Println(z)
}

// func AltMain() {
// 	g := gorgonia.NewGraph()

// 	var x, y, z *gorgonia.Node
// 	var err error

// 	// define the expression
// 	x = gorgonia.NewScalar(g, gorgonia.Float64, gorgonia.WithName("x"))
// 	y = gorgonia.NewScalar(g, gorgonia.Float64, gorgonia.WithName("y"))
// 	if z, err = gorgonia.Add(x, y); err != nil {
// 		log.Fatal(err)
// 	}

// 	// create a VM to run the program on
// 	machine := gorgonia.NewTapeMachine(g)
// 	defer machine.Close()

// 	// set initial values then run
// 	gorgonia.Let(x, 2.0)
// 	gorgonia.Let(y, 2.5)
// 	if err = machine.RunAll(); err != nil {
// 		log.Fatal(err)
// 	}

// 	fmt.Printf("%v", z.Value())
// }

func main() {
	flag.BoolVar(&altMainFlag, "altMainFlag", false,
		"boolean flag to run AltMain() instead of Main()")
	flag.Parse()

	if altMainFlag {
		AltMain()
	}

	if !altMainFlag {
		Main()
	}
}
