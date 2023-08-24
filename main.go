package main

import (
	"aranggitoar/go-ml/ann"
	m "aranggitoar/go-ml/matrix"
	"aranggitoar/go-ml/pprint"
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

	// Shuffle the data and labels in place.
	fmt.Println("Shuffling data ...")
	m.Shuffle(data, labels)

	fmt.Println("Splitting data ...")
	dataDev := data.Slice(0, devLen, 0, originalColumn)
	dataDev = dataDev.T()

	YDev := labels[0:devLen]
	XDev := *mat.DenseCopyOf(dataDev)

	fmt.Println("YDev (labels) shape:", len(YDev))
	XDevRow, XDevColumn := XDev.Dims()
	fmt.Println("XDev shape:", XDevRow, XDevColumn)

	dataTrain := data.Slice(devLen, originalRow, 0, originalColumn)
	dataTrain = dataTrain.T()

	YTrain := labels[devLen:originalRow]
	XTrain := *mat.DenseCopyOf(dataTrain)
	// fmt.Println(&dataTrain)
	// fmt.Println(XTrain)

	fmt.Println("YTrain (labels) shape:", len(YTrain))
	XTrainRow, XTrainColumn := XTrain.Dims()
	fmt.Println("XDev shape:", XTrainRow, XTrainColumn)

	var simpleANN ann.SimpleANN

	start := time.Now()

	alpha := 0.10
	iterations := 10
	simpleANN = simpleANN.GradientDescent(XTrain, YTrain, alpha, iterations)

	elapsed := time.Since(start)
	fmt.Printf("%v iterations of Gradient Descent took %s.\n", iterations, elapsed)
}

func AltMain() {
	data := []float64{-0.40980568674778084, -0.4478438036577872,
		-0.16848026470884153, 0.32873835641207383, 0.2771143330935003,
		-0.16497744529957276,

		-0.07715235103471674, -0.38723508934888284, -0.38453476977269846,
		0.2825873333948403, -0.29655332487065705, 0.006759313570438574,

		-0.03845435769594835, 0.2322391409205532, 0.10449514611616639,
		-0.10874236928168457, 0.31469110186698057, -0.4505522724658986,

		-0.2562325136554301, 0.4992344415903416, 0.3761251097325716,
		0.29287988516252195, -0.07927968825554949, -0.1742563231562701}

	// x := m.RandMatrix(4, 6, -0.5, 0.5)
	x := mat.NewDense(4, 6, data)
	y := x.T()
	// y := m.RandMatrix(6, 100, 0, 256)

	var z mat.Dense

	pprint.MatPrint(x)

	// xSoft := ann.Softmax(*x)

	// pprint.MatPrint(&xSoft)
	pprint.MatPrint(y)

	z.Product(x, y)

	pprint.MatPrint(&z)

	// z.Mul(x, y)
	// pprint.MatPrint(x)
	// fmt.Println()
	// pprint.MatPrint(y)
	// fmt.Println()
	// pprint.MatPrint(&z)
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
