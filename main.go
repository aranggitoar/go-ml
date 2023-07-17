package main

import (
	"aranggitoar/go-ml/ann"
	m "aranggitoar/go-ml/matrix"
	"bufio"
	"fmt"
	"os"
	"strconv"
	"strings"
	"time"
)

func CsvToArrayofArrays[T m.PossibleMatrixTypes](pathToFile string, labelIndex int) (m.Matrix[T], []T) {
	var data m.Matrix[T]
	var labels []T

	// Get file
	file, err := os.Open(pathToFile)
	if err != nil {
		fmt.Println(err)
	}
	defer file.Close()

	// Scan the file line by line
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
		convertedSplitLine := []T{}
		for i, v := range splitLine {
			j, err := strconv.ParseFloat(v, 8)
			if err != nil {
				panic(err)
			}

			if i == labelIndex {
				labels = append(labels, T(j))
			} else {
				convertedSplitLine = append(convertedSplitLine, T(j))
			}
		}

		// Append the array into the main data array
		data.Value = append(data.Value, convertedSplitLine)
	}

	if err := scanner.Err(); err != nil {
		fmt.Println(err)
	}

	return data, labels
}

func main() {
	fmt.Println("Loading data ...")
	data, labels := CsvToArrayofArrays[float64]("data/digit-recognizer/train.csv", 0)

	// Length of development
	devLen := 1000

	originalRow, _ := data.Shape()

	// fmt.Println("data shape:", data.SprintfShape())

	// Shuffle the data and labels in place.
	fmt.Println("Shuffling data ...")
	data.Shuffle(labels)

	fmt.Println("Splitting data ...")
	// var dataDev m.Matrix[float64]
	// dataDev.Value = data.Value[0:devLen]
	// dataDev = dataDev.Transpose()

	// YDev := labels[0:devLen]
	// var XDev m.Matrix[float64]
	// XDev = dataDev

	// fmt.Println("dataDev shape:", dataDev.SprintfShape())
	// fmt.Println("YDev (labels) shape:", len(YDev))
	// fmt.Println("XDev shape:", XDev.SprintfShape())

	var dataTrain m.Matrix[float64]
	dataTrain.Value = data.Value[devLen:originalRow]
	dataTrain = dataTrain.Transpose()

	YTrain := labels[devLen:originalRow]
	var XTrain m.Matrix[float64]
	XTrain = dataTrain

	// fmt.Println("dataTrain shape:", dataTrain.SprintfShape())
	// fmt.Println("YTrain (labels) shape:", len(YTrain))
	// fmt.Println("XTrain shape:", XTrain.SprintfShape())

	var simpleANN ann.SimpleANN[float64]

	start := time.Now()

	alpha := 0.10
	iterations := 50
	simpleANN = simpleANN.GradientDescent(XTrain, YTrain, alpha, iterations)

	elapsed := time.Since(start)
	fmt.Printf("%v iterations of Gradient Descent took %s.\n", iterations, elapsed)
}
