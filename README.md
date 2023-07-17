# Machine Learning in Go

**WORK IN PROGRESS**

This is an attempt to learn machine learning and the necessary mathematics
by way of coding.

The plan is to build a simple machine learning library using only standard
libraries and refine it as more complex usage arises.

## Functionalities
Math related:
- Handling ONLY 2D matrixes (for now)
- Matrix transposition
- Simple matrix arithmetics, scalar-matrix arithmetics and matrix dot
  product
- Matrix initializations with random numbers and zeros
- Retrieval of highest and lowest value inside a matrix
- Matrix shuffling
- Get matrix size and shape (as either integers or formatted string)

ANN related:
- Handling simple neuron layers consisting of weights and biases
- Initialize a simple ANN consisting of one input, hidden, and output
  layer
- Do Gradient Descent iterative optimization algorithm on a dataset

Data encoding related:
- Do one-hot encoding on the data's labels
