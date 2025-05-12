# Multilayer Perceptron (MLP) Implementation

This project implements a simple Multilayer Perceptron (MLP) neural network from scratch using Python. The implementation demonstrates the basic concepts of neural networks including forward propagation, backpropagation, and gradient descent.

## Project Structure

- `MLP.py`: Main implementation file containing the neural network code
- `XLNNTN.pdf`: Documentation file (in Vietnamese)

## Features

- Implementation of a 2-layer neural network (1 hidden layer)
- Sigmoid activation function
- Backpropagation algorithm
- Gradient descent optimization
- XOR problem demonstration

## Network Architecture

- Input Layer: 2 neurons
- Hidden Layer: 2 neurons
- Output Layer: 1 neuron
- Activation Function: Sigmoid

## Usage

To run the neural network:

```bash
python MLP.py
```

The program will:
1. Train the network on the XOR problem
2. Display training progress every 1000 epochs
3. Show final predictions for all input combinations

## Implementation Details

The implementation includes:
- Random weight initialization
- Forward propagation
- Backpropagation
- Weight updates using gradient descent
- Mean squared error loss function

## Requirements

- Python 3.x
- No external dependencies required (uses only standard library)

## Learning Parameters

- Learning Rate: 0.1
- Number of Epochs: 10000
- Training Data: XOR problem (4 input-output pairs) 