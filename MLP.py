import math
import random

# Sigmoid activation function and its derivative
def sigmoid(x):
    return 1 / (1 + math.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

# XOR inputs and outputs
X = [[0, 0], [0, 1], [1, 0], [1, 1]]
Y = [0, 1, 1, 0]

# Define the structure of the neural network
input_layer_neurons = 2  # Number of features (x1, x2)
hidden_layer_neurons = 2  # Number of neurons in the hidden layer
output_neurons = 1  # Single output neuron

# Randomly initialize weights and biases
weights_input_hidden = [[random.uniform(-1, 1) for _ in range(hidden_layer_neurons)] for _ in range(input_layer_neurons)] # 2x2 matrix for input to hidden
weights_hidden_output = [random.uniform(-1, 1) for _ in range(hidden_layer_neurons)] # 1x2 matrix for hidden to output
bias_hidden = [random.uniform(-1, 1) for _ in range(hidden_layer_neurons)] # Bias for the hidden layer
bias_output = random.uniform(-1, 1) # Bias for the output layer

# Learning rate
learning_rate = 0.1

# Training loop
epochs = 10000
for epoch in range(epochs):
    total_loss = 0
    for i in range(len(X)):
        # Forward pass
        x1, x2 = X[i]
        
        # Hidden layer calculations
        hidden_input = [
            x1 * weights_input_hidden[0][0] + x2 * weights_input_hidden[1][0] + bias_hidden[0],
            x1 * weights_input_hidden[0][1] + x2 * weights_input_hidden[1][1] + bias_hidden[1]
        ]
        hidden_output = [sigmoid(hidden_input[0]), sigmoid(hidden_input[1])]
        
        # Output layer calculations
        final_input = hidden_output[0] * weights_hidden_output[0] + hidden_output[1] * weights_hidden_output[1] + bias_output
        final_output = sigmoid(final_input)
        
        # Calculate the loss (mean squared error)
        error = Y[i] - final_output
        total_loss += error ** 2
        
        # Backpropagation
        # Calculate gradients for the output layer
        d_output = error * sigmoid_derivative(final_output)
        
        # Calculate gradients for the hidden layer
        d_hidden = [
            d_output * weights_hidden_output[0] * sigmoid_derivative(hidden_output[0]),
            d_output * weights_hidden_output[1] * sigmoid_derivative(hidden_output[1])
        ]
        
        # Update weights and biases
        # Update weights for hidden to output layer
        weights_hidden_output[0] += learning_rate * d_output * hidden_output[0]
        weights_hidden_output[1] += learning_rate * d_output * hidden_output[1]
        bias_output += learning_rate * d_output
        
        # Update weights for input to hidden layer
        weights_input_hidden[0][0] += learning_rate * d_hidden[0] * x1
        weights_input_hidden[1][0] += learning_rate * d_hidden[0] * x2
        weights_input_hidden[0][1] += learning_rate * d_hidden[1] * x1
        weights_input_hidden[1][1] += learning_rate * d_hidden[1] * x2
        bias_hidden[0] += learning_rate * d_hidden[0]
        bias_hidden[1] += learning_rate * d_hidden[1]

    if (epoch + 1) % 1000 == 0:
        print(f'Epoch {epoch + 1}, Loss: {total_loss / len(X):.4f}')

# Testing the model
print("\nPredictions:")
for i in range(len(X)):
    x1, x2 = X[i]
    hidden_input = [
        x1 * weights_input_hidden[0][0] + x2 * weights_input_hidden[1][0] + bias_hidden[0],
        x1 * weights_input_hidden[0][1] + x2 * weights_input_hidden[1][1] + bias_hidden[1]
    ]
    hidden_output = [sigmoid(hidden_input[0]), sigmoid(hidden_input[1])]
    final_input = hidden_output[0] * weights_hidden_output[0] + hidden_output[1] * weights_hidden_output[1] + bias_output
    final_output = sigmoid(final_input)
    print(f"Input: {X[i]}, Predicted Output: {round(final_output)}")
