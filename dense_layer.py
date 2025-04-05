import numpy as np
import nnfs
from nnfs.datasets import spiral_data

import loss_function

nnfs.init()


# Dense Layer
class Layer_Dense:

    # Layer initialization
    def __init__(self, n_inputs, n_neurons):
        self.weights = np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))

    # Forward pass
    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases


# ReLU Activation
class Activation_ReLU:

    # Forward pass
    def forward(self, inputs):
        # Calculate output values from inputs
        self.output = np.maximum(0, inputs)


# Softmax Activation
class Activation_Softmax:

    # Forward pass
    def forward(self, inputs):

        # Get unnormalized probabilities
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))

        # Normalize them for each sample
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)

        self.output = probabilities

# Create dataset
X, y = spiral_data(samples=100, classes=3)

# create Dense layer with 2 input features and 3 output values
dense1 = Layer_Dense(2, 3)

# Create ReLU activation
activation1 = Activation_ReLU()

# create second Dense layer with 3 input features (as we take output
# of previous layer here) and 3 output values
dense2 = Layer_Dense(3, 3)


# Create Softmax activation
activation2 = Activation_Softmax()

# Make a forward pass of training data throught this layer
dense1.forward(X)

# Make a forward pass through activation function
activation1.forward(dense1.output)

# Make a forward pass through second Dense layer
# it takes outputs of activation function of first layer as inputs
dense2.forward(activation1.output)

# Make a forward pass through activation function
# it takes the output of second dense layer here
activation2.forward(dense2.output)

print(activation2.output[:5])

# Perform a forward pass through the loss function
# it takes the output of second activation here and return loss
loss_function = loss_function.Loss_CategoricalCrossentropy()
loss = loss_function.calculate(activation2.output, y)

print('loss: ', loss)

# Convert one-hot encoded targets to sparse values using `np.argmax()`
predictions = np.argmax(activation2.output, axis=1)

if len(y.shape) == 2:
    y = np.argmax(y, axis=1)
    print('y', y)

accuracy = np.mean(predictions==y)

print('acc:', accuracy)

print(y)