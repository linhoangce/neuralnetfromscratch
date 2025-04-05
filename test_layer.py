from nnfs.datasets import spiral_data
import dense_layer

# Create dataset
X, y = spiral_data(samples=100, classes=3)

# Create Dense Layer with 2 input features and 3 output values
dense1 = dense_layer.Layer_Dense(2, 3)

# Perform a forward pass of our training data through this layer
dense1.forward(X)

print(dense1.output[:5])
