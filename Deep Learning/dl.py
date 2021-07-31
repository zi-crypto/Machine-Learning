"""
DL Library for Deep Learning.
- Create Your Own Neural Network
- Choose the number of Neurons and Layers!!
..This Module need NumPy Library.
"""
import numpy as np

class Layer():
    def __init__(self, n_inputs, n_neurons) -> None:
        self.weights = 0.10 * np.random.randn(n_inputs, n_neurons)
        self.biases  = np.zeros((1, n_neurons))

    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases
