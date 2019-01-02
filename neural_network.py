import numpy as np
import sys

# parameters
length_of_input = 784
number_of_hidden_layers = 2
nodes_per_hidden_layer = 128
number_of_ouputs = 10
learning_rate = 0.1

# Sigmoid activation: σ(x) = 1/(1+e^-x)
sigmoid = lambda x: 1.0 / (1.0 + np.exp(-x))
# Derrivative of sigmoid: σ′(x) = 1-σ(x)
derriv_sigmoid = lambda y: y * (1. - y)
# Squared error function. Takes in (expected, guess)
error = lambda x, y: x - y

# Creating the network
class Layer:
    # Just an object to hold the weights and biases of each neuron in a layer
    def __init__(self, name, num_of_nodes, previous_layer=None):
        self.name = name
        self.length = num_of_nodes
        # If the layer is not the input layer, generate the biases and the weights of the neurons from the preivous layers
        if previous_layer:
            # Randomly generate weights and biases from a standard normal distribution
            self.biases = np.random.randn(self.length)
            self.weights = np.random.randn(self.length, previous_layer.length)


class NeuralNetwork:
    # Our network
    def __init__(self):
        self.layers = []
        # This is the input layer
        self.layers.append(Layer("input", length_of_input))
        # Generate hidden layers
        for i in range(number_of_hidden_layers):
            self.layers.append(Layer("hidden_layer_"+str(i+1),
                                     nodes_per_hidden_layer,
                                     previous_layer=self.layers[i]))
        # Output layer added to network
        self.layers.append(Layer("output", number_of_ouputs, previous_layer=self.layers[number_of_hidden_layers]))
        # Print structure to console
        self.display_model()

    def display_model(self):
        # Print name and length of each layer
        for layer in self.layers:
            print(layer.name+":", layer.length, "nodes")
        print("")

    def feed_forward(self, activation):
        # We will store the activations of each layer in a list for backprop
        activations = [activation,]
        # For every layer other than input
        for i in range(len(self.layers) - 1):
            # Get weights and biases
            weights = self.layers[i+1].weights
            biases = self.layers[i+1].biases
            # Calculate next activation
            activation = sigmoid(np.dot(weights, activation) + biases)
            activations.append(activation)
        return activations

    def train(self, X, y, epochs=1, validation=None):
        for epoch in range(epochs):
            # Shuffle data
            p = np.random.permutation(len(X))
            X = X[p]
            y = y[p]
            # for every input
            for i in range(len(X)):
                sys.stdout.flush()
                sys.stdout.write("\rEpoch %i: %i/%i" % (epoch+1, i+1, len(X)))
                # Calculate the error of output nodes with given input
                layer_activations = self.feed_forward(X[i])
                output_errors = error(y[i], layer_activations[-1])
                # Backward pass between output and hidden layer
                delta = output_errors * derriv_sigmoid(layer_activations[-1]) * learning_rate
                self.layers[-1].biases += delta
                self.layers[-1].weights += np.dot(delta.reshape(number_of_ouputs, 1),
                                                  layer_activations[-2].reshape(1, nodes_per_hidden_layer))
                # Back propogation between hidden layers
                for j in range(2, len(self.layers)):
                    # Calculate delta
                    ds = derriv_sigmoid(layer_activations[-j])
                    delta = np.dot(self.layers[-j+1].weights.transpose(), delta) * ds * learning_rate
                    # Adjust biases and weights by delta
                    self.layers[-j].biases += delta
                    self.layers[-j].weights += np.dot(delta.reshape(self.layers[-j].length, 1),
                                                      layer_activations[-j-1].reshape(1, self.layers[-j-1].length))
            # If validation data is provided, then calculate the accuracy
            if validation:
                correct = 0
                for x in range(len(validation[0])):
                    prediction = self.predict(validation[0][x])
                    answer = np.argmax(validation[1][x])
                    if prediction == answer:
                        correct += 1
                acc = correct / len(validation[0])
                sys.stdout.write(" Accuracy: %f \n" % acc)
            else:
                sys.stdout.write("\n")

    def predict(self, x):
        output = self.feed_forward(x)[-1]
        result = np.argmax(output)
        return result
