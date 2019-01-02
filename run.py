import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from neural_network import NeuralNetwork
import pickle

# parameters
split_ratio = 0.8
epochs = 50

# Reading our training csv data
mnist_train = pd.read_csv("data/train.csv")
mnist_train.dropna()
labels = mnist_train['label'].values
X = mnist_train.drop('label', axis=1).values

# Normalize the dataset
X = X / 255.0

# Displays the image
def display_image(vector):
    vector = vector.reshape(28, 28)
    plt.imshow(vector, cmap=matplotlib.cm.binary, interpolation='nearest')
    plt.show()

# display_image(X.iloc[30000]) # Should display a 3

# Turn y into vectors
s = (len(labels), 10)
y = np.zeros(s)
for i in range(len(y)):
    y[i][labels[i]] = 1

# Split labeled dataset into testing and training
split = round(len(X) * split_ratio)
train_x, train_y = X[:split], y[:split]
test_x, test_y = X[split:], y[split:]

# Instantiate our network
network = NeuralNetwork()

# Start training
network.train(train_x, train_y, epochs=epochs, validation=[test_x, test_y])

# Save neural network
with open("nn.pkl", 'wb') as f:
    pickle.dump(network, f)
