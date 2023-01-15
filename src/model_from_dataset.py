import numpy as np
from game_objects import Game
import tensorflow as tf

tf.keras.backend.clear_session()
tf.random.set_seed(42)

config_path = input("Configuration file path: ")
results_path = input("Results file path: ")
number_of_layers = int(input("Number of layers in NN: "))
number_of_neurons = int(input("Number of neurons per layer: "))




configs = np.load(config_path)
results = np.load(results_path)


## We wil build a Multi Level Perceptron (MLP) using keras

## The input layer has 9 inputs (when flattened), the ouput layer has 1 continous output.

input_layer = tf.keras.layers.Flatten()
dense_layers = [tf.keras.layers.Dense(number_of_neurons, activation= "relu") for _ in range(number_of_layers)]
output_layer = tf.keras.layers.Dense(1, activation = "sigmoid")

model = tf.keras.Sequential([input_layer, *dense_layers, output_layer])

opt = tf.keras.optimizers.Adam(learning_rate=1e-3)
model.compile(loss = "mse", optimizer = opt, metrics = ["RootMeanSquaredError"])


model.fit(configs, results, epochs = 20)

save = input("Save the model? (y/n): ")

if save == "y":
    model.save("model.h5")