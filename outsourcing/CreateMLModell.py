import os
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from datetime import datetime


class CreateMLModel:

    def __init__(self, dense, output_shape, input_shape, layer, feature_names,
                 label_names, activation, output_activation):
        self.dense = dense
        self.layer = layer
        self.feature_names = feature_names
        self.label_names = label_names
        self.activation = activation
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.output_activation = output_activation

    def create_keras_model(self):

        self.input_shape = len(self.feature_names)

        return tf.keras.models.Sequential([
                tf.keras.layers.InputLayer(input_shape=(self.input_shape,)),
                self.create_hidden_layer(),
                tf.keras.layers.Layer(self.output_shape, activation=self.output_activation)


        ])

    def create_hidden_layer(self):
        for i in range(self.layer):
            tf.keras.layers.Dense(self.dense, activation=self.activation),

