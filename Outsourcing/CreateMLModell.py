import os
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from datetime import datetime


# class CreateMLModel:
#
#     def __init__(self, dense, layer, column_names, label_names, activation):
#         self.dense = dense
#         self.layer = layer
#         self.column_names = column_names
#         self.label_names = label_names
#         self.activation = activation
#         self.inputshape = 0
#
#     def create_keras_model(self):
#         return tf.keras.models.Sequential([
#                 tf.keras.layers.InputLayer(input_shape=(self.inputshape,))
#             for i in range(len(self.layer)):
#                 tf.keras.layers.Dense(self.dense, activation=self.activation),
#
#         ])
#
#     tf.keras.layers.Dense(10, activation=tf.nn.relu, input_shape=(4,)),  # input shape required
#     tf.keras.layers.Dense(10, activation=tf.nn.relu),
#     tf.keras.layers.Dense(3)