from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation
import tensorflow as tf
import tensorflow_federated as tff
from sklearn.preprocessing import Normalizer
from sklearn import preprocessing
from outsourcing import CustomMetrics
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.feature_selection import VarianceThreshold
from pandas.plotting import scatter_matrix
from yellowbrick.target import FeatureCorrelation
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
import os
import pathlib

root_project_path = pathlib.Path.cwd().parent
print(root_project_path)

print("imports ok")

# Run TensorFlow on CPU only
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

output_path = pathlib.Path('/hinkelmann')
label_name = "Label"
df = pd.read_csv(output_path / 'out.csv')

label = df.pop(label_name)
dataset = tf.data.Dataset.from_tensor_slices((df.values, label.values))

features, labels = next(iter(dataset))
print(features, labels)

# Create neural net

# Metrics
METRICS = [
        tf.keras.metrics.Recall(),
        tf.keras.metrics.Precision(),
        tf.keras.metrics.SensitivityAtSpecificity(0.5),
        tf.keras.metrics.SpecificityAtSensitivity(0.5),
        tf.keras.metrics.binary_accuracy()
]

print('Creating Neural Network')

# Measure accuracy

# CustomMetrics.mean_training_accuracy(history, "accuracy")
# CustomMetrics.plot_metric(history, "recall")
# CustomMetrics.plot_metric(history, "precision")
# CustomMetrics.plot_metric(history, "sensitivity_at_specificity")
# CustomMetrics.plot_metric(history, "specificity_at_sensitivity")
# CustomMetrics.subplot_metrics(history, "accuracy", "loss")
#
#
# prediction = model.predict(x_test)
# y_score = prediction
# pred = np.argmax(prediction, axis=1)
# y_eval = np.argmax(y_test, axis=1)
# score = accuracy_score(y_eval, pred)
# print("Validation score: {}".format(score))
# print(history.history.keys())
#
# # Confusion Matrix
# print("MATRIX")
# cfm = confusion_matrix(y_eval, pred)
# print(cfm)
#
# # Report
# cmp = classification_report(y_eval, pred)
# print(cmp)



def create_keras_model():
  return tf.keras.models.Sequential([
      tf.keras.layers.Dense(10, input_dim=X.shape[1], kernel_initializer='normal', activation='relu'),
      tf.keras.layers.Dense(50, input_dim=X.shape[1], kernel_initializer='normal', activation='relu'),
      tf.keras.layers.Dense(10, input_dim=X.shape[1], kernel_initializer='normal', activation='relu'),
      tf.keras.layers.Dense(1, kernel_initializer='normal'),
      tf.keras.layers.Dense(Y.shape[1], activation='softmax')
  ])


def model_fn():
  # We _must_ create a new model here, and _not_ capture it from an external
  # scope. TFF will call this within different graph contexts.
  keras_model = create_keras_model()
  return tff.learning.from_keras_model(
      keras_model,
      # Input information on what shape the input data will have
      # Must be from type tf.Type or tf.TensorSpec
      input_spec=split_dataset01.element_spec,
      loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
      metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])

keras_model = create_keras_model()

fed_avg = tff.learning.build_federated_averaging_process(
    model_fn=model_fn,
    client_optimizer_fn=lambda: tf.keras.optimizers.SGD(learning_rate=0.01),  # for each Client
    server_optimizer_fn=lambda: tf.keras.optimizers.SGD(learning_rate=1.0))  # for Global model


state = fed_avg.initialize()

epochs = 10

# Start Federated Learning process
for round_num in range(1, epochs):
    state, metrics = fed_avg.next(state, split_datasets)
    print('round {:2d}, metrics={}'.format(round_num, metrics))