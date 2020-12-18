import os
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from Outsourcing import DataPreprocessing
from Outsourcing import CustomMetrics
import numpy as np
import tensorflow.keras.backend as K
import pandas as pd

print("TensorFlow version: {}".format(tf.__version__))
print("Eager execution: {}".format(tf.executing_eagerly()))

# Tensorboard Command for CMD or Powershell
# tensorboard --logdir C:\\Users\\Stefan\\PycharmProjects\\thesis\\logs\\

# Filepaths
logfile_path = 'C:\\Users\\Stefan\\PycharmProjects\\thesis\\logs\\'
dataset_path_local = 'C:\\Users\\Stefan\\PycharmProjects\\thesis\\datasets\\iris_classification\\'
split_data_path = 'C:\\Users\\Stefan\\PycharmProjects\\thesis\\datasets\\iris_classification\\split\\'
saved_model_path = 'C:\\Users\\Stefan\\PycharmProjects\\thesis\\saved_model\\iris_model\\'

# Path to CSV from GITHUB
github_dataset = dataset_path_local + 'iris_training02.csv'
train_dataset_url = "https://storage.googleapis.com/download.tensorflow.org/data/"
test_url = "https://storage.googleapis.com/download.tensorflow.org/data/"

# Path for saving weights
checkpoint_path = saved_model_path + "cp.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)
batch_size = 32


# Tensorboard Writer
summary_writer = tf.summary.create_file_writer(logfile_path)

# Parameter
batch_size = 32
epochs = 200

# column order in CSV file
column_names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species']
feature_names = column_names[:-1]
label_name = column_names[-1]
class_names = ['Iris setosa', 'Iris versicolor', 'Iris virginica']

# Creating test and train datasets
# PreprocessData constructor usage
# url, filename, label_name, batch_size, title, shuffle_value=True,  column_names=None)
# Create Traindata
train_data = DataPreprocessing.PreprocessData(train_dataset_url, 'iris_training.csv', label_name, batch_size,
                                                  'Iris Train CSV Tensorflow', True, column_names)
train_data.get_dataset_by_url()
train_data.create_train_dataset()
train_data.make_graph()
train_data.map_dataset()
train_dataset = train_data.dataset

train_data_split04 = DataPreprocessing.PreprocessData(split_data_path + 'train\\', '4.csv', label_name, batch_size,
                                                  'Iris Train CSV Tensorflow Split04', True, column_names)
train_data_split04.get_local_dataset()
train_data_split04.create_train_dataset()
train_data_split04.make_graph()
train_data_split04.map_dataset()
train_dataset_split04 = train_data_split04.dataset
train_val = train_data_split04.dataset

train_data_123 = DataPreprocessing.PreprocessData(split_data_path + 'train\\', '123.csv', label_name, batch_size,
                                                  'Iris Train CSV Tensorflow Split04', True, column_names)
train_data_123.get_local_dataset()
train_data_123.create_train_dataset()
train_data_123.make_graph()
train_data_123.map_dataset()
train_dataset_123 = train_data_123.dataset

# Create Test Dataset
test_data = DataPreprocessing.PreprocessData(test_url, 'iris_test.csv', label_name, batch_size,
                                                  'Iris Test CSV Tensorflow', False, column_names)

test_data.get_dataset_by_url()
test_data.create_train_dataset()
test_data.make_graph()
test_data.map_dataset()
test_dataset = test_data.dataset

test_data_123 = DataPreprocessing.PreprocessData(split_data_path + 'test\\', '123.csv', label_name, batch_size,
                                                  'Iris Test CSV Tensorflow 123', False, column_names)
test_data_123.get_local_dataset()
test_data_123.create_train_dataset()
test_data_123.make_graph()
test_data_123.map_dataset()
test_dataset_123 = test_data_123.dataset
test_val = test_data_123.dataset
print(test_val)
print(train_dataset_123)

# Sorted Datasets
data_setosa = DataPreprocessing.PreprocessData(dataset_path_local, 'iris_setosa.csv', label_name, batch_size,
                                                  'Iris setosa Dataset', True, column_names)
data_setosa.get_local_dataset()
data_setosa.create_train_dataset()
data_setosa.make_graph()
data_setosa.map_dataset()
dataset_setosa = data_setosa.dataset

data_versicolor = DataPreprocessing.PreprocessData(dataset_path_local, 'iris_versicolor.csv', label_name, batch_size,
                                                  'Iris versicolor Dataset', True, column_names)
data_versicolor.get_local_dataset()
data_versicolor.create_train_dataset()
data_versicolor.make_graph()
data_versicolor.map_dataset()
dataset_versicolor = data_versicolor.dataset

data_virginica = DataPreprocessing.PreprocessData(dataset_path_local, 'iris_versicolor.csv', label_name, batch_size,
                                                  'Iris versicolor Dataset', True, column_names)
data_virginica.get_local_dataset()
data_virginica.create_train_dataset()
data_virginica.make_graph()
data_virginica.map_dataset()
dataset_virginica = data_virginica.dataset

model = tf.keras.Sequential([
  tf.keras.layers.Dense(10, activation=tf.nn.relu, input_shape=(4,)),  # input shape required
  tf.keras.layers.Dense(10, activation=tf.nn.relu),
  tf.keras.layers.Dense(3)
])

model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.01),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=["accuracy", CustomMetrics.recall, CustomMetrics.specificity, CustomMetrics.precision])

print(model.metrics)

# Display the model's architecture
model.summary()

# load model
# model.load_weights(checkpoint_path)

training_history = model.fit(train_val, epochs=epochs, validation_data=test_dataset)

# Model Evaluation
test_accuracy = tf.keras.metrics.Accuracy()

for (x, y) in test_dataset:
  # training=False is needed only if there are layers with different
  # behavior during training versus inference (e.g. Dropout).
  logits = model(x, training=False)
  prediction = tf.argmax(logits, axis=1, output_type=tf.int32)
  test_accuracy(prediction, y)

print("Test set accuracy: {:.3%}".format(test_accuracy.result()))

# print("Plotting metrics")
# CustomMetrics.subplot_metrics(training_history, "recall", "specificity")
# CustomMetrics.plot_metric(training_history, "precision")
# CustomMetrics.subplot_metrics(training_history, "accuracy", "loss")
# print("done")

model.save(saved_model_path + "\\keras_model\\", overwrite=True, include_optimizer=True)

train_acc = model.evaluate(train_dataset)
test_acc = model.evaluate(test_dataset)

