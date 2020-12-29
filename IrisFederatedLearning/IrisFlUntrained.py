import os
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
import tensorflow_federated as tff
from datetime import datetime
from Outsourcing import DataPreprocessing
from Outsourcing import CustomMetrics
import numpy as np

print("TensorFlow version: {}".format(tf.__version__))
print("Eager execution: {}".format(tf.executing_eagerly()))

# Tensorboard Command for CMD or Powershell
# tensorboard --logdir C:\\Users\\Stefan\\PycharmProjects\\Thesis\\Logs\\

# Filepaths
logfile_path = '/Logs\\'
dataset_path_local = '/Datasets/IrisClassification\\'
split_train_data_path = '/Datasets/IrisClassification\\split\\train\\'
split_test_data_path = '/Datasets/IrisClassification\\split\\test\\'
saved_model_path = '/Storage\\IrisModel\\'

# Path to CSV from GITHUB
github_dataset = dataset_path_local + 'iris_training02.csv'
train_dataset_url = "https://storage.googleapis.com/download.tensorflow.org/data/"
test_url = "https://storage.googleapis.com/download.tensorflow.org/data/"

# Parameter
batch_size = 30
epochs = 200

# column order in CSV file
column_names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species']

feature_names = column_names[:-1]
label_name = column_names[-1]

print("Features: {}".format(feature_names))
print("Label: {}".format(label_name))

class_names = ['Iris setosa', 'Iris versicolor', 'Iris virginica']

# Creating test and train Datasets
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

# Create Test Dataset
test_data = DataPreprocessing.PreprocessData(test_url, 'iris_test.csv', label_name, batch_size,
                                                  'Iris Test CSV Tensorflow', False, column_names)

test_data.get_dataset_by_url()
test_data.create_train_dataset()
test_data.make_graph()
test_data.map_dataset()
test_dataset = test_data.dataset

# Create split test data
split_test_data01 = DataPreprocessing.PreprocessData(split_test_data_path, '1.csv', label_name, batch_size,
                                                  'Split Test Dataset 01', True, column_names)
split_test_data01.get_local_dataset()
split_test_data01.create_train_dataset()
split_test_data01.make_graph()
split_test_data01.map_dataset()
split_testdataset01 = split_test_data01.dataset

split_test_data02 = DataPreprocessing.PreprocessData(split_test_data_path, '2.csv', label_name, batch_size,
                                                  'Split Test Dataset 02', True, column_names)
split_test_data02.get_local_dataset()
split_test_data02.create_train_dataset()
split_test_data02.make_graph()
split_test_data02.map_dataset()
split_testdataset02 = split_test_data02.dataset

split_test_data03 = DataPreprocessing.PreprocessData(split_test_data_path, '3.csv', label_name, batch_size,
                                                  'Split Test Dataset 03', True, column_names)
split_test_data03.get_local_dataset()
split_test_data03.create_train_dataset()
split_test_data03.make_graph()
split_test_data03.map_dataset()
split_testdataset03 = split_test_data01.dataset

split_test_data03 = DataPreprocessing.PreprocessData(split_test_data_path, '4.csv', label_name, batch_size,
                                                  'Split Test Dataset 04', True, column_names)
split_test_data03.get_local_dataset()
split_test_data03.create_train_dataset()
split_test_data03.make_graph()
split_test_data03.map_dataset()
split_testdataset04 = split_test_data01.dataset

# Create split train data
split_data01 = DataPreprocessing.PreprocessData(split_train_data_path, '1.csv', label_name, batch_size,
                                                  'Split Dataset 01', True, column_names)
split_data01.get_local_dataset()
split_data01.create_train_dataset()
split_data01.make_graph()
split_data01.map_dataset()
split_dataset01 = split_data01.dataset

split_data02 = DataPreprocessing.PreprocessData(split_train_data_path, '2.csv', label_name, batch_size,
                                                  'Split Dataset 02', True, column_names)
split_data02.get_local_dataset()
split_data02.create_train_dataset()
split_data02.make_graph()
split_data02.map_dataset()
split_dataset02 = split_data02.dataset

split_data03 = DataPreprocessing.PreprocessData(split_train_data_path, '3.csv', label_name, batch_size,
                                                  'Split Dataset 03', True, column_names)
split_data03.get_local_dataset()
split_data03.create_train_dataset()
split_data03.make_graph()
split_data03.map_dataset()
split_dataset03 = split_data03.dataset

split_data03 = DataPreprocessing.PreprocessData(split_train_data_path, '4.csv', label_name, batch_size,
                                                  'Split Dataset 04', True, column_names)
split_data03.get_local_dataset()
split_data03.create_train_dataset()
split_data03.make_graph()
split_data03.map_dataset()
split_dataset04 = split_data03.dataset

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


# Create Lists with Dataset per Client
sorted_datasets = [dataset_setosa, dataset_virginica, dataset_versicolor]
#test_datasets = [split_testdataset01, split_testdataset02, split_testdataset03]
test_datasets = [test_dataset, test_dataset, test_dataset]
split_datasets = [split_dataset01, split_dataset02, split_dataset03]

print("split datensatz")
print(split_datasets)

features, labels = next(iter(split_dataset01))
print(features)

# New model creation needed  TFF wont accept anything out of Scope e.g. Tensors
def create_keras_model():
  return tf.keras.models.Sequential([
      tf.keras.layers.Dense(10, activation=tf.nn.relu, input_shape=(4,)),  # input shape required
      tf.keras.layers.Dense(3,  activation=tf.nn.softmax)
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

# Start Federated Learning process
for round_num in range(1, epochs):
    state, metrics = fed_avg.next(state, split_datasets)
    print('round {:2d}, metrics={}'.format(round_num, metrics))

# Print content of metrics['train']
for name, metric in metrics['train'].items():
    print(name, metric)

# Evaluation
# Model Evaluation
test_accuracy = tf.keras.metrics.Accuracy()

evaluation = tff.learning.build_federated_evaluation(model_fn)

train_metrics = evaluation(state.model, test_datasets)
print(str(train_metrics))
