import os
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
import tensorflow_federated as tff
from datetime import datetime
from functions import functions

print("TensorFlow version: {}".format(tf.__version__))
print("Eager execution: {}".format(tf.executing_eagerly()))

# Tensorboard Command for CMD or Powershell
# tensorboard --logdir C:\\Users\\Stefan\\PycharmProjects\\thesis\\logs\\

# Filepaths
logfile_path = 'C:\\Users\\Stefan\\PycharmProjects\\thesis\\logs\\'
dataset_path_local = 'C:\\Users\\Stefan\\PycharmProjects\\thesis\\datasets\\iris_classification\\'
split_data_path = 'C:\\Users\\Stefan\\PycharmProjects\\thesis\\datasets\\iris_classification\\split\\'
# Path to CSV from GITHUB
github_dataset = dataset_path_local + 'iris_training02.csv'
train_dataset_url = "https://storage.googleapis.com/download.tensorflow.org/data/iris_training.csv"

train_dataset_fp = functions.get_dataset_by_url(train_dataset_url)

print("Local copy of the dataset file: {}".format(train_dataset_fp))

test_url = "https://storage.googleapis.com/download.tensorflow.org/data/iris_test.csv"

test_fp = functions.get_dataset_by_url(test_url)

# Tensorboard
now = datetime.now()
# Define the Keras TensorBoard callback.
logdir = logfile_path + "\\graph\\" + datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = keras.callbacks.TensorBoard(log_dir=logdir)

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

# Traindata CSV from Tensorflow
train_dataset_iris_tensorflow = functions.create_train_dataset_with_col_name(train_dataset_fp, column_names,
                                                                             label_name, batch_size)
# Traindata CSV from GITHUB
train_dataset_iris_github = functions.create_train_dataset_with_col_name(github_dataset, column_names,
                                                                             label_name, batch_size)
# Testdata CSV from Tensorflow
test_dataset = functions.create_train_dataset_with_col_name(github_dataset, column_names,
                                                                             'species', batch_size, shuffle=False)
    tf.data.experimental.make_csv_dataset(
    test_fp,
    batch_size,
    column_names=column_names,
    label_name='species',
    num_epochs=1,
    shuffle=False)

