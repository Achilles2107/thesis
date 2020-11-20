import os
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_federated as tff
import pandas as pd
import collections as col
import h5py
import numpy as np
import tensorflow_datasets as tfds
import random as rd

print("TensorFlow version: {}".format(tf.__version__))
print("Eager execution: {}".format(tf.executing_eagerly()))

train_dataset_url = "https://storage.googleapis.com/download.tensorflow.org/data/iris_training.csv"

train_dataset_fp = tf.keras.utils.get_file(fname=os.path.basename(train_dataset_url),
                                           origin=train_dataset_url)

print("Local copy of the dataset file: {}".format(train_dataset_fp))

train_dataset_local_copy = 'C:\\Users\\Stefan\\PycharmProjects\\thesis\\federated_learning\\iris_training.csv'

# column order in CSV file
column_names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species']

class_names = ['Iris setosa', 'Iris versicolor', 'Iris virginica']

label_name = column_names[-1]

clients = 2
client_id_colname = 'species'  # the column that represents client ID
SHUFFLE_BUFFER = 120
NUM_EPOCHS = 1
batch_size = 60


def create_client_list(client):
    l = []
    for i in range(client):
        l = ['client' + str(i)]
    return l


def preprocess(dataset):
    return (
        dataset.map('client01').batch(batch_size)
    )


# Create Dataset from csv
# Is type tf.data.Dataset
# num_epochs = number of loops with the dataset : none => infinite loop
train_dataset = tf.data.experimental.make_csv_dataset(
    train_dataset_fp,
    batch_size,
    column_names=column_names,
    label_name=label_name,
    num_epochs=1)

print(type(train_dataset))
df = pd.read_csv(train_dataset_local_copy)

# split client id into train and test clients
client_ids = df[client_id_colname].unique()
train_client_ids = rd.sample(client_ids.tolist(), clients)
test_client_ids = [x for x in client_ids if x not in train_client_ids]


def create_tf_dataset_for_client_fn(client_id):
  # a function which takes a client_id and returns a
  # tf.data.Dataset for that client
  client_data = df[df[client_id_colname] == client_id]
  #dataset = tf.data.Dataset.from_tensor_slices(client_data.to_dict('list'))
  dataset = train_dataset
  # dataset = dataset.shuffle(SHUFFLE_BUFFER).batch(batch_size).repeat(NUM_EPOCHS)
  dataset = dataset.batch(batch_size).repeat(NUM_EPOCHS)
  return dataset


train_data = tff.simulation.ClientData.from_clients_and_fn(
        client_ids=train_client_ids,
        create_tf_dataset_for_client_fn=create_tf_dataset_for_client_fn
    )
test_data = tff.simulation.ClientData.from_clients_and_fn(
        client_ids=test_client_ids,
        create_tf_dataset_for_client_fn=create_tf_dataset_for_client_fn
    )

example_dataset = train_data.create_tf_dataset_for_client(
        train_data.client_ids[0]
    )
print("Type example_Dataset")
print(type(example_dataset))
example_element = iter(example_dataset).next()
print("example_element")
print(example_element)


# client_data = preprocess(train_dataset1)
# print(client_data.element_spec)


# Wrap a Keras model for use with TFF.
def model_fn():
  model = tf.keras.models.Sequential([
      tf.keras.layers.Dense(10, activation=tf.nn.relu, input_shape=(4, )),  # input shape required
      tf.keras.layers.Dense(10, activation=tf.nn.relu),
      tf.keras.layers.Dense(3)
  ])
  return tff.learning.from_keras_model(
      model,
      input_spec=example_dataset.element_spec,
      loss=tf.keras.losses.SparseCategoricalCrossentropy(),
      metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])


# Simulate a few rounds of training with the selected client devices.
trainer = tff.learning.build_federated_averaging_process(
  model_fn,
  client_optimizer_fn=lambda: tf.keras.optimizers.SGD(0.1))
state = trainer.initialize()
for _ in range(5):
  state, metrics = trainer.next(state, example_dataset)
  print (metrics)

