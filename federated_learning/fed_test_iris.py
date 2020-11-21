import os
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_federated as tff
import pandas as pd
import collections as col
import numpy as np
import tensorflow_datasets as tfds
import random as rd

print("TensorFlow version: {}".format(tf.__version__))
print("Eager execution: {}".format(tf.executing_eagerly()))

train_dataset_local_copy = 'C:\\Users\\Stefan\\PycharmProjects\\thesis\\federated_learning\\iris_training.csv'

# column order in CSV file
column_names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species']
class_names = ['Iris setosa', 'Iris versicolor', 'Iris virginica']

feature_names = column_names[:-1]
label_name = column_names[-1]

clients = 2
batch_size = 120

df = pd.read_csv(train_dataset_local_copy)
len_csv = len(df)
# df['id'] = df.index
# df.to_csv(train_dataset_local_copy)

dataset = tf.data.Dataset.from_tensor_slices(df).range(df)
print(type(dataset))
print(dataset.element_spec)

client_id_colname = 'species'  # the column that represents client ID
SHUFFLE_BUFFER = 1000
NUM_EPOCHS = 1

# split client id into train and test clients
client_ids = df[client_id_colname].unique()
train_client_ids = client_ids.rd.sample(frac=0.5).tolist()
test_client_ids = [x for x in client_ids if x not in train_client_ids]


def create_tf_dataset_for_client_fn(client_id):
  # a function which takes a client_id and returns a
  # tf.data.Dataset for that client
  client_data = df[df[client_id_colname] == client_id]
  dataset = tf.data.Dataset.from_tensor_slices(client_data.to_dict('list'))
  dataset = dataset.shuffle(SHUFFLE_BUFFER).batch(1).repeat(NUM_EPOCHS)
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
print(type(example_dataset))
example_element = iter(example_dataset).next()
print(example_element)


# # Wrap a Keras model for use with TFF.
# def model_fn():
#   model = tf.keras.models.Sequential([
#       tf.keras.layers.Dense(10, tf.nn.softmax, input_shape=(784,),
#                             kernel_initializer='zeros')
#   ])
#   return tff.learning.from_keras_model(
#       model,
#       input_spec=train_dataset.element_spec,
#       loss=tf.keras.losses.SparseCategoricalCrossentropy(),
#       metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])
#
#
#
#
# # Simulate a few rounds of training with the selected client devices.
# trainer = tff.learning.build_federated_averaging_process(
#   model_fn,
#   client_optimizer_fn=lambda: tf.keras.optimizers.SGD(0.1))
# state = trainer.initialize()
# for _ in range(5):
#   state, metrics = trainer.next(state, train_dataset)
#   print (metrics)
