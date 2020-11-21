import os
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_federated as tff
import pandas as pd
import collections as col
import numpy as np


train_dataset_url = "https://storage.googleapis.com/download.tensorflow.org/data/iris_training.csv"

train_dataset_fp = tf.keras.utils.get_file(fname=os.path.basename(train_dataset_url),
                                           origin=train_dataset_url)

print("Local copy of the dataset file: {}".format(train_dataset_fp))

# column order in CSV file
column_names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species']

# Slicing the Dataset in Features and Label
feature_names = column_names[:-1]
label_name = column_names[-1]

print("Features: {}".format(feature_names))
print("Label: {}".format(label_name))

class_names = ['Iris setosa', 'Iris versicolor', 'Iris virginica']

batch_size = 32

# Create Dataset from csv
# num_epochs = number of loops with the dataset : none => infinite loop
train_dataset = tf.data.experimental.make_csv_dataset(
    train_dataset_fp,
    batch_size,
    column_names=column_names,
    label_name=label_name,
    num_epochs=1)

features, labels = next(iter(train_dataset))

print('Features: ')
print(features)
print('Features Ende')

# Create Graph for Features Groups
plt.scatter(features['petal_length'],
            features['sepal_length'],
            c=labels,
            cmap='viridis')

plt.xlabel("Petal length")
plt.ylabel("Sepal length")
plt.show()


# Packs the Features in a array and rises its dimension with tf.stack
def pack_features_vector(features, labels):
  """Pack the features into a single array."""
  features = tf.stack(list(features.values()), axis=1)
  return features, labels


# Add tf.stack features, label to train_dataset
train_dataset = train_dataset.map(pack_features_vector)

print('train dataset')
print(train_dataset)

features, labels = next(iter(train_dataset))

print(features[:-1])

NUM_CLIENTS = 2
NUM_EPOCHS = 1
BATCH_SIZE = 60
SHUFFLE_BUFFER = 120


def create_client_ids(clients):
    l = []
    for i in range(clients):
        l.append(i)
    return l


def preprocess(dataset):

  def batch_format_fn(element):
    """Flatten a batch `pixels` and return the features as an `OrderedDict`."""
    return col.OrderedDict(
        x=tf.reshape(element['features'], [-1, 4]),
        y=tf.reshape(element['label'], [-1, 1]))

  return dataset.repeat(NUM_EPOCHS).shuffle(SHUFFLE_BUFFER).batch(
      BATCH_SIZE).map(batch_format_fn)


def make_federated_data(client_data, client_ids):
  return [
      preprocess(client_data.create_tf_dataset_for_client(x))
      for x in client_ids
  ]


# sample_clients = emnist_train.client_ids[0:NUM_CLIENTS]
#
# federated_train_data = make_federated_data(emnist_train, sample_clients)
#
# print('Number of client datasets: {l}'.format(l=len(federated_train_data)))
# print('First dataset: {d}'.format(d=federated_train_data[0]))
#

