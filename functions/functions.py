import os
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
import tensorflow_federated as tff
from datetime import datetime


# For Datasets with Feature and Label Names in the actual File
def create_train_dataset(file_path, filename, label_name, batch_size):
    train_dataset = tf.data.experimental.make_csv_dataset(
    file_path + filename,
    batch_size,
    #column_names=column_names,
    label_name=label_name,
    num_epochs=1)
    return train_dataset


# For Datasets without Feature and Label Names in the actual File
def create_train_dataset_with_col_name(file_path, filename, column_names, label_name, batch_size, shuffle):
    train_dataset = tf.data.experimental.make_csv_dataset(
    file_path + filename,
    batch_size,
    column_names=column_names,
    label_name=label_name,
    num_epochs=1)
    shuffle
    return train_dataset


# Packs features in a single array
def pack_features_vector(features, labels):
  features = tf.stack(list(features.values()), axis=1)
  return features, labels


def make_graph(dataset, title):
    features, labels = next(iter(dataset))
    plt.scatter(features['petal_length'],
            features['sepal_length'],
            c=labels,
            cmap='viridis')

    plt.xlabel("Petal length")
    plt.ylabel("Sepal length")
    plt.title(title)
    plt.show()


def get_dataset_by_url(url):
    tf.keras.utils.get_file(fname=os.path.basename(url),
                            origin=url)

