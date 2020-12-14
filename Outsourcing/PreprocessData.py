import os
import matplotlib.pyplot as plt
import tensorflow as tf


class PreprocessData:

    def __init__(self, url, filename, label_name, batch_size, title, shuffle_value=True,  column_names=None):
        self.url = url
        self.filename = filename
        self.label_name = label_name
        self.column_names = column_names
        self.batch_size = batch_size
        self.shuffle_value = shuffle_value
        self.title = title
        self.file_path = 0
        self.features = 0
        self.labels = 0
        self.dataset = 0

    def get_dataset_by_url(self):
        keras_url = self.url + self.filename
        self.file_path = tf.keras.utils.get_file(fname=os.path.basename(keras_url),
                                origin=keras_url)
        return print(self.file_path)

    def get_local_dataset(self):
        self.file_path = self.url + self.filename
        return print(self.file_path)

    def create_train_dataset(self):
        dataset = tf.data.experimental.make_csv_dataset(
        self.file_path,
        self.batch_size,
        column_names=self.column_names,
        label_name=self.label_name,
        num_epochs=1,
        shuffle=self.shuffle_value)
        self.dataset = dataset
        print("Dataset created from " + str(self.filename))

    def make_graph(self):
        self.features, self.labels = next(iter(self.dataset))
        plt.scatter( self.features['petal_length'],
                     self.features['sepal_length'],
                     c=self.labels,
                     cmap='viridis')

        plt.xlabel("Petal length")
        plt.ylabel("Sepal length")
        plt.title(self.title)
        plt.show()

    @staticmethod
    # Packs features in a single array
    def pack_features_vector(features, labels):
      features = tf.stack(list(features.values()), axis=1)
      return features, labels

    def map_dataset(self):
        self.dataset = self.dataset.map(self.pack_features_vector)


