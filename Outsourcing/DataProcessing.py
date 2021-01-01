import os
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from Outsourcing import DataPreprocessing


class CreateDatasetLists:

    def __init__(self):
        self.dataset_list = []

    def add_dataset_to_list(self, dataset):
        self.dataset_list.append(dataset)
        print("Added " + str(self.dataset_list[-1]) + " to list")

    def remove_dataset_from_list(self, index):
        self.dataset_list.remove(index)
        print("Removed " + self.dataset_list[index] + " from list")

    def show_list(self):
        for i in self.dataset_list:
            print(self.dataset_list[i] + "\n")

    def get_dataset_list(self):
        return self.dataset_list

    def get_dataset_at_index(self, index):
        return self.dataset_list[index]


# Feature handling and label encoding
# to_categorical() is needed to put our labels
# in a binary matrix
def decode_label(dataset, num_classes):
    features, labels = next(iter(dataset))
    label_encoder = LabelEncoder()
    label_ids = label_encoder.fit_transform(labels)
    label_ids = tf.keras.utils.to_categorical(label_ids, num_classes=num_classes)
    return features, label_ids


# get features and labels from a
# dataset
def get_features_labels(dataset):
    features, labels = next(iter(dataset))
    return features, labels

