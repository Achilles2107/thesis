import os
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
import tensorflow_federated as tff
from datetime import datetime
from Outsourcing import DataPreprocessing
from Outsourcing import CustomMetrics
from Outsourcing import DataProcessing
import numpy as np
import glob

# File for Datasets for use in multiple models


def get_dataset_list(dataset_list):
    return dataset_list

# Iris datasets
# Filepaths
logfile_path = 'C:\\Users\\Stefan\\PycharmProjects\\thesis\\logs\\'
dataset_path_local = 'C:\\Users\\Stefan\\PycharmProjects\\thesis\\datasets\\iris_classification\\'
split_train_data_path = 'C:\\Users\\Stefan\\PycharmProjects\\thesis\\datasets\\iris_classification\\split\\train\\'
split_test_data_path = 'C:\\Users\\Stefan\\PycharmProjects\\thesis\\datasets\\iris_classification\\split\\test\\'
saved_model_path = 'C:\\Users\\Stefan\\PycharmProjects\\thesis\\saved_model\\iris_model\\'

# Urls and paths
github_dataset = dataset_path_local + 'iris_training02.csv'
train_dataset_url = "https://storage.googleapis.com/download.tensorflow.org/data/"
test_url = "https://storage.googleapis.com/download.tensorflow.org/data/"

# column order in CSV file
column_names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species']

feature_names = column_names[:-1]
label_name = column_names[-1]

class_names = ['Iris setosa', 'Iris versicolor', 'Iris virginica']

# Parameter
batch_size = 30
epochs = 200

# Dataset creation

iris_train = DataProcessing.CreateDatasets(train_dataset_url, 'iris_training.csv', label_name, batch_size,
                                                  'Iris Train CSV Tensorflow', True, column_names)
iris_dataset_train = iris_train.create_iris_train_dataset()

# Create Test Dataset
iris_test = DataProcessing.CreateDatasets(test_url, 'iris_test.csv', label_name, batch_size,
                                                  'Iris Test CSV Tensorflow', False, column_names)

iris_dataset_test = iris_train.create_iris_test_dataset()

cdl = DataProcessing.CreateDatasetLists()
cdl.add_dataset_to_list(iris_dataset_train)
cdl.add_dataset_to_list(iris_dataset_test)

