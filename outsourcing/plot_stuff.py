import tensorflow as tf
import os
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.preprocessing import Normalizer
from sklearn import preprocessing
from sklearn.metrics import classification_report,confusion_matrix
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import os

# Just a file to plot stuff for thesis dokumentation
# Thats why its pure chaos
# No relevance for main code

# # Parameter
# # Training iterations
# epochs = 200
# # Number of classes
# num_classes = 3
#
#
#
# train_dataset_url = "https://storage.googleapis.com/download.tensorflow.org/data/iris_training.csv"
#
# train_dataset_fp = tf.keras.utils.get_file(fname=os.path.basename(train_dataset_url),
#                                            origin=train_dataset_url)
#
# test_url = "https://storage.googleapis.com/download.tensorflow.org/data/iris_test.csv"
#
# test_fp = tf.keras.utils.get_file(fname=os.path.basename(test_url),
#                                   origin=test_url)
#
# # column order in CSV file
# column_names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species']
# class_names = ['Iris setosa', 'Iris versicolor', 'Iris virginica']
# feature_names = column_names[:-1]
# label_name = column_names[-1]
#
# print("Features: {}".format(feature_names))
# print("Label: {}".format(label_name))
#
# print("Local copy of the dataset file: {}".format(train_dataset_fp))
#
#
# df = pd.read_csv(train_dataset_fp)
#
#
# # ax = sns.boxplot(x=df["species"], y=df.index, labels=['Iris setosa', 'Iris versicolor', 'Iris virginica'])
# # ax.set(
# #     xlabel='Species',
# #     ylabel='Datensätze'
# # )
# # ax.set_title(
# #     "Zusammensetzung Iris Daten"
# # )
# # plt.show()
# x_pos = np.arange(len(class_names))
# height = 120
#
#
# df['species'].value_counts().sort_values().plot(kind='barh')
# plt.yticks(x_pos, class_names)
# plt.title('Label Zusammensetzung Iris Trainingsdaten')
# plt.xlabel('Datensätze')
# plt.ylabel('Species')
# plt.tight_layout()
# plt.show()
#
# df2 = pd.read_csv(test_fp)
#
# df2['species'].value_counts().sort_values().plot(kind='barh')
# plt.yticks(x_pos, class_names)
# plt.title('Label Zusammensetzung Iris Testdaten')
# plt.xlabel('Datensätze')
# plt.ylabel('Species')
# plt.tight_layout()
# plt.show()
#
# print("DF HEAD: \n" + str(df.head()))
#
# batch_size = 120
#
# train_dataset = tf.data.experimental.make_csv_dataset(
#     train_dataset_fp,
#     batch_size,
#     column_names=column_names,
#     label_name=label_name,
#     num_epochs=1)
#
# features, labels = next(iter(train_dataset))
#
# print(features)
#
# w = 120

# plt.scatter(labels,
#             w,
#             c=labels,
#             cmap='viridis')
#
# plt.xlabel("Petal length")
# plt.ylabel("Sepal length")
# plt.show()

# dtypes = []
#
# for i in range(0, 78):
#     dtypes.append('\'float64\',')
#     print(dtypes[i])

daten = 'C:\\Users\\Stefan\\Nextcloud\\Thesisstuff\\Datensätze\\MachineLearningCSV\\MachineLearningCVE\\'

# daten = '/home/stefan/daten/'

file = daten + 'Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv'

df = pd.read_csv(file, sep=r'\s*,\s*')

city = df['Label'].value_counts()

print(city)

# blub = ['1', '2']
#
# x_pos = np.arange(len(blub))
# height = 120
#
# print(df.head())
#
# df['Label'].value_counts().sort_values().plot(kind='barh')
# # plt.yticks(x_pos, 2)
# plt.title('Label Zusammensetzung Iris Trainingsdaten')
# plt.xlabel('Datensätze')
# plt.ylabel('Label')
# plt.tight_layout()
# plt.show()
