import os
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
import pandas as pd
from matplotlib import pyplot
from Outsourcing import CustomMetrics

# Filepaths
logfile_path = 'C:\\Users\\Stefan\\PycharmProjects\\thesis\\logs\\'
dataset_path_local = 'C:\\Users\\Stefan\\PycharmProjects\\thesis\\datasets\\iris_classification\\'
split_data_path = 'C:\\Users\\Stefan\\PycharmProjects\\thesis\\datasets\\iris_classification\\split\\'
saved_model_path = 'C:\\Users\\Stefan\\PycharmProjects\\thesis\\saved_model\\iris_model\\'

# Path to CSV from GITHUB
github_dataset = dataset_path_local + 'iris_training02.csv'
train_dataset_url_short = "https://storage.googleapis.com/download.tensorflow.org/data/"
test_url_short = "https://storage.googleapis.com/download.tensorflow.org/data/"

print("TensorFlow version: {}".format(tf.__version__))
print("Eager execution: {}".format(tf.executing_eagerly()))

train_dataset_url = "https://storage.googleapis.com/download.tensorflow.org/data/iris_training.csv"

train_dataset_fp = tf.keras.utils.get_file(fname=os.path.basename(train_dataset_url),
                                           origin=train_dataset_url)

print("Local copy of the dataset file: {}".format(train_dataset_fp))

test_url = "https://storage.googleapis.com/download.tensorflow.org/data/iris_test.csv"

test_fp = tf.keras.utils.get_file(fname=os.path.basename(test_url),
                                  origin=test_url)


# column order in CSV file
column_names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species']

feature_names = column_names[:-1]
label_name = column_names[-1]

print("Features: {}".format(feature_names))
print("Label: {}".format(label_name))

class_names = ['Iris setosa', 'Iris versicolor', 'Iris virginica']

batch_size = 32

def pack_features_vector(features, labels):
  """Pack the features into a single array."""
  features = tf.stack(list(features.values()), axis=1)
  return features, labels


train_dataset = tf.data.experimental.make_csv_dataset(
    train_dataset_fp,
    batch_size,
    column_names=column_names,
    label_name=label_name,
    num_epochs=1)

test_dataset = tf.data.experimental.make_csv_dataset(
    test_fp,
    batch_size,
    column_names=column_names,
    label_name='species',
    num_epochs=1,
    shuffle=False)

features, labels = next(iter(train_dataset))

print(features)

plt.scatter(features['petal_length'],
            features['sepal_length'],
            c=labels,
            cmap='viridis')

plt.xlabel("Petal length")
plt.ylabel("Sepal length")
plt.title("Train Dataset")
plt.show()

features, labels = next(iter(test_dataset))

print(features)

plt.scatter(features['petal_length'],
            features['sepal_length'],
            c=labels,
            cmap='viridis')

plt.xlabel("Petal length")
plt.ylabel("Sepal length")
plt.title("Test Dataset")
plt.show()

test_dataset = test_dataset.map(pack_features_vector)
train_dataset = train_dataset.map(pack_features_vector)

# features, labels = next(iter(train_dataset))
#
# print(features[:5])


# def precision(y_true, y_pred):
#     true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
#     predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
#     return true_positives / (predicted_positives + K.epsilon())
#
#
# def recall(y_true, y_pred):
#     true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
#     possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
#     return true_positives / (possible_positives + K.epsilon())
#
#
# def specificity(y_true, y_pred):
#     true_negatives = K.sum(K.round(K.clip((1 - y_true) * (1 - y_pred), 0, 1)))
#     possible_negatives = K.sum(K.round(K.clip(1 - y_true, 0, 1)))
#     return true_negatives / (possible_negatives + K.epsilon())
#
#
# def mean_squared_error(y_true, y_pred):
#     return K.mean(K.square(y_pred - y_true), axis=-1)
#
#
# def rmse(y_true, y_pred):
# 	return K.sqrt(K.mean(K.square(y_pred - y_true), axis=-1))


model = tf.keras.Sequential([
  tf.keras.layers.Dense(10, activation=tf.nn.relu, input_shape=(4,)),  # input shape required
  tf.keras.layers.Dense(10, activation=tf.nn.relu),
  tf.keras.layers.Dense(3)
])

model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.01),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              #metrics=[tf.keras.metrics.Precision()])
              metrics=[CustomMetrics.specificity, CustomMetrics.recall, CustomMetrics.precision, "accuracy"])


print(model.metrics)

# Display the model's architecture
model.summary()

# load model
# model.load_weights(checkpoint_path)

training_history = model.fit(train_dataset, epochs=200, validation_data=test_dataset)


# print("shape test data")
# print(test_dataset)
# print("shape train data")
# print(train_dataset)
#
# features, labels = next(iter(test_dataset))
#
# print(features[:5])
#
# print("dataset")
# print(test_dataset)

# # pandas test
# df = pd.read_csv(test_url_short + 'iris_test.csv')
# y_pred = model.predict(features)
# y_true = tf.convert_to_tensor(df, dtype="float32", dtype_hint=None, name=None)
#
# print(y_pred)
# print(y_true)
#
# print("y_true: \n", y_true)
#
# print("precision")
# print(precision(y_true, y_pred))
#
# print("recall/sensitivity")
# print(recall(y_true, y_pred))
#
# print("specitivity")
# print(specificity(y_true, y_pred))

# evaluate the model
train_acc = model.evaluate(train_dataset, verbose=0)
test_acc = model.evaluate(test_dataset, verbose=0)

print(model.metrics_names)
print("train acc: " + str(train_acc), "test acc" + str(test_acc))

pyplot.plot(training_history.history['recall'])
pyplot.title('recall')
pyplot.show()
pyplot.plot(training_history.history['specificity'])
pyplot.title('specificity')
pyplot.show()
pyplot.plot(training_history.history['precision'])
pyplot.title('precision')
pyplot.show()


