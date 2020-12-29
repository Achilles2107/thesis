import os
import tensorflow as tf
from Outsourcing import CustomMetrics
from Outsourcing import Datasets
from Outsourcing import DataProcessing
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn import datasets
import pandas as pd
import numpy as np
# label encoding the data
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import Normalizer
from sklearn import preprocessing

############################################
## This file is only for testing purposes ##
############################################

print("TensorFlow version: {}".format(tf.__version__))
print("Eager execution: {}".format(tf.executing_eagerly()))

# Filepaths
ROOT_DIR = os.path.abspath(os.curdir)
print("ROOT_DIR")
print(ROOT_DIR)
root_project_path = "C:\\Users\\Stefan\\PycharmProjects\\Thesis\\"

saved_model_path = root_project_path + '\\Storage\\IrisModel\\'
dataset_path_local = root_project_path + 'Datasets\\IrisClassification\\'
logfile_path = root_project_path + 'Datasets\\IrisClassification\\Logs\\'
split_train_data_path = root_project_path + '\\Datasets\\IrisClassification\\split\\train\\'
split_test_data_path = root_project_path + '\\Datasets\\IrisClassification\\split\\test\\'

# Urls and paths
github_dataset = dataset_path_local + 'iris_training02.csv'
train_dataset_url = "https://storage.googleapis.com/download.tensorflow.org/data/"
test_url = "https://storage.googleapis.com/download.tensorflow.org/data/"

# Path for saving weights
checkpoint_path = saved_model_path + "cp.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

# Parameter
batch_size = 120
epochs = 200

# column order in CSV file
column_names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species']
feature_names = column_names[:-1]
label_name = column_names[-1]
class_names = ['Iris setosa', 'Iris versicolor', 'Iris virginica']

# Datasets
iris_datasetlist = Datasets.iris_datalist
test_dataset = iris_datasetlist.get_dataset_at_index(1)

# train_features, train_labels = next(iter(iris_datasetlist.get_dataset_at_index(0)))
# print("train_features und train_labels: \n")
# print(train_features, train_labels)
#
# train_label_encoder=LabelEncoder()
# train_label_ids=train_label_encoder.fit_transform(train_labels)

iris_train = DataProcessing.CreateDatasets(train_dataset_url, 'iris_training.csv', label_name, batch_size,
                                                  'Iris Train CSV Tensorflow', True, column_names)
iris_dataset_train = iris_train.create_iris_url_dataset()

iris_train123 = DataProcessing.CreateDatasets(split_train_data_path, '123.csv', label_name, batch_size,
                                                  'Iris Train CSV Tensorflow', True, column_names)
iris_dataset_train123 = iris_train123.create_iris_local_dataset()

iris_train4 = DataProcessing.CreateDatasets(split_train_data_path, '4.csv', label_name, batch_size,
                                                  'Iris Train CSV Tensorflow', True, column_names)
iris_dataset_train4 = iris_train4.create_iris_local_dataset()

features, labels = next(iter(iris_dataset_train))
train_label_ids = DataProcessing.encode_label(labels)
train_features = features

test_features, test_labels = next(iter(iris_dataset_train))
test_label_ids = DataProcessing.encode_label(labels)
features_test = features

print(train_label_ids)

# df = pd.read_csv(dataset_path_local + "iris_training_with_cl_names.csv", index_col=False)
# print(df.head())
#
# features = df.copy()
# labels = features.pop('species')
#
# label_encoder=LabelEncoder()
# label_ids=label_encoder.fit_transform(labels)
#
# features = np.array(features)
# print(features)


Y = tf.keras.utils.to_categorical(train_label_ids, num_classes=3)
Y_test = tf.keras.utils.to_categorical(test_label_ids, num_classes=3)

print(train_features, Y)

# Create a callback that saves the model's weights
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 save_weights_only=True,
                                                 verbose=1)

METRICS = [
      tf.keras.metrics.TruePositives(name='tp'),
      tf.keras.metrics.FalsePositives(name='fp'),
      tf.keras.metrics.TrueNegatives(name='tn'),
      tf.keras.metrics.FalseNegatives(name='fn'),
      CustomMetrics.recall,
      CustomMetrics.specificity,
      "accuracy"
]


model = tf.keras.Sequential([
  tf.keras.layers.Dense(10, activation=tf.nn.relu, input_shape=(4,)),
  tf.keras.layers.Dense(3, activation=tf.nn.softmax)
])


model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
              loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
              metrics=["accuracy", CustomMetrics.recall, CustomMetrics.specificity,
                       CustomMetrics.mean_pred, CustomMetrics.precision])

print(model.metrics)

# Display the model's architecture
model.summary()

# load model
#model.load_weights(checkpoint_path)

training_history = model.fit(train_features, Y, epochs=100)


CustomMetrics.mean_training_accuracy(training_history, "accuracy")
CustomMetrics.plot_metric(training_history, "recall")
CustomMetrics.plot_metric(training_history, "specificity")
CustomMetrics.plot_metric(training_history, "mean_pred")
CustomMetrics.plot_metric(training_history, "precision")
CustomMetrics.subplot_metrics(training_history, "accuracy", "loss")


from sklearn.metrics import classification_report
import numpy as np

Y_test = np.argmax(Y_test, axis=1) # Convert one-hot to index
y_pred = np.argmax(model.predict(features_test), axis=-1)
print(classification_report(Y_test, y_pred))

# Model Evaluation
test_accuracy = tf.keras.metrics.Accuracy()

for (x, y) in test_dataset:
  # training=False is needed only if there are layers with different
  # behavior during training versus inference (e.g. Dropout).
  logits = model(x, training=False)
  prediction = tf.argmax(logits, axis=1, output_type=tf.int32)
  test_accuracy(prediction, y)

print("Test set accuracy: {:.3%}".format(test_accuracy.result()))



# print("Plotting metrics")
# CustomMetrics.subplot_metrics(training_history, "recall", "specificity")
# CustomMetrics.plot_metric(training_history, "precision")
# CustomMetrics.subplot_metrics(training_history, "accuracy", "loss")
# print("done")

#model.save(saved_model_path + "\\keras_model\\", overwrite=True, include_optimizer=True)

#train_acc = model.evaluate(features, Y)
#test_acc = model.evaluate(test_dataset)


# Prediction using the Model with unlabeled examples

predict_dataset = tf.convert_to_tensor([
    [5.1, 3.3, 1.7, 0.5, ],
    [5.9, 3.0, 4.2, 1.5, ],
    [6.9, 3.1, 5.4, 2.1]
])

predict_dataset_sorted = tf.convert_to_tensor([
    [5.0, 2.3, 3.3, 1.0, ],  # versicolor
    [7.0, 3.2, 4.7, 1.4, ],  # versicolor
    [5.0, 2.4, 3.8, 1.1, ],  # versicolor
    [7.7, 3.8, 6.7, 2.2, ],  # virginica
    [5.8, 2.8, 5.1, 2.4, ],  # virginica
    [6.3, 2.5, 5.0, 1.9]     # virginica
])

# training=False is needed only if there are layers with different
# behavior during training versus inference (e.g. Dropout).
predictions = model(predict_dataset_sorted, training=False)

for i, logits in enumerate(predict_dataset_sorted):
  class_idx = tf.argmax(logits).numpy()
  p = tf.nn.softmax(logits)[class_idx]
  name = class_names[class_idx]
  print("Example {} prediction: {} ({:4.1f}%)".format(i, name, 100*p))