import os
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
import tensorflow_federated as tff
from datetime import datetime
from Outsourcing import DataPreprocessing
from Outsourcing import CustomMetrics
from Outsourcing import DataProcessing
from Outsourcing import Datasets
import numpy as np

print("TensorFlow version: {}".format(tf.__version__))
print("Eager execution: {}".format(tf.executing_eagerly()))

# Filepaths
saved_model_path = '/Storage\\IrisModel\\'
dataset_path_local = '/Datasets/IrisClassification\\'
logfile_path = '/Logs\\'
split_train_data_path = '/Datasets/IrisClassification\\split\\train\\'
split_test_data_path = '/Datasets/IrisClassification\\split\\test\\'

# Urls and paths
github_dataset = dataset_path_local + 'iris_training02.csv'
train_dataset_url = "https://storage.googleapis.com/download.tensorflow.org/data/iris_training.csv"

train_dataset_fp = tf.keras.utils.get_file(fname=os.path.basename(train_dataset_url),
                                           origin=train_dataset_url)

test_url = "https://storage.googleapis.com/download.tensorflow.org/data/iris_test.csv"

test_fp = tf.keras.utils.get_file(fname=os.path.basename(test_url),
                                  origin=test_url)

print("Local copy of the dataset file: {}".format(train_dataset_fp))

# Path for saving weights
checkpoint_path = saved_model_path + "cp.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

# Parameter
batch_size = 60
epochs = 200

# column order in CSV file
column_names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species']
feature_names = column_names[:-1]
label_name = column_names[-1]
class_names = ['Iris setosa', 'Iris versicolor', 'Iris virginica']

# # Datasets
# iris_datasetlist = Datasets.iris_datalist
# test_dataset = iris_datasetlist.get_dataset_at_index(1)
#
# iris_train = DataProcessing.CreateDatasets(train_dataset_url, 'iris_training.csv', label_name, batch_size,
#                                                   'Iris Train CSV Tensorflow', True, column_names)
# iris_dataset_train = iris_train.create_iris_url_dataset()
# features, labels = next(iter(iris_dataset_train))
#
# train_label_ids = DataProcessing.encode_label(labels)
# train_features = features

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


# Y = tf.keras.utils.to_categorical(train_label_ids, num_classes=3)

train_dataset = tf.data.experimental.make_csv_dataset(
    train_dataset_fp,
    batch_size,
    column_names=column_names,
    label_name=label_name,
    num_epochs=1)

train_features, train_labels = next(iter(train_dataset))

print(train_features, train_labels)

def pack_features_vector(features, labels):
  """Pack the features into a single array."""
  features = tf.stack(list(features.values()), axis=1)
  return features, labels

train_dataset = train_dataset.map(pack_features_vector)

test_dataset = tf.data.experimental.make_csv_dataset(
    test_fp,
    batch_size,
    column_names=column_names,
    label_name='species',
    num_epochs=1,
    shuffle=False)

test_dataset = test_dataset.map(pack_features_vector)

keras_model = tf.keras.Sequential([
  tf.keras.layers.Dense(10, activation=tf.nn.relu, input_shape=(4,)),
  tf.keras.layers.Dense(3, activation=tf.nn.softmax)
])

client_data = [train_dataset, train_dataset]

print("traindataset")
print(train_dataset)

# Clone the keras_model inside `create_tff_model()`, which TFF will
# call to produce a new copy of the model inside the graph that it will
# serialize. Note: we want to construct all the necessary objects we'll need
# _inside_ this method.
def create_tff_model():
  # TFF uses an `input_spec` so it knows the types and shapes
  # that your model expects.
  input_spec = train_dataset.element_spec
  return tff.learning.from_keras_model(
      keras_model,
      input_spec=input_spec,
      loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
      metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])


# This command builds all the TensorFlow graphs and serializes them:
fed_avg = tff.learning.build_federated_averaging_process(
    create_tff_model,
    client_optimizer_fn=lambda: tf.keras.optimizers.SGD(lr=0.5),
    server_optimizer_fn=lambda: tf.keras.optimizers.SGD(lr=1.0))

state = fed_avg.initialize()
state, metrics = fed_avg.next(state, [train_dataset])
train_metrics = metrics['train']
print('loss={l:.3f}, accuracy={a:.3f}'.format(
    l=train_metrics['loss'], a=train_metrics['accuracy']))

#keras_model.save_weights()


# # We concatenate the test Datasets for evaluation with Keras by creating a
# # Dataset of Datasets, and then identity flat mapping across all the examples.
# test_dataset = tf.data.Dataset.from_tensor_slices(
#     [data(client, test_data) for client in clients]).flat_map(lambda x: x)

NUM_ROUNDS = 5

# # The state of the FL server, containing the model and optimization state.
# state = fed_avg.initialize()
#
# # Load our pre-trained Keras model weights into the global model state.
# state = tff.learning.state_with_new_model_weights(
#     state,
#     trainable_weights=[v.numpy() for v in keras_model.trainable_weights],
#     non_trainable_weights=[
#         v.numpy() for v in keras_model.non_trainable_weights
#     ])
#
#
# def keras_evaluate(state, round_num):
#   # Take our global model weights and push them back into a Keras model to
#   # use its standard `.evaluate()` method.
#   keras_model.compile(
#       loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
#       metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])
#   state.model.assign_weights_to(keras_model)
#   loss, accuracy = keras_model.evaluate(train_dataset, steps=2, verbose=0)
#   print('\tEval: loss={l:.3f}, accuracy={a:.3f}'.format(l=loss, a=accuracy))
#
#
# for round_num in range(NUM_ROUNDS):
#   print('Round {r}'.format(r=round_num))
#   keras_evaluate(state, round_num)
#   state, metrics = fed_avg.next(state, train_dataset)
#   train_metrics = metrics['train']
#   print('\tTrain: loss={l:.3f}, accuracy={a:.3f}'.format(
#       l=train_metrics['loss'], a=train_metrics['accuracy']))
#
# print('Final evaluation')
# keras_evaluate(state, NUM_ROUNDS + 1)


