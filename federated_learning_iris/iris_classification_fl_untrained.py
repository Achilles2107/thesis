import os
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
import tensorflow_federated as tff
from datetime import datetime

print("TensorFlow version: {}".format(tf.__version__))
print("Eager execution: {}".format(tf.executing_eagerly()))

# Tensorboard Command for CMD or Powershell
# tensorboard --logdir C:\\Users\\Stefan\\PycharmProjects\\thesis\\logs\\

# Filepaths
logfile_path = 'C:\\Users\\Stefan\\PycharmProjects\\thesis\\logs\\'
dataset_path_local = 'C:\\Users\\Stefan\\PycharmProjects\\thesis\\datasets\\iris_classification\\'
split_data_path = 'C:\\Users\\Stefan\\PycharmProjects\\thesis\\datasets\\iris_classification\\split\\'
# Path to CSV from GITHUB
github_dataset = dataset_path_local + 'iris_training02.csv'
train_dataset_url = "https://storage.googleapis.com/download.tensorflow.org/data/iris_training.csv"

train_dataset_fp = tf.keras.utils.get_file(fname=os.path.basename(train_dataset_url),
                                           origin=train_dataset_url)

print("Local copy of the dataset file: {}".format(train_dataset_fp))

test_url = "https://storage.googleapis.com/download.tensorflow.org/data/iris_test.csv"

test_fp = tf.keras.utils.get_file(fname=os.path.basename(test_url),
                                  origin=test_url)

# Tensorboard
now = datetime.now()
# Define the Keras TensorBoard callback.
logdir = logfile_path + "\\graph\\" + datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = keras.callbacks.TensorBoard(log_dir=logdir)

# Parameter
batch_size = 30
epochs = 200

# column order in CSV file
column_names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species']

feature_names = column_names[:-1]
label_name = column_names[-1]

print("Features: {}".format(feature_names))
print("Label: {}".format(label_name))

class_names = ['Iris setosa', 'Iris versicolor', 'Iris virginica']


# For Datasets with Feature and Label Names in the actual File
def create_train_dataset(file_path, filename):
    train_dataset = tf.data.experimental.make_csv_dataset(
    file_path + filename,
    batch_size,
    #column_names=column_names,
    label_name=label_name,
    num_epochs=1)
    return train_dataset


# For Datasets without Feature and Label Names in the actual File
def create_train_dataset_with_col_name(file_path, filename):
    train_dataset = tf.data.experimental.make_csv_dataset(
    file_path + filename,
    batch_size,
    column_names=column_names,
    label_name=label_name,
    num_epochs=1)
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


# Traindata CSV from Tensorflow
train_dataset_iris_tensorflow = tf.data.experimental.make_csv_dataset(
    train_dataset_fp,
    batch_size,
    column_names=column_names,
    label_name=label_name,
    num_epochs=1)

# Traindata CSV from Github
train_dataset_iris_github = tf.data.experimental.make_csv_dataset(
    github_dataset,
    batch_size,
    column_names=column_names,
    label_name=label_name,
    num_epochs=1)

# Testdata CSV from Tensorflow
test_dataset = tf.data.experimental.make_csv_dataset(
    test_fp,
    batch_size,
    column_names=column_names,
    label_name='species',
    num_epochs=1,
    shuffle=False)

# Dataset for Random CSV
train_dataset01 = create_train_dataset(dataset_path_local, 'iris_random01.csv')
train_dataset02 = create_train_dataset(dataset_path_local, 'iris_random02.csv')

# Datasets Sorted by Species
dataset_versicolor = create_train_dataset(dataset_path_local, 'iris_versicolor.csv')
dataset_setosa = create_train_dataset(dataset_path_local, 'iris_setosa.csv')
dataset_virginica = create_train_dataset(dataset_path_local, 'iris_virginica.csv')

# Splitted Datasets
split_dataset01 = create_train_dataset_with_col_name(split_data_path, '1.csv')
split_dataset02 = create_train_dataset_with_col_name(split_data_path, '2.csv')
split_dataset03 = create_train_dataset_with_col_name(split_data_path, '3.csv')


# Graphs for Dataset Features
# Tensorflow Dataset
make_graph(train_dataset_iris_tensorflow, 'Tensorflow Iris Dataset')
# Traindataset GITHUB CSV
make_graph(train_dataset_iris_github, 'Train Dataset GITHUB')
# Split Dataset 01
make_graph(split_dataset01, "Split Dataset 01")
make_graph(split_dataset02, "Split Dataset 02")
make_graph(split_dataset03, "Split Dataset 03")
# Sorted Datasets
make_graph(dataset_setosa, "Dataset Setosa")
make_graph(dataset_versicolor, "Dataset Versicolor")
make_graph(dataset_virginica, "Dataset Virgincia")

# Pack Datasets
train_dataset_iris_tensorflow = train_dataset_iris_tensorflow.map(pack_features_vector)
train_dataset_iris_github = train_dataset_iris_github.map(pack_features_vector)
train_dataset01 = train_dataset01.map(pack_features_vector)  # Random CSV Dataset
train_dataset02 = train_dataset02.map(pack_features_vector)  # Random CSV Dataset
test_dataset = test_dataset.map(pack_features_vector)

# Sorted Datasets
dataset_versicolor = dataset_versicolor.map(pack_features_vector)
dataset_virginica = dataset_virginica.map(pack_features_vector)
dataset_setosa = dataset_setosa.map(pack_features_vector)

# Split Datasets
split_dataset01 = split_dataset01.map(pack_features_vector)
split_dataset02 = split_dataset02.map(pack_features_vector)
split_dataset03 = split_dataset03.map(pack_features_vector)

# Create Lists with Dataset per Client
sorted_datasets = [dataset_setosa, dataset_virginica, dataset_versicolor]
train_datasets = [train_dataset_iris_tensorflow, train_dataset01]
test_datasets = [test_dataset, test_dataset]
split_datasets = [split_dataset01, split_dataset02, split_dataset03]


# New model creation needed  TFF wont accept anything out of Scope e.g. Tensors
def create_keras_model():
  return tf.keras.models.Sequential([
      tf.keras.layers.Dense(10, activation=tf.nn.relu, input_shape=(4,)),  # input shape required
      tf.keras.layers.Dense(10, activation=tf.nn.relu),
      tf.keras.layers.Dense(3)
  ])


def model_fn():
  # We _must_ create a new model here, and _not_ capture it from an external
  # scope. TFF will call this within different graph contexts.
  keras_model = create_keras_model()
  return tff.learning.from_keras_model(
      keras_model,
      # Input information on what shape the input data will have
      # Must be from type tf.Type or tf.TensorSpec
      input_spec=dataset_setosa.element_spec,
      loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
      metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])


iterative_process = tff.learning.build_federated_averaging_process(
    model_fn,
    client_optimizer_fn=lambda: tf.keras.optimizers.SGD(learning_rate=0.01),  # for each Client
    server_optimizer_fn=lambda: tf.keras.optimizers.SGD(learning_rate=1.0))  # for Global model

# Tensorboard Writer
summary_writer = tf.summary.create_file_writer(logfile_path)

tf.summary.trace_on(graph=True, profiler=True)
# Construct Server State
state = iterative_process.initialize()
with summary_writer.as_default():
  tf.summary.trace_export(
      name="Federated iterative process init",
      step=0,
      profiler_outdir=logfile_path)

# Start Federated Learning process
for round_num in range(1, epochs):
  state, metrics = iterative_process.next(state, sorted_datasets)
  with summary_writer.as_default():
      for name, metric in metrics['train'].items():
          tf.summary.scalar(name, metric, step=round_num)
  train_metrics = metrics['train']
  print('round ' + str(round_num) + ' loss={l:.3f}, accuracy={a:.3f}'.format(
        l=train_metrics['loss'], a=train_metrics['sparse_categorical_accuracy']))

# Print content of metrics['train']
# for name, metric in metrics['train'].items():
#     print(name, metric)

# Evaluation
# Model Evaluation
test_accuracy = tf.keras.metrics.Accuracy()

evaluation = tff.learning.build_federated_evaluation(model_fn)

train_metrics = evaluation(state.model, test_datasets)

print(str(train_metrics))

