import os
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
import tensorflow_federated as tff
from datetime import datetime
import shutil

print("TensorFlow version: {}".format(tf.__version__))
print("Eager execution: {}".format(tf.executing_eagerly()))

# Tensorboard Command
# tensorboard --logdir C:\\Users\\Stefan\\PycharmProjects\\thesis\\logs\\

# Filepaths
logfile_path = 'C:\\Users\\Stefan\\PycharmProjects\\thesis\\logs\\'
dataset_path_local = 'C:\\Users\\Stefan\\PycharmProjects\\thesis\\datasets\\iris_classification\\'

# Remove Graph Log files and directories
folder = str(logfile_path + 'graph\\')
for filename in os.listdir(logfile_path):
    file_path = os.path.join(logfile_path, filename)
    try:
        if os.path.isfile(file_path) or os.path.islink(file_path):
            os.remove(file_path)
    except Exception as e:
        print('Failed to delete %s. Reason: %s' % (file_path, e))

for filename in os.listdir(folder):
    file_path = os.path.join(folder, filename)
    try:
        if os.path.isfile(file_path) or os.path.islink(file_path):
            os.unlink(file_path)
        elif os.path.isdir(file_path):
            shutil.rmtree(file_path)
    except Exception as e:
        print('Failed to delete %s. Reason: %s' % (file_path, e))

print('all logs removed')

# Tensorboard
now = datetime.now()
# Define the Keras TensorBoard callback.
logdir = logfile_path +"\\graph\\" + datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = keras.callbacks.TensorBoard(log_dir=logdir)

# Parameter
batch_size = 32
epochs = 5

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


def create_train_dataset(file_path, filename):
    train_dataset = tf.data.experimental.make_csv_dataset(
    file_path + filename,
    batch_size,
    #column_names=column_names,
    label_name=label_name,
    num_epochs=1)
    return train_dataset


def pack_features_vector(features, labels):
  """Pack the features into a single array."""
  features = tf.stack(list(features.values()), axis=1)
  return features, labels


# CVS from Tensorflow
train_dataset_iris_tensorflow = tf.data.experimental.make_csv_dataset(
    train_dataset_fp,
    batch_size,
    column_names=column_names,
    label_name=label_name,
    num_epochs=1)

# CSV from Github
train_dataset_iris_github = tf.data.experimental.make_csv_dataset(
    train_dataset_fp,
    batch_size,
    column_names=column_names,
    label_name=label_name,
    num_epochs=1)


# CVS from Tensorflow
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

# Graphs
# Train Dataset Features
features, labels = next(iter(train_dataset_iris_tensorflow))

print(features)

plt.scatter(features['petal_length'],
            features['sepal_length'],
            c=labels,
            cmap='viridis')

plt.xlabel("Petal length")
plt.ylabel("Sepal length")
plt.title("Train Dataset")
plt.show()

# Train Dataset 01 Features
features, labels = next(iter(train_dataset01))

print(features)

plt.scatter(features['petal_length'],
            features['sepal_length'],
            c=labels,
            cmap='viridis')

plt.xlabel("Petal length")
plt.ylabel("Sepal length")
plt.title("Generated CSV File")
plt.show()

features, labels = next(iter(train_dataset02))

# Train Dataset02 Features
plt.scatter(features['petal_length'],
            features['sepal_length'],
            c=labels,
            cmap='viridis')

plt.xlabel("Petal length")
plt.ylabel("Sepal length")
plt.title("Train Dataset 02")
plt.show()

features, labels = next(iter(dataset_setosa))

# Dataset Setosa Features
plt.scatter(features['petal_length'],
            features['sepal_length'],
            c=labels,
            cmap='viridis')

plt.xlabel("Petal length")
plt.ylabel("Sepal length")
plt.title("Setosa")
plt.show()

features, labels = next(iter(dataset_versicolor))

# Dataset Versicolor Features
plt.scatter(features['petal_length'],
            features['sepal_length'],
            c=labels,
            cmap='viridis')

plt.xlabel("Petal length")
plt.ylabel("Sepal length")
plt.title("Versicolor")
plt.show()


features, labels = next(iter(dataset_virginica))

# Dataset Virginica Features
plt.scatter(features['petal_length'],
            features['sepal_length'],
            c=labels,
            cmap='viridis')

plt.xlabel("Petal length")
plt.ylabel("Sepal length")
plt.title("Virginica")
plt.show()

# Pack Datasets
train_dataset_iris_tensorflow = train_dataset_iris_tensorflow.map(pack_features_vector)
train_dataset_iris_github = train_dataset_iris_github.map(pack_features_vector)
train_dataset01 = train_dataset01.map(pack_features_vector)  # Random CSV Dataset
train_dataset02 = train_dataset02.map(pack_features_vector)  # Random CSV Dataset
test_dataset = test_dataset.map(pack_features_vector)

dataset_versicolor = dataset_versicolor.map(pack_features_vector)
dataset_virginica = dataset_virginica.map(pack_features_vector)
dataset_setosa = dataset_setosa.map(pack_features_vector)

features, labels = next(iter(train_dataset_iris_tensorflow))
print(features[:5])

# Create Lists with Dataset per Client
sorted_datasets = [dataset_setosa, dataset_virginica, dataset_versicolor]
train_datasets = [train_dataset_iris_tensorflow, train_dataset_iris_github]
test_datasets = [test_dataset, test_dataset]


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
      input_spec=train_dataset_iris_tensorflow.element_spec,
      loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
      metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])


iterative_process = tff.learning.build_federated_averaging_process(
    model_fn,
    client_optimizer_fn=lambda: tf.keras.optimizers.SGD(learning_rate=0.01),  # for each Client
    server_optimizer_fn=lambda: tf.keras.optimizers.SGD(learning_rate=1.0))  # for Global model

str(iterative_process.initialize.type_signature)

# Construct Server State
state = iterative_process.initialize()

# Tensorboard Writer
summary_writer = tf.summary.create_file_writer(logfile_path)

for round_num in range(epochs):
  state, metrics = iterative_process.next(state, train_datasets)
  with summary_writer.as_default():
      for name, metric in metrics['train'].items():
          tf.summary.scalar(name, metric, step=round_num)
  train_metrics = metrics['train']
  print('loss={l:.3f}, accuracy={a:.3f}'.format(
        l=train_metrics['loss'], a=train_metrics['sparse_categorical_accuracy']))

# Evaluation
# Model Evaluation
test_accuracy = tf.keras.metrics.Accuracy()

evaluation = tff.learning.build_federated_evaluation(model_fn)

train_metrics = evaluation(state.model, test_datasets)

print(str(train_metrics))

print(str(evaluation.type_signature))
