import os
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_federated as tff

print("TensorFlow version: {}".format(tf.__version__))
print("Eager execution: {}".format(tf.executing_eagerly()))

train_dataset_url = "https://storage.googleapis.com/download.tensorflow.org/data/iris_training.csv"

train_dataset_fp = tf.keras.utils.get_file(fname=os.path.basename(train_dataset_url),
                                           origin=train_dataset_url)

print("Local copy of the dataset file: {}".format(train_dataset_fp))

test_url = "https://storage.googleapis.com/download.tensorflow.org/data/iris_test.csv"

test_fp = tf.keras.utils.get_file(fname=os.path.basename(test_url),
                                  origin=test_url)
# Random generated CSV File
dataset_file01 = 'C:\\Users\\Stefan\\PycharmProjects\\thesis\\federated_learning_iris\\iris_random01.csv'

# column order in CSV file
column_names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species']

feature_names = column_names[:-1]
label_name = column_names[-1]

print("Features: {}".format(feature_names))
print("Label: {}".format(label_name))

class_names = ['Iris setosa', 'Iris versicolor', 'Iris virginica']

batch_size = 32

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

# Dataset for Random CSV
train_dataset01 = tf.data.experimental.make_csv_dataset(
    dataset_file01,
    batch_size,
    #column_names=column_names,
    label_name=label_name,
    num_epochs=1)

# Train Dataset Features
features, labels = next(iter(train_dataset))

print(features)

plt.scatter(features['petal_length'],
            features['sepal_length'],
            c=labels,
            cmap='viridis')

plt.xlabel("Petal length")
plt.ylabel("Sepal length")
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
plt.show()


def pack_features_vector(features, labels):
  """Pack the features into a single array."""
  features = tf.stack(list(features.values()), axis=1)
  return features, labels


#Pack Datasets
train_dataset = train_dataset.map(pack_features_vector)
train_dataset01 = train_dataset01.map(pack_features_vector)  # Random CSV Dataset
test_dataset = test_dataset.map(pack_features_vector)

features, labels = next(iter(train_dataset))

train_datasets = [train_dataset, train_dataset01]
test_datasets = [test_dataset, test_dataset]

print(features[:5])


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
      input_spec=train_dataset.element_spec,
      loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
      metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])


iterative_process = tff.learning.build_federated_averaging_process(
    model_fn,
    client_optimizer_fn=lambda: tf.keras.optimizers.SGD(learning_rate=0.01),
    server_optimizer_fn=lambda: tf.keras.optimizers.SGD(learning_rate=1.0))

str(iterative_process.initialize.type_signature)

# Construct Server State
state = iterative_process.initialize()


NUM_ROUNDS = 201
for round_num in range(1, NUM_ROUNDS):
  state, metrics = iterative_process.next(state, train_datasets)
  print('round {:2d}, metrics={}'.format(round_num, metrics))


# Evaluation
evaluation = tff.learning.build_federated_evaluation(model_fn)

train_metrics = evaluation(state.model, test_datasets)

print(str(train_metrics))

