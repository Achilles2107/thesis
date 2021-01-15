import nest_asyncio
from develop.IrisClientData import IrisClientData
# from develop.iris_model import iris_model
from develop.IrisModel02 import IrisModel
from outsourcing.Datasets import IrisDatasets
from outsourcing.CustomMetrics import *
import collections
import numpy as np
import tensorflow as tf
import tensorflow_federated as tff
import pathlib as path
import pandas as pd
from pathlib import Path
from outsourcing.Datasets import IrisDatasets
from outsourcing.DataProcessing import *
nest_asyncio.apply()

np.random.seed(0)

# Reads the data from the iris file
dataset = IrisClientData("./iris.data")
cwd = path.Path.cwd().parent
path = Path(cwd / 'storage/iris_fed_model')

print("Number of clients which produced data: " + str(len(dataset.client_ids)))

# Creates a single dataset for a given client_id
selected_client_id = dataset.client_ids[0]
example_dataset = dataset.create_tf_dataset_for_client(selected_client_id)

print("Dataset size for single client: " + str(len(example_dataset)))
[print(element) for element in example_dataset]
print('-' * 100)

# HERE STARTS THE FEDERATED LEARNING

NUM_CLIENTS = 8
NUM_EPOCHS = 5
BATCH_SIZE = 20
SHUFFLE_BUFFER = 100
PREFETCH_BUFFER = 10


def preprocess(dataset_to_preprocess):
    # Here is not much done, because the iris data is already in a perfect shape
    def batch_format_fn(element, label):
        return collections.OrderedDict(
            x=element,
            y=label)

    return dataset_to_preprocess.repeat(NUM_EPOCHS).shuffle(SHUFFLE_BUFFER).batch(
        BATCH_SIZE).map(batch_format_fn).prefetch(PREFETCH_BUFFER)


# This preprocessed dataset is used for the element_spec of the model_fn
preprocessed_dataset = preprocess(example_dataset)


def make_federated_data(client_data, client_ids):
    return [
        preprocess(client_data.create_tf_dataset_for_client(x))
        for x in client_ids
    ]


# Select the clients which are used for the test dataset
sample_clients = dataset.client_ids[0:NUM_CLIENTS]
federated_train_data = make_federated_data(dataset, sample_clients)

print('Number of client datasets: {l}'.format(l=len(federated_train_data)))
print('First dataset: {d}'.format(d=federated_train_data[0]))


def create_keras_model():
    return tf.keras.Sequential([
          tf.keras.layers.Dense(10, activation=tf.nn.relu, input_shape=(4,)),
          tf.keras.layers.Dense(3, activation=tf.nn.softmax)
    ])


def model_fn():
    # We _must_ create a new model here, and _not_ capture it from an external
    # scope. TFF will call this within different graph contexts.
    keras_model = create_keras_model()
    return tff.learning.from_keras_model(
        keras_model,
        input_spec=preprocessed_dataset.element_spec,
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])

iterative_process = tff.learning.build_federated_averaging_process(
    model_fn,
    client_optimizer_fn=lambda: tf.keras.optimizers.SGD(learning_rate=0.05))

print(str(iterative_process.initialize.type_signature))

server_state = iterative_process.initialize()


def evaluate(server_state):
  keras_model = create_keras_model()
  keras_model.compile(
      loss=tf.keras.losses.SparseCategoricalCrossentropy(),
      metrics=[tf.keras.metrics.SparseCategoricalAccuracy(),
               recall, specificity]
  )
  keras_model.set_weights(server_state)
  keras_model.evaluate(federated_train_data)


NUM_ROUNDS = 11
for round_num in range(1, NUM_ROUNDS):
    server_state, metrics = iterative_process.next(server_state, federated_train_data)
    print('round {:2d}, metrics={}'.format(round_num, metrics))


print('-' * 100)

# server_state.model.save_weights(path / "/keras_model/", overwrite=True, include_optimizer=False)

# HINT: Note the numbers look marginally better than what was reported by the last round of training above. By
# convention, the training metrics reported by the iterative training process generally reflect the performance of
# the model at the beginning of the training round, so the evaluation metrics will always be one step ahead.

evaluation = tff.learning.build_federated_evaluation(model_fn)
train_metrics = evaluation(server_state.model, federated_train_data)
print(train_metrics)

# Here the unused client:ids are used
sample_clients = dataset.client_ids[NUM_CLIENTS:]
federated_test_data = make_federated_data(dataset, sample_clients)
test_metrics = evaluation(server_state.model, federated_test_data)
print(test_metrics)

# Parameter
# Training iterations
epochs = 200
# Number of classes
num_classes = 3

# datasets
# Get datasets from IrisDataset class
train_dataset = IrisDatasets.train_dataset
print(train_dataset)

test_dataset = IrisDatasets.test_dataset
print(test_dataset)

# Print datasets
print("Train Dataset: \n")
print(get_features_labels(train_dataset))
print("Test Dataset: \n")
print(get_features_labels(train_dataset))

# Decode and put labels in a binary matrix and get features
# Binary matrix is need as we are classifying to more than two classes
train_features, train_labels = decode_label(train_dataset, num_classes)
test_features, test_labels = decode_label(test_dataset, num_classes)

METRICS = [
    recall,
    specificity,
    precision,
    ["accuracy"]
]

model = create_keras_model()

weights = model.get_weights()

print("Weights pre assign: \n", weights)

model.compile(
              optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
              loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
              metrics=METRICS
             )

server_state.model.assign_weights_to(model)

weights = model.get_weights()

print("Weights after assign: \n", weights)

# Display the model's architecture
model.summary()

# Start training and define optional validation dataset
history = model.fit(
                    train_features, train_labels,
                    validation_data=(test_features, test_labels),
                    epochs=epochs
                   )

# Get mean training accuracy and plot metrics
mean_training_accuracy(history, "accuracy")
subplot_metrics(history, "accuracy", "precision")
subplot_metrics(history, "recall", "specificity")
plot_metric(history, "loss")

# Model Evaluation
test_accuracy = tf.keras.metrics.Accuracy()

predictions = np.argmax(model.predict(train_features), axis=-1)
print(predictions)

for (x, y) in test_dataset:
  # training=False is needed only if there are layers with different
  # behavior during training versus inference (e.g. Dropout).
  logits = model(x, training=False)
  prediction = tf.argmax(logits, axis=1, output_type=tf.int32)
  test_accuracy(prediction, y)

print("Test set accuracy: {:.3%}".format(test_accuracy.result()))

train_acc = model.evaluate(train_features, train_labels)
test_acc = model.evaluate(test_features, test_labels)

