import nest_asyncio
from develop.IrisClientData import IrisClientData
from outsourcing.CustomMetrics import *
import collections
import tensorflow as tf
import tensorflow_federated as tff
import pathlib as path
from pathlib import Path
from outsourcing.Datasets import IrisDatasets
import os
nest_asyncio.apply()

print("TensorFlow version: {}".format(tf.__version__))
print("Eager execution: {}".format(tf.executing_eagerly()))

# Parameter
# Training iterations
epochs = 5
# Number of classes
num_classes = 3

# Root Path
root_project_path = path.Path.cwd().parent
print(root_project_path)
path = Path(root_project_path / 'storage/iris_model/federated_learning/')
filepath = Path(root_project_path / 'datasets/iris_classification/iris_training.csv')

# Reads the data from the iris file
dataset = IrisClientData(filepath, 1)

print("Number of clients which produced data: " + str(len(dataset.client_ids)))

# Creates a single dataset for a given client_id
selected_client_id = dataset.client_ids[0]
example_dataset = dataset.create_tf_dataset_for_client(selected_client_id)

print("Dataset size for single client: " + str(len(example_dataset)))
[print(element) for element in example_dataset]
print('-' * 100)

# HERE STARTS THE FEDERATED LEARNING

NUM_CLIENTS = 8
NUM_EPOCHS = 10
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

split01_dataset = IrisDatasets.split01_dataset
split02_dataset = IrisDatasets.split02_dataset
split03_dataset = IrisDatasets.split03_dataset

split_dataset = [split03_dataset, split02_dataset, split01_dataset]

print('Number of client datasets: {l}'.format(l=len(federated_train_data)))
print('First dataset: {d}'.format(d=federated_train_data[0]))


def create_keras_model():
    return tf.keras.models.Sequential([
        tf.keras.layers.Input(shape=(4,)),
        tf.keras.layers.Dense(10, kernel_initializer='zeros'),
        tf.keras.layers.Softmax(),
    ])


def model_fn():
    # We _must_ create a new model here, and _not_ capture it from an external
    # scope. TFF will call this within different graph contexts.
    keras_model = create_keras_model()
    return tff.learning.from_keras_model(
        keras_model,
        input_spec=split03_dataset.element_spec,
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])


fed_avg = tff.learning.build_federated_averaging_process(
    model_fn=model_fn,
    client_optimizer_fn=lambda: tf.keras.optimizers.SGD(learning_rate=0.01),  # for each Client
    server_optimizer_fn=lambda: tf.keras.optimizers.SGD(learning_rate=1.0))  # for Global model

# Keep results for plotting
train_loss_results = []
train_accuracy_results = []

# Create the checkpoint Manager
# ckpt_manager = tff.simulation.FileCheckpointManager(root_dir=path)
# # Save checkpoint for round N
# ckpt_manager.save_checkpoint(state, round_num=NUM_ROUNDS)
server_state = fed_avg.initialize()

# Start Federated Learning process
for round_num in range(1, epochs):
    server_state, metrics = fed_avg.next(server_state, split_dataset)
    print('round {:2d}, metrics={}'.format(round_num, metrics))
    # Track progress
    train_metrics = metrics['train']
    epoch_loss_avg = train_metrics['loss']
    epoch_accuracy = train_metrics['sparse_categorical_accuracy']
    # End epoch
    train_loss_results.append(epoch_loss_avg)
    train_accuracy_results.append(epoch_accuracy)



fig, axes = plt.subplots(2, sharex=True, figsize=(12, 8))
fig.suptitle('Training Metrics')

axes[0].set_ylabel("Loss", fontsize=14)
axes[0].plot(train_loss_results)

axes[1].set_ylabel("Accuracy", fontsize=14)
axes[1].set_xlabel("Epoch", fontsize=14)
axes[1].plot(train_accuracy_results)
plt.show()

# Load our pre-trained Keras model weights into the global model state.
state = tff.learning.state_with_new_model_weights(
    server_state,
    trainable_weights=[v.numpy() for v in keras_model.trainable_weights],
    non_trainable_weights=[
        v.numpy() for v in keras_model.non_trainable_weights
    ])

#
# # Evaluation
# evaluation = tff.learning.build_federated_evaluation(model_fn)
# train_metrics = evaluation(server_state.model, federated_train_data)
# print(train_metrics)
#
#
# # Here the unused client:ids are used
# sample_clients = dataset.client_ids[NUM_CLIENTS:]
# federated_test_data = make_federated_data(dataset, sample_clients)
# test_metrics = evaluation(server_state.model, federated_test_data)
# print(test_metrics)
