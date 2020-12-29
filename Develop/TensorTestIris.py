import nest_asyncio

from Develop.IrisClientData import IrisClientData
from Develop.IrisModel import IrisModel

nest_asyncio.apply()

import collections
import numpy as np
import tensorflow as tf
import tensorflow_federated as tff

np.random.seed(0)

# Reads the data from the iris file
dataset = IrisClientData("./iris.data")

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

print('Number of client Datasets: {l}'.format(l=len(federated_train_data)))
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
        input_spec=preprocessed_dataset.element_spec,
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])


iterative_process = tff.learning.build_federated_averaging_process(
    IrisModel,
    client_optimizer_fn=lambda: tf.keras.optimizers.SGD(learning_rate=0.05))

print(str(iterative_process.initialize.type_signature))

state = iterative_process.initialize()

NUM_ROUNDS = 11
for round_num in range(1, NUM_ROUNDS):
    state, metrics = iterative_process.next(state, federated_train_data)
    print(list(metrics['train']['value']))
    print('round {:2d}, metrics={}'.format(round_num, metrics))

print('-' * 100)

# HINT: Note the numbers look marginally better than what was reported by the last round of training above. By
# convention, the training metrics reported by the iterative training process generally reflect the performance of
# the model at the beginning of the training round, so the evaluation metrics will always be one step ahead.

evaluation = tff.learning.build_federated_evaluation(IrisModel)
train_metrics = evaluation(state.model, federated_train_data)
print(train_metrics)

# Here the unused client:ids are used
sample_clients = dataset.client_ids[NUM_CLIENTS:]
federated_test_data = make_federated_data(dataset, sample_clients)
test_metrics = evaluation(state.model, federated_test_data)
print(test_metrics)
