import collections
import numpy as np
from tensorflow.keras.datasets import mnist
from tensorflow import reshape, nest, config
from tensorflow.keras import losses, metrics, optimizers
import tensorflow_federated as tff
from matplotlib import pyplot as plt
from pathlib import Path

gpus = config.list_physical_devices('GPU')
if gpus:
  try:
    for gpu in gpus:
        config.experimental.set_memory_growth(gpu, True)
  except RuntimeError as e:
    print(e)


experiment_name = "mnist"
method = "tff_training"
client_lr = 1e-2
server_lr = 1e-2
split = 4
NUM_ROUNDS = 5
NUM_EPOCHS = 5
BATCH_SIZE = 20
PREFETCH_BUFFER = 10

this_dir = Path.cwd()
model_dir = this_dir / "saved_models" / experiment_name / method
output_dir = this_dir / "results" / experiment_name / method

if not model_dir.exists():
    model_dir.mkdir(parents=True)

if not output_dir.exists():
    output_dir.mkdir(parents=True)

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.astype(np.float32)
y_train = y_train.astype(np.int32)
x_test = x_test.astype(np.float32).reshape(10000, 28, 28, 1)
y_test = y_test.astype(np.int32).reshape(10000, 1)

total_image_count = len(x_train)
image_per_set = int(np.floor(total_image_count/split))

client_train_dataset = collections.OrderedDict()
for i in range(1, split+1):
    client_name = "client_" + str(i)
    start = image_per_set * (i-1)
    end = image_per_set * i

    print(f"Adding data from {start} to {end} for client : {client_name}")
    data = collections.OrderedDict((('label', y_train[start:end]), ('pixels', x_train[start:end])))
    client_train_dataset[client_name] = data

for k, v in client_train_dataset.items():
    print(k, v)

train_dataset = tff.simulation.FromTensorSlicesClientData(client_train_dataset)

sample_dataset = train_dataset.create_tf_dataset_for_client(train_dataset.client_ids[0])
sample_element = next(iter(sample_dataset))

SHUFFLE_BUFFER = image_per_set

def preprocess(dataset):

  def batch_format_fn(element):
    """Flatten a batch `pixels` and return the features as an `OrderedDict`."""

    return collections.OrderedDict(
        x=reshape(element['pixels'], [-1, 28, 28, 1]),
        y=reshape(element['label'], [-1, 1]))

  return dataset.repeat(NUM_EPOCHS).shuffle(SHUFFLE_BUFFER).batch(
      BATCH_SIZE).map(batch_format_fn).prefetch(PREFETCH_BUFFER)

preprocessed_sample_dataset = preprocess(sample_dataset)
sample_batch = nest.map_structure(lambda x: x.numpy(), next(iter(preprocessed_sample_dataset)))


def make_federated_data(client_data, client_ids):
    return [preprocess(client_data.create_tf_dataset_for_client(x)) for x in client_ids]

federated_train_data = make_federated_data(train_dataset, train_dataset.client_ids)

print('Number of client datasets: {l}'.format(l=len(federated_train_data)))
print('First dataset: {d}'.format(d=federated_train_data[0]))

