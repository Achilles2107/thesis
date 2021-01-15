import tensorflow as tf
import tensorflow_federated as tff
import pathlib as path
from pathlib import Path
import os

print("TensorFlow version: {}".format(tf.__version__))
print("Eager execution: {}".format(tf.executing_eagerly()))

# Filepaths
cwd = path.Path.cwd().parent
path = Path(cwd / 'storage/iris_model/fed/')

# Path for saving weights
checkpoint_path = path / "cp.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

# Load simulation data.
source, _ = tff.simulation.datasets.emnist.load_data()

print(type(source))


def client_data(n):
  return source.create_tf_dataset_for_client(source.client_ids[n]).map(
      lambda e: (tf.reshape(e['pixels'], [-1]), e['label'])
  ).repeat(10).batch(20)


# Pick a subset of client devices to participate in training.
train_data = [client_data(n) for n in range(3)]  # [[12312412124][2352345]]
# print(train_data)
# for i in train_data:
print('Elemet Spec')
print(type(train_data[0].element_spec))
print(train_data[0].element_spec)
print('Elemet Spec Ende')


# Wrap a Keras model for use with TFF.
def model_fn():
  model = tf.keras.models.Sequential([
      tf.keras.layers.Dense(10, tf.nn.softmax, input_shape=(784,),
                            kernel_initializer='zeros')
  ])
  return tff.learning.from_keras_model(
      model,
      input_spec=train_data[0].element_spec,
      loss=tf.keras.losses.SparseCategoricalCrossentropy(),
      metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])


# Simulate a few rounds of training with the selected client devices.
trainer = tff.learning.build_federated_averaging_process(
  model_fn,
  client_optimizer_fn=lambda: tf.keras.optimizers.SGD(0.1))
state = trainer.initialize()
for _ in range(5):
  state, metrics = trainer.next(state, train_data)
  print (metrics)
